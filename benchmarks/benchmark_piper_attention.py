"""Benchmark Piper Attention against SageAttention2++, SDPA, and controls.

Piper providers separate integer preprocessing from the fused attention launch,
so ``prepared_execution`` is the hot recurrence while ``operator_end_to_end``
includes K/V means, optional row ordering, quantization, and the mean-restoring
epilogue. Sage and SDPA providers retain their ordinary operator boundary.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import torch
from lib import (
    AttentionConfig,
    AttentionShape,
    BenchmarkProvider,
    BenchmarkRecord,
    EnvironmentInfo,
    OutputTarget,
    TritonCompilerRecord,
    add_compiler_inspection_arguments,
    add_output_arguments,
    add_profile_arguments,
    capture_environment,
    format_compiler_report,
    inspect_provider,
    measure_provider,
    measure_quality,
    output_target,
    profile_provider,
    write_records,
)

from piper_kernels._triton.targets import (
    supports_fp8_fp16_mma,
    supports_uint8_int8_mma,
)
from piper_kernels.attention import sage_attention_2pp
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_backend
from piper_kernels.attention.piper import triton as piper_backend
from piper_kernels.attention.piper.dispatch import _default_center_value

_CANONICAL_VERSION = "2.2.0"
_CANONICAL_REVISION = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
_PIPER = "piper"
_PIPER_CENTERED = "piper-centered"
_PIPER_UNCENTERED = "piper-uncentered"
_PIPER_AFFINE = "piper-affine"
_PURE_TRITON_SAGE2PP = "pure-triton-sage2pp"
_SDPA = "pytorch-sdpa"
_CANONICAL_SAGE2PP = "canonical-cuda-sage2pp"
_CANONICAL_SAGE2 = "canonical-cuda-sage2"
_PROVIDER_NAMES = (
    _PIPER,
    _PIPER_CENTERED,
    _PIPER_UNCENTERED,
    _PIPER_AFFINE,
    _PURE_TRITON_SAGE2PP,
    _SDPA,
    _CANONICAL_SAGE2PP,
    _CANONICAL_SAGE2,
)
_PIPER_PROVIDERS = (_PIPER, _PIPER_CENTERED, _PIPER_UNCENTERED, _PIPER_AFFINE)
_FP8_SAGE_PROVIDERS = (_PURE_TRITON_SAGE2PP, _CANONICAL_SAGE2PP, _CANONICAL_SAGE2)

type AttentionInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
type CanonicalSage = Callable[..., torch.Tensor]
type AttentionProvider = BenchmarkProvider[object, torch.Tensor]


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _canonical_qk_granularity(capability: tuple[int, int]) -> str:
    return "per_warp" if capability[0] == 12 else "per_thread"


def _load_canonical(capability: tuple[int, int]) -> CanonicalSage:
    try:
        module = importlib.import_module("sageattention")
    except (ImportError, OSError) as error:
        architecture = f"{capability[0]}.{capability[1]}"
        raise SystemExit(
            "Canonical SageAttention is unavailable. Build the pinned benchmark "
            "dependency with:\n"
            f"  TORCH_CUDA_ARCH_LIST={architecture} uv sync --group benchmark"
        ) from error
    return cast(CanonicalSage, module.sageattn_qk_int8_pv_fp8_cuda)


def _sdpa(
    inputs: AttentionInputs,
    *,
    scale: float | None,
    is_causal: bool,
) -> torch.Tensor:
    query, key, value = inputs
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        is_causal=is_causal,
    )


def _piper_jit_functions(
    capability: tuple[int, int],
    *,
    sort_value_rows: bool,
) -> dict[str, object]:
    qk_kernels = (
        {
            "quantize-query-per-warp": qk_backend.quantize_query_per_warp_kernel,
            "quantize-key-per-block": (
                piper_backend._quantize_ordered_key_per_block_kernel
                if sort_value_rows
                else qk_backend.quantize_key_per_block_kernel
            ),
        }
        if capability[0] == 12
        else {
            "quantize-query-per-thread": qk_backend.quantize_query_per_thread_kernel,
            "quantize-key-per-thread": qk_backend.quantize_key_per_thread_kernel,
        }
    )
    kernels = {
        "kv-mean-partial": piper_backend._kv_mean_partial_kernel,
        "kv-mean-finish": piper_backend._kv_mean_finalize_kernel,
        **qk_kernels,
        "quantize-value-per-key": piper_backend._quantize_value_per_key_kernel,
        "attention": piper_backend._piper_attention_kernel,
    }
    if sort_value_rows:
        kernels["centered-value-row-range"] = piper_backend._centered_value_row_range_kernel
    return kernels


def _make_piper_provider(
    name: str,
    inputs: AttentionInputs,
    *,
    config: AttentionConfig,
    capability: tuple[int, int],
    center_value: bool,
    native_uint8: bool,
) -> AttentionProvider:
    query, key, value = inputs
    scale = config.scale if config.scale is not None else query.shape[-1] ** -0.5
    sort_value_rows = piper_backend._should_sort_value_rows(
        center_value=center_value,
        capability=capability,
        nvidia_cuda=True,
        is_causal=config.is_causal,
        head_dim=query.shape[-1],
        key_length=key.shape[2],
    )

    def prepare() -> object:
        return piper_backend._prepare_piper_attention(
            query,
            key,
            value,
            scale,
            config.is_causal,
            center_value,
            native_uint8=native_uint8,
            sort_value_rows=sort_value_rows,
        )

    def run(prepared: object) -> torch.Tensor:
        return piper_backend._launch_piper_attention(
            cast(piper_backend._PreparedPiperAttention, prepared)
        )

    return BenchmarkProvider(
        name=name,
        prepare=prepare,
        run=run,
        synchronize=torch.cuda.synchronize,
        configuration={
            **config.as_dict(),
            "implementation": "pure_triton",
            "algorithm": "piper_attention",
            "qk_quantization": _canonical_qk_granularity(capability),
            "probability_dtype": "uint8",
            "value_dtype": "int8",
            "value_scale": "per_key",
            "center_value": center_value,
            "value_row_order": "centered_range_ascending" if sort_value_rows else "original",
            "mixed_sign_mma": "native" if native_uint8 else "affine_proxy",
        },
        triton_jit_functions=_piper_jit_functions(
            capability,
            sort_value_rows=sort_value_rows,
        ),
    )


def _make_providers(
    inputs: AttentionInputs,
    *,
    provider_names: Sequence[str],
    config: AttentionConfig,
    capability: tuple[int, int],
) -> dict[str, AttentionProvider]:
    query, _, _ = inputs
    default_centering = _default_center_value(query, inputs[1], config.is_causal)
    providers: dict[str, AttentionProvider] = {}
    piper_settings = {
        _PIPER: (default_centering, True),
        _PIPER_CENTERED: (True, True),
        _PIPER_UNCENTERED: (False, True),
        _PIPER_AFFINE: (default_centering, False),
    }
    for name, (center_value, native_uint8) in piper_settings.items():
        if name in provider_names:
            providers[name] = _make_piper_provider(
                name,
                inputs,
                config=config,
                capability=capability,
                center_value=center_value,
                native_uint8=native_uint8,
            )

    def prepare_inputs() -> object:
        return inputs

    if _PURE_TRITON_SAGE2PP in provider_names:
        providers[_PURE_TRITON_SAGE2PP] = BenchmarkProvider(
            name=_PURE_TRITON_SAGE2PP,
            prepare=prepare_inputs,
            run=lambda prepared: sage_attention_2pp(
                *cast(AttentionInputs, prepared),
                scale=config.scale,
                is_causal=config.is_causal,
            ),
            synchronize=torch.cuda.synchronize,
            configuration={
                **config.as_dict(),
                "implementation": "pure_triton",
                "algorithm": "sage_attention_2pp",
                "qk_quantization": _canonical_qk_granularity(capability),
                "pv_accumulation": "fp32+fp16",
            },
        )
    if _SDPA in provider_names:
        providers[_SDPA] = BenchmarkProvider(
            name=_SDPA,
            prepare=prepare_inputs,
            run=lambda prepared: _sdpa(
                cast(AttentionInputs, prepared),
                scale=config.scale,
                is_causal=config.is_causal,
            ),
            synchronize=torch.cuda.synchronize,
            configuration={
                **config.as_dict(),
                "implementation": "pytorch",
                "algorithm": "scaled_dot_product_attention",
            },
        )

    canonical = (
        _load_canonical(capability)
        if _CANONICAL_SAGE2PP in provider_names or _CANONICAL_SAGE2 in provider_names
        else None
    )
    if canonical is not None:
        canonical_configuration = {
            **config.as_dict(),
            "implementation": "canonical_cuda",
            "canonical_version": _CANONICAL_VERSION,
            "canonical_revision": _CANONICAL_REVISION,
            "qk_quantization": _canonical_qk_granularity(capability),
        }

        def canonical_run(pv_accum_dtype: str) -> Callable[[object], torch.Tensor]:
            def run(prepared: object) -> torch.Tensor:
                query, key, value = cast(AttentionInputs, prepared)
                return canonical(
                    query,
                    key,
                    value,
                    tensor_layout="HND",
                    is_causal=config.is_causal,
                    qk_quant_gran=_canonical_qk_granularity(capability),
                    sm_scale=config.scale,
                    pv_accum_dtype=pv_accum_dtype,
                    smooth_k=True,
                    smooth_v=False,
                    return_lse=False,
                )

            return run

        providers[_CANONICAL_SAGE2PP] = BenchmarkProvider(
            name=_CANONICAL_SAGE2PP,
            prepare=prepare_inputs,
            run=canonical_run("fp32+fp16"),
            synchronize=torch.cuda.synchronize,
            configuration={
                **canonical_configuration,
                "algorithm": "sage_attention_2pp",
                "pv_accumulation": "fp32+fp16",
            },
        )
        providers[_CANONICAL_SAGE2] = BenchmarkProvider(
            name=_CANONICAL_SAGE2,
            prepare=prepare_inputs,
            run=canonical_run("fp32+fp32"),
            synchronize=torch.cuda.synchronize,
            configuration={
                **canonical_configuration,
                "algorithm": "sage_attention2",
                "pv_accumulation": "fp32+fp32",
            },
        )
    return {name: providers[name] for name in provider_names}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        choices=_PROVIDER_NAMES,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="add both revision-pinned canonical CUDA providers",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--kv-sequence", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-provider", choices=_PROVIDER_NAMES, default=_PIPER)
    add_compiler_inspection_arguments(parser)
    add_profile_arguments(parser)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _resolved_provider_names(
    args: argparse.Namespace,
    *,
    fp8_supported: bool,
) -> tuple[str, ...]:
    requested = (
        args.providers
        if args.providers is not None
        else [
            _PIPER,
            _PIPER_UNCENTERED,
            *([_PURE_TRITON_SAGE2PP] if fp8_supported else []),
            _SDPA,
        ]
    )
    names = list(dict.fromkeys(requested))
    if args.canonical:
        for name in (_CANONICAL_SAGE2PP, _CANONICAL_SAGE2):
            if name not in names:
                names.append(name)
    return tuple(names)


def _validate_provider_support(
    provider_names: Sequence[str],
    device: torch.device,
) -> None:
    if any(name in _PIPER_PROVIDERS for name in provider_names) and not supports_uint8_int8_mma(
        device
    ):
        raise SystemExit(
            "Piper Attention providers require NVIDIA SM8x or consumer Blackwell SM12x"
        )
    if any(name in _FP8_SAGE_PROVIDERS for name in provider_names) and not supports_fp8_fp16_mma(
        device
    ):
        raise SystemExit(
            "Sage 8+8 providers require NVIDIA FP8 tensor cores; "
            "the canonical RTX 30 fallback is a different FP16-PV algorithm"
        )


def _validate_args(args: argparse.Namespace, provider_names: Sequence[str]) -> None:
    if any(length <= 0 for length in args.sequence):
        raise SystemExit("query sequence lengths must be positive")
    if args.kv_sequence is not None and args.kv_sequence <= 0:
        raise SystemExit("key/value sequence length must be positive")
    if args.batch_size <= 0 or args.heads <= 0:
        raise SystemExit("batch size and heads must be positive")
    if (
        args.causal
        and args.kv_sequence is not None
        and any(length != args.kv_sequence for length in args.sequence)
    ):
        raise SystemExit("causal attention requires equal query and key/value lengths")
    compiler_requested = (
        args.compiler_report
        or args.compiler_json is not None
        or args.compiler_jsonl is not None
    )
    if (args.profile or compiler_requested) and len(args.sequence) != 1:
        raise SystemExit("profiling and compiler inspection require exactly one query length")
    if args.profile and args.profile_provider not in provider_names:
        raise SystemExit("--profile-provider must be selected by --providers or --canonical")
    if compiler_requested and tuple(provider_names) != (_PIPER,):
        raise SystemExit("compiler inspection requires only the default Piper provider")


def _output_targets(args: argparse.Namespace) -> tuple[OutputTarget | None, OutputTarget | None]:
    benchmark_target = output_target(args)
    compiler_target = output_target(args, option_prefix="compiler")
    if args.profile and benchmark_target is not None:
        raise SystemExit("--profile cannot produce benchmark --json/--jsonl records")
    if (
        benchmark_target is not None
        and compiler_target is not None
        and benchmark_target.path.resolve() == compiler_target.path.resolve()
    ):
        raise SystemExit("benchmark and compiler output paths must be different")
    return benchmark_target, compiler_target


def _make_inputs(
    shape: AttentionShape,
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> AttentionInputs:
    query = torch.randn(
        (shape.batch_size, shape.num_query_heads, shape.query_length, shape.head_dim),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    key = torch.randn(
        (
            shape.batch_size,
            shape.effective_num_key_value_heads,
            shape.key_value_length,
            shape.head_dim,
        ),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    value = torch.randn(key.shape, device="cuda", dtype=dtype, generator=generator)
    return query, key, value


def _compiler_report(
    args: argparse.Namespace,
    provider: AttentionProvider,
    environment: EnvironmentInfo,
    target: OutputTarget | None,
) -> TritonCompilerRecord | None:
    if not args.compiler_report and target is None:
        return None
    report = inspect_provider(
        provider,
        environment,
        include_sass=args.sass,
        nvdisasm=args.nvdisasm,
        require_isolated_jit_cache=False,
    )
    write_records([report], target)
    if args.compiler_report:
        print(format_compiler_report(report))
    return report


def _print_measurement(record: BenchmarkRecord) -> None:
    timings = record.timings
    assert timings.first_call_ms is not None
    assert timings.preparation is not None
    assert timings.operator_end_to_end is not None
    assert record.quality is not None
    print(
        f"| {record.shape['query_length']} | {record.shape['key_value_length']} "
        f"| {record.provider} | {timings.first_call_ms:.3f} "
        f"| {timings.preparation.display(3)} "
        f"| {timings.prepared_execution.display(3)} "
        f"| {timings.operator_end_to_end.display(3)} "
        f"| {record.quality.mean_absolute_error:.6f} "
        f"| {record.quality.sqnr_db:.2f} |"
    )


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("Piper Attention benchmarking requires a CUDA-capable GPU")
    device = torch.device("cuda")
    provider_names = _resolved_provider_names(
        args,
        fp8_supported=supports_fp8_fp16_mma(device),
    )
    _validate_args(args, provider_names)
    _validate_provider_support(provider_names, device)
    benchmark_target, compiler_target = _output_targets(args)
    capability = torch.cuda.get_device_capability(device)

    environment = capture_environment(Path(__file__).resolve().parents[1])
    config = AttentionConfig(
        dtype=args.dtype,
        is_causal=args.causal,
        scale=args.scale,
        qkv_layout="BHSD",
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records: list[BenchmarkRecord] = []
    compiler_provider: AttentionProvider | None = None

    print(
        f"device: {environment.gpu_name}; architecture: {environment.gpu_architecture}; "
        f"torch: {environment.torch_version}; triton: {environment.triton_version}"
    )
    print(
        "| query | key/value | provider | first call (ms) | preparation wall p50 "
        "[p20, p80] (ms) | hot device p50 [p20, p80] (ms) | complete wall p50 "
        "[p20, p80] (ms) | mean abs error | SQNR (dB) |"
    )
    print("|---:|---:|:---|---:|---:|---:|---:|---:|---:|")

    for query_length in args.sequence:
        key_value_length = args.kv_sequence or query_length
        shape = AttentionShape(
            batch_size=args.batch_size,
            num_query_heads=args.heads,
            query_length=query_length,
            key_value_length=key_value_length,
            head_dim=args.head_dim,
        )
        inputs = _make_inputs(shape, dtype=_dtype(args.dtype), generator=generator)
        providers = _make_providers(
            inputs,
            provider_names=provider_names,
            config=config,
            capability=capability,
        )
        if args.profile:
            profile = profile_provider(
                providers[args.profile_provider],
                iterations=args.profile_iterations,
                warmup_iterations=args.profile_warmup_iterations,
                phase=args.profile_phase,
                range_name=args.profile_range_name,
                include_setup=args.profile_include_setup,
            )
            print(
                f"profiled provider={profile.provider} phase={profile.phase.value} "
                f"iterations={profile.iterations} range={profile.range_name!r}"
            )
            if _PIPER in providers:
                _compiler_report(args, providers[_PIPER], environment, compiler_target)
            return

        measurements = [
            measure_provider(
                provider,
                warmup_ms=args.warmup_ms,
                measurement_time_ms=args.measurement_time_ms,
            )
            for provider in providers.values()
        ]
        expected = _sdpa(inputs, scale=config.scale, is_causal=config.is_causal)
        torch.cuda.synchronize()
        for measurement in measurements:
            record = BenchmarkRecord(
                benchmark="piper-attention",
                provider=measurement.provider,
                shape=shape.as_dict(),
                configuration=measurement.configuration,
                timings=measurement.timings,
                quality=measure_quality(measurement.output, expected),
                environment=environment,
            )
            records.append(record)
            _print_measurement(record)
        compiler_provider = providers.get(_PIPER)

    if compiler_provider is not None:
        _compiler_report(args, compiler_provider, environment, compiler_target)
    write_records(records, benchmark_target)


def main() -> None:
    """Run the Piper Attention benchmark CLI."""
    _main()


if __name__ == "__main__":
    main()
