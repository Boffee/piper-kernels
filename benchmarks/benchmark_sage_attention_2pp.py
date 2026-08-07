"""Benchmark canonical SageAttention2++ providers and PyTorch SDPA.

The pure-Triton provider is the production implementation in this package.
Canonical CUDA SageAttention2 and SageAttention2++ are optional, revision-pinned
benchmark dependencies; they are never imported by the installed package.
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

from piper_kernels.attention import sage_attention_2pp
from piper_kernels.attention._sage2pp.backends import triton as triton_backend

_CANONICAL_VERSION = "2.2.0"
_CANONICAL_REVISION = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
_PURE_TRITON = "pure-triton-sage2pp"
_SDPA = "pytorch-sdpa"
_CANONICAL_SAGE2PP = "canonical-cuda-sage2pp"
_CANONICAL_SAGE2 = "canonical-cuda-sage2"
_PROVIDER_NAMES = (_PURE_TRITON, _SDPA, _CANONICAL_SAGE2PP, _CANONICAL_SAGE2)

type AttentionInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
type CanonicalSage = Callable[..., torch.Tensor]


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


def _triton_jit_functions(capability: tuple[int, int]) -> dict[str, object]:
    qk_kernels = (
        {
            "quantize-query-per-warp": triton_backend._quantize_query_per_warp_kernel,
            "quantize-key-per-block": triton_backend._quantize_key_per_block_kernel,
        }
        if capability[0] == 12
        else {
            "quantize-query-per-thread": triton_backend._quantize_query_per_thread_kernel,
            "quantize-key-per-thread": triton_backend._quantize_key_per_thread_kernel,
        }
    )
    return {
        "kv-statistics-partial": triton_backend._kv_statistics_partial_kernel,
        "kv-statistics-finish": triton_backend._finish_kv_statistics_kernel,
        **qk_kernels,
        "quantize-value-per-channel": triton_backend._quantize_value_kernel,
        "attention": triton_backend._sage_attention_2pp_kernel,
    }


def _make_providers(
    inputs: AttentionInputs,
    *,
    provider_names: Sequence[str],
    config: AttentionConfig,
    capability: tuple[int, int],
) -> dict[str, BenchmarkProvider[AttentionInputs, torch.Tensor]]:
    canonical = (
        _load_canonical(capability)
        if _CANONICAL_SAGE2PP in provider_names or _CANONICAL_SAGE2 in provider_names
        else None
    )

    def prepare() -> AttentionInputs:
        return inputs

    def run_triton(prepared: AttentionInputs) -> torch.Tensor:
        query, key, value = prepared
        return sage_attention_2pp(
            query,
            key,
            value,
            scale=config.scale,
            is_causal=config.is_causal,
        )

    def run_sdpa(prepared: AttentionInputs) -> torch.Tensor:
        return _sdpa(prepared, scale=config.scale, is_causal=config.is_causal)

    def canonical_run(pv_accum_dtype: str) -> Callable[[AttentionInputs], torch.Tensor]:
        def run(prepared: AttentionInputs) -> torch.Tensor:
            assert canonical is not None
            query, key, value = prepared
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

    common_configuration = config.as_dict()
    providers = {
        _PURE_TRITON: BenchmarkProvider(
            name=_PURE_TRITON,
            prepare=prepare,
            run=run_triton,
            synchronize=torch.cuda.synchronize,
            configuration={
                **common_configuration,
                "implementation": "pure_triton",
                "algorithm": "sage_attention_2pp",
                "qk_quantization": _canonical_qk_granularity(capability),
                "pv_accumulation": "fp32+fp16",
            },
            triton_jit_functions=_triton_jit_functions(capability),
        ),
        _SDPA: BenchmarkProvider(
            name=_SDPA,
            prepare=prepare,
            run=run_sdpa,
            synchronize=torch.cuda.synchronize,
            configuration={
                **common_configuration,
                "implementation": "pytorch",
                "algorithm": "scaled_dot_product_attention",
            },
        ),
    }
    if canonical is not None:
        canonical_configuration = {
            **common_configuration,
            "implementation": "canonical_cuda",
            "canonical_version": _CANONICAL_VERSION,
            "canonical_revision": _CANONICAL_REVISION,
            "qk_quantization": _canonical_qk_granularity(capability),
        }
        providers[_CANONICAL_SAGE2PP] = BenchmarkProvider(
            name=_CANONICAL_SAGE2PP,
            prepare=prepare,
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
            prepare=prepare,
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
        default=[_PURE_TRITON, _SDPA],
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
        default=[512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument(
        "--kv-sequence",
        type=int,
        help="fixed key/value length; defaults to each query length",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile-provider",
        choices=_PROVIDER_NAMES,
        default=_PURE_TRITON,
    )
    add_compiler_inspection_arguments(parser)
    add_profile_arguments(parser)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _resolved_provider_names(args: argparse.Namespace) -> tuple[str, ...]:
    names = list(dict.fromkeys(args.providers))
    if args.canonical:
        for name in (_CANONICAL_SAGE2PP, _CANONICAL_SAGE2):
            if name not in names:
                names.append(name)
    return tuple(names)


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
        raise SystemExit("--profile-provider must also be selected by --providers or --canonical")
    if compiler_requested and _PURE_TRITON not in provider_names:
        raise SystemExit("compiler inspection requires the pure-Triton provider")
    if args.profile and compiler_requested and args.profile_provider != _PURE_TRITON:
        raise SystemExit(
            "combined profiling and compiler inspection requires the pure-Triton "
            "profile provider"
        )


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
        (
            shape.batch_size,
            shape.num_query_heads,
            shape.query_length,
            shape.head_dim,
        ),
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
    value = torch.randn(
        key.shape,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    return query, key, value


def _compiler_report(
    args: argparse.Namespace,
    provider: BenchmarkProvider[AttentionInputs, torch.Tensor],
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
        f"| {timings.prepared_execution.display(3)} "
        f"| {timings.operator_end_to_end.display(3)} "
        f"| {record.quality.mean_absolute_error:.6f} "
        f"| {record.quality.sqnr_db:.2f} |"
    )


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    provider_names = _resolved_provider_names(args)
    _validate_args(args, provider_names)
    benchmark_target, compiler_target = _output_targets(args)
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None:
        raise SystemExit("SageAttention2++ benchmarking requires an NVIDIA CUDA GPU")

    capability = torch.cuda.get_device_capability()
    if _PURE_TRITON in provider_names and capability != (8, 9) and capability[0] != 12:
        raise SystemExit("the pure-Triton SageAttention2++ provider requires SM89 or SM12x")

    repository = Path(__file__).resolve().parents[1]
    environment = capture_environment(repository)
    config = AttentionConfig(
        dtype=args.dtype,
        is_causal=args.causal,
        scale=args.scale,
        qkv_layout="BHSD",
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records: list[BenchmarkRecord] = []
    compiler_provider: BenchmarkProvider[AttentionInputs, torch.Tensor] | None = None

    print(
        f"device: {environment.gpu_name}; architecture: {environment.gpu_architecture}; "
        f"torch: {environment.torch_version}; triton: {environment.triton_version}"
    )
    print(
        "| query | key/value | provider | first call (ms) "
        "| device p50 [p20, p80] (ms) | wall p50 [p20, p80] (ms) "
        "| mean abs error | SQNR (dB) |"
    )
    print("|---:|---:|:---|---:|---:|---:|---:|---:|")

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
            if _PURE_TRITON in providers:
                _compiler_report(
                    args,
                    providers[_PURE_TRITON],
                    environment,
                    compiler_target,
                )
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
            attention_pairs = shape.query_length * shape.key_value_length
            if config.is_causal:
                attention_pairs = shape.query_length * (shape.query_length + 1) // 2
            operations = (
                4
                * shape.batch_size
                * shape.num_query_heads
                * attention_pairs
                * shape.head_dim
            )
            effective_tflops = (
                operations
                / (measurement.timings.prepared_execution.median_ms * 1e-3)
                / 1e12
            )
            record = BenchmarkRecord(
                benchmark="sage-attention-2pp",
                provider=measurement.provider,
                shape=shape.as_dict(),
                configuration=measurement.configuration,
                timings=measurement.timings,
                quality=measure_quality(measurement.output, expected),
                environment=environment,
                extra={"effective_tflops": effective_tflops},
            )
            records.append(record)
            _print_measurement(record)
        compiler_provider = providers.get(_PURE_TRITON)

    if compiler_provider is not None:
        _compiler_report(args, compiler_provider, environment, compiler_target)
    write_records(records, benchmark_target)


def main() -> None:
    """Run the SageAttention2++ benchmark CLI."""
    _main()


if __name__ == "__main__":
    main()
