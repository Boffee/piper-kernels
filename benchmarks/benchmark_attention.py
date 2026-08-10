"""Benchmark full-attention providers with shared shapes, timing, and quality metrics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from lib.attention import AttentionConfig, AttentionShape, make_attention_inputs
from lib.attention_providers import (
    PROVIDER_NAMES,
    TRITON_PROVIDERS,
    AttentionProvider,
    make_attention_providers,
    resolve_provider_names,
    run_sdpa,
    validate_provider_support,
)
from lib.environment import EnvironmentInfo, capture_environment
from lib.profiling import add_profile_arguments, profile_provider
from lib.providers import measure_provider
from lib.quality import measure_quality
from lib.reporting import (
    BenchmarkRecord,
    OutputTarget,
    add_output_arguments,
    output_target,
    write_records,
)
from lib.triton_inspection import (
    TritonCompilerRecord,
    add_compiler_inspection_arguments,
    format_compiler_report,
    inspect_provider,
)

from piper_kernels._triton.targets import AcceleratorTarget


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        choices=PROVIDER_NAMES,
        nargs="+",
        metavar="NAME",
        help=(
            "providers to compare; defaults to a useful hardware-aware set; "
            f"choices: {', '.join(PROVIDER_NAMES)}"
        ),
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="add both revision-pinned canonical CUDA SageAttention providers",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="one or more query sequence lengths",
    )
    parser.add_argument(
        "--kv-sequence",
        type=int,
        help="fixed key/value length; defaults to each query length",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile-provider",
        choices=PROVIDER_NAMES,
        metavar="NAME",
        help="provider to profile; defaults to the first selected provider",
    )
    parser.add_argument(
        "--compiler-provider",
        choices=TRITON_PROVIDERS,
        metavar="NAME",
        help="Triton provider to inspect; inferred when exactly one is selected",
    )
    add_compiler_inspection_arguments(parser)
    add_profile_arguments(parser)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _compiler_requested(args: argparse.Namespace) -> bool:
    return args.compiler_report or args.compiler_json is not None or args.compiler_jsonl is not None


def _validate_args(args: argparse.Namespace, provider_names: Sequence[str]) -> None:
    if any(length <= 0 for length in args.sequence):
        raise SystemExit("query sequence lengths must be positive")
    if args.kv_sequence is not None and args.kv_sequence <= 0:
        raise SystemExit("key/value sequence length must be positive")
    if args.batch_size <= 0 or args.heads <= 0:
        raise SystemExit("batch size and heads must be positive")
    if args.warmup_ms < 0 or args.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    if (
        args.causal
        and args.kv_sequence is not None
        and any(length != args.kv_sequence for length in args.sequence)
    ):
        raise SystemExit("causal attention requires equal query and key/value lengths")
    if (args.profile or _compiler_requested(args)) and len(args.sequence) != 1:
        raise SystemExit("profiling and compiler inspection require exactly one query length")
    if args.profile_provider is not None and args.profile_provider not in provider_names:
        raise SystemExit("--profile-provider must also be selected by --providers or --canonical")
    if args.profile_provider is not None and not args.profile:
        raise SystemExit("--profile-provider requires --profile")
    if args.compiler_provider is not None and not _compiler_requested(args):
        raise SystemExit("--compiler-provider requires a compiler report output option")
    if args.compiler_provider is not None and args.compiler_provider not in provider_names:
        raise SystemExit("--compiler-provider must also be selected by --providers")


def _profile_provider_name(
    args: argparse.Namespace,
    provider_names: Sequence[str],
) -> str | None:
    if not args.profile:
        return None
    return args.profile_provider or provider_names[0]


def _compiler_provider_name(
    args: argparse.Namespace,
    provider_names: Sequence[str],
) -> str | None:
    if not _compiler_requested(args):
        return None
    if args.compiler_provider is not None:
        return args.compiler_provider
    selected = tuple(name for name in provider_names if name in TRITON_PROVIDERS)
    if not selected:
        raise SystemExit("compiler inspection requires a selected Triton provider")
    if len(selected) > 1:
        raise SystemExit(
            "compiler inspection requires --compiler-provider when multiple Triton "
            "providers are selected"
        )
    return selected[0]


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


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _effective_tflops(
    shape: AttentionShape,
    config: AttentionConfig,
    latency_ms: float,
) -> float:
    attention_pairs = shape.query_length * shape.key_value_length
    if config.is_causal:
        attention_pairs = shape.query_length * (shape.query_length + 1) // 2
    operations = 4 * shape.batch_size * shape.num_query_heads * attention_pairs * shape.head_dim
    return operations / (latency_ms * 1e-3) / 1e12


def _inspect_compiler(
    args: argparse.Namespace,
    provider: AttentionProvider,
    environment: EnvironmentInfo,
) -> TritonCompilerRecord:
    return inspect_provider(
        provider,
        environment,
        include_sass=args.sass,
        nvdisasm=args.nvdisasm,
    )


def _write_compiler_report(
    args: argparse.Namespace,
    report: TritonCompilerRecord,
    target: OutputTarget | None,
) -> None:
    write_records([report], target)
    if args.compiler_report:
        print(format_compiler_report(report))


def _print_environment(environment: EnvironmentInfo) -> None:
    print(
        f"device: {environment.gpu_name}; architecture: {environment.gpu_architecture}; "
        f"torch: {environment.torch_version}; triton: {environment.triton_version}"
    )


def _print_table_header() -> None:
    print(
        "| query | key/value | provider | first call (ms) | preparation wall p50 "
        "[p20, p80] (ms) | hot device p50 [p20, p80] (ms) | complete wall p50 "
        "[p20, p80] (ms) | effective TFLOP/s | mean abs error | SQNR (dB) |"
    )
    print("|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|")


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
        f"| {record.extra['effective_tflops']:.2f} "
        f"| {record.quality.mean_absolute_error:.6f} "
        f"| {record.quality.sqnr_db:.2f} |"
    )


def _profile(
    args: argparse.Namespace,
    providers: dict[str, AttentionProvider],
    profile_name: str,
    compiler_name: str | None,
    environment: EnvironmentInfo,
    compiler_target: OutputTarget | None,
) -> None:
    if compiler_name is not None and compiler_name != profile_name:
        raise SystemExit("combined profiling and compiler inspection must select the same provider")
    profile = profile_provider(
        providers[profile_name],
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
    if compiler_name is not None:
        report = _inspect_compiler(
            args,
            providers[compiler_name],
            environment,
        )
        _write_compiler_report(args, report, compiler_target)


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("attention benchmarking requires a CUDA or ROCm accelerator")

    device = torch.device("cuda")
    target = AcceleratorTarget.from_device(device)
    provider_names = resolve_provider_names(
        args.providers,
        include_canonical=args.canonical,
        piper_attention_supported=target.supports_uint8_int8_mma,
        sage_attention_2pp_supported=target.supports_fp8_fp16_mma,
    )
    _validate_args(args, provider_names)
    validate_provider_support(provider_names, target)
    profile_name = _profile_provider_name(args, provider_names)
    compiler_name = _compiler_provider_name(args, provider_names)
    benchmark_target, compiler_target = _output_targets(args)

    environment = capture_environment(Path(__file__).resolve().parents[1])
    config = AttentionConfig(
        dtype=args.dtype,
        is_causal=args.causal,
        scale=args.scale,
        qkv_layout="BHSD",
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records: list[BenchmarkRecord] = []
    compiler_report: TritonCompilerRecord | None = None

    _print_environment(environment)
    if profile_name is None:
        _print_table_header()
    for query_length in args.sequence:
        key_value_length = args.kv_sequence or query_length
        shape = AttentionShape(
            batch_size=args.batch_size,
            num_query_heads=args.heads,
            query_length=query_length,
            key_value_length=key_value_length,
            head_dim=args.head_dim,
        )
        inputs = make_attention_inputs(
            shape,
            dtype=_dtype(args.dtype),
            device=device,
            generator=generator,
        )
        providers = make_attention_providers(
            inputs,
            provider_names=provider_names,
            config=config,
            target=target,
        )
        if profile_name is not None:
            _profile(
                args,
                providers,
                profile_name,
                compiler_name,
                environment,
                compiler_target,
            )
            return

        measurement_order = list(providers)
        if compiler_name is not None:
            measurement_order.remove(compiler_name)
            measurement_order.insert(0, compiler_name)
        measurements = {}
        for provider_name in measurement_order:
            provider = providers[provider_name]
            measurement = measure_provider(
                provider,
                warmup_ms=args.warmup_ms,
                measurement_time_ms=args.measurement_time_ms,
            )
            measurements[provider_name] = measurement
            if provider.name == compiler_name:
                compiler_report = _inspect_compiler(args, provider, environment)

        expected = run_sdpa(inputs, config)
        torch.cuda.synchronize()
        for provider_name in provider_names:
            measurement = measurements[provider_name]
            tflops = _effective_tflops(
                shape,
                config,
                measurement.timings.prepared_execution.median_ms,
            )
            record = BenchmarkRecord(
                benchmark="attention",
                provider=measurement.provider,
                shape=shape.as_dict(),
                configuration=measurement.configuration,
                timings=measurement.timings,
                quality=measure_quality(measurement.output, expected),
                environment=environment,
                extra={"effective_tflops": tflops},
            )
            records.append(record)
            _print_measurement(record)

    write_records(records, benchmark_target)
    if compiler_report is not None:
        _write_compiler_report(args, compiler_report, compiler_target)


def main() -> None:
    """Run the full-attention benchmark CLI."""
    _main()


if __name__ == "__main__":
    main()
