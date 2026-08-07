"""Benchmark native and affine-proxy integer P@V dot products.

All variants run with stock Triton. The ``u8-s8-native`` variant uses Piper's
compiler extension to select NVIDIA's native mixed-sign integer MMA. The affine
proxy models the exact UINT8 identity used by the attention experiment: its
precomputed ``128 * sum(V)`` correction is loaded as the integer MMA accumulator.
"""

# Triton's JIT pointer arguments intentionally omit Python annotations.
# ruff: noqa: ANN001, ANN202

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl
from _lib import (
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
    measure_saturation,
    output_target,
    profile_provider,
    write_records,
)

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook, uint8_int8_dot


@triton.jit
def _dot_kernel(
    a_ptr,
    b_ptr,
    correction_ptr,
    output_ptr,
    use_affine_proxy: tl.constexpr,
    use_native_uint8: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    tile = tl.program_id(0)
    offsets_m = tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    a = tl.load(
        a_ptr
        + tile * block_m * block_k
        + offsets_m[:, None] * block_k
        + offsets_k[None, :]
    )
    b = tl.load(
        b_ptr
        + tile * block_k * block_n
        + offsets_k[:, None] * block_n
        + offsets_n[None, :]
    )
    if use_affine_proxy:
        a = (a.to(tl.int32) - 128).to(tl.int8)
        correction = tl.load(correction_ptr + tile * block_n + offsets_n)
        accumulator = tl.zeros((block_m, block_n), tl.int32) + correction[None, :]
        result = tl.dot(a, b, accumulator, out_dtype=tl.int32)
    elif use_native_uint8:
        result = uint8_int8_dot(a, b)
    else:
        result = tl.dot(a, b, out_dtype=tl.int32)
    tl.store(
        output_ptr
        + tile * block_m * block_n
        + offsets_m[:, None] * block_n
        + offsets_n[None, :],
        result,
    )


@dataclass(frozen=True, slots=True)
class PreparedDot:
    """Per-invocation metadata and output storage."""

    correction: torch.Tensor
    output: torch.Tensor


def _reference_output(
    probability: torch.Tensor,
    value: torch.Tensor,
    *,
    tile_batch: int = 32,
) -> torch.Tensor:
    """Compute every integer P@V tile on CPU with bounded temporary memory."""
    if probability.ndim != 3 or value.ndim != 3:
        raise ValueError("probability and value must be rank-three batched matrices")
    if probability.shape[0] != value.shape[0] or probability.shape[2] != value.shape[1]:
        raise ValueError("probability and value batch or reduction dimensions do not match")
    if tile_batch <= 0:
        raise ValueError("tile batch must be positive")

    expected = torch.empty(
        (probability.shape[0], probability.shape[1], value.shape[2]),
        device="cpu",
        dtype=torch.int32,
    )
    for start in range(0, probability.shape[0], tile_batch):
        stop = min(start + tile_batch, probability.shape[0])
        probability_batch = probability[start:stop].cpu().to(torch.int32)
        value_batch = value[start:stop].cpu().to(torch.int32)
        expected[start:stop] = probability_batch @ value_batch
    return expected


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        choices=("s8-s8", "u8-s8-native", "u8-s8-affine-proxy"),
    )
    parser.add_argument("--tiles", type=int, default=2048)
    parser.add_argument("--block-m", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--block-n", type=int, choices=(64, 128), default=128)
    parser.add_argument("--block-k", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--measurement-time-ms", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    add_compiler_inspection_arguments(parser)
    add_profile_arguments(parser)
    add_output_arguments(parser)
    return parser.parse_args(argv)


def _run_profile_if_requested(
    args: argparse.Namespace,
    provider: BenchmarkProvider[PreparedDot, torch.Tensor],
    repository: Path,
) -> bool:
    if not args.profile:
        return False
    profile = profile_provider(
        provider,
        iterations=args.profile_iterations,
        warmup_iterations=args.profile_warmup_iterations,
        phase=args.profile_phase,
        range_name=args.profile_range_name,
        include_setup=args.profile_include_setup,
    )
    print(
        f"profiled provider={profile.provider} phase={profile.phase.value} "
        f"iterations={profile.iterations} range={profile.range_name!r} "
        f"include_setup={profile.include_setup}"
    )
    compiler_report = _compiler_report_if_requested(
        args,
        provider,
        capture_environment(repository),
    )
    _print_compiler_report_if_requested(args, compiler_report)
    return True


def _compiler_report_if_requested(
    args: argparse.Namespace,
    provider: BenchmarkProvider[PreparedDot, torch.Tensor],
    environment: EnvironmentInfo,
) -> TritonCompilerRecord | None:
    compiler_output = output_target(args, option_prefix="compiler")
    if not args.compiler_report and compiler_output is None:
        return None
    report = inspect_provider(
        provider,
        environment,
        include_sass=args.sass,
        nvdisasm=args.nvdisasm,
    )
    write_records([report], compiler_output)
    return report


def _print_compiler_report_if_requested(
    args: argparse.Namespace,
    report: TritonCompilerRecord | None,
) -> None:
    if args.compiler_report:
        assert report is not None
        print(format_compiler_report(report))


def _benchmark_output(args: argparse.Namespace) -> OutputTarget | None:
    target = output_target(args)
    if args.profile and target is not None:
        raise SystemExit("--profile cannot produce benchmark --json/--jsonl records")
    compiler_target = output_target(args, option_prefix="compiler")
    if (
        target is not None
        and compiler_target is not None
        and target.path.resolve() == compiler_target.path.resolve()
    ):
        raise SystemExit("benchmark and compiler output paths must be different")
    return target


def _configure_variant_runtime(variant: str) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("integer P@V benchmarking requires a Triton-supported GPU")
    if variant == "u8-s8-native":
        install_uint8_int8_dot_hook()


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_variant_runtime(args.variant)
    benchmark_output = _benchmark_output(args)

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape_a = (args.tiles, args.block_m, args.block_k)
    shape_b = (args.tiles, args.block_k, args.block_n)
    if args.variant == "s8-s8":
        a = torch.randint(
            -128,
            128,
            shape_a,
            device=device,
            dtype=torch.int8,
            generator=generator,
        )
    else:
        a = torch.randint(
            0,
            256,
            shape_a,
            device=device,
            dtype=torch.uint8,
            generator=generator,
        )
    b = torch.randint(
        -128,
        128,
        shape_b,
        device=device,
        dtype=torch.int8,
        generator=generator,
    )

    def prepare() -> PreparedDot:
        if args.variant == "u8-s8-affine-proxy":
            correction = (128 * b.to(torch.int32).sum(dim=1)).to(torch.int32)
        else:
            correction = torch.empty(1, device=device, dtype=torch.int32)
        output = torch.empty(
            (args.tiles, args.block_m, args.block_n),
            device=device,
            dtype=torch.int32,
        )
        return PreparedDot(correction, output)

    def launch(prepared: PreparedDot) -> torch.Tensor:
        _dot_kernel[(args.tiles,)](
            a,
            b,
            prepared.correction,
            prepared.output,
            use_affine_proxy=args.variant == "u8-s8-affine-proxy",
            use_native_uint8=args.variant == "u8-s8-native",
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.num_warps,
        )
        return prepared.output

    lhs_dtype = "int8" if args.variant == "s8-s8" else "uint8"
    implementation = (
        "affine-proxy" if args.variant == "u8-s8-affine-proxy" else "native"
    )
    configuration = {
        "lhs_dtype": lhs_dtype,
        "rhs_dtype": "int8",
        "accumulator_dtype": "int32",
        "implementation": implementation,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "num_warps": args.num_warps,
        "seed": args.seed,
    }
    provider = BenchmarkProvider(
        name=f"triton-{implementation}",
        prepare=prepare,
        run=launch,
        synchronize=torch.cuda.synchronize,
        configuration=configuration,
        triton_jit_functions={"integer-pv-dot": _dot_kernel},
    )
    repository = Path(__file__).resolve().parents[1]
    if _run_profile_if_requested(args, provider, repository):
        return

    measurement = measure_provider(
        provider,
        warmup_ms=args.warmup_ms,
        measurement_time_ms=args.measurement_time_ms,
    )

    expected = _reference_output(a, b)
    actual = measurement.output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    operations = 2 * args.tiles * args.block_m * args.block_n * args.block_k
    tops = (
        operations / (measurement.timings.prepared_execution.median_ms * 1e-3) / 1e12
    )
    environment = capture_environment(repository)
    compiler_report = _compiler_report_if_requested(args, provider, environment)

    print(
        f"device: {environment.gpu_name}; backend: {environment.accelerator_backend}; "
        f"architecture: {environment.gpu_architecture}"
    )
    print(f"triton: {environment.triton_version}")
    print(
        f"variant={args.variant} operation={lhs_dtype}xint8->int32 tiles={args.tiles} "
        f"tile={args.block_m}x{args.block_n}x{args.block_k} warps={args.num_warps}"
    )
    assert measurement.timings.first_call_ms is not None
    assert measurement.timings.preparation is not None
    assert measurement.timings.operator_end_to_end is not None
    print(f"first_call_synchronized_wall_ms={measurement.timings.first_call_ms:.6f}")
    print(
        "preparation_synchronized_wall_p50_p20_p80_ms="
        f"{measurement.timings.preparation.display(6)}"
    )
    print(
        "prepared_execution_device_event_p50_p20_p80_ms="
        f"{measurement.timings.prepared_execution.display(6)} "
        f"effective_tops={tops:.2f}"
    )
    print(
        "operator_end_to_end_synchronized_wall_p50_p20_p80_ms="
        f"{measurement.timings.operator_end_to_end.display(6)}"
    )
    _print_compiler_report_if_requested(args, compiler_report)
    probability_limits = (-128, 127) if args.variant == "s8-s8" else (0, 255)
    saturation = {
        "probability": measure_saturation(a, *probability_limits),
        "value": measure_saturation(b, -128, 127),
    }
    record = BenchmarkRecord(
        benchmark="integer-pv-dot",
        provider=measurement.provider,
        shape={
            "tiles": args.tiles,
            "probability_rows": args.block_m,
            "key_tile": args.block_k,
            "value_features": args.block_n,
        },
        configuration=measurement.configuration,
        timings=measurement.timings,
        quality=measure_quality(actual, expected, saturation=saturation),
        environment=environment,
        extra={"effective_tops": tops},
    )
    write_records([record], benchmark_output)


if __name__ == "__main__":
    _main()
