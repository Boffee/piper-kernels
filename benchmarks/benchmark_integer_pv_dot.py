"""Benchmark native and affine-proxy integer P@V dot products.

The ``s8-s8`` and ``u8-s8-affine-proxy`` variants run with stock Triton. The
``u8-s8-native`` variant requires compiler and target support for a mixed-sign
integer dot. The affine proxy models the exact UINT8 identity used by the
attention experiment: its precomputed ``128 * sum(V)`` correction is loaded as
the integer MMA accumulator.
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
    add_output_arguments,
    capture_environment,
    measure_provider,
    measure_quality,
    measure_saturation,
    output_target,
    write_records,
)


@triton.jit
def _dot_kernel(
    a_ptr,
    b_ptr,
    correction_ptr,
    output_ptr,
    use_affine_proxy: tl.constexpr,
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
    add_output_arguments(parser)
    return parser.parse_args(argv)


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("integer P@V benchmarking requires a Triton-supported GPU")

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
    measurement = measure_provider(
        BenchmarkProvider(
            name=f"triton-{implementation}",
            prepare=prepare,
            run=launch,
            synchronize=torch.cuda.synchronize,
            configuration=configuration,
        ),
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
    environment = capture_environment(Path(__file__).resolve().parents[1])

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
    print(f"first_call_ms={measurement.timings.first_call_ms:.6f}")
    print(f"preparation_p50_p20_p80_ms={measurement.timings.preparation.display(6)}")
    print(
        "prepared_execution_p50_p20_p80_ms="
        f"{measurement.timings.prepared_execution.display(6)} "
        f"effective_tops={tops:.2f}"
    )
    print(
        "operator_end_to_end_p50_p20_p80_ms="
        f"{measurement.timings.operator_end_to_end.display(6)}"
    )
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
    write_records([record], output_target(args))


if __name__ == "__main__":
    _main()
