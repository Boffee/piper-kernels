"""Isolate per-K64 INT32-to-FP32 PV conversion overhead."""

# Triton JIT pointer arguments intentionally have no Python runtime types, and
# the benchmark callbacks are consumed before the next sequence-loop iteration.
# ruff: noqa: ANN001, ANN202, B023

import argparse
import json
from collections.abc import Sequence

import torch
import triton
import triton.language as tl
import triton.testing
from profile_sage_pv_variant import _compiler_report


@triton.jit
def _convert_each_tile_kernel(
    probability_ptr,
    value_ptr,
    output_ptr,
    tiles,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    program = tl.program_id(0)
    offsets_m = tl.arange(0, block_m)
    offsets_k = tl.arange(0, block_k)
    offsets_d = tl.arange(0, head_dim)
    accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    for tile in tl.range(0, tiles, disable_licm=True):
        probability = tl.load(
            probability_ptr
            + tile * block_m * block_k
            + offsets_m[:, None] * block_k
            + offsets_k[None, :]
        )
        value = tl.load(
            value_ptr
            + tile * block_k * head_dim
            + offsets_k[:, None] * head_dim
            + offsets_d[None, :]
        )
        partial = tl.dot(probability, value, out_dtype=tl.int32)
        accumulator += partial.to(tl.float32)
    tl.store(
        output_ptr
        + program * block_m * head_dim
        + offsets_m[:, None] * head_dim
        + offsets_d[None, :],
        accumulator,
    )


@triton.jit
def _persistent_int32_kernel(
    probability_ptr,
    value_ptr,
    output_ptr,
    tiles,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    program = tl.program_id(0)
    offsets_m = tl.arange(0, block_m)
    offsets_k = tl.arange(0, block_k)
    offsets_d = tl.arange(0, head_dim)
    accumulator = tl.zeros((block_m, head_dim), dtype=tl.int32)
    for tile in tl.range(0, tiles, disable_licm=True):
        probability = tl.load(
            probability_ptr
            + tile * block_m * block_k
            + offsets_m[:, None] * block_k
            + offsets_k[None, :]
        )
        value = tl.load(
            value_ptr
            + tile * block_k * head_dim
            + offsets_k[:, None] * head_dim
            + offsets_d[None, :]
        )
        accumulator = tl.dot(
            probability,
            value,
            accumulator,
            out_dtype=tl.int32,
        )
    tl.store(
        output_ptr
        + program * block_m * head_dim
        + offsets_m[:, None] * head_dim
        + offsets_d[None, :],
        accumulator.to(tl.float32),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--block-m", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--num-warps", type=int, choices=[4, 8], default=4)
    parser.add_argument("--num-stages", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--warmup-ms", type=int, default=300)
    parser.add_argument("--repeat-ms", type=int, default=2000)
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    """Benchmark conversion placement with bounded exact integer accumulation."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    block_k = 64
    for sequence in args.sequence:
        tiles = triton.cdiv(sequence, block_k)
        programs = args.heads * triton.cdiv(sequence, args.block_m)
        torch.manual_seed(1900 + sequence)
        probability = torch.randint(
            0,
            16,
            (tiles, args.block_m, block_k),
            device="cuda",
            dtype=torch.int8,
        )
        value = torch.randint(
            -8,
            8,
            (tiles, block_k, args.head_dim),
            device="cuda",
            dtype=torch.int8,
        )
        converted_output = torch.empty(
            (programs, args.block_m, args.head_dim),
            device="cuda",
            dtype=torch.float32,
        )
        persistent_output = torch.empty_like(converted_output)
        launch_options = {"num_warps": args.num_warps, "num_stages": args.num_stages}

        def convert_each_tile() -> None:
            _convert_each_tile_kernel[(programs,)](
                probability,
                value,
                converted_output,
                tiles,
                head_dim=args.head_dim,
                block_m=args.block_m,
                block_k=block_k,
                **launch_options,
            )

        def persistent_int32() -> None:
            _persistent_int32_kernel[(programs,)](
                probability,
                value,
                persistent_output,
                tiles,
                head_dim=args.head_dim,
                block_m=args.block_m,
                block_k=block_k,
                **launch_options,
            )

        convert_each_tile()
        persistent_int32()
        torch.cuda.synchronize()
        torch.testing.assert_close(converted_output, persistent_output, atol=0.0, rtol=0.0)
        convert_ms = float(
            triton.testing.do_bench(
                convert_each_tile,
                warmup=args.warmup_ms,
                rep=args.repeat_ms,
            )
        )
        persistent_ms = float(
            triton.testing.do_bench(
                persistent_int32,
                warmup=args.warmup_ms,
                rep=args.repeat_ms,
            )
        )
        print(
            json.dumps(
                {
                    "sequence": sequence,
                    "tiles": tiles,
                    "programs": programs,
                    "convert_each_tile": _compiler_report(
                        _convert_each_tile_kernel,
                        convert_ms,
                    ),
                    "persistent_int32": _compiler_report(
                        _persistent_int32_kernel,
                        persistent_ms,
                    ),
                    "persistent_speedup_percent": (convert_ms / persistent_ms - 1.0) * 100.0,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
