"""Compare split, masked, and per-tile-branch INT8 tails, keeping 128x256 tiles.

The original split and fully masked launchers are benchmark-only controls for
the production single-launch kernel. All three use caller-owned buffers and CUDA graph replay.
Full-linear timings include rotation/quantization. Acquire the shared GPU gate first.
"""

# Triton's launch options are not part of the kernel's Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import torch
from lib.convrot_int8_legacy import legacy_matmul
from lib.environment import capture_environment
from triton.testing import do_bench_cudagraph

from piper_kernels.linear.convrot.int8._nvidia import triton as nvidia


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        nargs="+",
        type=_positive_int,
        default=[1025, 3073, 8193, 32769, 65537, 100000, 100001, 100096],
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=_positive_int,
        action="append",
        metavar=("K", "N"),
        help="custom K/N shape; repeat to compare multiple shapes",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        choices=(16, 64, 256),
        help="default: largest supported divisor of K",
    )
    parser.add_argument("--rep-ms", type=_positive_int, default=60)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=871)
    return parser.parse_args(argv)


def _benchmark_shape(m: int, k: int, n: int, args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(args.seed)
    value = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=value.dtype)
    group_size = args.group_size or next(g for g in (256, 64, 16) if k % g == 0)
    qdata, scale = nvidia.prepare_input(weight, group_size)
    prepared = nvidia.prepare_input(value, group_size)
    split_out = torch.empty(m, n, device=value.device, dtype=value.dtype)
    single_out = torch.empty_like(split_out)
    branch_out = torch.empty_like(split_out)
    plan = nvidia.default_execution_plan(qdata)
    if (plan.matmul_block_m, plan.matmul_block_n, plan.matmul_block_k) != (128, 256, 128):
        raise AssertionError("this experiment requires the production large-tile plan")

    def split_gemm() -> torch.Tensor:
        return legacy_matmul(prepared, qdata, scale, split_out, plan)

    def single_gemm() -> torch.Tensor:
        return legacy_matmul(prepared, qdata, scale, single_out, plan, split_tail=False)

    def branch_gemm() -> torch.Tensor:
        return nvidia.execute_prepared_linear(
            *prepared, qdata, scale, None, value.dtype, plan, out=branch_out
        )

    def split_linear() -> torch.Tensor:
        nvidia.prepare_input(value, group_size, out=prepared)
        return split_gemm()

    def single_linear() -> torch.Tensor:
        nvidia.prepare_input(value, group_size, out=prepared)
        return single_gemm()

    def branch_linear() -> torch.Tensor:
        nvidia.prepare_input(value, group_size, out=prepared)
        return branch_gemm()

    if not torch.equal(split_linear(), single_linear()):
        raise AssertionError("single-launch tail differs from the split tail")
    if not torch.equal(split_out, branch_linear()):
        raise AssertionError("per-tile branch differs from the split tail")
    operations: dict[str, Callable[[], object]] = {
        "split_gemm": split_gemm,
        "single_gemm": single_gemm,
        "branch_gemm": branch_gemm,
        "split_linear": split_linear,
        "single_linear": single_linear,
        "branch_linear": branch_linear,
    }
    samples: dict[str, list[float]] = {name: [] for name in operations}
    for repeat in range(args.repeats):
        order = list(operations.items())
        if repeat % 2:
            order.reverse()
        for name, operation in order:
            samples[name].append(
                1000
                * cast(float, do_bench_cudagraph(operation, rep=args.rep_ms, return_mode="median"))
            )
    median_us = {name: statistics.median(values) for name, values in samples.items()}
    print(
        json.dumps(
            {
                "shape_mkn": (m, k, n),
                "plan": plan.as_dict(),
                "group_size": group_size,
                "exact_agreement": True,
                "tail_rows": m % 128,
                "median_us": median_us,
                "samples_us": samples,
            }
        ),
        flush=True,
    )
    return median_us


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("requires an NVIDIA CUDA GPU")
    print(
        json.dumps(
            {
                "environment": capture_environment(Path(__file__).resolve().parents[1]).as_dict(),
                "arguments": vars(args),
                "clock": "cuda_graph_device_event",
                "dtype": "bfloat16",
                "group_size": args.group_size or "largest supported divisor of K",
                "buffers": "preallocated for all three providers",
                "inputs": "synthetic normal tensors",
            }
        ),
        flush=True,
    )
    shapes = args.shape or [(1024, 1024), (3072, 1024), (5376, 14336), (272, 257)]
    if any(k % (args.group_size or 16) for k, _ in shapes):
        raise SystemExit("K must be divisible by a supported ConvRot group size (at least 16)")
    for m in args.rows:
        for k, n in shapes:
            _benchmark_shape(m, k, n, args)


if __name__ == "__main__":
    main()
