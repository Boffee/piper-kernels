"""CUDA-graph ConvRot INT8 schedule and BF16 comparisons across projection shapes.

Includes input rotation/quantization in full-linear timings. Acquire the local
model server's GPU gate before running on a shared GPU (see benchmarks/README.md).
JSON lines on stdout include environment, per-shape timings, and seven-call totals.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from lib.environment import capture_environment
from triton.testing import do_bench_cudagraph

from piper_kernels.linear.convrot.int8._nvidia import triton as nvidia
from piper_kernels.linear.convrot.int8._plan import LinearExecutionPlan
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

# K, N, projections sharing the shape. Totals count k/v and gate/up twice.
SHAPES = (
    (1024, 2048, ("q",)),
    (1024, 1024, ("k", "v")),
    (2048, 1024, ("o",)),
    (1024, 3072, ("gate", "up")),
    (3072, 1024, ("down",)),
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", nargs="+", type=_positive_int, default=[128, 256, 384, 512, 1024, 3072]
    )
    parser.add_argument("--rep-ms", type=_positive_int, default=100)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=871)
    parser.add_argument(
        "--order-offset", type=int, default=0, help="alternate timing order across processes"
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=_positive_int,
        action="append",
        metavar=("K", "N"),
        help="custom K/N shape; repeat to compare multiple shapes",
    )
    parser.add_argument("--skip-bf16", action="store_true", help="time only INT8 schedules")
    parser.add_argument(
        "--paired", action="store_true", help="share preparation across two projections"
    )
    parser.add_argument(
        "--compare-schedules",
        action="store_true",
        help="compare the 32x64, 64x64, and 128x256 plans at every row count",
    )
    return parser.parse_args(argv)


def _measure_operations(
    operations: dict[str, Callable[[], object]], args: argparse.Namespace
) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {name: [] for name in operations}
    for repeat in range(args.repeats):
        order = list(operations.items())
        if (repeat + args.order_offset) % 2:
            order.reverse()
        for name, operation in order:
            samples[name].append(
                1000
                * cast(float, do_bench_cudagraph(operation, rep=args.rep_ms, return_mode="median"))
            )
    return samples


def _reference_projection(
    prepared: tuple[torch.Tensor, torch.Tensor],
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    *,
    paired: bool,
) -> torch.Tensor:
    m, n = prepared[0].shape[0], qdata.shape[0]
    padded = torch.nn.functional.pad(prepared[0], (0, 0, 0, (-m) % 32))
    padded_weight = torch.nn.functional.pad(qdata, (0, 0, 0, (-n) % 8))
    expected = torch._int_mm(padded, padded_weight.T)[:m, :n].float()
    expected.mul_(prepared[1][:, None]).mul_(scale.reshape(1, n))
    expected = expected.to(dtype)
    if paired:
        expected = torch.cat((expected, -expected), dim=-1)
    return expected


def _benchmark_shape(m: int, k: int, n: int, args: argparse.Namespace) -> dict[str, float]:
    started_at = datetime.now(UTC).isoformat()
    torch.manual_seed(args.seed)
    value = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    dense_weight = torch.randn(n, k, device="cuda", dtype=value.dtype)
    group_size = next(g for g in (256, 64, 16) if k % g == 0)
    qdata, scale = nvidia.prepare_input(dense_weight, group_size)
    weight = ConvRotInt8Tensor.from_quantized(
        qdata, scale.reshape(n, 1), group_size=group_size, logical_dtype=value.dtype
    )
    second_projection = (-qdata, scale.clone(), None) if args.paired else None
    second_dense_weight = -dense_weight if args.paired else None
    prepared = nvidia.prepare_input(value, group_size)
    output = torch.empty(m, n * (2 if args.paired else 1), device=value.device, dtype=value.dtype)
    selected = nvidia.default_execution_plan(
        qdata, rows=m, projection_count=2 if args.paired else 1
    )
    production = selected
    if args.compare_schedules:
        production = replace(
            production,
            matmul_block_m=32,
            matmul_block_n=64,
            matmul_block_k=128,
            matmul_num_warps=8,
            matmul_num_stages=4,
        )
    previous = replace(
        production,
        matmul_block_m=128,
        matmul_block_n=256,
        matmul_block_k=128,
        matmul_num_warps=8,
        matmul_num_stages=3,
    )
    medium = replace(
        previous,
        matmul_block_m=64,
        matmul_block_n=64,
        matmul_num_warps=4,
    )

    def with_plan(plan: LinearExecutionPlan) -> torch.Tensor:
        if second_projection is None:
            return nvidia.run_linear(value, qdata, scale, None, group_size, execution_plan=plan)
        return nvidia.execute_prepared_linear(
            *nvidia.prepare_input(value, group_size),
            qdata,
            scale,
            None,
            value.dtype,
            plan,
            second_projection=second_projection,
        )

    def linear() -> torch.Tensor:
        return with_plan(production) if args.compare_schedules else policy_linear()

    def previous_linear() -> torch.Tensor:
        return with_plan(previous)

    def medium_linear() -> torch.Tensor:
        return with_plan(medium)

    def policy_linear() -> torch.Tensor:
        if second_projection is None:
            return torch.nn.functional.linear(value, weight)
        return nvidia.linear_prepared(
            *nvidia.prepare_input(value, group_size),
            qdata,
            scale,
            None,
            value.dtype,
            second_projection=second_projection,
        )

    def bf16_linear() -> object:
        first = torch.nn.functional.linear(value, dense_weight)
        if second_dense_weight is None:
            return first
        return first, torch.nn.functional.linear(value, second_dense_weight)

    def gemm() -> torch.Tensor:
        return nvidia.execute_prepared_linear(
            *prepared,
            qdata,
            scale,
            None,
            value.dtype,
            production,
            out=output,
            second_projection=second_projection,
        )

    # Independently check the exact INT32 accumulation and ordered FP32 scaling,
    # plus bitwise agreement with the original production kernel schedule.
    expected = _reference_projection(prepared, qdata, scale, value.dtype, paired=args.paired)
    for result in (linear(), previous_linear(), gemm()):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    if args.compare_schedules:
        torch.testing.assert_close(medium_linear(), expected, atol=0, rtol=0)
        torch.testing.assert_close(policy_linear(), expected, atol=0, rtol=0)

    operations: dict[str, Callable[[], object]] = {
        "linear": linear,
        "previous_linear": previous_linear,
    }
    if not args.skip_bf16:
        operations["bf16_linear"] = bf16_linear
    if args.compare_schedules:
        operations["medium_linear"] = medium_linear
        operations["policy_linear"] = policy_linear
    if not args.compare_schedules:
        operations["prepared_gemm"] = gemm
        operations["preparation"] = lambda: nvidia.prepare_input(value, group_size, out=prepared)
    samples = _measure_operations(operations, args)
    median_us = {name: statistics.median(values) for name, values in samples.items()}
    record = {
        "shape_mkn": (m, k, n),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "plan": production.as_dict(),
        "selected_plan": selected.as_dict(),
        "previous_plan": previous.as_dict(),
        "medium_plan": medium.as_dict() if args.compare_schedules else None,
        "group_size": group_size,
        "projection_count": 2 if args.paired else 1,
        "exact_agreement": True,
        "median_us": median_us,
        "samples_us": samples,
        "linear_tops": 2 * m * k * n * (2 if args.paired else 1) / (median_us["linear"] * 1e6),
    }
    print(json.dumps(record), flush=True)
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
                "group_size": "largest supported divisor of K",
                "inputs": "synthetic normal tensors; custom dimensions"
                if args.shape
                else "synthetic normal tensors; Harrier projection dimensions",
            }
        ),
        flush=True,
    )
    shapes = [(k, n, ("custom",)) for k, n in args.shape] if args.shape else SHAPES
    if any(k % 16 for k, _n, _projections in shapes):
        raise SystemExit("K must be divisible by a supported ConvRot group size (at least 16)")
    for m in args.rows:
        totals: dict[str, float] = dict.fromkeys(("linear", "previous_linear"), 0.0)
        if not args.skip_bf16:
            totals["bf16_linear"] = 0.0
        if args.compare_schedules:
            totals["medium_linear"] = 0.0
            totals["policy_linear"] = 0.0
        for k, n, projections in shapes:
            timing = _benchmark_shape(m, k, n, args)
            for name in totals:
                totals[name] += len(projections) * timing[name]
        total_key = "projection_total_us" if args.shape else "seven_projection_total_us"
        if args.paired:
            total_key = total_key.replace("projection_", "projection_pair_")
        print(json.dumps({"rows": m, total_key: totals}), flush=True)


if __name__ == "__main__":
    main()
