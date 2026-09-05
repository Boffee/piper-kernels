"""Benchmark the production AMD INT8 path, separately reporting preparation and GEMM.

Uses normal BF16 tensors, the actual ConvRot preparation, caller-owned phase
buffers, exact INT32-GEMM verification, and both cache-flushed and graph timings.
No power, voltage, or clock settings are modified. JSON lines go to stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict, cast

import torch
import triton
from lib.environment import capture_environment
from triton.testing import do_bench, do_bench_cudagraph

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._amd import policy
from piper_kernels.linear.convrot.int8._amd import triton as amd


class PhaseTiming(TypedDict):
    cache_flushed_median_ms: float
    cache_flushed_samples_ms: list[float]
    graph_median_ms: float


def _shape(value: str) -> tuple[int, int, int]:
    try:
        m, k, n = (int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use M,K,N, for example 8192,6144,4096") from error
    if min(m, k, n) < 1 or k % 256:
        raise argparse.ArgumentTypeError("dimensions must be positive and K divisible by 256")
    return m, k, n


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_shape, action="append", help="M,K,N; repeatable")
    parser.add_argument("--warmup-ms", type=_positive_int, default=60)
    parser.add_argument("--rep-ms", type=_positive_int, default=200)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--graph-rep-ms", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=871)
    return parser.parse_args(argv)


def _measure(operation: Callable[[], object], args: argparse.Namespace) -> PhaseTiming:
    cold = [
        cast(
            float, do_bench(operation, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="median")
        )
        for _ in range(args.repeats)
    ]
    graph = cast(float, do_bench_cudagraph(operation, rep=args.graph_rep_ms, return_mode="median"))
    return {
        "cache_flushed_median_ms": statistics.median(cold),
        "cache_flushed_samples_ms": cold,
        "graph_median_ms": graph,
    }


def _benchmark_shape(shape: tuple[int, int, int], args: argparse.Namespace) -> None:
    m, k, n = shape
    torch.manual_seed(args.seed)
    value = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=value.dtype)
    prepared = amd.prepare_input(value, 256)
    weight_qdata, weight_scale = amd.prepare_input(weight, 256)
    output = torch.empty(m, n, device=value.device, dtype=value.dtype)

    def prepare() -> tuple[torch.Tensor, torch.Tensor]:
        return amd.prepare_input(value, 256, out=prepared)

    def project() -> torch.Tensor:
        return amd.linear_prepared(
            *prepared, weight_qdata, weight_scale, None, value.dtype, out=output
        )

    def linear() -> torch.Tensor:
        return amd.run_linear(value, weight_qdata, weight_scale, None, 256)

    # Float32 arithmetic on the exact integer accumulator matches the kernel's
    # scale order, without comparing two launches of the same implementation.
    padded_input = torch.nn.functional.pad(prepared[0], (0, 0, 0, (-m) % 32))
    padded_weight = torch.nn.functional.pad(weight_qdata, (0, 0, 0, (-n) % 8))
    expected = torch._int_mm(padded_input, padded_weight.T)[:m, :n].float()
    expected.mul_(prepared[1].reshape(m, 1)).mul_(weight_scale.reshape(1, n))
    expected = expected.to(value.dtype)
    if not torch.equal(project(), expected) or not torch.equal(linear(), expected):
        raise AssertionError("prepared/full ConvRot does not match the exact INT32 reference")
    for phase, operation in [("prepare", prepare), ("prepared_gemm", project), ("linear", linear)]:
        timing = _measure(operation, args)
        record = {
            "shape_mkn": shape,
            "phase": phase,
            "exact": True,
            "execution_plan": asdict(amd.default_execution_plan(weight_qdata)),
            **timing,
        }
        if phase != "prepare":
            # Dense convention: an INT8 multiply and add count as two operations.
            record["dense_tops"] = 2 * m * k * n / float(timing["cache_flushed_median_ms"]) / 1e9
            record["graph_dense_tops"] = 2 * m * k * n / float(timing["graph_median_ms"]) / 1e9
        print(json.dumps(record), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit("requires ROCm PyTorch and a supported AMD GPU; use HIP_VISIBLE_DEVICES=0")
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not policy.supports_target(target):
        raise SystemExit(f"unsupported AMD target: {target}")
    print(
        json.dumps(
            {
                "environment": capture_environment(Path(__file__).resolve().parents[1]).as_dict(),
                "gpu": torch.cuda.get_device_name(),
                "target": str(target),
                "torch": torch.__version__,
                "hip_component": torch.version.hip,
                "triton": triton.__version__,
                "dtype": "bfloat16",
                "group_size": 256,
                "seed": args.seed,
                "large_matmul_group_m": amd._LARGE_MATMUL_GROUP_M_TILES,
                "warmup_ms": args.warmup_ms,
                "rep_ms": args.rep_ms,
                "repeats": args.repeats,
                "graph_rep_ms": args.graph_rep_ms,
            }
        ),
        flush=True,
    )
    for shape in args.shape or [(8192, 6144, 4096)]:
        _benchmark_shape(shape, args)


if __name__ == "__main__":
    main()
