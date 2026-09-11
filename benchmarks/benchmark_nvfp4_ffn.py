"""Benchmark the current plain and ConvRot NVFP4 fused FFNs."""

from __future__ import annotations

import argparse
import gc
import math
import os
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from lib.environment import EnvironmentInfo, capture_environment
from lib.reporting import BenchmarkRecord, add_output_arguments, output_target, write_records
from lib.timing import ClockDomain, PhaseTimings, Timing, _linear_quantile

from piper_kernels._triton import nvfp4 as nvfp4_primitives
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import _preparation as rotated_preparation
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import triton as convrot_ffn
from piper_kernels.fusions.nvfp4_swiglu_ffn import _core
from piper_kernels.fusions.nvfp4_swiglu_ffn._preparation import StandardPreparation
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4
from piper_kernels.linear.nvfp4 import triton as nvfp4

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
_DEFAULT_SHAPES = [(1_024, 2_048, 8_192), (1_797, 2_048, 8_192), (4_096, 5_376, 14_336)]


@dataclass(frozen=True, slots=True)
class Case:
    """One complete FFN with independently supplied gate/value weights."""

    shape: tuple[int, int, int]
    dtype: torch.dtype
    dynamic: bool
    group_size: int | None
    chunk_rows: int
    high_first: bool
    seed: int


@dataclass(frozen=True, slots=True)
class GraphTiming:
    """Timing windows and call counts, excluding graph capture and compilation."""

    calls_per_graph: int = 16
    warmup_rounds: int = 6
    samples: int = 11
    sample_ms: int = 80


def _preparation(case: Case) -> _core.PreparationBackend:
    if case.group_size is None:
        return StandardPreparation(case.high_first, case.high_first)
    return convrot_ffn._preparation(
        case.group_size,
        case.group_size,
        case.group_size,
        case.high_first,
        case.high_first,
        case.high_first,
    )


def _workload(case: Case) -> tuple[torch.Tensor, tuple[_core.LinearOperands, ...]]:
    rows, input_features, intermediate_features = case.shape
    torch.manual_seed(case.seed)
    source = torch.randn(rows, input_features, device="cuda", dtype=case.dtype)
    weights = []
    biases = []
    for width, height in (
        (input_features, intermediate_features),
        (input_features, intermediate_features),
        (intermediate_features, input_features),
    ):
        dense = torch.randn(height, width, device="cuda", dtype=case.dtype) / math.sqrt(width)
        packed = (
            nvfp4.prepare_static(
                dense, nvfp4_primitives.dynamic_scale(dense), high_first=case.high_first
            )
            if case.group_size is None
            else convrot_nvfp4.prepare_dynamic(dense, case.group_size, high_first=case.high_first)
        )
        weights.append(packed)
        biases.append(torch.randn(height, device="cuda", dtype=case.dtype) * 0.1)
    source_scale = _preparation(case).dynamic_source_scale(source)
    down_scale = torch.tensor(0.01, device="cuda")
    linears = tuple(
        _core.LinearOperands(
            weight_qdata=weight[0],
            weight_scale=weight[1],
            weight_per_tensor_scale=weight[2],
            activation_per_tensor_scale=(
                None if case.dynamic else (source_scale if index < 2 else down_scale)
            ),
            bias=bias,
            dynamic_activation_scale=case.dynamic,
            high_first=case.high_first,
        )
        for index, (weight, bias) in enumerate(zip(weights, biases, strict=True))
    )
    return source, linears


def _require_exclusive_gpu() -> None:
    selected_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    others = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, gpu_uuid = (field.strip() for field in line.split(",", maxsplit=1))
        if gpu_uuid == selected_uuid and int(pid) != os.getpid():
            others.append(pid)
    if others:
        raise RuntimeError(f"GPU is shared with process(es) {', '.join(others)}; retry when free")


@contextmanager
def _workspace_limit(limit_mib: int | None) -> Iterator[None]:
    previous = rotated_preparation._ROTATED_WORKSPACE_BYTES
    if limit_mib is not None:
        rotated_preparation._ROTATED_WORKSPACE_BYTES = limit_mib * 1024 * 1024
    try:
        yield
    finally:
        rotated_preparation._ROTATED_WORKSPACE_BYTES = previous


def _capture_graph(function: Callable[[], torch.Tensor], calls: int) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            function()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _elapsed_ms(graph: torch.cuda.CUDAGraph, replays: int, calls: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (replays * calls)


def _measure_graph(graph: torch.cuda.CUDAGraph, config: GraphTiming) -> tuple[list[float], int]:
    estimated_ms = _elapsed_ms(graph, 5, config.calls_per_graph)
    replays = max(1, math.ceil(config.sample_ms / (estimated_ms * config.calls_per_graph)))
    samples = []
    for round_index in range(config.warmup_rounds + config.samples):
        elapsed = _elapsed_ms(graph, replays, config.calls_per_graph)
        if round_index >= config.warmup_rounds:
            samples.append(elapsed)
    return samples, replays * config.calls_per_graph


def _distribution(samples: list[float]) -> Timing:
    ordered = sorted(samples)
    return Timing(
        median_ms=_linear_quantile(ordered, 0.5),
        p20_ms=_linear_quantile(ordered, 0.2),
        p80_ms=_linear_quantile(ordered, 0.8),
        clock=ClockDomain.DEVICE_EVENT,
    )


def benchmark_case(
    case: Case, timing: GraphTiming, environment: EnvironmentInfo
) -> BenchmarkRecord:
    """Measure the complete current FFN on a fixed synthetic workload."""
    _require_exclusive_gpu()
    source, linears = _workload(case)
    preparation = _preparation(case)

    def run() -> torch.Tensor:
        return _core.run_chunked_swiglu_ffn(
            source, linears[0], linears[1], linears[2], case.chunk_rows, preparation
        )

    output = run()
    if output.shape != source.shape or output.dtype != case.dtype:
        raise RuntimeError("FFN output has an unexpected shape or dtype")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("FFN output contains non-finite values")
    del output
    graph = _capture_graph(run, timing.calls_per_graph)
    _require_exclusive_gpu()
    samples, iterations = _measure_graph(graph, timing)
    _require_exclusive_gpu()
    distribution = _distribution(samples)
    rows, input_features, intermediate_features = case.shape
    record = BenchmarkRecord(
        benchmark="nvfp4-ffn",
        provider="piper",
        shape={
            "rows": rows,
            "input_features": input_features,
            "intermediate_features": intermediate_features,
            "output_features": input_features,
        },
        configuration={
            "format": "nvfp4" if case.group_size is None else "convrot-nvfp4",
            "dtype": str(case.dtype),
            "dynamic_scale": case.dynamic,
            "group_size": case.group_size,
            "chunk_rows": case.chunk_rows,
            "high_first": case.high_first,
            "seed": case.seed,
            "static_down_scale": None if case.dynamic else 0.01,
            "rotated_workspace_limit_bytes": (
                rotated_preparation._ROTATED_WORKSPACE_BYTES
                if case.group_size is not None
                else None
            ),
            "timing_mode": "cuda_graph_replay",
            **asdict(timing),
        },
        timings=PhaseTimings(
            warmup_ms=timing.warmup_rounds * timing.sample_ms,
            measurement_time_ms=timing.samples * timing.sample_ms,
            first_call_ms=None,
            preparation=None,
            prepared_execution=distribution,
            operator_end_to_end=None,
        ),
        environment=environment,
        extra={
            "samples_ms": samples,
            "iterations_per_sample": iterations,
            "output_finite": True,
        },
    )
    label = f"{case.shape} {case.dtype} {'dynamic' if case.dynamic else 'static'}"
    print(f"{label}: {distribution.display(precision=6)} ms", flush=True)
    return record


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["nvfp4", "convrot-nvfp4"], default="convrot-nvfp4")
    parser.add_argument("--shape", type=int, nargs=3, action="append", metavar=("M", "K", "N"))
    parser.add_argument("--dtype", nargs="+", choices=list(_DTYPES), default=list(_DTYPES))
    parser.add_argument(
        "--scaling", nargs="+", choices=["static", "dynamic"], default=["static", "dynamic"]
    )
    parser.add_argument("--group-size", type=int, choices=[16, 64, 256], default=256)
    parser.add_argument("--chunk-rows", type=int, default=_core.DEFAULT_CHUNK_ROWS)
    parser.add_argument("--high-first", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--calls-per-graph", type=int, default=16)
    parser.add_argument("--warmup-rounds", type=int, default=6)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--sample-ms", type=int, default=80)
    parser.add_argument(
        "--rotated-workspace-mib",
        type=int,
        help=(
            "benchmark-only ConvRot scratch limit; 0 forces recomputation "
            "(default: production limit)"
        ),
    )
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    args.shape = args.shape or _DEFAULT_SHAPES
    if any(m <= 0 or k <= 0 or n <= 0 or k % 64 or n % 64 for m, k, n in args.shape):
        parser.error("M must be positive; K and N must be positive multiples of 64")
    if args.format == "convrot-nvfp4" and any(
        k % args.group_size or n % args.group_size for _, k, n in args.shape
    ):
        parser.error("K and N must be divisible by the rotation group size")
    if args.chunk_rows < 128 or args.chunk_rows % 128:
        parser.error("--chunk-rows must be a positive multiple of 128")
    if min(args.calls_per_graph, args.samples, args.sample_ms) <= 0 or args.warmup_rounds < 0:
        parser.error("timing counts must be positive and warmup rounds non-negative")
    if args.rotated_workspace_mib is not None and (
        args.rotated_workspace_mib < 0 or args.format != "convrot-nvfp4"
    ):
        parser.error("--rotated-workspace-mib must be non-negative and is only for ConvRot")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("This benchmark requires NVIDIA CUDA and exact SM120")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit("This benchmark requires exact SM120")
    torch.set_num_threads(8)
    environment = capture_environment(Path(__file__).resolve().parents[1])
    timing = GraphTiming(args.calls_per_graph, args.warmup_rounds, args.samples, args.sample_ms)
    records: list[BenchmarkRecord] = []
    with torch.inference_mode(), _workspace_limit(args.rotated_workspace_mib):
        for shape in args.shape:
            for dtype in args.dtype:
                for scaling in args.scaling:
                    group_size = args.group_size if args.format == "convrot-nvfp4" else None
                    case = Case(
                        tuple(shape),
                        _DTYPES[dtype],
                        scaling == "dynamic",
                        group_size,
                        args.chunk_rows,
                        args.high_first,
                        args.seed + sum(shape) + (group_size or 0),
                    )
                    records.append(benchmark_case(case, timing, environment))
                    write_records(records, output_target(args))
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
