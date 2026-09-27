"""Representative ConvRot INT8 linear benchmark that finishes in a few minutes.

Cases follow MiniMax H3. Transformer blocks use hidden width 5376, 56x128 attention, and a
tanh-GELU FFN of width 14336 at 8K-131K rows, with Q/K/V sharing one prepared input and
GELU fused into the down-projection preparation. The H3 VAE linears and short-M projection
mixes cover the smaller tile configurations, and the eight primary M/N/K anchors keep
continuity with the other ConvRot benchmarks.

Each case captures one CUDA graph per variant over preallocated buffers: the production
plan, the original plan (shared preparation and fixed 128x256 tiles), and BF16 cuBLAS. It
then replays the graphs interleaved over several rounds in one process, so a large call
runs once per round instead of once per estimation, warmup, and retry. BF16 multiplies the
same inputs without a separate GELU pass, which favors BF16 for the down projection. INT8
outputs are compared bitwise with the original plan in row chunks. JSON lines go to stdout
and a summary table to stderr. Acquire the shared GPU gate first (see benchmarks/README.md).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lib.environment import capture_environment

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._nvidia import dispatch as nvidia
from piper_kernels.linear.convrot.int8._nvidia import policy
from piper_kernels.linear.convrot.int8._plan import LinearExecutionPlan

_GROUP_SIZE = 256
_CHECK_ROWS = 16_384
_VARIANTS = ("production", "original", "bf16")
# Stage, K, N, projections sharing one prepared input, and fused input activation.
_H3_BLOCK = (
    ("qkv", 5376, 7168, 3, None),
    ("out", 7168, 5376, 1, None),
    ("ffn_up", 5376, 14336, 1, None),
    ("ffn_down", 14336, 5376, 1, "gelu_tanh"),
)
_H3_VAE = ((2048, 2048), (2048, 16384), (8192, 2048))
_PROJECTION_MIX = ((1024, 2048), (1024, 1024), (2048, 1024), (1024, 3072), (3072, 1024))
_H3_WIDTHS = ((5376, 7168), (5376, 14336), (14336, 5376))
_ANCHOR_WIDTHS = ((6144, 4096), (6144, 16384), (14336, 4096), (14336, 16384))


@dataclass(frozen=True, slots=True)
class Case:
    """One linear stage: rows, widths, projections per preparation, and activation."""

    group: str
    stage: str
    rows: int
    in_features: int
    out_features: int
    projections: int = 1
    activation: str | None = None

    @property
    def operations(self) -> int:
        return 2 * self.rows * self.in_features * self.out_features * self.projections


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h3-rows", nargs="+", type=_positive_int, default=[8192, 32768, 131072, 131073]
    )
    parser.add_argument("--vae-rows", nargs="+", type=_positive_int, default=[1797, 7188])
    parser.add_argument("--mix-rows", nargs="+", type=_positive_int, default=[1, 16, 128, 1024])
    parser.add_argument("--h3-short-rows", nargs="+", type=_positive_int, default=[1, 64, 512])
    parser.add_argument("--anchor-rows", nargs="*", type=_positive_int, default=[8192, 32768])
    parser.add_argument("--rounds", type=_positive_int, default=3)
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=10.0,
        help="sustained load before timing so power-limited clocks reach steady state",
    )
    parser.add_argument(
        "--sample-ms",
        type=float,
        default=250.0,
        help="minimum graph duration per sample, up to 1,000 calls; shorter bursts run above "
        "sustained clocks",
    )
    parser.add_argument("--bias", action="store_true", help="add BF16 biases")
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _cases(args: argparse.Namespace) -> list[Case]:
    cases = [
        Case("h3_block", stage, rows, k, n, projections, activation)
        for rows in args.h3_rows
        for stage, k, n, projections, activation in _H3_BLOCK
    ]
    cases += [Case("h3_vae", f"{k}->{n}", rows, k, n) for rows in args.vae_rows for k, n in _H3_VAE]
    cases += [
        Case("projection_mix", f"{k}->{n}", rows, k, n)
        for rows in args.mix_rows
        for k, n in _PROJECTION_MIX
    ]
    cases += [
        Case("h3_short", f"{k}->{n}", rows, k, n)
        for rows in args.h3_short_rows
        for k, n in _H3_WIDTHS
    ]
    cases += [
        Case("anchor", f"{k}->{n}", rows, k, n)
        for rows in args.anchor_rows
        for k, n in _ANCHOR_WIDTHS
    ]
    return cases


def _capture(function: Callable[[], object], sample_ms: float) -> tuple[torch.cuda.CUDAGraph, int]:
    """Capture enough calls, at most 1,000, to fill ``sample_ms``; large calls run once."""
    function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    torch.cuda.synchronize()
    calls = max(1, min(1000, math.ceil(sample_ms / max(start.elapsed_time(end), 1e-3))))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            function()
    graph.replay()
    torch.cuda.synchronize()
    return graph, calls


def _warm_up(seconds: float) -> None:
    """Hold the GPU at load so early cases are not timed at cold-start boost clocks."""
    matrix = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    product = torch.empty_like(matrix)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            torch.mm(matrix, matrix, out=product)
        torch.cuda.synchronize()


def _replay_us(graph: torch.cuda.CUDAGraph, calls: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end) / calls


def _run_case(case: Case, args: argparse.Namespace, target: AcceleratorTarget) -> dict[str, Any]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(
        args.seed + case.rows + case.out_features
    )
    k, n, rows = case.in_features, case.out_features, case.rows
    value = torch.randn(rows, k, device=device, dtype=torch.bfloat16, generator=generator)
    dense = [
        torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=generator) * k**-0.5
        for _ in range(case.projections)
    ]
    weights = [nvidia.prepare_input(weight, _GROUP_SIZE) for weight in dense]
    weights = [(qdata, scale.reshape(n, 1)) for qdata, scale in weights]
    bias = (
        torch.randn(n, device=device, dtype=torch.bfloat16, generator=generator)
        if args.bias
        else None
    )
    prepared = (
        torch.empty(rows, k, device=device, dtype=torch.int8),
        torch.empty(rows, device=device, dtype=torch.float32),
    )
    outputs = [torch.empty(rows, n, device=device, dtype=torch.bfloat16) for _ in dense]
    production = nvidia.default_execution_plan(weights[0][0], rows=rows)
    original = policy.baseline_execution_plan(in_features=k)

    def int8(
        plan: LinearExecutionPlan,
        source: torch.Tensor = value,
        buffers: tuple[torch.Tensor, torch.Tensor] = prepared,
        destinations: list[torch.Tensor] = outputs,
    ) -> None:
        nvidia.prepare_input_with_plan(
            source,
            k,
            _GROUP_SIZE,
            activation_fn=case.activation,
            execution_plan=plan,
            target=target,
            out=buffers,
        )
        for (qdata, scale), destination in zip(weights, destinations, strict=True):
            nvidia.execute_prepared_linear(
                *buffers, qdata, scale, bias, value.dtype, plan, out=destination
            )

    def bf16() -> None:
        for weight, destination in zip(dense, outputs, strict=True):
            if bias is None:
                torch.mm(value, weight.T, out=destination)
            else:
                torch.addmm(bias, value, weight.T, out=destination)

    # Production outputs stay in place; the original plan is recomputed in row chunks,
    # which match whole-tensor results because preparation and GEMM rows are independent.
    int8(production)
    exact = True
    for start in range(0, rows, _CHECK_ROWS):
        stop = min(rows, start + _CHECK_ROWS)
        chunk = (
            torch.empty_like(prepared[0][start:stop]),
            torch.empty_like(prepared[1][start:stop]),
        )
        chunk_outputs = [torch.empty_like(output[start:stop]) for output in outputs]
        int8(original, value[start:stop], chunk, chunk_outputs)
        exact &= all(
            torch.equal(expected, output[start:stop])
            for expected, output in zip(chunk_outputs, outputs, strict=True)
        )

    functions: dict[str, Callable[[], object]] = {
        "production": lambda: int8(production),
        "original": lambda: int8(original),
    }
    if not args.skip_bf16:
        functions["bf16"] = bf16
    graphs = {name: _capture(function, args.sample_ms) for name, function in functions.items()}
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    for round_index in range(args.rounds):
        order = list(graphs) if round_index % 2 == 0 else list(reversed(graphs))
        for name in order:
            samples[name].append(_replay_us(*graphs[name]))
    median_us = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "group": case.group,
        "stage": case.stage,
        "rows": rows,
        "in_features": k,
        "out_features": n,
        "projections": case.projections,
        "activation": case.activation,
        "bias": args.bias,
        "production_execution_plan": production.as_dict(),
        "exact_vs_original": exact,
        "median_us": median_us,
        "spread": {name: max(v) / min(v) - 1 for name, v in samples.items()},
        "samples_us": samples,
        "tops": {name: case.operations / us / 1e6 for name, us in median_us.items()},
    }


def _summaries(records: list[dict[str, Any]], cases: list[Case]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[tuple[Case, dict[str, Any]]]] = {}
    for case, record in zip(cases, records, strict=True):
        groups.setdefault((case.group, case.rows), []).append((case, record))
    summaries = []
    for (group, rows), members in groups.items():
        variants = [name for name in _VARIANTS if name in members[0][1]["median_us"]]
        total_us = {
            name: sum(record["median_us"][name] for _, record in members) for name in variants
        }
        operations = sum(case.operations for case, _ in members)
        summaries.append(
            {
                "group": group,
                "rows": rows,
                "cases": len(members),
                "exact_vs_original": all(record["exact_vs_original"] for _, record in members),
                "total_us": total_us,
                "tops": {name: operations / us / 1e6 for name, us in total_us.items()},
                "speedup_vs_original": total_us["original"] / total_us["production"],
                "speedup_vs_bf16": total_us["bf16"] / total_us["production"]
                if "bf16" in total_us
                else None,
            }
        )
    return summaries


def _print_table(summaries: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    lines = [
        "| Group | Rows | Production us (TOPS) | Original us (TOPS) | BF16 us (TFLOPS) "
        "| vs original | vs BF16 | Exact |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        total, tops = summary["total_us"], summary["tops"]
        bf16 = f"{total['bf16']:.1f} ({tops['bf16']:.0f})" if "bf16" in total else "-"
        vs_bf16 = summary["speedup_vs_bf16"]
        lines.append(
            f"| {summary['group']} | {summary['rows']} "
            f"| {total['production']:.1f} ({tops['production']:.0f}) "
            f"| {total['original']:.1f} ({tops['original']:.0f}) | {bf16} "
            f"| {summary['speedup_vs_original']:.2f}x "
            f"| {'-' if vs_bf16 is None else f'{vs_bf16:.2f}x'} | {summary['exact_vs_original']} |"
        )
    ratios = [r["median_us"]["original"] / r["median_us"]["production"] for r in records]
    lines.append(
        f"Geometric-mean speedup over {len(records)} cases vs the original plan: "
        f"{math.exp(sum(map(math.log, ratios)) / len(ratios)):.2f}x"
    )
    print("\n".join(lines), file=sys.stderr, flush=True)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("requires an NVIDIA CUDA GPU")
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "environment": capture_environment(Path(__file__).resolve().parents[1]).as_dict(),
                "arguments": vars(args),
                "clock": "cuda_graph_device_event",
                "dtype": "bfloat16",
                "group_size": _GROUP_SIZE,
            }
        ),
        flush=True,
    )
    cases = _cases(args)
    _warm_up(args.warmup_s)
    records = []
    for case in cases:
        record = _run_case(case, args, target)
        records.append(record)
        print(json.dumps(record), flush=True)
        torch.cuda.empty_cache()
    summaries = _summaries(records, cases)
    for summary in summaries:
        print(json.dumps({"summary": summary}), flush=True)
    _print_table(summaries, records)
    print(f"elapsed {time.perf_counter() - started:.0f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
