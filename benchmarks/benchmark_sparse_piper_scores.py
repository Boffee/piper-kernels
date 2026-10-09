"""Compare selected FP32 minmax scoring with the two-GEMM Torch baseline.

Uses the shared lower video case by default. --case selects another attention
or pipeline workload; custom shape flags are standalone diagnostics.
--query-blocks measures explicit routing chunks. K retains the sparse-prefix view of the padded
sequence allocation. Scores are checked against FP64 before timing. Timings
include score allocation and the maximum epilogue, but not summary generation
or route selection. These are FP32 operations, not INT8 TOPS.
"""

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from lib.cases import AttentionCase, PipelineCase, named_case
from lib.environment import EnvironmentInfo, capture_environment
from lib.reporting import BenchmarkRecord, add_output_arguments, output_target, write_records
from lib.suite_types import normal_tensor
from lib.timing import ClockDomain, PhaseTimings, Timing, triton_benchmark

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._routing import routing_scores
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING
from piper_kernels.fusions.sparse_piper._output import DEFAULT_QUERY_CHUNK_ROWS

_WARMUP_MS = 20


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="shared attention or pipeline case")
    parser.add_argument("--sequence", type=int, nargs="+")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int, choices=(64, 128))
    parser.add_argument("--query-blocks", type=int, nargs="+")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    workload = ("sequence", "batch", "heads", "kv_heads", "head_dim", "seed")
    custom = any(getattr(args, key) is not None for key in workload)
    if args.case is not None and custom:
        parser.error("--case cannot be combined with custom workload flags")
    if args.case is None and not custom:
        args.case = "sparse-attention-video-low-half"
    try:
        case = named_case(args.case or "sparse-attention-video-low-half")
    except ValueError as error:
        parser.error(str(error))
    if not isinstance(case, (AttentionCase, PipelineCase)):
        parser.error("--case requires an attention or pipeline case")
    defaults = {
        "sequence": [case.sequence],
        "batch": case.batch,
        "heads": case.heads,
        "kv_heads": case.kv_heads,
        "head_dim": case.head_dim,
        "seed": case.seed,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if min(args.sequence) < 64 or args.samples < 1 or args.rep_ms < 1 or args.device < 0:
        parser.error("requires sequence >= 64, positive samples/rep-ms, and a nonnegative device")
    if args.query_blocks is not None and min(args.query_blocks) < 1:
        parser.error("query blocks must be positive")
    if min(args.batch, args.heads, args.kv_heads) < 1 or args.heads % args.kv_heads:
        parser.error("requires a positive batch and query heads divisible by positive KV heads")
    return args


def _query_chunks(sequence: int) -> list[int]:
    blocks = (sequence + 63) // 64
    chunk_blocks = DEFAULT_QUERY_CHUNK_ROWS // 64
    full, tail = divmod(blocks, chunk_blocks)
    return ([chunk_blocks] if full else []) + ([tail] if tail else [])


def _torch_scores(
    query: torch.Tensor, primary: torch.Tensor, auxiliary: torch.Tensor
) -> torch.Tensor:
    batch, heads, rows, width = query.shape
    keys = primary.shape[2]
    kv_heads = primary.shape[1]
    grouped_query = query.reshape(batch, kv_heads, heads // kv_heads, rows, width)
    scores = grouped_query @ primary[:, :, None].transpose(-1, -2)
    other = grouped_query @ auxiliary[:, :, None].transpose(-1, -2)
    return torch.maximum(scores, other, out=scores).reshape(batch, heads, rows, keys)


def _benchmark(
    args: argparse.Namespace, sequence: int, rows: int, environment: EnvironmentInfo
) -> list[BenchmarkRecord]:
    device = torch.device("cuda", args.device)
    keys, storage = sequence // 64, (sequence + 63) // 64
    query = normal_tensor(
        (args.batch, args.heads, rows, args.head_dim),
        dtype=torch.float32,
        device=device,
        seed=args.seed,
        scale=0.5,
    )
    key_shape = (args.batch, args.kv_heads, storage, args.head_dim)
    primary = (
        normal_tensor(key_shape, dtype=torch.float32, device=device, seed=args.seed + 1, scale=0.5)
        + 2
    )
    auxiliary = (
        normal_tensor(key_shape, dtype=torch.float32, device=device, seed=args.seed + 2, scale=0.5)
        - 2
    )
    primary, auxiliary = primary[:, :, :keys], auxiliary[:, :, :keys]
    selected = _backend.select_minmax_scores(query, primary, auxiliary)
    functions = {
        "torch": lambda: _torch_scores(query, primary, auxiliary),
        "selected": lambda: routing_scores(query, primary, auxiliary, _MINMAX_ROUTING),
    }
    groups = args.heads // args.kv_heads
    reference = torch.maximum(
        query.double() @ primary.double().repeat_interleave(groups, dim=1).transpose(-1, -2),
        query.double() @ auxiliary.double().repeat_interleave(groups, dim=1).transpose(-1, -2),
    )
    errors = {}
    for name, function in functions.items():
        actual = function()
        torch.testing.assert_close(actual.double(), reference, rtol=2e-5, atol=2e-5)
        errors[name] = float((actual.double() - reference).norm() / reference.norm())
        del actual
    del reference
    order = list(functions)
    rng = random.Random(args.seed)
    samples: dict[str, list[dict[str, float | str | None]]] = {name: [] for name in order}
    medians: dict[str, list[float]] = {name: [] for name in order}
    for _ in range(args.samples):
        rng.shuffle(order)
        for name in order:
            timing = triton_benchmark(functions[name], _WARMUP_MS, args.rep_ms)
            samples[name].append(timing.as_dict())
            medians[name].append(timing.median_ms)
    return [
        BenchmarkRecord(
            benchmark="sparse_piper_scores",
            provider=name,
            shape={
                "sequence": sequence,
                "score_shape_bhqk": [args.batch, args.heads, rows, keys],
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "key_storage_blocks": storage,
            },
            configuration={
                "seed": args.seed,
                "case_id": args.case,
                "diagnostic": True,
                "query_stride": query.stride(),
                "key_stride": primary.stride(),
                "implementation": (
                    selected.__module__ if name == "selected" and selected is not None else "torch"
                ),
                "input_source": "synthetic_fp32_summaries",
                "scope": "summary scoring only; allocation and maximum epilogue included",
                "measurement_order": "shuffled_paired_panels",
                "panel_count": args.samples,
                "timing_summary": "quantiles_of_panel_medians",
            },
            timings=PhaseTimings(
                warmup_ms=_WARMUP_MS,
                measurement_time_ms=args.rep_ms,
                first_call_ms=None,
                preparation=None,
                prepared_execution=Timing.from_samples(medians[name], ClockDomain.DEVICE_EVENT),
                operator_end_to_end=None,
            ),
            environment=environment,
            extra={"relative_l2_vs_fp64": errors[name], "panels": samples[name]},
        )
        for name in functions
    ]


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("requires a CUDA or ROCm GPU")
    with device_context(torch.device("cuda", args.device)):
        environment = capture_environment(Path(__file__).resolve().parents[1])
        records = []
        for sequence in args.sequence:
            for rows in args.query_blocks or _query_chunks(sequence):
                for record in _benchmark(args, sequence, rows, environment):
                    records.append(record)
                    print(
                        f"S={sequence} Q={rows} {record.provider}: "
                        f"{record.timings.prepared_execution.display()} ms (device_event; "
                        "p50 [p20, p80] of panel medians)",
                        flush=True,
                    )
        write_records(records, output_target(args))


if __name__ == "__main__":
    main()
