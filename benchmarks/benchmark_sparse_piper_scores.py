"""Compare selected FP32 minmax scoring with the two-GEMM Torch baseline.

Uses synthetic summaries at H3's B1/H56/D128, with full and final query chunks
from a 4096-token fused pipeline. K retains the sparse-prefix view of the padded
sequence allocation. Scores are checked against FP64 before timing. Timings
include score allocation and the maximum epilogue, but not summary generation
or route selection. These are FP32 operations, not INT8 TOPS.
"""

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from lib.environment import EnvironmentInfo, capture_environment
from lib.reporting import BenchmarkRecord, add_output_arguments, output_target, write_records
from lib.timing import ClockDomain, PhaseTimings, Timing, triton_benchmark

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._routing import routing_scores
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING

_WARMUP_MS = 20


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, nargs="+", default=[8192, 32768, 100000])
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=273)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    if min(args.sequence) < 64 or args.samples < 1 or args.rep_ms < 1 or args.device < 0:
        parser.error("requires sequence >= 64, positive samples/rep-ms, and a nonnegative device")
    return args


def _query_chunks(sequence: int) -> list[int]:
    blocks = (sequence + 63) // 64
    full, tail = divmod(blocks, 64)
    return ([64] if full else []) + ([tail] if tail else [])


def _torch_scores(
    query: torch.Tensor, primary: torch.Tensor, auxiliary: torch.Tensor
) -> torch.Tensor:
    batch, heads, rows, width = query.shape
    keys = primary.shape[2]
    flat_query = query.reshape(batch * heads, rows, width)
    scores = torch.bmm(flat_query, primary.reshape(batch * heads, keys, width).transpose(1, 2))
    other = torch.bmm(flat_query, auxiliary.reshape(batch * heads, keys, width).transpose(1, 2))
    return torch.maximum(scores, other, out=scores).reshape(batch, heads, rows, keys)


def _benchmark(
    args: argparse.Namespace, sequence: int, rows: int, environment: EnvironmentInfo
) -> list[BenchmarkRecord]:
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed + sequence)
    keys, storage = sequence // 64, (sequence + 63) // 64
    query = torch.randn((1, 56, rows, 128), device=device, generator=generator) * 0.5
    primary = torch.randn((1, 56, storage, 128), device=device, generator=generator) * 0.5 + 2
    auxiliary = torch.randn((1, 56, storage, 128), device=device, generator=generator) * 0.5 - 2
    primary, auxiliary = primary[:, :, :keys], auxiliary[:, :, :keys]
    selected = _backend.select_minmax_scores(query, primary, auxiliary)
    functions = {
        "torch": lambda: _torch_scores(query, primary, auxiliary),
        "selected": lambda: routing_scores(query, primary, auxiliary, _MINMAX_ROUTING),
    }
    reference = torch.maximum(
        query.double() @ primary.double().transpose(-1, -2),
        query.double() @ auxiliary.double().transpose(-1, -2),
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
    samples: dict[str, list[dict[str, float | str]]] = {name: [] for name in order}
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
                "score_shape_bhqk": [1, 56, rows, keys],
                "head_dim": 128,
                "key_storage_blocks": storage,
            },
            configuration={
                "seed": args.seed,
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
            for rows in _query_chunks(sequence):
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
