"""Measure fixed-budget forced routing at MiniMax-H3 video sizes.

H3: 24 fps; valid output frames 17*n+5; latent frames 5*n+2; spatial
compression 16 and 1x2x2 patchification. 1280x768 gives a 24x40 token grid.
Frame tiling has 15 K64 tiles per frame; 4x4x4 tiling has 60 per temporal slab.
Forcing a whole slab covers every frame represented in a cube query tile.

Uses synthetic BF16 Q/K/V, B1/H56/D128, a configurable text suffix plus stereo
audio at 40 latent rows/sec/channel. No checkpoint, model quality measurement,
projection/output fusion, VAE, or complete generation is included. Each phase
uses shuffled paired timing panels. Public timings include routing, preparation,
allocation and attention; separately timed phases are not additive.
"""

import argparse
import math
import random
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from lib.environment import EnvironmentInfo, capture_environment
from lib.reporting import BenchmarkRecord, add_output_arguments, output_target, write_records
from lib.timing import (
    ClockDomain,
    PhaseTimings,
    SampleTimings,
    Timing,
    time_first_call,
    triton_benchmark,
)

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._routes import PackedRoutes
from piper_kernels.attention.sparse_piper_attention._routes_triton import (
    tiled_radix_select_packed_routes,
)
from piper_kernels.attention.sparse_piper_attention._routing import (
    packed_routes_from_summaries,
    routing_scores,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import routing_mode_from_name
from piper_kernels.attention.sparse_piper_attention._summaries import sequence_block_summaries


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, nargs="+", default=[5, 10, 15])
    parser.add_argument("--tiling", choices=["frame", "cube"], nargs="+", default=["frame", "cube"])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--routing", choices=["mean", "minmax"], default="mean")
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5341)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["selector", "routing", "attention", "public"],
        default=["selector", "routing", "attention", "public"],
    )
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    if min(args.seconds) <= 0 or any(not 0 < r < 1 for r in args.ratios):
        parser.error("requires positive durations and ratios in (0,1)")
    if args.samples < 1 or args.rep_ms < 1 or args.text_tokens < 0:
        parser.error("requires positive samples/duration and nonnegative text tokens")
    return args


def h3_shape(seconds: float, tiling: str, text_tokens: int = 256) -> dict[str, Any]:
    """Nearest valid H3 duration, with no conditioning/reference video."""
    chunks = max(1, round((seconds * 24 - 5) / 17))
    frames, latent_frames = 17 * chunks + 5, 5 * chunks + 2
    duration = frames / 24
    forced_count = 15 if tiling == "frame" else 60
    groups = latent_frames if tiling == "frame" else math.ceil(latent_frames / 4)
    video_blocks = groups * forced_count
    audio_tokens = 2 * round(duration * 40)
    dense_rows = text_tokens + audio_tokens
    total_blocks = video_blocks + math.ceil(dense_rows / 64)
    return {
        "requested_seconds": seconds,
        "actual_seconds": duration,
        "pixel_frames": frames,
        "width": 1280,
        "height": 768,
        "fps": 24,
        "latent_frames": latent_frames,
        "token_grid": [latent_frames, 24, 40],
        "tiling": tiling,
        "video_tokens": latent_frames * 960,
        "sparse_key_blocks": video_blocks,
        "forced_tiles_per_video_query": forced_count,
        "audio_tokens": audio_tokens,
        "text_tokens": text_tokens,
        "dense_rows": dense_rows,
        "query_blocks": total_blocks,
        "physical_sequence": total_blocks * 64,
        "heads": 56,
        "head_dim": 128,
        "batch": 1,
    }


def _force_inputs(shape: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    blocks, keys = shape["query_blocks"], shape["sparse_key_blocks"]
    count = shape["forced_tiles_per_video_query"]
    q = torch.arange(blocks, device=device, dtype=torch.int32)
    ids = (q // count * count)[:, None] + torch.arange(count, device=device, dtype=torch.int32)
    counts = torch.where(q < keys, count, 0).to(torch.int32)
    lengths = torch.full((blocks,), 64, device=device, dtype=torch.int32)
    if shape["tiling"] == "cube" and shape["latent_frames"] % 4:
        lengths[keys - count : keys] = (shape["latent_frames"] % 4) * 16
    if shape["dense_rows"] % 64:
        lengths[-1] = shape["dense_rows"] % 64
    return ids, counts, lengths


def _check_routes(routes: PackedRoutes, ids: torch.Tensor, counts: torch.Tensor, keys: int) -> None:
    """Validate sampled rows against explicit sets outside the timing region."""
    indices = routes.indices.cpu().long()
    offsets = routes.route_head_offsets.cpu().tolist()
    ids, counts = ids.cpu(), counts.cpu()
    for q in sorted(
        {0, min(14, indices.shape[1] - 1), indices.shape[1] // 2, indices.shape[1] - 1}
    ):
        required = set(ids[q, : counts[q]].tolist())
        for h in (0, 27, 55):
            selected = indices[0, q, offsets[h] : offsets[h + 1]].tolist()
            assert len(set(selected)) == len(selected)
            assert required.issubset(selected)
            assert min(selected) >= 0
            assert max(selected) < keys


def _measure(
    phase: str,
    functions: dict[str, Callable[[], Any]],
    args: argparse.Namespace,
    shape: dict[str, Any],
    ratio: float,
    records: list[BenchmarkRecord[Any]],
    environment: EnvironmentInfo,
) -> None:
    for function in functions.values():
        function()
    torch.cuda.synchronize()
    panels = {name: [] for name in functions}
    rng = random.Random(args.seed)
    for _ in range(args.samples):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            if phase == "public":
                elapsed = time_first_call(functions[name], torch.cuda.synchronize)[1]
            else:
                timing = triton_benchmark(functions[name], 10, args.rep_ms)
                elapsed = timing.median_ms
            panels[name].append(elapsed)
    for name, samples in panels.items():
        clock = ClockDomain.SYNCHRONIZED_WALL if phase == "public" else ClockDomain.DEVICE_EVENT
        timing = Timing.from_samples(samples, clock)
        records.append(
            BenchmarkRecord[Any](
                benchmark="sparse_piper_forced",
                provider=name,
                shape=shape,
                configuration={
                    "phase": phase,
                    "keep_ratio": ratio,
                    "routing": args.routing,
                    "input_source": "synthetic_bf16_qkv",
                    "seed": args.seed,
                    "force_scope": "shared_across_batch_and_heads",
                    "measurement_order": "shuffled_paired_panels",
                    "timing_summary": "quantiles_of_single_call_samples"
                    if phase == "public"
                    else "quantiles_of_panel_medians",
                    "fusion": "standalone_attention_only",
                },
                timings=SampleTimings(warmup_calls=1, samples_ms=tuple(samples))
                if phase == "public"
                else PhaseTimings(
                    warmup_ms=10,
                    measurement_time_ms=args.rep_ms,
                    first_call_ms=None,
                    preparation=None,
                    prepared_execution=timing,
                    operator_end_to_end=None,
                ),
                environment=environment,
                extra={"panel_medians_ms": samples},
            )
        )
        print(
            f"{shape['tiling']} {shape['requested_seconds']:g}s r={ratio:g} "
            f"{phase}/{name}: {timing.display()} ms",
            flush=True,
        )
    write_records(records, output_target(args))


def _benchmark(
    args: argparse.Namespace,
    shape: dict[str, Any],
    ratio: float,
    records: list[BenchmarkRecord[Any]],
    environment: EnvironmentInfo,
) -> None:
    device = torch.device("cuda", args.device)
    keys, blocks = shape["sparse_key_blocks"], shape["query_blocks"]
    ids, counts, lengths = _force_inputs(shape, device)
    zero_counts = torch.zeros_like(counts)
    layout = _resolve_route_layout(_normalize_head_keep_ratios((ratio,) * 56), keys, device)
    budgets = layout.head_keep_blocks.cpu().tolist()
    if min(budgets) < shape["forced_tiles_per_video_query"]:
        raise ValueError("forced tiles exceed the smallest physical head budget")
    shape = {
        **shape,
        "head_keep_blocks": budgets,
        "membership_bytes": blocks * math.ceil(keys / 32) * 4,
        "force_input_bytes": (ids.numel() + counts.numel()) * 4,
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    q, k, v = [
        torch.randn(
            (1, shape["physical_sequence"], 56, 128),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    ]
    mode = routing_mode_from_name(args.routing)
    summaries = sequence_block_summaries(
        q.transpose(1, 2), k.transpose(1, 2)[:, :, : keys * 64], mode, lengths
    )
    route_functions = {
        "baseline": partial(packed_routes_from_summaries, *summaries, layout, mode),
        "empty_force": partial(
            packed_routes_from_summaries,
            *summaries,
            layout,
            mode,
            force_tile_ids=ids,
            force_tile_counts=zero_counts,
        ),
        "same_frame": partial(
            packed_routes_from_summaries,
            *summaries,
            layout,
            mode,
            force_tile_ids=ids,
            force_tile_counts=counts,
        ),
    }
    if "selector" in args.phases:
        mask = _backend.build_force_tile_bitmask(ids, counts, keys)
        empty_mask = torch.zeros_like(mask)
        routes = torch.empty(
            (1, blocks, layout.routes_per_query), device=device, dtype=torch.uint16
        )
        for chunk in (64, 384):
            scores = routing_scores(summaries[0][:, :, :chunk], summaries[1], summaries[2], mode)
            select = partial(
                tiled_radix_select_packed_routes,
                scores,
                routes,
                layout.head_keep_blocks,
                layout.route_head_offsets,
                query_block_offset=0,
            )
            _measure(
                "selector",
                {
                    "baseline": select,
                    "empty_force": partial(select, force_tile_mask=empty_mask),
                    "same_frame": partial(select, force_tile_mask=mask),
                },
                args,
                {**shape, "score_query_blocks": chunk},
                ratio,
                records,
                environment,
            )
        _measure(
            "membership",
            {"build": partial(_backend.build_force_tile_bitmask, ids, counts, keys)},
            args,
            shape,
            ratio,
            records,
            environment,
        )
        del mask, empty_mask, routes, scores, select
    if "routing" in args.phases:
        _measure("routing", route_functions, args, shape, ratio, records, environment)
    if "attention" in args.phases:
        baseline_routes, forced_routes = (
            route_functions["baseline"](),
            route_functions["same_frame"](),
        )
        _check_routes(forced_routes, ids, counts, keys)
        backend = _backend.require_attention_backend(q)
        operands = backend.prepare(
            q.transpose(1, 2),
            128**-0.5,
            sparse_key_blocks=keys,
            combined_key=k.transpose(1, 2),
            combined_value=v.transpose(1, 2),
            block_lengths=lengths,
            sparse_query_blocks=keys,
        )
        output = torch.empty_like(q)
        launches = {}
        for name, routes in (("baseline", baseline_routes), ("same_frame", forced_routes)):
            prepared = operands.with_routes(
                routes.indices, routes.head_keep_blocks, routes.route_head_offsets
            )
            launches[name] = partial(
                backend.bind_context(prepared.context), prepared, output.transpose(1, 2)
            )
        _measure("attention", launches, args, shape, ratio, records, environment)
        del operands, output, prepared, routes, launches, baseline_routes, forced_routes
    del route_functions, summaries
    if "public" in args.phases:
        attention = SparsePiperAttention((ratio,) * 56, routing=args.routing)
        run = partial(
            attention,
            q,
            k,
            v,
            sparse_key_blocks=keys,
            sparse_query_blocks=keys,
            block_lengths=lengths,
        )
        _measure(
            "public",
            {
                "baseline": run,
                "empty_force": partial(run, force_tile_ids=ids, force_tile_counts=zero_counts),
                "same_frame": partial(run, force_tile_ids=ids, force_tile_counts=counts),
            },
            args,
            shape,
            ratio,
            records,
            environment,
        )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("requires a native sparse-attention GPU")
    with torch.cuda.device(args.device):
        environment = capture_environment(Path(__file__).resolve().parents[1])
        print(environment.as_dict(), flush=True)
        records = []
        for seconds in args.seconds:
            for tiling in args.tiling:
                for ratio in args.ratios:
                    _benchmark(
                        args,
                        h3_shape(seconds, tiling, args.text_tokens),
                        ratio,
                        records,
                        environment,
                    )
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
