"""Benchmark fused sparse Piper separately from routing and input preparation.

Kernel TOPS counts only the useful selected QK and PV multiply/add operations;
it excludes softmax operations and does not count skipped blocks as work.
Graph timings reuse prepared Q/K/V, routes, and caller-owned output storage.
Public-call wall timings include routing, quantization, allocations, and launch.
Routing and preparation wall timings isolate those phases; independently timed
phases need not add up exactly to the complete public call.
Inputs are seeded synthetic BF16 tensors, not captured model activations.
"""

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import cast

import torch
from lib.environment import capture_environment
from lib.sparse_piper import assert_equal_finite, check_query_samples, useful_integer_operations
from lib.timing import synchronized_wall_benchmark
from triton.testing import do_bench, do_bench_cudagraph

from piper_kernels import SparsePiperAttention
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.sparse_piper_attention._backend import require_attention_backend
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._interfaces import LaunchAttention
from piper_kernels.attention.sparse_piper_attention._prepared import _PreparedSparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import routing_mode_from_name
from piper_kernels.attention.sparse_piper_attention.reference import (
    reference_sparse_piper_attention,
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=_positive_int, nargs="+", default=[1024, 4096])
    parser.add_argument("--heads", type=_positive_int, default=8)
    parser.add_argument("--batch", type=_positive_int, default=1)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.25, 1.0])
    parser.add_argument("--routing", choices=["minmax", "mean"], default="minmax")
    parser.add_argument("--rep-ms", type=_positive_int, default=100)
    parser.add_argument("--samples", type=_positive_int, default=3)
    parser.add_argument("--reference-max-sequence", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=931)
    args = parser.parse_args(argv)
    if min(args.sequence) < 64:
        parser.error("benchmark sequence lengths must be at least 64")
    if any(not 0 < ratio <= 1 for ratio in args.ratios):
        parser.error("ratios must be in (0, 1]")
    return args


def _benchmark(args: argparse.Namespace, sequence: int, ratio: float) -> None:
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.batch, sequence, args.heads, 128)
    query, key, value = [
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    ]
    backend = require_attention_backend(query)
    attention = SparsePiperAttention([ratio] * args.heads, routing=args.routing)
    blocks = sequence // 64
    layout = _resolve_route_layout(
        _normalize_head_keep_ratios([ratio] * args.heads), blocks, device
    )
    route_call = partial(
        packed_routes_from_sequences,
        query.transpose(1, 2),
        key.transpose(1, 2)[:, :, : blocks * 64],
        layout,
        routing_mode_from_name(args.routing),
    )
    routes = route_call()
    prepare_call = partial(
        backend.prepare,
        query.transpose(1, 2),
        routes.indices,
        routes.head_keep_blocks,
        128**-0.5,
        sparse_key_blocks=blocks,
        route_head_offsets=routes.route_head_offsets,
        combined_key=key.transpose(1, 2),
        combined_value=value.transpose(1, 2),
    )

    def prepare_execution(
        prepare_inputs: Callable[[], _PreparedSparsePiperAttention] = prepare_call,
    ) -> tuple[_PreparedSparsePiperAttention, LaunchAttention]:
        state = prepare_inputs()
        return state, backend.bind_context(state.context)

    prepared, bound_launch = prepare_execution()
    output = torch.empty_like(query)

    launch = partial(bound_launch, prepared, output.transpose(1, 2))

    def public_call() -> torch.Tensor:
        return attention(query, key, value, sparse_key_blocks=blocks)

    launch()
    assert_equal_finite(public_call(), output)
    reference_ms = None
    error = None
    if sequence <= args.reference_max_sequence:
        reference_call = partial(
            reference_sparse_piper_attention,
            query,
            key,
            value,
            routes,
            sparse_key_blocks=blocks,
            scale=128**-0.5,
        )

        reference = reference_call()
        error = float((output.float() - reference.float()).norm() / reference.float().norm())
        assert error < 0.015, error
        reference_ms = synchronized_wall_benchmark(
            reference_call, 0, args.rep_ms, synchronize=torch.cuda.synchronize
        ).median_ms
        del reference_call, reference
    cold = [
        cast(float, do_bench(launch, warmup=60, rep=args.rep_ms, return_mode="median"))
        for _ in range(args.samples)
    ]
    graph = [
        cast(float, do_bench_cudagraph(launch, rep=args.rep_ms, return_mode="median"))
        for _ in range(args.samples)
    ]
    samples = check_query_samples(prepared, output)
    routing = synchronized_wall_benchmark(
        route_call, 60, args.rep_ms, synchronize=torch.cuda.synchronize
    )
    preparation = synchronized_wall_benchmark(
        prepare_execution, 60, args.rep_ms, synchronize=torch.cuda.synchronize
    )
    selected_blocks = routes.head_keep_blocks.cpu().tolist()
    operations = useful_integer_operations(sequence, selected_blocks, args.batch)
    # Measure real public-call memory with no extra prepared benchmark state or
    # output retained. This matters at 100k tokens on a 16-GB accelerator.
    del launch, bound_launch, prepared, output, routes, route_call, prepare_execution, prepare_call
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    public = synchronized_wall_benchmark(
        public_call, 60, args.rep_ms, synchronize=torch.cuda.synchronize
    )
    print(
        json.dumps(
            {
                "shape_bnhd": shape,
                "input_source": "synthetic_bf16_normal",
                "ratio": ratio,
                "routing": args.routing,
                "selected_blocks_per_head": selected_blocks,
                "relative_l2_vs_quantized_reference": error,
                "reference_wall_ms_excluding_routing": reference_ms,
                "kernel_cache_flushed_samples_ms": cold,
                "kernel_graph_samples_ms": graph,
                "kernel_graph_median_ms": statistics.median(graph),
                "useful_integer_operations": operations,
                "kernel_graph_effective_tops": operations / statistics.median(graph) / 1e9,
                "routing_wall": routing.as_dict(),
                "preparation_wall": preparation.as_dict(),
                "preparation_includes_backend_packing": True,
                "public_call_wall": public.as_dict(),
                "sampled_fp64_reference_checks": samples,
                "public_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("requires a native sparse-attention GPU backend")
    with device_context(torch.device("cuda", args.device)):
        print(
            json.dumps(
                {
                    "environment": capture_environment(
                        Path(__file__).resolve().parents[1]
                    ).as_dict(),
                    "arguments": vars(args),
                }
            ),
            flush=True,
        )
        for sequence in args.sequence:
            for ratio in args.ratios:
                _benchmark(args, sequence, ratio)


if __name__ == "__main__":
    main()
