"""Measure complete fused sparse-Piper Q/K/V projection phases at H3 dimensions.

Inputs are seeded synthetic BF16 tensors prepared with ConvRot INT8, not model
activations. Timings exclude input/weight preparation and attention. Q/K include
RMSNorm, RoPE, routing summaries, rotation and quantization; V includes projected
global means, block means and centered quantization. The combined measurement
also includes the represented-input mean reduction used by V.

Output buffers are reused. TOPS counts only 2*S*K*N INT8 multiply/add operations
per projection, not the FP32 epilogues or mean reduction. Independently timed
phases need not sum exactly to the combined measurement.
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
from lib.timing import synchronized_wall_benchmark
from triton.testing import do_bench, do_bench_cudagraph

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.sparse_piper_attention._routing_modes import routing_mode_from_name
from piper_kernels.fusions.convrot_int8_sparse_piper import _backend, key, query, value
from piper_kernels.linear.convrot.int8 import _ops
from piper_kernels.linear.convrot.int8._backend import select_preparation_backend


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=_positive_int, nargs="+", default=[8192])
    parser.add_argument("--heads", type=_positive_int, default=56)
    parser.add_argument("--input-features", type=_positive_int, default=5376)
    parser.add_argument("--batch", type=_positive_int, default=1)
    parser.add_argument("--routing", choices=["minmax", "mean"], default="minmax")
    parser.add_argument("--rep-ms", type=_positive_int, default=100)
    parser.add_argument("--samples", type=_positive_int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=882)
    args = parser.parse_args(argv)
    if min(args.sequence) < 64 or args.input_features % 256:
        parser.error("requires at least 64 tokens and input features divisible by 256")
    return args


def _measure(
    function: Callable[[], object], operations: int, args: argparse.Namespace
) -> dict[str, object]:
    cold = [
        cast(float, do_bench(function, warmup=60, rep=args.rep_ms, return_mode="median"))
        for _ in range(args.samples)
    ]
    graph = [
        cast(float, do_bench_cudagraph(function, rep=args.rep_ms, return_mode="median"))
        for _ in range(args.samples)
    ]
    wall = synchronized_wall_benchmark(
        function, 60, args.rep_ms, synchronize=torch.cuda.synchronize
    )
    return {
        "cache_flushed_samples_ms": cold,
        "graph_samples_ms": graph,
        "graph_median_ms": statistics.median(graph),
        "wall": wall.as_dict(),
        "integer_operations": operations,
        "cache_flushed_effective_tops": (
            operations / statistics.median(cold) / 1e9 if operations else None
        ),
        "graph_effective_tops": operations / statistics.median(graph) / 1e9 if operations else None,
    }


def _benchmark(args: argparse.Namespace, sequence: int) -> None:
    device = torch.device("cuda", args.device)
    backend = _backend.require_projection_backend(torch.empty(0, device=device))
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def prepare(shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        source = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        return select_preparation_backend(source).prepare_input(source, 256)

    qdata, scale = prepare((args.batch, sequence, args.input_features))
    weights = [prepare((args.heads * 128, args.input_features)) for _ in range(3)]
    weights = [(data, weight_scale.reshape(-1, 1)) for data, weight_scale in weights]
    norm = torch.rand(128, device=device, dtype=torch.bfloat16, generator=generator) + 0.5
    angles = torch.rand((sequence, 96), device=device, generator=generator)
    cos, sin = angles.cos(), angles.sin()
    del angles
    routing = routing_mode_from_name(args.routing)
    mean_call = partial(_ops.dequantized_input_mean, qdata, scale)
    mean = mean_call()
    q_args = (qdata, scale, *weights[0], norm, cos, sin, 1e-5, 128**-0.5, routing)
    k_args = (qdata, scale, *weights[1], norm, cos, sin, 1e-5, routing)
    q = query._launch_query_projection(*q_args)
    k = key._launch_key_projection(*k_args)
    v = value._launch_value_projection(qdata, scale, mean, *weights[2], None, emit_block_mean=True)
    q_call = partial(
        backend.project_query, *q_args, None, chunk_start=0, chunk_rows=sequence, out=q
    )
    k_call = partial(backend.project_key, *k_args, None, out=k)

    def v_call(input_mean: torch.Tensor = mean) -> None:
        backend.project_value(
            qdata, scale, input_mean, *weights[2], None, emit_block_mean=True, out=v
        )

    def combined_call() -> None:
        input_mean = mean_call()
        q_call()
        k_call()
        v_call(input_mean)

    combined_call()
    torch.cuda.synchronize()
    for output in (q, k, v):
        assert all(torch.isfinite(tensor).all() for tensor in output[1:])
    operations = 2 * args.batch * sequence * args.input_features * args.heads * 128
    phases = {
        "query": _measure(q_call, operations, args),
        "key": _measure(k_call, operations, args),
        "value": _measure(v_call, operations, args),
        "input_mean": _measure(mean_call, 0, args),
        "mean_and_qkv": _measure(combined_call, operations * 3, args),
    }
    print(
        json.dumps(
            {
                "shape_bskn": [args.batch, sequence, args.input_features, args.heads * 128],
                "input_source": "synthetic_bf16_normal",
                "routing": args.routing,
                "emit_value_block_means": True,
                "phases": phases,
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("requires a fused sparse-projection GPU backend")
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
            _benchmark(args, sequence)


if __name__ == "__main__":
    main()
