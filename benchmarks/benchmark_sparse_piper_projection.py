"""Diagnose sparse-Piper Q/K/V projection phases for one catalog pipeline case.

Inputs use the parent case's dimensions, dtype, and seed with ConvRot INT8 weights.
Timings exclude input/weight preparation and attention. Q/K include
RMSNorm, RoPE, routing summaries, rotation and quantization; V includes projected
global means, block means and centered quantization. The combined measurement
also includes the represented-input mean reduction used by V.

Output buffers are reused. TOPS counts only 2*S*K*N INT8 multiply/add operations
per projection, not the FP32 epilogues or mean reduction. Independently timed
phases need not sum exactly to the combined measurement.
"""

import argparse
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import cast

import torch
from lib.case_cli import require_case
from lib.cases import PipelineCase, named_case
from lib.environment import EnvironmentInfo, capture_environment
from lib.inputs import normal_tensor
from lib.reporting import BenchmarkRecord, add_output_arguments, output_target, write_records
from lib.suite_pipeline import _GROUP_SIZE, _projection
from lib.timing import DeviceTimings, measure_device, synchronized_wall_benchmark

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.sparse_piper_attention._routing_modes import routing_mode_from_name
from piper_kernels.fusions.convrot_int8_sparse_piper import _backend, key, query, value
from piper_kernels.linear.convrot.int8 import _ops
from piper_kernels.linear.convrot.int8._backend import select_preparation_backend
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--case", default="sparse-pipeline-small", help="parent pipeline case identity"
    )
    parser.add_argument("--routing", choices=["minmax", "mean"], default="minmax")
    parser.add_argument(
        "--measurement-time-ms", "--rep-ms", dest="rep_ms", type=_positive_int, default=100
    )
    parser.add_argument("--warmup-ms", type=int, default=60)
    parser.add_argument("--samples", type=_positive_int, default=3)
    parser.add_argument("--device", type=int, default=0)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    case = require_case(args.case, PipelineCase)
    if case.sequence < 64 or case.width % _GROUP_SIZE:
        parser.error("requires at least 64 tokens and input width divisible by 64")
    if args.device < 0 or args.warmup_ms < 0:
        parser.error("requires non-negative device and warmup")
    return args


def _measure(
    function: Callable[[], object], operations: int, args: argparse.Namespace
) -> tuple[DeviceTimings, dict[str, object]]:
    timing = measure_device(
        function, warmup_ms=args.warmup_ms, measurement_time_ms=args.rep_ms, samples=args.samples
    )
    wall = synchronized_wall_benchmark(
        function, args.warmup_ms, args.rep_ms, synchronize=torch.cuda.synchronize
    )
    return timing, {
        "wall": wall.as_dict(),
        "integer_operations": operations,
        "cache_flushed_effective_tops": (
            operations / timing.cache_flushed.median_ms / 1e9 if operations else None
        ),
        "graph_effective_tops": operations / timing.graph.median_ms / 1e9 if operations else None,
    }


def _benchmark(
    args: argparse.Namespace,
    case: PipelineCase,
    environment: EnvironmentInfo,
) -> list[BenchmarkRecord[DeviceTimings]]:
    device = torch.device("cuda", args.device)
    backend = _backend.require_projection_backend(
        torch.empty(0, device=device), head_dim=case.head_dim
    )
    sequence, dtype = case.sequence, getattr(torch, case.dtype)
    source = normal_tensor(
        (case.batch, sequence, case.width), dtype=dtype, device=device, seed=case.seed
    )
    qdata, scale = select_preparation_backend(source).prepare_input(source, _GROUP_SIZE)
    del source
    weights = [
        cast(
            ConvRotInt8Tensor,
            _projection(
                case.width,
                heads * case.head_dim,
                dtype=dtype,
                device=device,
                seed=case.seed + offset,
            ).weight,
        )
        for offset, heads in enumerate((case.heads, case.kv_heads, case.kv_heads), start=1)
    ]
    query_norm, key_norm = (
        normal_tensor(
            (case.head_dim,), dtype=dtype, device=device, seed=case.seed + offset, scale=0.1
        )
        + 1
        for offset in (5, 6)
    )
    angles = normal_tensor(
        (sequence, case.rotary_dim),
        dtype=torch.float32,
        device=torch.device("cpu"),
        seed=case.seed + 7,
    )
    cos, sin = angles.cos().to(device), angles.sin().to(device)
    del angles
    routing = routing_mode_from_name(args.routing)
    mean_call = partial(_ops.dequantized_input_mean, qdata, scale)
    mean = mean_call()
    q_args = (
        qdata,
        scale,
        weights[0].qdata,
        weights[0].scale,
        query_norm,
        cos,
        sin,
        1e-5,
        case.head_dim**-0.5,
        routing,
    )
    k_args = (
        qdata,
        scale,
        weights[1].qdata,
        weights[1].scale,
        key_norm,
        cos,
        sin,
        1e-5,
        routing,
    )
    q = query._launch_query_projection(*q_args)
    k = key._launch_key_projection(*k_args)
    v = value._launch_value_projection(
        qdata,
        scale,
        mean,
        weights[2].qdata,
        weights[2].scale,
        None,
        emit_block_mean=True,
        head_dim=case.head_dim,
    )
    q_call = partial(
        backend.project_query, *q_args, None, chunk_start=0, chunk_rows=sequence, out=q
    )
    k_call = partial(backend.project_key, *k_args, None, out=k)

    def v_call(input_mean: torch.Tensor = mean) -> None:
        backend.project_value(
            qdata,
            scale,
            input_mean,
            weights[2].qdata,
            weights[2].scale,
            None,
            emit_block_mean=True,
            out=v,
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
    per_head_operations = 2 * case.batch * sequence * case.width * case.head_dim
    query_operations = per_head_operations * case.heads
    kv_operations = per_head_operations * case.kv_heads
    phases = {
        "query": (q_call, query_operations),
        "key": (k_call, kv_operations),
        "value": (v_call, kv_operations),
        "input_mean": (mean_call, 0),
        "mean_and_qkv": (combined_call, query_operations + 2 * kv_operations),
    }
    records = []
    for phase, (operation, integer_operations) in phases.items():
        timing, extra = _measure(operation, integer_operations, args)
        records.append(
            BenchmarkRecord(
                case_id=case.id,
                benchmark="sparse_piper_projection",
                provider="piper-convrot",
                shape=case.as_dict(),
                configuration={
                    "input_source": "cpu_seeded_synthetic_activations_and_packed_weights",
                    "routing": args.routing,
                    "emit_value_block_means": True,
                    "seed": case.seed,
                    "dtype": case.dtype,
                    "group_size": _GROUP_SIZE,
                    "phase": phase,
                    "output_allocation": "mean_only"
                    if phase in ("input_mean", "mean_and_qkv")
                    else "preallocated",
                },
                timings=timing,
                environment=environment,
                extra=extra,
            )
        )
        print(
            f"S={sequence} {phase}: cache-flushed {timing.cache_flushed.display()} ms; "
            f"graph {timing.graph.display()} ms",
            flush=True,
        )
    return records


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("requires a fused sparse-projection GPU backend")
    with device_context(torch.device("cuda", args.device)):
        environment = capture_environment(Path(__file__).resolve().parents[1])
        print(f"GPU: {environment.gpu_name}; backend: {environment.accelerator_backend}")
        case = named_case(args.case)
        assert isinstance(case, PipelineCase)
        records = _benchmark(args, case, environment)
        write_records(records, output_target(args))


if __name__ == "__main__":
    main()
