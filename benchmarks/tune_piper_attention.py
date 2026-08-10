"""Tune a small Piper Attention schedule search offline."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import cast

import torch
from lib import (
    AttentionShape,
    BenchmarkProvider,
    TuningCandidate,
    TuningPhase,
    TuningRecord,
    UnsupportedTuningCandidateError,
    add_output_arguments,
    capture_environment,
    make_attention_inputs,
    measure_quality,
    output_target,
    tune_candidates,
    write_records,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import triton as piper_attention_backend
from piper_kernels.attention.piper_attention.dispatch import _default_center_value

_BLOCK_M_VALUES = (32, 64, 128)
_NUM_WARPS_VALUES = (2, 4, 8)
_NUM_STAGES_VALUES = (1, 2, 3, 4)
_LOAD_PATHS = {"pointer": False, "tensor-descriptor": True}
_CAUSAL_BLOCK_ORDERS = {"forward": False, "reverse": True}
_LOOP_LICM = {"disabled": True, "enabled": False}
_OPTIONAL_LOOP_STAGES = ("none", "1", "2", "3", "4")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--kv-sequence", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--center-value",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the production centering policy",
    )
    parser.add_argument("--load-path", choices=tuple(_LOAD_PATHS), nargs="+")
    parser.add_argument("--block-m", type=int, choices=_BLOCK_M_VALUES, nargs="+")
    parser.add_argument("--num-warps", type=int, choices=_NUM_WARPS_VALUES, nargs="+")
    parser.add_argument("--num-stages", type=int, choices=_NUM_STAGES_VALUES, nargs="+")
    parser.add_argument(
        "--causal-block-order",
        choices=tuple(_CAUSAL_BLOCK_ORDERS),
        nargs="+",
    )
    parser.add_argument("--loop-num-stages", choices=_OPTIONAL_LOOP_STAGES, nargs="+")
    parser.add_argument("--loop-licm", choices=tuple(_LOOP_LICM), nargs="+")
    parser.add_argument(
        "--phase",
        type=TuningPhase,
        choices=tuple(TuningPhase),
        default=TuningPhase.PREPARED_EXECUTION,
    )
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measurement-time-ms", type=int, default=200)
    parser.add_argument("--minimum-sqnr-db", type=float, default=20.0)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    add_output_arguments(parser, record_name="tuning candidate")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    lengths = (args.sequence, args.kv_sequence or args.sequence)
    if any(length <= 0 for length in lengths):
        raise SystemExit("attention sequence lengths must be positive")
    if args.batch_size <= 0 or args.heads <= 0:
        raise SystemExit("batch size and heads must be positive")
    if args.causal and lengths[0] != lengths[1]:
        raise SystemExit("causal attention requires equal query and key/value lengths")
    if (
        not args.causal
        and args.causal_block_order is not None
        and "reverse" in args.causal_block_order
    ):
        raise SystemExit("reverse causal-block order requires causal attention")
    if args.warmup_ms < 0 or args.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    if not math.isfinite(args.minimum_sqnr_db):
        raise SystemExit("minimum SQNR must be finite")
    if args.max_candidates <= 0:
        raise SystemExit("maximum candidate count must be positive")


def _unique[T](values: Sequence[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))


def _axis[T](values: Sequence[T] | None, production_value: T) -> tuple[T, ...]:
    return (production_value,) if values is None else _unique(values)


def _mapped_axis[T](
    values: Sequence[str] | None,
    production_value: T,
    mapping: Mapping[str, T],
) -> tuple[T, ...]:
    return (
        (production_value,)
        if values is None
        else _unique(tuple(mapping[value] for value in values))
    )


def _loop_stage_axis(
    values: Sequence[str] | None,
    production_value: int | None,
) -> tuple[int | None, ...]:
    if values is None:
        return (production_value,)
    return _unique(tuple(None if value == "none" else int(value) for value in values))


def _candidate_plans(
    args: argparse.Namespace,
    production_plan: piper_attention_backend._PiperAttentionExecutionPlan,
) -> tuple[piper_attention_backend._PiperAttentionExecutionPlan, ...]:
    """Build an explicit bounded search around production policy."""
    axes = (
        _axis(args.block_m, production_plan.block_m),
        _axis(args.num_warps, production_plan.num_warps),
        _axis(args.num_stages, production_plan.num_stages),
        _mapped_axis(
            args.load_path,
            production_plan.use_tensor_descriptors,
            _LOAD_PATHS,
        ),
        _mapped_axis(
            args.causal_block_order,
            production_plan.reverse_causal_blocks,
            _CAUSAL_BLOCK_ORDERS,
        ),
        _loop_stage_axis(args.loop_num_stages, production_plan.loop_num_stages),
        _mapped_axis(
            args.loop_licm,
            production_plan.disable_loop_licm,
            _LOOP_LICM,
        ),
    )
    plans = tuple(
        replace(
            production_plan,
            block_m=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
            use_tensor_descriptors=use_tensor_descriptors,
            reverse_causal_blocks=reverse_causal_blocks,
            loop_num_stages=loop_num_stages,
            disable_loop_licm=disable_loop_licm,
        )
        for (
            block_m,
            num_warps,
            num_stages,
            use_tensor_descriptors,
            reverse_causal_blocks,
            loop_num_stages,
            disable_loop_licm,
        ) in product(*axes)
    )
    if len(plans) > args.max_candidates:
        raise SystemExit(
            f"search expands to {len(plans)} candidates; narrow the axes or increase "
            "--max-candidates"
        )
    return plans


def _plan_name(plan: piper_attention_backend._PiperAttentionExecutionPlan) -> str:
    load_path = "descriptor" if plan.use_tensor_descriptors else "pointer"
    loop_stages = plan.loop_num_stages if plan.loop_num_stages is not None else "none"
    block_order = "reverse" if plan.reverse_causal_blocks else "forward"
    licm = "licm" if not plan.disable_loop_licm else "no-licm"
    return (
        f"{load_path}-m{plan.block_m}-w{plan.num_warps}-s{plan.num_stages}-"
        f"{block_order}-loop{loop_stages}-{licm}"
    )


def _make_candidate(
    plan: piper_attention_backend._PiperAttentionExecutionPlan,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    scale: float,
    is_causal: bool,
    center_value: bool,
    target: AcceleratorTarget,
    common_configuration: Mapping[str, object],
) -> TuningCandidate[object, torch.Tensor]:
    name = _plan_name(plan)

    def make_provider() -> BenchmarkProvider[object, torch.Tensor]:
        if plan.use_tensor_descriptors and not target.is_cuda_capability(12):
            raise UnsupportedTuningCandidateError(
                "the tensor-descriptor candidate currently targets SM12x"
            )
        query, key, value = inputs

        def prepare() -> object:
            return piper_attention_backend._prepare_piper_attention(
                query,
                key,
                value,
                scale,
                is_causal,
                center_value,
                execution_plan=plan,
            )

        def run(prepared: object) -> torch.Tensor:
            return piper_attention_backend._launch_piper_attention(
                cast(piper_attention_backend._PreparedPiperAttention, prepared)
            )

        return BenchmarkProvider(
            name=f"piper_attention_{name.replace('-', '_')}",
            prepare=prepare,
            run=run,
            synchronize=torch.cuda.synchronize,
        )

    return TuningCandidate(
        name=name,
        configuration={
            **common_configuration,
            "algorithm": "piper_attention",
            **plan.as_dict(),
            "center_value": center_value,
        },
        make_provider=make_provider,
    )


def _print_results(records: Sequence[TuningRecord]) -> None:
    print("| candidate | status | selected | p50 (ms) | SQNR (dB) | reason |")
    print("|:---|:---|:---:|---:|---:|:---|")
    for record in records:
        timing = "-" if record.timing is None else f"{record.timing.median_ms:.3f}"
        quality = "-" if record.quality is None else f"{record.quality.sqnr_db:.2f}"
        print(
            f"| {record.candidate} | {record.status.value} | {record.selected} "
            f"| {timing} | {quality} | {record.reason or ''} |"
        )


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("Piper Attention tuning requires an available NVIDIA GPU")
    device = torch.device("cuda")
    target = AcceleratorTarget.from_device(device)
    if not target.supports_uint8_int8_mma:
        raise SystemExit("Piper Attention tuning requires NVIDIA SM8x or SM12x")

    key_value_length = args.kv_sequence or args.sequence
    shape = AttentionShape(
        args.batch_size,
        args.heads,
        args.sequence,
        key_value_length,
        args.head_dim,
    )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    inputs = make_attention_inputs(shape, dtype=dtype, device=device, generator=generator)
    query, key, value = inputs
    scale = args.head_dim**-0.5
    center_value = (
        _default_center_value(query, key, args.causal, target)
        if args.center_value is None
        else args.center_value
    )
    production_plan = piper_attention_backend._default_piper_attention_execution_plan(
        query,
        key,
        args.causal,
        center_value,
        target=target,
        native_uint8=True,
    )
    common_configuration = {
        "dtype": args.dtype,
        "is_causal": args.causal,
        "scale": scale,
        "seed": args.seed,
    }
    candidates = tuple(
        _make_candidate(
            plan,
            inputs,
            scale=scale,
            is_causal=args.causal,
            center_value=center_value,
            target=target,
            common_configuration=common_configuration,
        )
        for plan in _candidate_plans(args, production_plan)
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=args.causal,
        scale=scale,
    )
    run = tune_candidates(
        candidates,
        tuning="piper_attention_execution_plan",
        shape=shape.as_dict(),
        environment=capture_environment(Path(__file__).resolve().parents[1]),
        phase=args.phase,
        warmup_ms=args.warmup_ms,
        measurement_time_ms=args.measurement_time_ms,
        measure_candidate_quality=lambda output: measure_quality(output, expected),
        quality_gate=lambda quality: (
            quality.nonfinite_mismatch_count == 0 and quality.sqnr_db >= args.minimum_sqnr_db
        ),
    )
    _print_results(run.records)
    write_records(run.records, output_target(args))
    if run.winner is None:
        raise SystemExit("no tuning candidate passed the quality gate")
    print(f"selected: {run.winner.candidate}")


def main() -> None:
    """Run the Piper Attention schedule search."""
    _main()


if __name__ == "__main__":
    main()
