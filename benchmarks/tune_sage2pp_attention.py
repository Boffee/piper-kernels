"""Search SageAttention2++ execution plans offline on the active GPU."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import torch
from lib import (
    AttentionConfig,
    AttentionInputs,
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
from lib.attention_providers import qk_quantization_granularity

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage2pp import triton as sage_backend

_BLOCK_M_VALUES = (32, 64, 128)
_NUM_WARPS_VALUES = (2, 4, 8)
_NUM_STAGES_VALUES = (1, 2, 3, 4)
_OPTIONAL_LOOP_STAGES = ("none", "1", "2", "3", "4")
_LOAD_PATHS = {"pointer": False, "tensor-descriptor": True}
_KV_QUANTIZATION = {"separate": False, "fused": True}
_QUERY_QUANTIZATION = {"separate": False, "fused": True}
_SCORE_RECURRENCE = {"scaled": False, "unscaled": True}
_CAUSAL_BLOCK_ORDERS = {"forward": False, "reverse": True}
_LOOP_LICM = {"disabled": True, "enabled": False}
_PROBABILITY_CONVERSIONS = {"stock": False, "packed": True}


@dataclass(frozen=True, slots=True)
class _Sage2ppTuningChoice:
    """Tunable fields layered over the target's production execution plan."""

    block_m: int
    num_warps: int
    num_stages: int
    use_tensor_descriptors: bool
    fuse_kv_quantization: bool
    fuse_query_quantization: bool
    use_unscaled_score_recurrence: bool
    reverse_causal_blocks: bool
    loop_num_stages: int | None
    disable_loop_licm: bool
    use_packed_probability_conversion: bool

    @property
    def name(self) -> str:
        """Return a stable compact identifier for reports and terminal output."""
        tokens = (
            f"m{self.block_m}",
            f"w{self.num_warps}",
            f"s{self.num_stages}",
            "descriptor" if self.use_tensor_descriptors else "pointer",
            "fused-kv" if self.fuse_kv_quantization else "separate-kv",
            "fused-q" if self.fuse_query_quantization else "separate-q",
            "unscaled-score" if self.use_unscaled_score_recurrence else "scaled-score",
            "reverse" if self.reverse_causal_blocks else "forward",
            f"loop{self.loop_num_stages or 'none'}",
            "licm" if not self.disable_loop_licm else "no-licm",
            "packed-p" if self.use_packed_probability_conversion else "stock-p",
        )
        return "-".join(tokens)

    def as_dict(self) -> dict[str, object]:
        """Return requested choices even when the candidate is unsupported."""
        return asdict(self)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--kv-sequence", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--block-m", type=int, choices=_BLOCK_M_VALUES, nargs="+")
    parser.add_argument("--num-warps", type=int, choices=_NUM_WARPS_VALUES, nargs="+")
    parser.add_argument("--num-stages", type=int, choices=_NUM_STAGES_VALUES, nargs="+")
    parser.add_argument("--load-path", choices=tuple(_LOAD_PATHS), nargs="+")
    parser.add_argument("--kv-quantization", choices=tuple(_KV_QUANTIZATION), nargs="+")
    parser.add_argument("--query-quantization", choices=tuple(_QUERY_QUANTIZATION), nargs="+")
    parser.add_argument("--score-recurrence", choices=tuple(_SCORE_RECURRENCE), nargs="+")
    parser.add_argument(
        "--causal-block-order",
        choices=tuple(_CAUSAL_BLOCK_ORDERS),
        nargs="+",
    )
    parser.add_argument("--loop-num-stages", choices=_OPTIONAL_LOOP_STAGES, nargs="+")
    parser.add_argument("--loop-licm", choices=tuple(_LOOP_LICM), nargs="+")
    parser.add_argument(
        "--probability-conversion",
        choices=tuple(_PROBABILITY_CONVERSIONS),
        nargs="+",
    )
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


def _candidate_choices(
    args: argparse.Namespace,
    production_plan: sage_backend._Sage2ppExecutionPlan,
) -> tuple[_Sage2ppTuningChoice, ...]:
    """Form the explicit Cartesian search, defaulting omitted axes to production."""
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
            args.kv_quantization,
            production_plan.fuse_kv_quantization,
            _KV_QUANTIZATION,
        ),
        _mapped_axis(
            args.query_quantization,
            production_plan.fuse_query_quantization,
            _QUERY_QUANTIZATION,
        ),
        _mapped_axis(
            args.score_recurrence,
            production_plan.use_unscaled_score_recurrence,
            _SCORE_RECURRENCE,
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
        _mapped_axis(
            args.probability_conversion,
            production_plan.use_packed_probability_conversion,
            _PROBABILITY_CONVERSIONS,
        ),
    )
    candidate_count = math.prod(len(axis) for axis in axes)
    if candidate_count > args.max_candidates:
        raise SystemExit(
            f"search expands to {candidate_count} candidates; narrow the axes or increase "
            "--max-candidates"
        )
    return tuple(_Sage2ppTuningChoice(*values) for values in product(*axes))


def _resolve_plan(
    choice: _Sage2ppTuningChoice,
    production_plan: sage_backend._Sage2ppExecutionPlan,
    *,
    target: AcceleratorTarget,
    head_dim: int,
    is_causal: bool,
) -> sage_backend._Sage2ppExecutionPlan:
    if choice.use_tensor_descriptors and (
        not target.is_cuda_capability(12) or choice.block_m != 128 or head_dim != 128
    ):
        raise UnsupportedTuningCandidateError(
            "tensor descriptors require an SM12x D128 M128 specialization"
        )
    if choice.reverse_causal_blocks and not is_causal:
        raise UnsupportedTuningCandidateError("reverse block order requires causal attention")
    try:
        return replace(
            production_plan,
            block_m=choice.block_m,
            num_warps=choice.num_warps,
            num_stages=choice.num_stages,
            use_tensor_descriptors=choice.use_tensor_descriptors,
            fuse_kv_quantization=choice.fuse_kv_quantization,
            fuse_query_quantization=choice.fuse_query_quantization,
            use_unscaled_score_recurrence=choice.use_unscaled_score_recurrence,
            reverse_causal_blocks=choice.reverse_causal_blocks,
            loop_num_stages=choice.loop_num_stages,
            disable_loop_licm=choice.disable_loop_licm,
            use_packed_probability_conversion=choice.use_packed_probability_conversion,
        )
    except ValueError as error:
        raise UnsupportedTuningCandidateError(str(error)) from error


def _make_candidate(
    choice: _Sage2ppTuningChoice,
    inputs: AttentionInputs,
    *,
    production_plan: sage_backend._Sage2ppExecutionPlan,
    config: AttentionConfig,
    target: AcceleratorTarget,
    seed: int,
) -> TuningCandidate[AttentionInputs, torch.Tensor]:
    query, _, _ = inputs
    scale = config.scale if config.scale is not None else query.shape[-1] ** -0.5

    def make_provider() -> BenchmarkProvider[AttentionInputs, torch.Tensor]:
        plan = _resolve_plan(
            choice,
            production_plan,
            target=target,
            head_dim=query.shape[-1],
            is_causal=config.is_causal,
        )

        def prepare() -> AttentionInputs:
            # Match benchmark_attention.py: the hot device phase is the complete
            # public Sage operator, including statistics and quantization kernels.
            return inputs

        def run(prepared: AttentionInputs) -> torch.Tensor:
            prepared_query, prepared_key, prepared_value = prepared
            return sage_backend._run_sage_attention_2pp(
                prepared_query,
                prepared_key,
                prepared_value,
                scale,
                config.is_causal,
                execution_plan=plan,
            )

        return BenchmarkProvider(
            name=f"sage2pp-{choice.name}",
            prepare=prepare,
            run=run,
            synchronize=torch.cuda.synchronize,
            configuration={
                **config.as_dict(),
                "implementation": "pure_triton",
                "algorithm": "sage_attention_2pp",
                "qk_quantization": qk_quantization_granularity(target),
                "pv_accumulation": "fp32+fp16",
                "block_n": int(sage_backend._BLOCK_N),
                **plan.as_dict(),
                "seed": seed,
            },
        )

    return TuningCandidate(
        name=choice.name,
        configuration={
            **config.as_dict(),
            **choice.as_dict(),
            "implementation": "pure_triton",
            "algorithm": "sage_attention_2pp",
            "seed": seed,
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
        raise SystemExit("SageAttention2++ tuning requires an available accelerator GPU")
    device = torch.device("cuda")
    target = AcceleratorTarget.from_device(device)
    if not target.supports_fp8_fp16_mma:
        if target.is_amd_hip:
            raise SystemExit(
                "SageAttention2++ has no optimized HIP backend yet; gfx1200/gfx1201 "
                "cannot be tuned until the PTX conversion and FP8 MMA path are ported"
            )
        raise SystemExit("SageAttention2++ tuning requires NVIDIA SM89 or newer")

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
    config = AttentionConfig(dtype=args.dtype, is_causal=args.causal, scale=scale)
    production_plan = sage_backend._default_sage2pp_execution_plan(
        query,
        key,
        args.causal,
        target=target,
    )
    choices = _candidate_choices(args, production_plan)
    candidates = tuple(
        _make_candidate(
            choice,
            inputs,
            production_plan=production_plan,
            config=config,
            target=target,
            seed=args.seed,
        )
        for choice in choices
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
        tuning="sage2pp-execution-plan",
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
    """Run the SageAttention2++ execution-plan search."""
    _main()


if __name__ == "__main__":
    main()
