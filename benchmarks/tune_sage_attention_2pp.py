"""Search SageAttention2++ execution plans offline on the active GPU."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import torch
from lib.attention import (
    AttentionConfig,
    AttentionInputs,
    AttentionShape,
    attention_dtype,
    make_attention_inputs,
    run_sdpa,
)
from lib.attention_providers import qk_quantization_granularity
from lib.attention_tuning import (
    add_attention_tuning_arguments,
    validate_attention_tuning_arguments,
)
from lib.environment import capture_environment
from lib.providers import BenchmarkProvider
from lib.quality import measure_quality
from lib.reporting import output_target
from lib.tuning import (
    TuningCandidate,
    UnsupportedTuningCandidateError,
    boolean_tuning_axis,
    meets_minimum_sqnr,
    report_tuning_run,
    tune_candidates,
    tuning_axis,
    validate_tuning_candidate_count,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage_attention_2pp import _policy as sage_attention_2pp_policy
from piper_kernels.attention.sage_attention_2pp import triton as sage_attention_2pp_backend

_validate_args = validate_attention_tuning_arguments


@dataclass(frozen=True, slots=True)
class _SageAttention2ppTuningChoice:
    """Tunable fields layered over the target's production execution plan."""

    block_m: int
    num_warps: int
    num_stages: int
    use_tensor_descriptors: bool
    reverse_causal_blocks: bool
    loop_num_stages: int | None
    loop_licm: bool
    use_packed_probability_conversion: bool

    @property
    def name(self) -> str:
        """Return a stable compact identifier for reports and terminal output."""
        tokens = (
            f"m{self.block_m}",
            f"w{self.num_warps}",
            f"s{self.num_stages}",
            "descriptor" if self.use_tensor_descriptors else "pointer",
            "reverse" if self.reverse_causal_blocks else "forward",
            f"loop{self.loop_num_stages or 'default'}",
            "licm" if self.loop_licm else "no-licm",
            "packed-p" if self.use_packed_probability_conversion else "stock-p",
        )
        return "-".join(tokens)

    def as_dict(self) -> dict[str, object]:
        """Return requested choices even when the candidate is unsupported."""
        return asdict(self)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_attention_tuning_arguments(parser)
    return parser.parse_args(argv)


def _candidate_choices(
    args: argparse.Namespace,
    production_plan: sage_attention_2pp_policy.SageAttention2ppExecutionPlan,
) -> tuple[_SageAttention2ppTuningChoice, ...]:
    """Form the explicit Cartesian search, defaulting omitted axes to production."""
    axes = (
        tuning_axis(args.block_m, production_plan.block_m),
        tuning_axis(args.num_warps, production_plan.num_warps),
        tuning_axis(args.num_stages, production_plan.num_stages),
        boolean_tuning_axis(
            args.use_tensor_descriptors,
            production_plan.use_tensor_descriptors,
        ),
        boolean_tuning_axis(
            args.reverse_causal_blocks,
            production_plan.reverse_causal_blocks,
        ),
        tuning_axis(args.loop_num_stages, production_plan.loop_num_stages),
        boolean_tuning_axis(args.loop_licm, production_plan.loop_licm),
        boolean_tuning_axis(
            args.use_packed_probability_conversion,
            production_plan.use_packed_probability_conversion,
        ),
    )
    validate_tuning_candidate_count(axes, args.max_candidates)
    return tuple(_SageAttention2ppTuningChoice(*values) for values in product(*axes))


def _resolve_plan(
    choice: _SageAttention2ppTuningChoice,
    production_plan: sage_attention_2pp_policy.SageAttention2ppExecutionPlan,
    *,
    target: AcceleratorTarget,
    head_dim: int,
    is_causal: bool,
) -> sage_attention_2pp_policy.SageAttention2ppExecutionPlan:
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
            reverse_causal_blocks=choice.reverse_causal_blocks,
            loop_num_stages=choice.loop_num_stages,
            loop_licm=choice.loop_licm,
            use_packed_probability_conversion=choice.use_packed_probability_conversion,
        )
    except ValueError as error:
        raise UnsupportedTuningCandidateError(str(error)) from error


def _make_candidate(
    choice: _SageAttention2ppTuningChoice,
    inputs: AttentionInputs,
    *,
    production_plan: sage_attention_2pp_policy.SageAttention2ppExecutionPlan,
    config: AttentionConfig,
    target: AcceleratorTarget,
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
            # public SageAttention2++ operator, including statistics and quantization kernels.
            return inputs

        def run(prepared: AttentionInputs) -> torch.Tensor:
            prepared_query, prepared_key, prepared_value = prepared
            return sage_attention_2pp_backend._run_sage_attention_2pp(
                prepared_query,
                prepared_key,
                prepared_value,
                scale,
                config.is_causal,
                execution_plan=plan,
            )

        return BenchmarkProvider(
            name=f"sage_attention_2pp_{choice.name.replace('-', '_')}",
            prepare=prepare,
            run=run,
            synchronize=torch.cuda.synchronize,
            configuration={
                **config.as_dict(),
                "implementation": "pure_triton",
                "algorithm": "sage_attention_2pp",
                "qk_quantization": qk_quantization_granularity(target),
                "pv_accumulation": "fp32+fp16",
                "block_n": int(sage_attention_2pp_backend._BLOCK_N),
                **plan.as_dict(),
            },
        )

    return TuningCandidate(
        name=choice.name,
        configuration={
            **config.as_dict(),
            **choice.as_dict(),
            "implementation": "pure_triton",
            "algorithm": "sage_attention_2pp",
        },
        make_provider=make_provider,
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
    scale = args.head_dim**-0.5
    config = AttentionConfig(
        dtype=attention_dtype(args.dtype),
        is_causal=args.causal,
        scale=scale,
        seed=args.seed,
    )
    inputs = make_attention_inputs(shape, config=config, device=device)
    query, _, _ = inputs
    production_plan = sage_attention_2pp_backend._default_sage_attention_2pp_execution_plan(
        query,
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
        )
        for choice in choices
    )
    expected = run_sdpa(inputs, config)
    run = tune_candidates(
        candidates,
        tuning="sage_attention_2pp_execution_plan",
        shape=shape.as_dict(),
        environment=capture_environment(Path(__file__).resolve().parents[1]),
        phase=args.phase,
        warmup_ms=args.warmup_ms,
        measurement_time_ms=args.measurement_time_ms,
        measure_candidate_quality=lambda output: measure_quality(output, expected),
        quality_gate=lambda quality: meets_minimum_sqnr(quality, args.minimum_sqnr_db),
    )
    report_tuning_run(run, output_target(args))


def main() -> None:
    """Run the SageAttention2++ execution-plan search."""
    _main()


if __name__ == "__main__":
    main()
