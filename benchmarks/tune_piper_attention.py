"""Tune a small Piper Attention schedule search offline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import cast

import torch
from lib.attention import (
    AttentionConfig,
    AttentionInputs,
    AttentionShape,
    attention_dtype,
    make_attention_inputs,
    run_sdpa,
)
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
from piper_kernels.attention.piper_attention import _policy as piper_attention_policy
from piper_kernels.attention.piper_attention import triton as piper_attention_backend

_validate_args = validate_attention_tuning_arguments


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_attention_tuning_arguments(parser)
    return parser.parse_args(argv)


def _candidate_plans(
    args: argparse.Namespace,
    production_plan: piper_attention_policy.PiperAttentionExecutionPlan,
) -> tuple[piper_attention_policy.PiperAttentionExecutionPlan, ...]:
    """Build an explicit bounded search around production policy."""
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
    plans = tuple(
        replace(
            production_plan,
            block_m=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
            use_tensor_descriptors=use_tensor_descriptors,
            reverse_causal_blocks=reverse_causal_blocks,
            loop_num_stages=loop_num_stages,
            loop_licm=loop_licm,
            use_packed_probability_conversion=use_packed_probability_conversion,
        )
        for (
            block_m,
            num_warps,
            num_stages,
            use_tensor_descriptors,
            reverse_causal_blocks,
            loop_num_stages,
            loop_licm,
            use_packed_probability_conversion,
        ) in product(*axes)
    )
    return plans


def _plan_name(plan: piper_attention_policy.PiperAttentionExecutionPlan) -> str:
    load_path = "descriptor" if plan.use_tensor_descriptors else "pointer"
    loop_stages = plan.loop_num_stages if plan.loop_num_stages is not None else "default"
    block_order = "reverse" if plan.reverse_causal_blocks else "forward"
    licm = "licm" if plan.loop_licm else "no-licm"
    probability_conversion = "packed-p" if plan.use_packed_probability_conversion else "stock-p"
    return (
        f"{load_path}-m{plan.block_m}-w{plan.num_warps}-s{plan.num_stages}-"
        f"{block_order}-loop{loop_stages}-{licm}-{probability_conversion}"
    )


def _make_candidate(
    plan: piper_attention_policy.PiperAttentionExecutionPlan,
    inputs: AttentionInputs,
    *,
    config: AttentionConfig,
    target: AcceleratorTarget,
) -> TuningCandidate[object, torch.Tensor]:
    name = _plan_name(plan)
    query, key, value = inputs
    scale = config.scale if config.scale is not None else query.shape[-1] ** -0.5

    def make_provider() -> BenchmarkProvider[object, torch.Tensor]:
        if plan.use_tensor_descriptors and not target.is_cuda_capability(12):
            raise UnsupportedTuningCandidateError(
                "the tensor-descriptor candidate currently targets SM12x"
            )

        def prepare() -> object:
            return piper_attention_backend._prepare_piper_attention(
                query,
                key,
                value,
                scale,
                config.is_causal,
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
            **config.as_dict(),
            "algorithm": "piper_attention",
            **plan.as_dict(),
        },
        make_provider=make_provider,
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
    scale = args.head_dim**-0.5
    config = AttentionConfig(
        dtype=attention_dtype(args.dtype),
        is_causal=args.causal,
        scale=scale,
        seed=args.seed,
    )
    inputs = make_attention_inputs(shape, config=config, device=device)
    query, key, _ = inputs
    production_plan = piper_attention_backend._default_piper_attention_execution_plan(
        query,
        key,
        args.causal,
        target=target,
    )
    candidates = tuple(
        _make_candidate(
            plan,
            inputs,
            config=config,
            target=target,
        )
        for plan in _candidate_plans(args, production_plan)
    )
    expected = run_sdpa(inputs, config)
    run = tune_candidates(
        candidates,
        tuning="piper_attention_execution_plan",
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
    """Run the Piper Attention schedule search."""
    _main()


if __name__ == "__main__":
    main()
