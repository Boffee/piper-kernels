"""Search ConvRot INT8 forward-linear execution plans offline on the active GPU."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from itertools import product
from pathlib import Path

import torch
from lib.convrot import (
    CONVROT_DTYPE_NAMES,
    DENSE_LINEAR_ANCHOR_IN_FEATURES,
    DENSE_LINEAR_ANCHOR_OUT_FEATURES,
    DENSE_LINEAR_ANCHOR_ROWS,
    ConvRotConfig,
    ConvRotInputs,
    ConvRotShape,
    convrot_dtype,
)
from lib.convrot_providers import (
    ConvRotWorkload,
    make_convrot_workload,
    make_planned_convrot_provider,
    planned_convrot_configuration,
)
from lib.environment import capture_environment
from lib.providers import BenchmarkProvider
from lib.quality import measure_quality
from lib.reporting import output_target
from lib.tuning import (
    TuningCandidate,
    add_tuning_arguments,
    boolean_tuning_axis,
    meets_minimum_sqnr,
    report_tuning_run,
    tune_candidates,
    tuning_axis,
    validate_tuning_arguments,
    validate_tuning_candidate_count,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot._rotation import SUPPORTED_GROUP_SIZES
from piper_kernels.linear.convrot.int8 import _policy as convrot_policy


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=DENSE_LINEAR_ANCHOR_ROWS[0],
        help="activation rows M (default: 8192)",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=DENSE_LINEAR_ANCHOR_OUT_FEATURES[0],
        help="linear output width N (default: 4096)",
    )
    parser.add_argument(
        "--in-features",
        type=int,
        default=DENSE_LINEAR_ANCHOR_IN_FEATURES[0],
        help="linear/weight width K (default: 6144)",
    )
    parser.add_argument("--group-size", type=int, choices=SUPPORTED_GROUP_SIZES, default=256)
    parser.add_argument(
        "--input-activation",
        choices=("swiglu",),
        default=None,
    )
    parser.add_argument(
        "--bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include bias (default: disabled)",
    )
    parser.add_argument(
        "--dtype",
        choices=CONVROT_DTYPE_NAMES,
        default="bfloat16",
    )
    parser.add_argument(
        "--fuse-rotation-quantization",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fused-num-warps",
        type=int,
        choices=convrot_policy._FUSED_NUM_WARPS_VALUES,
        nargs="+",
    )
    parser.add_argument(
        "--rotation-num-warps",
        type=int,
        choices=convrot_policy._ROTATION_NUM_WARPS_VALUES,
        nargs="+",
    )
    parser.add_argument(
        "--quantization-num-warps",
        type=int,
        choices=convrot_policy._QUANTIZATION_NUM_WARPS_VALUES,
        nargs="+",
    )
    parser.add_argument(
        "--matmul-block-m",
        type=int,
        choices=convrot_policy._MATMUL_BLOCK_M_VALUES,
        nargs="+",
    )
    parser.add_argument(
        "--matmul-block-n",
        type=int,
        choices=convrot_policy._MATMUL_BLOCK_N_VALUES,
        nargs="+",
    )
    parser.add_argument(
        "--matmul-block-k",
        type=int,
        choices=convrot_policy._MATMUL_BLOCK_K_VALUES,
        nargs="+",
    )
    parser.add_argument(
        "--matmul-num-warps",
        type=int,
        choices=convrot_policy._MATMUL_NUM_WARPS_VALUES,
        nargs="+",
    )
    parser.add_argument(
        "--matmul-num-stages",
        type=int,
        choices=convrot_policy._MATMUL_NUM_STAGES_VALUES,
        nargs="+",
    )
    add_tuning_arguments(parser)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if any(value <= 0 for value in (args.rows, args.out_features, args.in_features)):
        raise SystemExit("rows, out_features, and in_features must all be positive")
    if args.in_features % args.group_size:
        raise SystemExit("in_features must be divisible by --group-size")
    validate_tuning_arguments(args)


def _candidate_plans(
    args: argparse.Namespace,
    production_plan: convrot_policy.ConvRotInt8LinearExecutionPlan,
) -> tuple[convrot_policy.ConvRotInt8LinearExecutionPlan, ...]:
    """Build a bounded explicit search around the production execution plan."""
    fusion_axis = boolean_tuning_axis(
        args.fuse_rotation_quantization,
        production_plan.fuse_rotation_quantization,
    )
    if not fusion_axis[0] and args.fused_num_warps is not None:
        raise SystemExit("--fused-num-warps requires --fuse-rotation-quantization")
    if fusion_axis[0] and (
        args.rotation_num_warps is not None or args.quantization_num_warps is not None
    ):
        raise SystemExit("split preparation warp axes require --no-fuse-rotation-quantization")
    fused_warps_axis = tuning_axis(
        args.fused_num_warps if fusion_axis[0] else None,
        production_plan.fused_num_warps,
    )
    rotation_warps_axis = tuning_axis(
        args.rotation_num_warps if not fusion_axis[0] else None,
        production_plan.rotation_num_warps,
    )
    quantization_warps_axis = tuning_axis(
        args.quantization_num_warps if not fusion_axis[0] else None,
        production_plan.quantization_num_warps,
    )
    axes = (
        fusion_axis,
        fused_warps_axis,
        rotation_warps_axis,
        quantization_warps_axis,
        tuning_axis(args.matmul_block_m, production_plan.matmul_block_m),
        tuning_axis(args.matmul_block_n, production_plan.matmul_block_n),
        tuning_axis(args.matmul_block_k, production_plan.matmul_block_k),
        tuning_axis(args.matmul_num_warps, production_plan.matmul_num_warps),
        tuning_axis(args.matmul_num_stages, production_plan.matmul_num_stages),
    )
    validate_tuning_candidate_count(axes, args.max_candidates)
    return tuple(
        replace(
            production_plan,
            fuse_rotation_quantization=fuse_rotation_quantization,
            fused_num_warps=fused_num_warps,
            rotation_num_warps=rotation_num_warps,
            quantization_num_warps=quantization_num_warps,
            matmul_block_m=block_m,
            matmul_block_n=block_n,
            matmul_block_k=block_k,
            matmul_num_warps=num_warps,
            matmul_num_stages=num_stages,
        )
        for (
            fuse_rotation_quantization,
            fused_num_warps,
            rotation_num_warps,
            quantization_num_warps,
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
        ) in product(*axes)
    )


def _plan_name(plan: convrot_policy.ConvRotInt8LinearExecutionPlan) -> str:
    preparation = (
        f"fused-pw{plan.fused_num_warps}"
        if plan.fuse_rotation_quantization
        else (f"split-rw{plan.rotation_num_warps}-qw{plan.quantization_num_warps}")
    )
    return (
        f"{preparation}-"
        f"m{plan.matmul_block_m}-n{plan.matmul_block_n}-k{plan.matmul_block_k}-"
        f"w{plan.matmul_num_warps}-s{plan.matmul_num_stages}"
    )


def _make_candidate(
    plan: convrot_policy.ConvRotInt8LinearExecutionPlan,
    workload: ConvRotWorkload,
) -> TuningCandidate[ConvRotInputs, torch.Tensor]:
    """Wrap one plan around the complete production-paid ConvRot device path."""
    name = _plan_name(plan)
    configuration = planned_convrot_configuration(workload, plan)

    def make_provider() -> BenchmarkProvider[ConvRotInputs, torch.Tensor]:
        return make_planned_convrot_provider(
            workload,
            plan,
            name=f"convrot_int8_linear_{name.replace('-', '_')}",
        )

    return TuningCandidate(
        name=name,
        configuration=configuration,
        make_provider=make_provider,
    )


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot INT8 linear tuning requires an available NVIDIA GPU")
    device = torch.device("cuda")
    target = AcceleratorTarget.from_device(device)
    if not target.is_nvidia_cuda or not target.cuda_capability_at_least(7, 5):
        raise SystemExit("ConvRot INT8 linear tuning requires NVIDIA SM75 or newer")

    shape = ConvRotShape(
        "custom",
        args.rows,
        args.out_features,
        args.in_features,
        args.input_activation,
        args.bias,
    )
    config = ConvRotConfig(
        dtype=convrot_dtype(args.dtype),
        group_size=args.group_size,
        seed=args.seed,
    )
    workload = make_convrot_workload(
        shape,
        config,
        device=device,
        target=target,
    )
    candidates = tuple(
        _make_candidate(plan, workload) for plan in _candidate_plans(args, workload.production_plan)
    )

    expected = workload.reference()
    run = tune_candidates(
        candidates,
        tuning="convrot_int8_linear_execution_plan",
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
    """Run the ConvRot INT8 forward-linear execution-plan search."""
    _main()


if __name__ == "__main__":
    main()
