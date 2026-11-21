"""Search ConvRot INT8 forward-linear execution plans offline on the active GPU."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, cast

import torch
from lib.case_cli import apply_case
from lib.convrot import (
    CONVROT_DTYPE_NAMES,
    DENSE_LINEAR_ANCHOR_IN_FEATURES,
    DENSE_LINEAR_ANCHOR_OUT_FEATURES,
    DENSE_LINEAR_ANCHOR_ROWS,
    ConvRotConfig,
    ConvRotInputs,
    ConvRotShape,
    convrot_dtype,
    make_convrot_inputs,
)
from lib.environment import capture_environment
from lib.providers import Implementation, Operation
from lib.quality import measure_quality
from lib.reporting import output_target
from lib.tuning import (
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
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8.reference import linear as reference_linear
from piper_kernels.weights.convrot import SUPPORTED_GROUP_SIZES

_MATMUL_KERNEL_VALUES = ("triton", "gluon_async_copy", "gluon_tma")
_MATMUL_GROUP_M_VALUES = (0, 8, 16)


@dataclass(frozen=True, slots=True)
class ConvRotInt8Workload:
    """Tensors and production policy for one ConvRot INT8 tuning case."""

    shape: ConvRotShape
    config: ConvRotConfig
    inputs: ConvRotInputs
    production_plan: linear_backend.LinearExecutionPlan
    backend: linear_backend.LinearBackend

    def common_configuration(self) -> dict[str, object]:
        """Return provider-neutral workload metadata."""
        logical_input_layout = "up_gate" if self.shape.input_activation == "swiglu" else "plain"
        return {
            **self.config.as_dict(),
            "input_activation": self.shape.input_activation or "none",
            "logical_input_layout": logical_input_layout,
            "provider_input_layout": logical_input_layout,
            "has_bias": self.shape.has_bias,
            "prepared_execution_scope": "complete_operator_on_fixed_source_tensors",
        }

    def reference(self) -> torch.Tensor:
        """Evaluate the complete workload with the shared portable reference."""
        activation, qdata, scale, bias = self.inputs
        return reference_linear(
            activation,
            qdata,
            scale,
            self.config.group_size,
            bias,
            activation_fn=self.shape.input_activation,
        )


def make_convrot_int8_workload(
    shape: ConvRotShape,
    config: ConvRotConfig,
    *,
    device: torch.device,
    target: AcceleratorTarget | None = None,
) -> ConvRotInt8Workload:
    """Create tensors and resolve policy; an explicit target permits offline inspection."""
    resolved_target = AcceleratorTarget.from_device(device) if target is None else target
    backend = linear_backend.select_linear_backend_for_target(resolved_target)
    if backend is None:
        raise ValueError(f"ConvRot INT8 tuning has no optimized backend for {resolved_target}")
    inputs = make_convrot_inputs(shape, config, device=device)
    production_plan = backend.default_execution_plan(
        inputs[1], target=resolved_target, rows=shape.rows
    )
    return ConvRotInt8Workload(
        shape=shape,
        config=config,
        inputs=inputs,
        production_plan=production_plan,
        backend=backend,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--case", help="use an unchanged linear case from the shared catalog")
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
        choices=("gelu_tanh", "swiglu"),
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
    # The selected backend's plan validates this union of CUDA/ROCm tuning axes.
    parser.add_argument(
        "--fused-num-warps",
        type=int,
        choices=(1, 2, 4, 8, 16, 32),
        nargs="+",
    )
    parser.add_argument(
        "--rotation-num-warps",
        type=int,
        choices=(1, 2, 4, 8, 16, 32),
        nargs="+",
    )
    parser.add_argument(
        "--quantization-num-warps",
        type=int,
        choices=(1, 2, 4, 8, 16, 32),
        nargs="+",
    )
    parser.add_argument(
        "--matmul-block-m",
        type=int,
        choices=(16, 32, 64, 128, 256),
        nargs="+",
    )
    parser.add_argument(
        "--matmul-block-n",
        type=int,
        choices=(16, 32, 64, 128, 256),
        nargs="+",
    )
    parser.add_argument(
        "--matmul-block-k",
        type=int,
        choices=(16, 32, 64, 128, 256),
        nargs="+",
    )
    parser.add_argument(
        "--matmul-num-warps",
        type=int,
        choices=(2, 4, 8),
        nargs="+",
    )
    parser.add_argument(
        "--matmul-num-stages",
        type=int,
        choices=(1, 2, 3, 4),
        nargs="+",
    )
    parser.add_argument(
        "--matmul-kernel",
        choices=_MATMUL_KERNEL_VALUES,
        nargs="+",
        help="NVIDIA implementations to compare; omitted axes retain the production value",
    )
    parser.add_argument(
        "--matmul-group-m",
        type=int,
        choices=_MATMUL_GROUP_M_VALUES,
        nargs="+",
        help="NVIDIA GEMM row-tile grouping; zero disables grouping",
    )
    add_tuning_arguments(parser)
    return apply_case(parser.parse_args(argv), argv)


def _validate_args(args: argparse.Namespace) -> None:
    if any(value <= 0 for value in (args.rows, args.out_features, args.in_features)):
        raise SystemExit("rows, out_features, and in_features must all be positive")
    if args.in_features % args.group_size:
        raise SystemExit("in_features must be divisible by --group-size")
    validate_tuning_arguments(args)


def _candidate_plans(
    args: argparse.Namespace,
    production_plan: linear_backend.LinearExecutionPlan,
) -> tuple[linear_backend.LinearExecutionPlan, ...]:
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
    axes: dict[str, tuple[object, ...]] = {
        "fuse_rotation_quantization": fusion_axis,
        "fused_num_warps": fused_warps_axis,
        "rotation_num_warps": rotation_warps_axis,
        "quantization_num_warps": quantization_warps_axis,
        "matmul_block_m": tuning_axis(args.matmul_block_m, production_plan.matmul_block_m),
        "matmul_block_n": tuning_axis(args.matmul_block_n, production_plan.matmul_block_n),
        "matmul_block_k": tuning_axis(args.matmul_block_k, production_plan.matmul_block_k),
        "matmul_num_warps": tuning_axis(args.matmul_num_warps, production_plan.matmul_num_warps),
        "matmul_num_stages": tuning_axis(args.matmul_num_stages, production_plan.matmul_num_stages),
    }
    if isinstance(production_plan, linear_backend.nvidia_policy.NvidiaExecutionPlan):
        axes.update(
            matmul_kernel=tuning_axis(args.matmul_kernel, production_plan.matmul_kernel),
            matmul_group_m=tuning_axis(args.matmul_group_m, production_plan.matmul_group_m),
        )
    elif args.matmul_kernel is not None or args.matmul_group_m is not None:
        raise SystemExit("--matmul-kernel and --matmul-group-m require an NVIDIA execution plan")
    validate_tuning_candidate_count(tuple(axes.values()), args.max_candidates)
    plans = []
    skipped: dict[str, int] = {}
    for values in product(*axes.values()):
        try:
            # All backend execution plans are frozen dataclasses validated on construction.
            plan = cast(
                linear_backend.LinearExecutionPlan,
                replace(cast(Any, production_plan), **dict(zip(axes, values, strict=True))),
            )
        except ValueError as error:
            reason = str(error)
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            plans.append(plan)
    for reason, count in skipped.items():
        print(f"Skipped {count} unsupported candidate(s): {reason}", file=sys.stderr)
    if not plans:
        raise SystemExit("no supported execution plans in the requested search")
    return tuple(plans)


def _plan_name(plan: linear_backend.LinearExecutionPlan) -> str:
    preparation = (
        f"fused-pw{plan.fused_num_warps}"
        if plan.fuse_rotation_quantization
        else (f"split-rw{plan.rotation_num_warps}-qw{plan.quantization_num_warps}")
    )
    implementation = (
        f"{plan.matmul_kernel}-g{plan.matmul_group_m}-"
        if isinstance(plan, linear_backend.nvidia_policy.NvidiaExecutionPlan)
        else ""
    )
    return (
        f"{preparation}-{implementation}"
        f"m{plan.matmul_block_m}-n{plan.matmul_block_n}-k{plan.matmul_block_k}-"
        f"w{plan.matmul_num_warps}-s{plan.matmul_num_stages}"
    )


def _make_candidate(
    plan: linear_backend.LinearExecutionPlan,
    workload: ConvRotInt8Workload,
) -> Implementation[Operation[torch.Tensor]]:
    """Wrap one plan around the complete production-paid ConvRot device path."""
    name = _plan_name(plan)
    configuration: dict[str, object] = {
        **workload.common_configuration(),
        "algorithm": "convrot_int8_linear",
        **plan.as_dict(),
    }

    def build_operation() -> Operation[torch.Tensor]:
        activation, qdata, scale, bias = workload.inputs

        def run() -> torch.Tensor:
            return workload.backend.run_linear(
                activation,
                qdata,
                scale,
                bias,
                workload.config.group_size,
                activation_fn=workload.shape.input_activation,
                execution_plan=plan,
            )

        return Operation(
            run=run,
            configuration=configuration,
            synchronize=torch.cuda.synchronize,
        )

    return Implementation(
        name=name,
        configuration=configuration,
        build=build_operation,
    )


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("ConvRot INT8 linear tuning requires an available CUDA or ROCm GPU")
    device = torch.device("cuda")

    shape = ConvRotShape(
        args.case or "custom",
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
    workload = make_convrot_int8_workload(
        shape,
        config,
        device=device,
    )
    candidates = tuple(
        _make_candidate(plan, workload) for plan in _candidate_plans(args, workload.production_plan)
    )

    expected = workload.reference()
    run = tune_candidates(
        candidates,
        case_id=args.case,
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
