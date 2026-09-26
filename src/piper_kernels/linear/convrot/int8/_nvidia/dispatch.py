"""NVIDIA ConvRot orchestration over shared preparation and explicit GEMM implementations."""

import math

import torch

from piper_kernels._input_activations import apply_input_activation, input_activation_width
from piper_kernels._triton import convrot as convrot_backend
from piper_kernels._triton.targets import AcceleratorTarget

from .._plan import LinearExecutionPlan
from . import gluon_async_copy, policy
from . import triton as triton_kernels


def default_execution_plan(
    weight_qdata: torch.Tensor,
    *,
    target: AcceleratorTarget | None = None,
    rows: int | None = None,
) -> policy.NvidiaExecutionPlan:
    """Resolve production policy, accepting an explicit target for offline tuning."""
    target = AcceleratorTarget.from_device(weight_qdata.device) if target is None else target
    return policy.select_execution_plan(
        target,
        in_features=weight_qdata.shape[1],
        rows=rows,
        out_features=weight_qdata.shape[0],
    )


def prepare_input_with_plan(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    in_features: int,
    group_size: int,
    *,
    activation_fn: str | None,
    input_scale: torch.Tensor | None = None,
    execution_plan: LinearExecutionPlan,
    target: AcceleratorTarget,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and quantize an input, optionally into caller-owned storage."""
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    m = input_2d.shape[0]
    if out is None:
        input_qdata = torch.empty(
            (m, in_features),
            device=input.device,
            dtype=torch.int8,
        )
        row_scales = torch.empty(m, device=input.device, dtype=torch.float32)
        result = (
            input_qdata.reshape(*input.shape[:-1], in_features),
            row_scales.reshape(input.shape[:-1]),
        )
    else:
        result = out
    input_qdata = result[0].reshape(m, in_features)
    row_scales = result[1].reshape(m)
    if execution_plan.fuse_rotation_quantization:
        triton_kernels.fused_rotate_quantize_input(
            input_2d,
            input_qdata,
            row_scales,
            group_size,
            activation_fn=activation_fn,
            input_scale=input_scale,
            num_warps=execution_plan.fused_num_warps,
            target=target,
        )
    else:
        transformed_input = apply_input_activation(input_2d, activation_fn)
        rotated = torch.empty_like(transformed_input)
        convrot_backend.rotate_input(
            transformed_input,
            rotated,
            group_size,
            num_warps=execution_plan.rotation_num_warps,
        )
        triton_kernels.quantize_input(
            rotated,
            input_qdata,
            row_scales,
            input_scale=input_scale,
            num_warps=execution_plan.quantization_num_warps,
        )
    return result


def execute_prepared_linear(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
    execution_plan: LinearExecutionPlan,
    *,
    out: torch.Tensor | None = None,
    second_projection: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Project prepared INT8 input, optionally to two equal-width independent weights.

    Paired projections share a launch and write adjacent output columns without
    packing weights. ``out`` may provide caller-owned, row-strided storage.
    """
    leading_shape = input_qdata.shape[:-1]
    m = math.prod(leading_shape)
    k = input_qdata.shape[-1]
    n = weight_qdata.shape[0]
    paired = second_projection is not None
    second_weight, second_scale, second_bias = (
        (weight_qdata, weight_scale, None) if second_projection is None else second_projection
    )
    if second_weight.shape != weight_qdata.shape:
        raise ValueError("paired INT8 projections must have matching weight shapes")
    output_features = n * (2 if paired else 1)
    if out is None:
        output = torch.empty(
            (m, output_features),
            device=input_qdata.device,
            dtype=logical_dtype,
        )
        result = output.reshape(*leading_shape, output_features)
    else:
        result = out
    input_qdata_2d = input_qdata.reshape(m, k)
    input_scale_1d = input_scale.reshape(m)
    output = result.reshape(m, output_features)
    if output.stride(1) != 1:
        raise ValueError("prepared INT8 GEMM output must be column-contiguous")
    if not m or not n:
        return result
    if not isinstance(execution_plan, policy.NvidiaExecutionPlan):
        raise TypeError("NVIDIA ConvRot execution requires a NvidiaExecutionPlan")
    plan = execution_plan
    launcher = triton_kernels.launch_int8_matmul
    if plan.matmul_kernel == "gluon_async_copy":
        if gluon_async_copy.operands_aligned(input_qdata_2d, weight_qdata, second_weight):
            launcher = gluon_async_copy.launch_int8_matmul
        else:
            plan = policy.grouped_triton_plan(plan)
    launcher(
        input_qdata_2d,
        weight_qdata,
        output,
        input_scale_1d,
        weight_scale,
        bias,
        second_weight,
        second_scale,
        second_bias,
        paired=paired,
        plan=plan,
    )
    return result


def run_linear(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    *,
    activation_fn: str | None = None,
    input_scale: torch.Tensor | None = None,
    execution_plan: LinearExecutionPlan | None = None,
) -> torch.Tensor:
    """Run ConvRot input preparation and INT8 GEMM under one plan."""
    original_shape = input.shape
    k = weight_qdata.shape[1]
    expected_width = k * input_activation_width(activation_fn)
    if original_shape[-1] != expected_width:
        operation = "activated linear input" if activation_fn is not None else "linear input"
        raise ValueError(
            f"{operation} has {original_shape[-1]} features, expected {expected_width}"
        )
    target = AcceleratorTarget.from_device(input.device)
    plan = (
        execution_plan
        if execution_plan is not None
        else default_execution_plan(
            weight_qdata,
            target=target,
            rows=math.prod(original_shape[:-1]),
        )
    )
    input_qdata, row_scales = prepare_input_with_plan(
        input,
        k,
        group_size,
        activation_fn=activation_fn,
        input_scale=input_scale,
        execution_plan=plan,
        target=target,
    )
    return execute_prepared_linear(
        input_qdata,
        row_scales,
        weight_qdata,
        weight_scale,
        bias,
        input.dtype,
        plan,
    )


def linear(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    activation_fn: str | None = None,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ConvRot input rotation, quantization, and INT8 GEMM."""
    return run_linear(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        activation_fn=activation_fn,
        input_scale=input_scale,
    )


def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
    input_scale: torch.Tensor | None = None,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then rotate and quantize a linear input."""
    in_features = input.shape[-1] // input_activation_width(activation_fn)
    target = AcceleratorTarget.from_device(input.device)
    plan = policy.select_execution_plan(
        target,
        in_features=in_features,
    )
    return prepare_input_with_plan(
        input,
        in_features,
        group_size,
        activation_fn=activation_fn,
        input_scale=input_scale,
        execution_plan=plan,
        target=target,
        out=out,
    )


def linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
    second_projection: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Apply one weight to an input prepared by the matching operator."""
    plan = default_execution_plan(weight_qdata, rows=math.prod(input_qdata.shape[:-1]))
    return execute_prepared_linear(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        logical_dtype,
        plan,
        out=out,
        second_projection=second_projection,
    )
