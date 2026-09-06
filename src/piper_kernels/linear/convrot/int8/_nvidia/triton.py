"""NVIDIA kernels and launchers for ConvRot INT8; custom ops live in ``.._ops``."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import math

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._input_activations import (
    apply_input_activation,
    input_activation_width,
)
from piper_kernels.linear.convrot import triton as convrot_backend

from .._kernels.triton import (
    int8_matmul_kernel,
    quantize_rows_kernel,
    rotate_quantize_rows_kernel,
)
from .._plan import LinearExecutionPlan, fused_preparation_chunks
from . import policy

_LARGE_MATMUL_GROUP_M_TILES = 16
_MEAN_BLOCK_M = 256
_MEAN_BLOCK_K = 128


@triton.jit
def _dequantized_input_mean_partial_kernel(
    input_ptr,
    input_scale_ptr,
    partial_ptr,
    block_lengths_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Sum one row block of the activation represented by prepared ConvRot storage."""
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    feature_offsets = feature_block * block_k + tl.arange(0, block_k)
    valid = (sequence_offsets[:, None] < sequence_length) & (
        feature_offsets[None, :] < input_features
    )
    if mask_block_lengths:
        block_lengths = tl.load(
            block_lengths_ptr + sequence_offsets // 64,
            mask=sequence_offsets < sequence_length,
            other=0,
        )
        valid &= sequence_offsets[:, None] % 64 < block_lengths[:, None]
    values = tl.load(
        input_ptr
        + (batch * sequence_length + sequence_offsets[:, None]) * input_features
        + feature_offsets[None, :],
        mask=valid,
        other=0,
    ).to(tl.float32)
    scales = tl.load(
        input_scale_ptr + batch * sequence_length + sequence_offsets,
        mask=sequence_offsets < sequence_length,
        other=0.0,
    )
    partial = tl.sum(values * scales[:, None], axis=0)
    tl.store(
        partial_ptr + (batch * row_block_count + row_block) * input_features + feature_offsets,
        partial,
        mask=feature_offsets < input_features,
    )


@triton.jit
def _dequantized_input_mean_reduce_kernel(
    partial_ptr,
    output_ptr,
    valid_count_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    reduction_rows: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    block_k: tl.constexpr,
):
    """Reduce prepared-input partial sums to one FP32 feature mean per batch."""
    feature_block = tl.program_id(0)
    batch = tl.program_id(1)
    row_offsets = tl.arange(0, reduction_rows)
    feature_offsets = feature_block * block_k + tl.arange(0, block_k)
    partial = tl.load(
        partial_ptr
        + (batch * row_block_count + row_offsets[:, None]) * input_features
        + feature_offsets[None, :],
        mask=(row_offsets[:, None] < row_block_count) & (feature_offsets[None, :] < input_features),
        other=0.0,
    )
    valid_count = tl.load(valid_count_ptr) if mask_block_lengths else sequence_length
    mean = tl.sum(partial, axis=0) / valid_count
    tl.store(
        output_ptr + batch * input_features + feature_offsets,
        mean,
        mask=feature_offsets < input_features,
    )


def quantize_input(
    rotated: torch.Tensor,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    logical_dtype_code: int,
    *,
    num_warps: int,
) -> None:
    """Apply the portable split-path rowwise quantization."""
    m, k = rotated.shape
    with device_context(rotated.device):
        quantize_rows_kernel[(m,)](
            rotated,
            input_qdata,
            input_scale,
            k,
            block_size=max(128, triton.next_power_of_2(k)),
            logical_dtype_code=logical_dtype_code,
            num_warps=num_warps,
        )


def fused_rotate_quantize_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    group_size: int,
    logical_dtype_code: int,
    *,
    activation_fn: str | None = None,
    num_warps: int,
    target: AcceleratorTarget | None = None,
) -> None:
    """Rotate and quantize to ``input_qdata`` without a rotated intermediate.

    ``input_qdata`` defines the activated row width. SwiGLU requires a raw
    ``[up | gate]`` input with twice that width.
    """
    if input.ndim != 2 or input_qdata.ndim != 2:
        raise ValueError(
            "fused preparation tensors must be 2-D, "
            f"got shapes {tuple(input.shape)} and {tuple(input_qdata.shape)}"
        )
    m, k = input_qdata.shape
    expected_input_shape = (m, k * input_activation_width(activation_fn))
    if tuple(input.shape) != expected_input_shape:
        raise ValueError(
            f"fused preparation input must have shape {expected_input_shape}, "
            f"got {tuple(input.shape)}"
        )
    target = AcceleratorTarget.from_device(input.device) if target is None else target
    if not policy.supports_preparation_target(target):
        raise ValueError(f"ConvRot INT8 preparation has no optimized policy for {target}")
    fused_chunks = fused_preparation_chunks(k)
    if fused_chunks is None:
        raise ValueError(f"fused preparation does not support row width {k}")
    chunk_count, chunk_size = fused_chunks
    with device_context(input.device):
        rotate_quantize_rows_kernel[(m,)](
            input,
            input_qdata,
            input_scale,
            k,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            logical_dtype_code=logical_dtype_code,
            activation_fn=activation_fn,
            accelerator_backend=target.backend,
            gguf_quant_type=-1,
            num_warps=num_warps,
        )


def default_execution_plan(
    weight_qdata: torch.Tensor,
    *,
    target: AcceleratorTarget | None = None,
) -> LinearExecutionPlan:
    """Resolve production policy, accepting an explicit target for offline tuning."""
    target = AcceleratorTarget.from_device(weight_qdata.device) if target is None else target
    return policy.select_execution_plan(
        target,
        in_features=weight_qdata.shape[1],
    )


def prepare_input_with_plan(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    in_features: int,
    group_size: int,
    *,
    activation_fn: str | None,
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
        input_scale = torch.empty(m, device=input.device, dtype=torch.float32)
        result = (
            input_qdata.reshape(*input.shape[:-1], in_features),
            input_scale.reshape(input.shape[:-1]),
        )
    else:
        result = out
    input_qdata = result[0].reshape(m, in_features)
    input_scale = result[1].reshape(m)
    logical_dtype_code = convrot_backend.logical_dtype_code(input.dtype)
    if execution_plan.fuse_rotation_quantization:
        fused_rotate_quantize_input(
            input_2d,
            input_qdata,
            input_scale,
            group_size,
            logical_dtype_code,
            activation_fn=activation_fn,
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
        quantize_input(
            rotated,
            input_qdata,
            input_scale,
            logical_dtype_code,
            num_warps=execution_plan.quantization_num_warps,
        )
    return result


def _prepare_input_with_production_plan(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    *,
    activation_fn: str | None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare an ordinary or activated input under production policy."""
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
        execution_plan=plan,
        target=target,
        out=out,
    )


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
    plan = execution_plan
    num_n_tiles = triton.cdiv(n, plan.matmul_block_n) * (2 if paired else 1)
    # Cache grouping and the M-tail split are intrinsic to the selected large-tile family,
    # rather than additional execution-plan axes.
    group_m = (
        _LARGE_MATMUL_GROUP_M_TILES
        if plan.matmul_block_m == 128 and plan.matmul_block_n == 256
        else 0
    )
    bias_pointer = bias if bias is not None else output

    with device_context(input_qdata.device):

        def launch_tiles(row_block_count: int, row_block_offset: int, *, aligned_m: bool) -> None:
            grid = (row_block_count * num_n_tiles,) if group_m else (row_block_count, num_n_tiles)
            int8_matmul_kernel[grid](
                input_qdata_2d,
                weight_qdata,
                output,
                input_scale_1d,
                weight_scale,
                bias_pointer,
                second_weight,
                second_scale,
                second_bias if second_bias is not None else output,
                m,
                n,
                k,
                output.stride(0),
                row_block_offset,
                block_m=plan.matmul_block_m,
                block_n=plan.matmul_block_n,
                block_k=plan.matmul_block_k,
                has_bias=bias is not None,
                paired=paired,
                second_has_bias=second_bias is not None,
                aligned_tiles=aligned_m
                and (n % plan.matmul_block_n == 0)
                and (k % plan.matmul_block_k == 0),
                group_m=group_m,
                num_stages=plan.matmul_num_stages,
                num_warps=plan.matmul_num_warps,
            )

        full_row_blocks = m // plan.matmul_block_m
        if group_m:
            if full_row_blocks:
                launch_tiles(full_row_blocks, 0, aligned_m=True)
            if m % plan.matmul_block_m:
                launch_tiles(1, full_row_blocks, aligned_m=False)
        else:
            launch_tiles(
                (m + plan.matmul_block_m - 1) // plan.matmul_block_m,
                0,
                aligned_m=False,
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
        else default_execution_plan(weight_qdata, target=target)
    )
    input_qdata, input_scale = prepare_input_with_plan(
        input,
        k,
        group_size,
        activation_fn=activation_fn,
        execution_plan=plan,
        target=target,
    )
    return execute_prepared_linear(
        input_qdata,
        input_scale,
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
) -> torch.Tensor:
    """Run ConvRot input rotation, dynamic quantization, and INT8 GEMM."""
    return run_linear(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        activation_fn=activation_fn,
    )


def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then rotate and quantize a linear input."""
    return _prepare_input_with_production_plan(
        input,
        group_size,
        activation_fn=activation_fn,
        out=out,
    )


def dequantized_input_mean(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the FP32 compact or valid-front padded prepared-input mean."""
    if input_qdata.ndim != 3 or input_qdata.dtype is not torch.int8:
        raise ValueError("ConvRot mean input must be [batch,sequence,features] INT8")
    batch, sequence_length, input_features = input_qdata.shape
    if input_scale.shape != (batch, sequence_length) or input_scale.dtype is not torch.float32:
        raise ValueError("ConvRot mean scale must be a batch/sequence FP32 matrix")
    if input_qdata.device.type != "cuda" or input_scale.device != input_qdata.device:
        raise ValueError("ConvRot mean operands must share a CUDA device")
    if not input_qdata.is_contiguous() or not input_scale.is_contiguous():
        raise ValueError("ConvRot mean operands must be contiguous")
    if block_lengths is not None and (
        sequence_length % 64
        or block_lengths.shape != (sequence_length // 64,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != input_qdata.device
        or not block_lengths.is_contiguous()
    ):
        raise ValueError("ConvRot mean block lengths must be one contiguous device INT32 per K64")
    row_block_count = int(triton.cdiv(sequence_length, _MEAN_BLOCK_M))
    partial = torch.empty(
        (batch, row_block_count, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    output = torch.empty(
        (batch, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else input_scale
    valid_count = block_lengths.sum(dtype=torch.float32) if has_block_lengths else input_scale
    with device_context(input_qdata.device):
        _dequantized_input_mean_partial_kernel[
            (row_block_count, triton.cdiv(input_features, _MEAN_BLOCK_K), batch)
        ](
            input_qdata,
            input_scale,
            partial,
            block_lengths_ptr,
            sequence_length,
            input_features=input_features,
            row_block_count=row_block_count,
            mask_block_lengths=has_block_lengths,
            block_m=_MEAN_BLOCK_M,
            block_k=_MEAN_BLOCK_K,
            num_warps=8,
        )
        _dequantized_input_mean_reduce_kernel[(triton.cdiv(input_features, _MEAN_BLOCK_K), batch)](
            partial,
            output,
            valid_count,
            sequence_length,
            input_features=input_features,
            row_block_count=row_block_count,
            reduction_rows=triton.next_power_of_2(row_block_count),
            mask_block_lengths=has_block_lengths,
            block_k=_MEAN_BLOCK_K,
            num_warps=8,
        )
        return output


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
    plan = default_execution_plan(weight_qdata)
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
