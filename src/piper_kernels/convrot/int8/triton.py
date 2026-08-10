"""Triton implementation of rotated INT8 W8A8 linear layers."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.stochastic_quantization import (
    _seed_argument,
    _stochastic_round_to_int,
)
from piper_kernels._triton.targets import AcceleratorTarget

from . import _policy
from .reference import _empty_inner_linear


@triton.jit
def _hadamard_stage_factorized(values, block_size: tl.constexpr, stride: tl.constexpr):
    """Apply one H4 factor with eight additions per independent quartet."""
    outer: tl.constexpr = block_size // (4 * stride)
    grouped = tl.reshape(values, (outer, 4, stride))
    quartets = tl.permute(grouped, (0, 2, 1))
    paired = tl.reshape(quartets, (outer, stride, 2, 2))
    ac, bd = tl.split(paired)
    a, c = tl.split(ac)
    b, d = tl.split(bd)
    p = a + b
    q = a - b
    r = c + d
    s = c - d
    y0 = p + s
    y1 = p - s
    y2 = q + r
    y3 = r - q
    y02 = tl.join(y0, y2)
    y13 = tl.join(y1, y3)
    transformed = tl.reshape(tl.join(y02, y13), (outer, stride, 4))
    transformed = tl.permute(transformed, (0, 2, 1))
    return tl.reshape(transformed, (block_size,))


@triton.jit
def _rotate_hadamard_groups(
    values,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
):
    """Apply every H4 factor within independent ConvRot groups."""
    values = _hadamard_stage_factorized(values, block_size, 1)
    values = _hadamard_stage_factorized(values, block_size, 4)
    if group_size >= 64:
        values = _hadamard_stage_factorized(values, block_size, 16)
    if group_size >= 256:
        values = _hadamard_stage_factorized(values, block_size, 64)
    return values


@triton.jit
def _normalize_for_int8(values, scale, logical_dtype_code: tl.constexpr):
    """Normalize values without dividing by an underflowed logical scale."""
    if logical_dtype_code == 1:
        logical_scale = scale.to(tl.float16)
        safe_logical_scale = tl.where(logical_scale == 0, 1.0, logical_scale).to(tl.float16)
        scaled = (values / safe_logical_scale).to(tl.float16)
        return tl.where(
            logical_scale == 0,
            values.to(tl.float32) / scale,
            scaled.to(tl.float32),
        )
    elif logical_dtype_code == 2:
        return (values / scale.to(tl.bfloat16)).to(tl.bfloat16)
    else:
        return values / scale


@triton.jit
def _rotate_groups_kernel(
    x_ptr,
    out_ptr,
    row_width,
    groups_per_row,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
):
    group_id = tl.program_id(0)
    row = group_id // groups_per_row
    group = group_id % groups_per_row
    offsets = tl.arange(0, group_size)
    row_offset = row.to(tl.int64) * row_width
    pointers = x_ptr + row_offset + group * group_size + offsets
    values = tl.load(pointers).to(tl.float32)

    values = _rotate_hadamard_groups(values, group_size, group_size)

    tl.store(out_ptr + row_offset + group * group_size + offsets, values * inverse_sqrt_group)


@triton.jit
def _rotate_quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    apply_swiglu: tl.constexpr,
):
    """Rotate and quantize one complete row without a global-memory intermediate."""
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    input_row_width = row_width * (2 if apply_swiglu else 1)
    input_row_offset = row_i64 * input_row_width
    output_row_offset = row_i64 * row_width

    if apply_swiglu:
        up = tl.load(
            x_ptr + input_row_offset + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            x_ptr + input_row_offset + row_width + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        activated_gate = gate / (1.0 + tl.exp(-gate))
        if logical_dtype_code == 1:
            activated_gate = activated_gate.to(tl.float16).to(tl.float32)
            values = (up * activated_gate).to(tl.float16).to(tl.float32)
        elif logical_dtype_code == 2:
            activated_gate = activated_gate.to(tl.bfloat16).to(tl.float32)
            values = (up * activated_gate).to(tl.bfloat16).to(tl.float32)
        else:
            values = up * activated_gate
    else:
        values = tl.load(
            x_ptr + input_row_offset + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    values = _rotate_hadamard_groups(values, block_size, group_size)
    values *= inverse_sqrt_group
    if logical_dtype_code == 1:
        values = values.to(tl.float16)
    elif logical_dtype_code == 2:
        values = values.to(tl.bfloat16)
    scale = tl.maximum(tl.max(tl.abs(values).to(tl.float32), axis=0) / 127.0, 1e-30)
    scaled = _normalize_for_int8(values, scale, logical_dtype_code)
    quantized = tl.clamp(libdevice.rint(scaled.to(tl.float32)), -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + output_row_offset + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row_i64, scale)


@triton.jit
def _quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block_size: tl.constexpr,
    logical_dtype_code: tl.constexpr,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    row_offset = row_i64 * row_width
    values = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0) / 127.0, 1e-30)
    scaled = _normalize_for_int8(values, scale, logical_dtype_code)
    quantized = tl.clamp(libdevice.rint(scaled.to(tl.float32)), -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + row_offset + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row_i64, scale)


@triton.jit
def _requantize_addmm_rows_kernel(
    q_ptr,
    scale_ptr,
    update_ptr,
    row_width,
    stride_q_row,
    stride_q_col,
    stride_scale_row,
    stride_update_row,
    stride_update_col,
    beta,
    alpha,
    rounding_seed,
    block_size: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    has_base: tl.constexpr,
    has_update: tl.constexpr,
    stochastic: tl.constexpr,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    offsets_i64 = offsets.to(tl.int64)
    mask = offsets < row_width
    if has_base:
        quantized = tl.load(
            q_ptr + row_i64 * stride_q_row + offsets_i64 * stride_q_col,
            mask=mask,
            other=0,
        )
        old_scale = tl.load(scale_ptr + row_i64 * stride_scale_row)
        values = beta * quantized.to(tl.float32) * old_scale
    else:
        values = tl.zeros((block_size,), dtype=tl.float32)
    if has_update:
        update = tl.load(
            update_ptr + row_i64 * stride_update_row + offsets_i64 * stride_update_col,
            mask=mask,
            other=0.0,
        )
        values += alpha * update.to(tl.float32)
    if logical_dtype_code == 1:
        values = values.to(tl.float16)
    elif logical_dtype_code == 2:
        values = values.to(tl.bfloat16)
    scale = tl.maximum(tl.max(tl.abs(values).to(tl.float32), axis=0) / 127.0, 1e-30)
    scaled = _normalize_for_int8(values, scale, logical_dtype_code)
    quantized = tl.clamp(libdevice.rint(scaled.to(tl.float32)), -128.0, 127.0).to(tl.int8)
    if stochastic:
        stochastic_scaled = values.to(tl.float32) / scale
        logical_offsets = row_i64 * row_width + offsets_i64
        quantized = _stochastic_round_to_int(
            stochastic_scaled,
            quantized,
            rounding_seed,
            logical_offsets,
            -128,
            127,
        ).to(tl.int8)
    tl.store(
        q_ptr + row_i64 * stride_q_row + offsets_i64 * stride_q_col,
        quantized,
        mask=mask,
    )
    tl.store(scale_ptr + row_i64 * stride_scale_row, scale)


@triton.jit
def _int8_matmul_kernel(
    activation_ptr,
    weight_ptr,
    output_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    offsets_k_i64 = offsets_k.to(tl.int64)

    activation_pointers = (
        activation_ptr + offsets_m_i64[:, None] * stride_am + offsets_k_i64[None, :] * stride_ak
    )
    weight_pointers = (
        weight_ptr + offsets_n_i64[None, :] * stride_wn + offsets_k_i64[:, None] * stride_wk
    )
    accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

    for k_offset in range(tl.cdiv(k, block_k)):
        activation = tl.load(
            activation_pointers,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k - k_offset * block_k),
            other=0,
        )
        weight = tl.load(
            weight_pointers,
            mask=(offsets_n[None, :] < n) & (offsets_k[:, None] < k - k_offset * block_k),
            other=0,
        )
        accumulator += tl.dot(activation, weight)
        activation_pointers += block_k * stride_ak
        weight_pointers += block_k * stride_wk

    activation_scale = tl.load(activation_scale_ptr + offsets_m, mask=offsets_m < m, other=0.0)
    weight_scale = tl.load(weight_scale_ptr + offsets_n, mask=offsets_n < n, other=0.0)
    result = accumulator.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]
    if has_bias:
        bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0)
        result += bias[None, :]

    output_pointers = (
        output_ptr + offsets_m_i64[:, None] * stride_om + offsets_n_i64[None, :] * stride_on
    )
    tl.store(
        output_pointers,
        result,
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


def _logical_dtype_code(dtype: torch.dtype) -> int:
    """Encode the logical floating-point dtype used while requantizing."""
    if dtype is torch.float16:
        return 1
    if dtype is torch.bfloat16:
        return 2
    return 0


def _rotate_activations(
    activation: torch.Tensor,
    rotated: torch.Tensor,
    group_size: int,
) -> None:
    """Apply the portable split-path activation rotation."""
    m, k = activation.shape
    groups_per_row = k // group_size
    _rotate_groups_kernel[(m * groups_per_row,)](
        activation,
        rotated,
        k,
        groups_per_row,
        group_size=group_size,
        inverse_sqrt_group=group_size**-0.5,
        num_warps=4,
    )


def _quantize_activations(
    rotated: torch.Tensor,
    activation_qdata: torch.Tensor,
    activation_scale: torch.Tensor,
    logical_dtype_code: int,
) -> None:
    """Apply the portable split-path rowwise quantization."""
    m, k = rotated.shape
    _quantize_rows_kernel[(m,)](
        rotated,
        activation_qdata,
        activation_scale,
        k,
        block_size=max(128, triton.next_power_of_2(k)),
        logical_dtype_code=logical_dtype_code,
        num_warps=8,
    )


def _fused_rotate_quantize_activations(
    activation: torch.Tensor,
    activation_qdata: torch.Tensor,
    activation_scale: torch.Tensor,
    group_size: int,
    logical_dtype_code: int,
    *,
    apply_swiglu: bool = False,
    plan: _policy.PreparationPlan | None = None,
) -> None:
    """Rotate and quantize to ``activation_qdata`` without a rotated intermediate.

    ``activation_qdata`` defines the logical row width. ``apply_swiglu`` requires
    a raw ``[up | gate]`` input with twice that width.
    """
    if activation.ndim != 2 or activation_qdata.ndim != 2:
        raise ValueError(
            "fused preparation tensors must be 2-D, "
            f"got shapes {tuple(activation.shape)} and {tuple(activation_qdata.shape)}"
        )
    m, k = activation_qdata.shape
    expected_input_shape = (m, k * (2 if apply_swiglu else 1))
    if tuple(activation.shape) != expected_input_shape:
        raise ValueError(
            f"fused preparation input must have shape {expected_input_shape}, "
            f"got {tuple(activation.shape)}"
        )
    if plan is None:
        plan = _policy.select_preparation_plan(
            AcceleratorTarget.from_device(activation.device),
            rows=m,
            in_features=k,
            group_size=group_size,
            dtype=activation.dtype,
            swiglu=apply_swiglu,
        )
    _rotate_quantize_rows_kernel[(m,)](
        activation,
        activation_qdata,
        activation_scale,
        k,
        block_size=plan.fused_block_size,
        group_size=group_size,
        inverse_sqrt_group=group_size**-0.5,
        logical_dtype_code=logical_dtype_code,
        apply_swiglu=apply_swiglu,
        num_warps=plan.fused_num_warps,
    )


def _int8_linear_from_quantized(
    source_activation: torch.Tensor,
    activation_qdata: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run the existing portable INT8 GEMM schedule on prepared activations."""
    m, k = activation_qdata.shape
    n = weight_qdata.shape[0]
    output = torch.empty(
        (m, n),
        device=source_activation.device,
        dtype=source_activation.dtype,
    )
    block_m = 32 if m < 64 else 64
    block_n = 64 if n < 128 else 128
    block_k = 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias_pointer = bias if bias is not None else source_activation
    _int8_matmul_kernel[grid](
        activation_qdata,
        weight_qdata,
        output,
        activation_scale,
        weight_scale,
        bias_pointer,
        m,
        n,
        k,
        activation_qdata.stride(0),
        activation_qdata.stride(1),
        weight_qdata.stride(0),
        weight_qdata.stride(1),
        output.stride(0),
        output.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        has_bias=bias is not None,
        num_stages=3,
        num_warps=4,
    )
    return output.reshape(*source_activation.shape[:-1], n)


def triton_convrot_int8_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Run ConvRot activation rotation, dynamic quantization, and INT8 GEMM."""
    original_shape = activation.shape
    n, k = qdata.shape
    if original_shape[-1] != k:
        raise ValueError(f"linear input has {original_shape[-1]} features, expected {k}")
    if k == 0:
        return _empty_inner_linear(activation, n, bias)
    activation_2d = activation.reshape(-1, original_shape[-1]).contiguous()
    m = activation_2d.shape[0]
    if m == 0 or n == 0:
        return activation.new_empty((*original_shape[:-1], n))

    logical_dtype_code = _logical_dtype_code(activation.dtype)
    activation_qdata = torch.empty_like(activation_2d, dtype=torch.int8)
    activation_scale = torch.empty(m, device=activation.device, dtype=torch.float32)
    plan = _policy.select_preparation_plan(
        AcceleratorTarget.from_device(activation.device),
        rows=m,
        in_features=k,
        group_size=group_size,
        dtype=activation.dtype,
    )
    if plan.fuse_rotation_quantization:
        _fused_rotate_quantize_activations(
            activation_2d,
            activation_qdata,
            activation_scale,
            group_size,
            logical_dtype_code,
            plan=plan,
        )
    else:
        rotated = torch.empty_like(activation_2d)
        _rotate_activations(activation_2d, rotated, group_size)
        _quantize_activations(
            rotated,
            activation_qdata,
            activation_scale,
            logical_dtype_code,
        )
    return _int8_linear_from_quantized(
        activation,
        activation_qdata,
        activation_scale,
        qdata,
        scale,
        bias,
    )


def triton_convrot_int8_swiglu_linear(
    activation: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Fuse ``[up | gate]`` SwiGLU with ConvRot preparation and INT8 GEMM."""
    original_shape = activation.shape
    n, k = qdata.shape
    if original_shape[-1] != 2 * k:
        raise ValueError(f"fused SwiGLU input has {original_shape[-1]} features, expected {2 * k}")
    if k == 0:
        return _empty_inner_linear(activation, n, bias)
    activation_2d = activation.reshape(-1, original_shape[-1]).contiguous()
    m = activation_2d.shape[0]
    if m == 0 or n == 0:
        return activation.new_empty((*original_shape[:-1], n))

    activation_qdata = torch.empty((m, k), device=activation.device, dtype=torch.int8)
    activation_scale = torch.empty(m, device=activation.device, dtype=torch.float32)
    plan = _policy.select_preparation_plan(
        AcceleratorTarget.from_device(activation.device),
        rows=m,
        in_features=k,
        group_size=group_size,
        dtype=activation.dtype,
        swiglu=True,
    )
    _fused_rotate_quantize_activations(
        activation_2d,
        activation_qdata,
        activation_scale,
        group_size,
        _logical_dtype_code(activation.dtype),
        apply_swiglu=True,
        plan=plan,
    )
    return _int8_linear_from_quantized(
        activation,
        activation_qdata,
        activation_scale,
        qdata,
        scale,
        bias,
    )


def triton_convrot_int8_addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Apply an addmm update in the rotated basis and requantize the weight in place."""
    out_features, in_features = qdata.shape
    if in_features == 0:
        scale.fill_(1e-30)
        return
    if out_features == 0:
        return
    has_update = alpha != 0 and mat1.shape[1] != 0
    if has_update:
        mat2_contiguous = mat2.contiguous()
        rotated_mat2 = torch.empty_like(mat2_contiguous)
        _rotate_activations(mat2_contiguous, rotated_mat2, group_size)
        update = torch.mm(mat1, rotated_mat2)
    else:
        update = qdata

    logical_dtype_code = _logical_dtype_code(mat1.dtype)
    requant_block = max(128, triton.next_power_of_2(in_features))
    _requantize_addmm_rows_kernel[(out_features,)](
        qdata,
        scale,
        update,
        in_features,
        qdata.stride(0),
        qdata.stride(1),
        scale.stride(0),
        update.stride(0),
        update.stride(1),
        beta,
        alpha,
        _seed_argument(rounding_seed),
        block_size=requant_block,
        logical_dtype_code=logical_dtype_code,
        has_base=beta != 0,
        has_update=has_update,
        stochastic=rounding_seed is not None,
        num_warps=8,
    )
