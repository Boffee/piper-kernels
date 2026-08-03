"""Triton backend for rotated INT8 W8A8 linear layers."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _hadamard_stage(values, offsets, stride: tl.constexpr):
    """Apply one H4 Kronecker factor to a flattened regular Hadamard row."""
    digit = (offsets // stride) % 4
    base = offsets - digit * stride
    a = tl.gather(values, base, 0)
    b = tl.gather(values, base + stride, 0)
    c = tl.gather(values, base + 2 * stride, 0)
    d = tl.gather(values, base + 3 * stride, 0)
    return tl.where(
        digit == 0,
        a + b + c - d,
        tl.where(
            digit == 1,
            a + b - c + d,
            tl.where(digit == 2, a - b + c + d, -a + b + c + d),
        ),
    )


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
    pointers = x_ptr + row * row_width + group * group_size + offsets
    values = tl.load(pointers).to(tl.float32)

    values = _hadamard_stage(values, offsets, 1)
    values = _hadamard_stage(values, offsets, 4)
    if group_size >= 64:
        values = _hadamard_stage(values, offsets, 16)
    if group_size >= 256:
        values = _hadamard_stage(values, offsets, 64)

    tl.store(out_ptr + row * row_width + group * group_size + offsets, values * inverse_sqrt_group)


@triton.jit
def _quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block_size: tl.constexpr,
    input_dtype_code: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    values = tl.load(x_ptr + row * row_width + offsets, mask=mask, other=0.0)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0) / 127.0, 1e-30)
    if input_dtype_code == 1:
        scaled = (values / scale.to(tl.float16)).to(tl.float16)
    elif input_dtype_code == 2:
        scaled = (values / scale.to(tl.bfloat16)).to(tl.bfloat16)
    else:
        scaled = values / scale
    quantized = tl.clamp(libdevice.rint(scaled.to(tl.float32)), -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + row * row_width + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row, scale)


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

    activation_pointers = (
        activation_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    )
    weight_pointers = weight_ptr + offsets_n[None, :] * stride_wn + offsets_k[:, None] * stride_wk
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

    output_pointers = output_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on
    tl.store(
        output_pointers,
        result,
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


@torch.library.custom_op("piper_kernels::int8_convrot_linear", mutates_args=())
def triton_int8_convrot_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Run ConvRot activation rotation, dynamic quantization, and INT8 GEMM."""
    original_shape = activation.shape
    activation_2d = activation.reshape(-1, original_shape[-1]).contiguous()
    m, k = activation_2d.shape
    n = weight.shape[0]
    if m == 0 or n == 0:
        return activation.new_empty((*original_shape[:-1], n))

    groups_per_row = k // group_size
    rotated = torch.empty_like(activation_2d)
    _rotate_groups_kernel[(m * groups_per_row,)](
        activation_2d,
        rotated,
        k,
        groups_per_row,
        group_size=group_size,
        inverse_sqrt_group=group_size**-0.5,
        num_warps=4,
    )

    quantized = torch.empty_like(rotated, dtype=torch.int8)
    activation_scale = torch.empty(m, device=activation.device, dtype=torch.float32)
    quant_block = max(128, triton.next_power_of_2(k))
    if activation.dtype is torch.float16:
        input_dtype_code = 1
    elif activation.dtype is torch.bfloat16:
        input_dtype_code = 2
    else:
        input_dtype_code = 0
    _quantize_rows_kernel[(m,)](
        rotated,
        quantized,
        activation_scale,
        k,
        block_size=quant_block,
        input_dtype_code=input_dtype_code,
        num_warps=8,
    )

    output = torch.empty((m, n), device=activation.device, dtype=activation.dtype)
    block_m = 32 if m < 64 else 64
    block_n = 64 if n < 128 else 128
    block_k = 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    bias_pointer = bias if bias is not None else activation
    _int8_matmul_kernel[grid](
        quantized,
        weight,
        output,
        activation_scale,
        weight_scale,
        bias_pointer,
        m,
        n,
        k,
        quantized.stride(0),
        quantized.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        has_bias=bias is not None,
        num_stages=3,
        num_warps=4,
    )
    return output.reshape(*original_shape[:-1], n)


@triton_int8_convrot_linear.register_fake
def _triton_int8_convrot_linear_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
) -> torch.Tensor:
    return activation.new_empty((*activation.shape[:-1], weight.shape[0]))
