"""Triton implementation of rotated INT8 W8A8 linear layers."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..._rotation import build_hadamard


@triton.jit
def _hadamard_stage_factorized(values, block_size: tl.constexpr, stride: tl.constexpr):
    """Apply H4 using its optimal eight-add factorization per quartet."""
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
def _rotate_group_blocks_kernel(
    x_ptr,
    out_ptr,
    row_width,
    groups_per_row,
    group_size: tl.constexpr,
    groups_per_program: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
):
    """Rotate several adjacent groups in one program to amortize scheduling."""
    group_block_id = tl.program_id(0)
    blocks_per_row = groups_per_row // groups_per_program
    row = group_block_id // blocks_per_row
    group_block = group_block_id % blocks_per_row
    block_size: tl.constexpr = group_size * groups_per_program
    offsets = tl.arange(0, block_size)
    row_offset = row.to(tl.int64) * row_width
    pointers = x_ptr + row_offset + group_block * block_size + offsets
    values = tl.load(pointers).to(tl.float32)
    values = _rotate_hadamard_groups(values, block_size, group_size)
    tl.store(
        out_ptr + row_offset + group_block * block_size + offsets,
        values * inverse_sqrt_group,
    )


@triton.jit
def _rotate_groups_matmul_kernel(
    x_ptr,
    hadamard_ptr,
    out_ptr,
    group_rows,
    group_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Apply ConvRot groups as a dense tensor-core matrix multiplication."""
    offsets_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    offsets_n = tl.program_id(1) * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    offsets_k_i64 = offsets_k.to(tl.int64)
    x_pointers = x_ptr + offsets_m_i64[:, None] * group_size + offsets_k_i64[None, :]
    h_pointers = hadamard_ptr + offsets_k_i64[:, None] * group_size + offsets_n_i64[None, :]
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_offset in range(tl.cdiv(group_size, block_k)):
        values = tl.load(
            x_pointers,
            mask=(offsets_m[:, None] < group_rows)
            & (offsets_k[None, :] < group_size - k_offset * block_k),
            other=0.0,
        )
        hadamard = tl.load(
            h_pointers,
            mask=(offsets_n[None, :] < group_size)
            & (offsets_k[:, None] < group_size - k_offset * block_k),
            other=0.0,
        )
        accumulator += tl.dot(values, hadamard)
        x_pointers += block_k
        h_pointers += block_k * group_size
    tl.store(
        out_ptr + offsets_m_i64[:, None] * group_size + offsets_n_i64[None, :],
        accumulator,
        mask=(offsets_m[:, None] < group_rows) & (offsets_n[None, :] < group_size),
    )


@triton.jit
def _rotate_quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    input_dtype_code: tl.constexpr,
    input_act_code: tl.constexpr,
):
    """Rotate and quantize one complete row without an HBM intermediate."""
    row = tl.program_id(0)
    # A 128K SwiGLU input contains more than 2**31 BF16 elements, so form the
    # row base in 64 bits before multiplying by the raw 2K input width.
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    input_row_width = row_width * (2 if input_act_code == 1 else 1)
    input_row_offset = row_i64 * input_row_width
    output_row_offset = row_i64 * row_width
    values = tl.load(x_ptr + input_row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    if input_act_code == 1:
        up = tl.load(
            x_ptr + input_row_offset + row_width + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        activated_gate = values / (1.0 + tl.exp(-values))
        # Preserve both logical dtype boundaries in SiLU(gate) * up before
        # rotation, matching the materialized PyTorch expression.
        if input_dtype_code == 1:
            activated_gate = activated_gate.to(tl.float16).to(tl.float32)
            values = (activated_gate * up).to(tl.float16).to(tl.float32)
        elif input_dtype_code == 2:
            activated_gate = activated_gate.to(tl.bfloat16).to(tl.float32)
            values = (activated_gate * up).to(tl.bfloat16).to(tl.float32)
        else:
            values = activated_gate * up

    values = _rotate_hadamard_groups(values, block_size, group_size)
    values *= inverse_sqrt_group

    # Match the logical rotation boundary of the split path before choosing
    # the row-wide scale. This preserves its FP16/BF16 quantization semantics.
    if input_dtype_code == 1:
        values = values.to(tl.float16)
    elif input_dtype_code == 2:
        values = values.to(tl.bfloat16)
    scale = tl.maximum(tl.max(tl.abs(values).to(tl.float32), axis=0) / 127.0, 1e-30)
    scaled = _normalize_for_int8(values, scale, input_dtype_code)
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
    input_dtype_code: tl.constexpr,
):
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    row_offset = row_i64 * row_width
    values = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0) / 127.0, 1e-30)
    scaled = _normalize_for_int8(values, scale, input_dtype_code)
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
    block_size: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    has_base: tl.constexpr,
    has_update: tl.constexpr,
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
    tl.store(
        q_ptr + row_i64 * stride_q_row + offsets_i64 * stride_q_col,
        quantized,
        mask=mask,
    )
    tl.store(scale_ptr + row_i64 * stride_scale_row, scale)


@triton.jit
def _int8_matmul_kernel(  # noqa: PLR0912
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
    even_m: tl.constexpr,
    even_n: tl.constexpr,
    even_k: tl.constexpr,
    group_m: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    actual_group_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % num_pid_in_group) % actual_group_m
    pid_n = (pid % num_pid_in_group) // actual_group_m
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    # M * N exceeds signed 32-bit indexing for 128K-token QKV and FC1 outputs.
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
        if even_m and even_k:
            activation = tl.load(activation_pointers)
        else:
            activation = tl.load(
                activation_pointers,
                mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k - k_offset * block_k),
                other=0,
            )
        if even_n and even_k:
            weight = tl.load(weight_pointers)
        else:
            weight = tl.load(
                weight_pointers,
                mask=(offsets_n[None, :] < n) & (offsets_k[:, None] < k - k_offset * block_k),
                other=0,
            )
        accumulator += tl.dot(activation, weight)
        activation_pointers += block_k * stride_ak
        weight_pointers += block_k * stride_wk

    if even_m:
        activation_scale = tl.load(activation_scale_ptr + offsets_m)
    else:
        activation_scale = tl.load(activation_scale_ptr + offsets_m, mask=offsets_m < m, other=0.0)
    if even_n:
        weight_scale = tl.load(weight_scale_ptr + offsets_n)
    else:
        weight_scale = tl.load(weight_scale_ptr + offsets_n, mask=offsets_n < n, other=0.0)
    result = accumulator.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]
    if has_bias:
        if even_n:
            bias = tl.load(bias_ptr + offsets_n)
        else:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0)
        result += bias[None, :]

    output_pointers = (
        output_ptr + offsets_m_i64[:, None] * stride_om + offsets_n_i64[None, :] * stride_on
    )
    if even_m and even_n:
        tl.store(output_pointers, result)
    else:
        tl.store(
            output_pointers,
            result,
            mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
        )


@triton.jit
def _int8_matvec_kernel(
    activation_ptr,
    weight_ptr,
    output_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    n,
    k: tl.constexpr,
    stride_wn,
    stride_wk,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
):
    """Compute one INT8 output row without padding it to a tensor-core tile."""
    offsets_n = tl.program_id(0) * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    offsets_n_i64 = offsets_n.to(tl.int64)
    offsets_k_i64 = offsets_k.to(tl.int64)
    accumulator = tl.zeros((block_n,), dtype=tl.int32)
    for k_offset in range(tl.cdiv(k, block_k)):
        remaining_k = k - k_offset * block_k
        activation = tl.load(
            activation_ptr + k_offset * block_k + offsets_k,
            mask=offsets_k < remaining_k,
            other=0,
        ).to(tl.int32)
        weight = tl.load(
            weight_ptr
            + offsets_n_i64[:, None] * stride_wn
            + (k_offset * block_k + offsets_k_i64[None, :]) * stride_wk,
            mask=(offsets_n[:, None] < n) & (offsets_k[None, :] < remaining_k),
            other=0,
        ).to(tl.int32)
        accumulator += tl.sum(weight * activation[None, :], axis=1)

    result = accumulator.to(tl.float32) * tl.load(activation_scale_ptr)
    result *= tl.load(weight_scale_ptr + offsets_n, mask=offsets_n < n, other=0.0)
    if has_bias:
        result += tl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0)
    tl.store(output_ptr + offsets_n, result, mask=offsets_n < n)


def _is_blackwell(device: torch.device) -> bool:
    """Return whether SM120-specific launch heuristics were measured here."""
    return (
        torch.version.hip is None
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) >= (12, 0)
    )


def _rotate_activations(
    activation: torch.Tensor,
    rotated: torch.Tensor,
    group_size: int,
) -> None:
    m, k = activation.shape
    groups_per_row = k // group_size
    use_blackwell = _is_blackwell(activation.device)
    if use_blackwell and group_size == 256 and activation.dtype is not torch.float32 and m >= 512:
        # At this scale, expressing each rotation as a small dense matmul lets
        # Blackwell tensor cores outrun the elementwise butterfly network.
        hadamard = build_hadamard(group_size, activation.device, activation.dtype)
        group_rows = m * groups_per_row
        if group_rows <= 8192:
            block_m, block_n, block_k = 256, 64, 64
            num_warps = 8
        else:
            block_m, block_n, block_k = 64, 128, 64
            num_warps = 4
        _rotate_groups_matmul_kernel[
            (triton.cdiv(group_rows, block_m), triton.cdiv(group_size, block_n))
        ](
            activation,
            hadamard,
            rotated,
            group_rows,
            group_size=group_size,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_stages=3,
            num_warps=num_warps,
        )
    elif use_blackwell and group_size == 256 and m >= 256 and groups_per_row % 2 == 0:
        _rotate_group_blocks_kernel[(m * groups_per_row // 2,)](
            activation,
            rotated,
            k,
            groups_per_row,
            group_size=group_size,
            groups_per_program=2,
            inverse_sqrt_group=group_size**-0.5,
            num_warps=4,
        )
    else:
        _rotate_groups_kernel[(m * groups_per_row,)](
            activation,
            rotated,
            k,
            groups_per_row,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            num_warps=4,
        )


def _int8_matmul_config(
    m: int,
    n: int,
    k: int,
    use_blackwell: bool,
) -> tuple[int, int, int, int, int]:
    if not use_blackwell:
        return (32 if m < 64 else 64), (64 if n < 128 else 128), 32, 3, 4
    if m <= 64:
        return 64, 64, 128, 3, 4
    if m < 512:
        return 64, 128, 128, 4, 4
    if m < 768:
        return 128, 128, 128, 3, 4
    if k > 4096:
        return 128, 256, 128, 3, 8
    return 128, 256, 64, 3, 8


def _quantize_activations(
    rotated: torch.Tensor,
    quantized: torch.Tensor,
    activation_scale: torch.Tensor,
    input_dtype_code: int,
) -> None:
    m, k = rotated.shape
    quant_warps = 8 if m < 512 else (4 if m < 3072 else 2)
    _quantize_rows_kernel[(m,)](
        rotated,
        quantized,
        activation_scale,
        k,
        block_size=max(128, triton.next_power_of_2(k)),
        input_dtype_code=input_dtype_code,
        num_warps=quant_warps,
    )


def _can_fuse_rotation_quantization(
    m: int,
    k: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Return whether the measured one-row fused kernel is a safe fast path."""
    block_size = max(128, 1 << (k - 1).bit_length())
    return (
        group_size == 256
        and _is_blackwell(device)
        and dtype in (torch.float16, torch.bfloat16)
        and m >= 512
        and block_size <= 16_384
    )


def _fused_rotate_quantize_activations(
    activation: torch.Tensor,
    quantized: torch.Tensor,
    activation_scale: torch.Tensor,
    group_size: int,
    input_dtype_code: int,
    input_act_code: int = 0,
) -> None:
    """Rotate, find each row scale, and quantize without an HBM intermediate."""
    m, input_width = activation.shape
    k = input_width // 2 if input_act_code == 1 else input_width
    block_size = max(128, 1 << (k - 1).bit_length())
    large_swiglu = input_act_code == 1 and block_size == 16_384
    use_blackwell_wide_cta = large_swiglu and m >= 8192 and _is_blackwell(activation.device)
    num_warps = 16 if use_blackwell_wide_cta else (8 if large_swiglu else 4)
    _rotate_quantize_rows_kernel[(m,)](
        activation,
        quantized,
        activation_scale,
        k,
        block_size=block_size,
        group_size=group_size,
        inverse_sqrt_group=group_size**-0.5,
        input_dtype_code=input_dtype_code,
        input_act_code=input_act_code,
        num_warps=num_warps,
    )


def _input_dtype_code(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return 1
    if dtype is torch.bfloat16:
        return 2
    return 0


def _int8_linear_from_quantized(
    activation: torch.Tensor,
    quantized: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_prefix: tuple[int, ...],
) -> torch.Tensor:
    """Run the shared INT8 GEMM and scaled output epilogue."""
    m, k = quantized.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=activation.device, dtype=activation.dtype)
    bias_pointer = bias if bias is not None else activation
    if m == 1:
        block_n = 4
        _int8_matvec_kernel[(triton.cdiv(n, block_n),)](
            quantized,
            weight,
            output,
            activation_scale,
            weight_scale,
            bias_pointer,
            n,
            k,
            weight.stride(0),
            weight.stride(1),
            block_n=block_n,
            block_k=512,
            has_bias=bias is not None,
            num_warps=4,
        )
        return output.reshape(*output_prefix, n)

    use_blackwell = _is_blackwell(activation.device)
    block_m, block_n, block_k, num_stages, num_warps = _int8_matmul_config(
        m,
        n,
        k,
        use_blackwell,
    )
    num_m_tiles = triton.cdiv(m, block_m)
    num_n_tiles = triton.cdiv(n, block_n)
    # Flatten and group the output-tile grid so nearby programs reuse the
    # activation tile across N. Narrow outputs benefit from strict M-major
    # ordering; wider expansions retain a band of M tiles for weight reuse.
    group_m = 1 if not use_blackwell or num_n_tiles <= 32 else 64
    grid = (num_m_tiles * num_n_tiles,)
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
        even_m=m % block_m == 0,
        even_n=n % block_n == 0,
        even_k=k % block_k == 0,
        group_m=group_m,
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=2 if use_blackwell and m >= 768 and k <= 4096 else 1,
    )
    return output.reshape(*output_prefix, n)


@torch.library.custom_op("piper_kernels::convrot_int8_linear", mutates_args=())
def triton_convrot_int8_linear(
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

    input_dtype_code = _input_dtype_code(activation.dtype)
    quantized = torch.empty_like(activation_2d, dtype=torch.int8)
    activation_scale = torch.empty(m, device=activation.device, dtype=torch.float32)
    if _can_fuse_rotation_quantization(
        m,
        k,
        group_size,
        activation_2d.dtype,
        activation_2d.device,
    ):
        # The factorized H4 stages expose quartet reuse to Triton, which keeps
        # a complete row on chip while its scale is reduced and consumed.
        _fused_rotate_quantize_activations(
            activation_2d,
            quantized,
            activation_scale,
            group_size,
            input_dtype_code,
        )
    else:
        rotated = torch.empty_like(activation_2d)
        _rotate_activations(activation_2d, rotated, group_size)
        _quantize_activations(rotated, quantized, activation_scale, input_dtype_code)
    return _int8_linear_from_quantized(
        activation,
        quantized,
        activation_scale,
        weight,
        weight_scale,
        bias,
        tuple(original_shape[:-1]),
    )


@torch.library.custom_op("piper_kernels::convrot_int8_swiglu_linear", mutates_args=())
def triton_convrot_int8_swiglu_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Fuse SwiGLU, ConvRot activation preparation, and the INT8 GEMM."""
    original_shape = activation.shape
    activation_2d = activation.reshape(-1, original_shape[-1]).contiguous()
    m = activation_2d.shape[0]
    n, k = weight.shape
    if activation_2d.shape[1] != 2 * k:
        raise ValueError(
            f"fused SwiGLU input has {activation_2d.shape[1]} features, expected {2 * k}"
        )
    if m == 0 or n == 0:
        return activation.new_empty((*original_shape[:-1], n))

    quantized = torch.empty((m, k), device=activation.device, dtype=torch.int8)
    activation_scale = torch.empty(m, device=activation.device, dtype=torch.float32)
    _fused_rotate_quantize_activations(
        activation_2d,
        quantized,
        activation_scale,
        group_size,
        _input_dtype_code(activation.dtype),
        input_act_code=1,
    )
    return _int8_linear_from_quantized(
        activation,
        quantized,
        activation_scale,
        weight,
        weight_scale,
        bias,
        tuple(original_shape[:-1]),
    )


def triton_convrot_int8_addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
) -> None:
    """Apply an addmm update in the rotated basis and requantize the weight in place."""
    out_features, in_features = qdata.shape
    has_update = alpha != 0 and mat1.shape[1] != 0
    if has_update:
        mat2_contiguous = mat2.contiguous()
        rotated_mat2 = torch.empty_like(mat2_contiguous)
        groups_per_row = in_features // group_size
        _rotate_groups_kernel[(mat2.shape[0] * groups_per_row,)](
            mat2_contiguous,
            rotated_mat2,
            in_features,
            groups_per_row,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            num_warps=4,
        )
        update = torch.mm(mat1, rotated_mat2)
    else:
        update = qdata

    if mat1.dtype is torch.float16:
        logical_dtype_code = 1
    elif mat1.dtype is torch.bfloat16:
        logical_dtype_code = 2
    else:
        logical_dtype_code = 0
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
        block_size=requant_block,
        logical_dtype_code=logical_dtype_code,
        has_base=beta != 0,
        has_update=has_update,
        num_warps=8,
    )


@triton_convrot_int8_linear.register_fake
def _triton_convrot_int8_linear_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
) -> torch.Tensor:
    return activation.new_empty((*activation.shape[:-1], weight.shape[0]))


@triton_convrot_int8_swiglu_linear.register_fake
def _triton_convrot_int8_swiglu_linear_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
) -> torch.Tensor:
    return activation.new_empty((*activation.shape[:-1], weight.shape[0]))
