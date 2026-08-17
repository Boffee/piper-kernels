"""Triton implementation of rotated INT8 W8A8 linear layers."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.stochastic_quantization import (
    seed_argument,
    stochastic_round_to_int,
)
from piper_kernels._triton.targets import AcceleratorTarget

from . import _policy

_LARGE_MATMUL_GROUP_M_TILES = 16


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
def rotate_groups_kernel(
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
def rotate_quantize_rows_kernel(
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
def quantize_rows_kernel(
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
        quantized = stochastic_round_to_int(
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
    input_ptr,
    weight_ptr,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    m,
    n,
    k,
    row_block_count,
    row_block_offset,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    aligned_tiles: tl.constexpr,
    group_m: tl.constexpr,
):
    if group_m:
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(n, block_n)
        num_pid_in_group = group_m * num_pid_n
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * group_m
        actual_group_m = tl.minimum(row_block_count - first_pid_m, group_m)
        pid_m = first_pid_m + pid_in_group % actual_group_m + row_block_offset
        pid_n = pid_in_group // actual_group_m
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    offsets_k_i64 = offsets_k.to(tl.int64)

    input_pointers = input_ptr + offsets_m_i64[:, None] * k + offsets_k_i64[None, :]
    weight_pointers = weight_ptr + offsets_n_i64[None, :] * k + offsets_k_i64[:, None]
    accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

    for k_offset in range(tl.cdiv(k, block_k)):
        if aligned_tiles:
            input_values = tl.load(input_pointers)
            weight = tl.load(weight_pointers)
        else:
            input_values = tl.load(
                input_pointers,
                mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k - k_offset * block_k),
                other=0,
            )
            weight = tl.load(
                weight_pointers,
                mask=(offsets_n[None, :] < n) & (offsets_k[:, None] < k - k_offset * block_k),
                other=0,
            )
        accumulator += tl.dot(input_values, weight)
        input_pointers += block_k
        weight_pointers += block_k

    if aligned_tiles:
        input_scale = tl.load(input_scale_ptr + offsets_m)
        weight_scale = tl.load(weight_scale_ptr + offsets_n)
    else:
        input_scale = tl.load(
            input_scale_ptr + offsets_m,
            mask=offsets_m < m,
            other=0.0,
        )
        weight_scale = tl.load(
            weight_scale_ptr + offsets_n,
            mask=offsets_n < n,
            other=0.0,
        )
    result = accumulator.to(tl.float32) * input_scale[:, None] * weight_scale[None, :]
    if has_bias:
        if aligned_tiles:
            bias = tl.load(bias_ptr + offsets_n)
        else:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0)
        result += bias[None, :]

    output_pointers = output_ptr + offsets_m_i64[:, None] * n + offsets_n_i64[None, :]
    if aligned_tiles:
        tl.store(output_pointers, result)
    else:
        tl.store(
            output_pointers,
            result,
            mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
        )


def dtype_code(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return 1
    if dtype is torch.bfloat16:
        return 2
    return 0


def _uses_swiglu(activation_fn: str | None) -> bool:
    if activation_fn not in (None, "swiglu"):
        raise ValueError(
            f"ConvRot input activation must be 'swiglu' or None, got {activation_fn!r}"
        )
    return activation_fn == "swiglu"


def rotate_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    rotated: torch.Tensor,
    group_size: int,
    *,
    num_warps: int,
) -> None:
    """Apply the split-path input rotation."""
    m, k = input.shape
    groups_per_row = k // group_size
    rotate_groups_kernel[(m * groups_per_row,)](
        input,
        rotated,
        k,
        groups_per_row,
        group_size=group_size,
        inverse_sqrt_group=group_size**-0.5,
        num_warps=num_warps,
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
    apply_swiglu: bool = False,
    num_warps: int,
) -> None:
    """Rotate and quantize to ``input_qdata`` without a rotated intermediate.

    ``input_qdata`` defines the logical row width. ``apply_swiglu`` requires
    a raw ``[up | gate]`` input with twice that width.
    """
    if input.ndim != 2 or input_qdata.ndim != 2:
        raise ValueError(
            "fused preparation tensors must be 2-D, "
            f"got shapes {tuple(input.shape)} and {tuple(input_qdata.shape)}"
        )
    m, k = input_qdata.shape
    expected_input_shape = (m, k * (2 if apply_swiglu else 1))
    if tuple(input.shape) != expected_input_shape:
        raise ValueError(
            f"fused preparation input must have shape {expected_input_shape}, "
            f"got {tuple(input.shape)}"
        )
    block_size = max(128, triton.next_power_of_2(k))
    rotate_quantize_rows_kernel[(m,)](
        input,
        input_qdata,
        input_scale,
        k,
        block_size=block_size,
        group_size=group_size,
        inverse_sqrt_group=group_size**-0.5,
        logical_dtype_code=logical_dtype_code,
        apply_swiglu=apply_swiglu,
        num_warps=num_warps,
    )


def default_convrot_int8_execution_plan(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    group_size: int,
    *,
    activation_fn: str | None = None,
    target: AcceleratorTarget | None = None,
) -> _policy.ConvRotInt8LinearExecutionPlan:
    """Resolve production policy for execution, benchmarks, and offline tuning."""
    rows = math.prod(input.shape[:-1])
    apply_swiglu = _uses_swiglu(activation_fn)
    target = AcceleratorTarget.from_device(input.device) if target is None else target
    return _policy.select_execution_plan(
        target,
        rows=rows,
        out_features=weight_qdata.shape[0],
        in_features=weight_qdata.shape[1],
        group_size=group_size,
        dtype=input.dtype,
        swiglu=apply_swiglu,
    )


def _prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    in_features: int,
    group_size: int,
    *,
    apply_swiglu: bool,
    execution_plan: _policy.ConvRotInt8LinearExecutionPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and dynamically quantize a linear input for one or more weights."""
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    m = input_2d.shape[0]
    input_qdata = torch.empty(
        (m, in_features),
        device=input.device,
        dtype=torch.int8,
    )
    input_scale = torch.empty(m, device=input.device, dtype=torch.float32)
    logical_dtype_code = dtype_code(input.dtype)
    if execution_plan.fuse_rotation_quantization:
        fused_rotate_quantize_input(
            input_2d,
            input_qdata,
            input_scale,
            group_size,
            logical_dtype_code,
            apply_swiglu=apply_swiglu,
            num_warps=execution_plan.fused_num_warps,
        )
    else:
        transformed_input = input_2d
        if apply_swiglu:
            up, gate = input_2d.chunk(2, dim=-1)
            transformed_input = up * torch.nn.functional.silu(gate)
        rotated = torch.empty_like(transformed_input)
        rotate_input(
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
    return (
        input_qdata.reshape(*input.shape[:-1], in_features),
        input_scale.reshape(input.shape[:-1]),
    )


def _prepare_input_with_production_plan(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    *,
    activation_fn: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare an ordinary or packed-SwiGLU input under production policy."""
    apply_swiglu = _uses_swiglu(activation_fn)
    in_features = input.shape[-1] // (2 if apply_swiglu else 1)
    plan = _policy.select_execution_plan(
        AcceleratorTarget.from_device(input.device),
        rows=math.prod(input.shape[:-1]),
        out_features=0,
        in_features=in_features,
        group_size=group_size,
        dtype=input.dtype,
        swiglu=apply_swiglu,
    )
    return _prepare_input(
        input,
        in_features,
        group_size,
        apply_swiglu=apply_swiglu,
        execution_plan=plan,
    )


def _convrot_int8_linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
    execution_plan: _policy.ConvRotInt8LinearExecutionPlan,
) -> torch.Tensor:
    """Allocate and run one INT8 GEMM from a rotated and quantized input."""
    leading_shape = input_qdata.shape[:-1]
    m = math.prod(leading_shape)
    k = input_qdata.shape[-1]
    n = weight_qdata.shape[0]
    output = torch.empty(
        (m, n),
        device=input_qdata.device,
        dtype=logical_dtype,
    )
    input_qdata_2d = input_qdata.reshape(m, k)
    input_scale_1d = input_scale.reshape(m)
    plan = execution_plan
    num_n_tiles = triton.cdiv(n, plan.matmul_block_n)
    # Cache grouping and the M-tail split are intrinsic to the selected large-tile family,
    # rather than additional execution-plan axes.
    group_m = (
        _LARGE_MATMUL_GROUP_M_TILES
        if plan.matmul_block_m == 128 and plan.matmul_block_n == 256
        else 0
    )
    bias_pointer = bias if bias is not None else output

    def launch_tiles(row_block_count: int, row_block_offset: int, *, aligned_m: bool) -> None:
        grid = (row_block_count * num_n_tiles,) if group_m else (row_block_count, num_n_tiles)
        _int8_matmul_kernel[grid](
            input_qdata_2d,
            weight_qdata,
            output,
            input_scale_1d,
            weight_scale,
            bias_pointer,
            m,
            n,
            k,
            row_block_count,
            row_block_offset,
            block_m=plan.matmul_block_m,
            block_n=plan.matmul_block_n,
            block_k=plan.matmul_block_k,
            has_bias=bias is not None,
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
    return output.reshape(*leading_shape, n)


def run_convrot_int8_linear(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    *,
    activation_fn: str | None = None,
    execution_plan: _policy.ConvRotInt8LinearExecutionPlan | None = None,
) -> torch.Tensor:
    """Run ConvRot input preparation and INT8 GEMM under one plan."""
    original_shape = input.shape
    k = weight_qdata.shape[1]
    apply_swiglu = _uses_swiglu(activation_fn)
    expected_width = k * (2 if apply_swiglu else 1)
    if original_shape[-1] != expected_width:
        operation = "fused SwiGLU input" if apply_swiglu else "linear input"
        raise ValueError(
            f"{operation} has {original_shape[-1]} features, expected {expected_width}"
        )
    plan = (
        execution_plan
        if execution_plan is not None
        else default_convrot_int8_execution_plan(
            input,
            weight_qdata,
            group_size,
            activation_fn=activation_fn,
        )
    )
    input_qdata, input_scale = _prepare_input(
        input,
        k,
        group_size,
        apply_swiglu=apply_swiglu,
        execution_plan=plan,
    )
    return _convrot_int8_linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        input.dtype,
        plan,
    )


@torch.library.custom_op("piper_kernels::convrot_int8_linear", mutates_args=())
def convrot_int8_linear(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    activation_fn: str | None = None,
) -> torch.Tensor:
    """Run ConvRot input rotation, dynamic quantization, and INT8 GEMM."""
    return run_convrot_int8_linear(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        activation_fn=activation_fn,
    )


@convrot_int8_linear.register_fake
def _convrot_int8_linear_fake(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    _activation_fn: str | None = None,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], weight_qdata.shape[0]))


@torch.library.custom_op("piper_kernels::convrot_int8_prepare_input", mutates_args=())
def convrot_int8_prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then rotate and quantize a linear input."""
    return _prepare_input_with_production_plan(
        input,
        group_size,
        activation_fn=activation_fn,
    )


@convrot_int8_prepare_input.register_fake
def _convrot_int8_prepare_input_fake(
    input: torch.Tensor,  # noqa: A002
    _group_size: int,
    activation_fn: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_width = input.shape[-1] // (2 if _uses_swiglu(activation_fn) else 1)
    return (
        input.new_empty((*input.shape[:-1], input_width), dtype=torch.int8),
        input.new_empty(input.shape[:-1], dtype=torch.float32),
    )


@torch.library.custom_op("piper_kernels::convrot_int8_linear_prepared", mutates_args=())
def convrot_int8_linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply one weight to an input prepared by the matching operator."""
    leading_shape = input_qdata.shape[:-1]
    out_features, in_features = weight_qdata.shape
    rows = math.prod(leading_shape)
    plan = _policy.select_execution_plan(
        AcceleratorTarget.from_device(input_qdata.device),
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        dtype=logical_dtype,
    )
    return _convrot_int8_linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        logical_dtype,
        plan,
    )


@convrot_int8_linear_prepared.register_fake
def _convrot_int8_linear_prepared_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    return input_qdata.new_empty(
        (*input_qdata.shape[:-1], weight_qdata.shape[0]),
        dtype=logical_dtype,
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_addmm_",
    mutates_args=("qdata", "scale"),
)
def convrot_int8_addmm_(
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
    has_update = alpha != 0 and mat1.shape[1] != 0
    if has_update:
        mat2_contiguous = mat2.contiguous()
        rotated_mat2 = torch.empty_like(mat2_contiguous)
        rotate_input(
            mat2_contiguous,
            rotated_mat2,
            group_size,
            num_warps=4,
        )
        update = torch.mm(mat1, rotated_mat2)
    else:
        update = qdata

    logical_dtype_code = dtype_code(mat1.dtype)
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
        seed_argument(rounding_seed),
        block_size=requant_block,
        logical_dtype_code=logical_dtype_code,
        has_base=beta != 0,
        has_update=has_update,
        stochastic=rounding_seed is not None,
        num_warps=8,
    )


@convrot_int8_addmm_.register_fake
def _convrot_int8_addmm_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _mat1: torch.Tensor,
    _mat2: torch.Tensor,
    _group_size: int,
    _beta: float,
    _alpha: float,
    _rounding_seed: int | None = None,
) -> None:
    return None
