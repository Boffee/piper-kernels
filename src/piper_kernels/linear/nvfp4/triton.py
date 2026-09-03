"""Triton utilities for prepared NVFP4 activation storage."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import _layout

_NVFP4_BLOCK_SIZE = _layout.BLOCK_SIZE
_NVFP4_QDATA_BLOCK_SIZE = _layout.QDATA_BLOCK_SIZE
_NVFP4_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_BLOCK_SIZE)
_NVFP4_QDATA_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_QDATA_BLOCK_SIZE)
_PREPARE_BLOCKS = 32
_EPILOGUE_BLOCK_SIZE = 256
_MEAN_BLOCK_M = 256
_MEAN_BLOCK_K = 128
_PROJECTION_BLOCK_N = 64
_PROJECTION_BLOCK_K = 128
_AMAX_REDUCTION_BLOCK_SIZE = 1_024


@triton.jit
def _decode_fp4_code(code):
    magnitude = code & 0x7
    value = tl.where(
        magnitude <= 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
    )
    return tl.where(code & 0x8 == 0, value, -value)


@triton.jit
def _decode_fp4(packed, logical_offsets):
    code = tl.where(logical_offsets % 2 == 0, packed & 0xF, packed >> 4)
    return _decode_fp4_code(code)


@triton.jit
def swizzled_scale_offsets(rows, scale_columns, column_blocks: tl.constexpr):
    row_block = rows // 128
    row_inner = rows % 128
    column_block = scale_columns // 4
    return (
        ((row_block * column_blocks + column_block) * 32 + row_inner % 32) * 16
        + (row_inner // 32) * 4
        + scale_columns % 4
    )


@triton.jit
def pack_e2m1_pairs(low, high):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 packed;
            cvt.rn.satfinite.e2m1x2.f32 packed, $2, $1;
            cvt.u32.u8 $0, packed;
        }
        """,
        constraints="=r,f,f",
        args=[low, high],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def encode_nvfp4_blocks(
    values,
    per_tensor_scale,
    block_count: tl.constexpr,
):
    """Encode FP32 values into canonical E2M1 pairs and FP8 block scales."""
    values = values.to(tl.float32)
    block_amax = tl.max(tl.abs(values), axis=1)
    encoded_scale = tl.clamp(
        block_amax * (1.0 / 6.0) / per_tensor_scale,
        0.015625,
        448.0,
    ).to(tl.float8e4nv)
    reciprocal_scale = (1.0 / per_tensor_scale) / encoded_scale.to(tl.float32)
    scaled = tl.clamp(values * reciprocal_scale[:, None], -6.0, 6.0)
    paired = tl.reshape(
        scaled,
        (block_count, _NVFP4_QDATA_BLOCK_SIZE_TL, 2),
    )
    low, high = tl.split(paired)
    return pack_e2m1_pairs(low, high), encoded_scale


@triton.jit
def _prepare_static_kernel(
    input_ptr,
    per_tensor_scale_ptr,
    source_global_scale_ptr,
    source_bias_ptr,
    qdata_ptr,
    scale_ptr,
    block_count,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    swiglu: tl.constexpr,
    apply_source_affine: tl.constexpr,
    has_source_bias: tl.constexpr,
    blocks_per_program: tl.constexpr,
):
    """Quantize static-scale activations directly into both hardware layouts."""
    block_offsets = tl.program_id(0) * blocks_per_program + tl.arange(0, blocks_per_program)
    valid_blocks = block_offsets < block_count
    scale_columns = block_offsets % (output_features // _NVFP4_BLOCK_SIZE_TL)
    rows = block_offsets // (output_features // _NVFP4_BLOCK_SIZE_TL)
    element_offsets = tl.arange(0, _NVFP4_BLOCK_SIZE_TL)
    input_offsets = (
        rows[:, None] * input_features
        + scale_columns[:, None] * _NVFP4_BLOCK_SIZE_TL
        + element_offsets[None, :]
    )
    values = tl.load(
        input_ptr + input_offsets,
        mask=valid_blocks[:, None],
        other=0.0,
    ).to(tl.float32)
    gate = values
    if swiglu:
        gate = tl.load(
            input_ptr + input_offsets + output_features,
            mask=valid_blocks[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    if apply_source_affine:
        source_global_scale = tl.load(source_global_scale_ptr).to(tl.float32)
        values *= source_global_scale
        if swiglu:
            gate *= source_global_scale
        if has_source_bias:
            source_columns = scale_columns[:, None] * _NVFP4_BLOCK_SIZE_TL + element_offsets
            values += tl.load(source_bias_ptr + source_columns).to(tl.float32)
            if swiglu:
                gate += tl.load(source_bias_ptr + source_columns + output_features).to(tl.float32)
        values = values.to(tl.bfloat16).to(tl.float32)
        if swiglu:
            gate = gate.to(tl.bfloat16).to(tl.float32)
    if swiglu:
        values *= gate / (1.0 + libdevice.exp(-gate))  # pyright: ignore[reportOperatorIssue]

    per_tensor_scale = tl.load(per_tensor_scale_ptr).to(tl.float32)
    packed, encoded_scale = encode_nvfp4_blocks(  # pyright: ignore[reportGeneralTypeIssues]
        values,
        per_tensor_scale,
        blocks_per_program,
    )
    qdata_offsets = (
        rows[:, None] * (output_features // 2)
        + scale_columns[:, None] * _NVFP4_QDATA_BLOCK_SIZE_TL
        + tl.arange(0, _NVFP4_QDATA_BLOCK_SIZE_TL)[None, :]
    )
    tl.store(qdata_ptr + qdata_offsets, packed, mask=valid_blocks[:, None])
    scale_offsets = swizzled_scale_offsets(
        rows,
        scale_columns,
        scale_column_blocks,
    )
    tl.store(scale_ptr + scale_offsets, encoded_scale, mask=valid_blocks)


@triton.jit
def _projection_epilogue_kernel(
    input_ptr,
    global_scale_ptr,
    bias_ptr,
    output_ptr,
    elements,
    features: tl.constexpr,
    has_bias: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = (tl.program_id(0) * block_size + tl.arange(0, block_size)).to(tl.int64)
    valid = offsets < elements
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    values *= tl.load(global_scale_ptr).to(tl.float32)
    if has_bias:
        values += tl.load(bias_ptr + offsets % features, mask=valid, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, values, mask=valid)


@triton.jit
def _amax_partial_kernel(
    input_ptr,
    partial_ptr,
    elements,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    values = tl.load(input_ptr + offsets, mask=offsets < elements, other=0.0)
    tl.store(partial_ptr + tl.program_id(0), tl.max(tl.abs(values).to(tl.float32), axis=0))


@triton.jit
def _amax_scale_kernel(
    input_ptr,
    per_tensor_scale_ptr,
    elements,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    values = tl.load(input_ptr + offsets, mask=offsets < elements, other=0.0)
    amax = tl.max(tl.abs(values).to(tl.float32), axis=0)
    tl.store(per_tensor_scale_ptr, amax * (1.0 / (448.0 * 6.0)))


def dynamic_scale(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce a logical tensor to its exact NVFP4 global scale."""
    if input.numel() < 1:
        raise ValueError("dynamic NVFP4 scale requires a nonempty tensor")
    if not input.is_floating_point():
        raise ValueError("dynamic NVFP4 scale requires a floating tensor")
    if out is None:
        per_tensor_scale = torch.empty((), device=input.device, dtype=torch.float32)
    else:
        if (
            out.shape != ()
            or out.dtype is not torch.float32
            or out.device != input.device
            or not out.is_contiguous()
        ):
            raise ValueError("dynamic NVFP4 scale output must be a contiguous FP32 scalar")
        per_tensor_scale = out

    values = input.contiguous().view(-1)
    while values.numel() > _AMAX_REDUCTION_BLOCK_SIZE:
        partial_count = (
            values.numel() + _AMAX_REDUCTION_BLOCK_SIZE - 1
        ) // _AMAX_REDUCTION_BLOCK_SIZE
        partial = torch.empty(partial_count, device=input.device, dtype=torch.float32)
        _amax_partial_kernel[(partial_count,)](
            values,
            partial,
            values.numel(),
            block_size=_AMAX_REDUCTION_BLOCK_SIZE,
            num_warps=8,
        )
        values = partial
    _amax_scale_kernel[(1,)](
        values,
        per_tensor_scale,
        values.numel(),
        block_size=triton.next_power_of_2(values.numel()),
        num_warps=8,
    )
    return per_tensor_scale


def _prepare_static_storage(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    *,
    swiglu: bool,
    source_global_scale: torch.Tensor | None,
    source_bias: torch.Tensor | None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    contiguous_input = input.contiguous()
    input_features = int(contiguous_input.shape[-1])
    output_features = input_features // 2 if swiglu else input_features
    rows = int(contiguous_input.numel() // input_features)
    qdata, scale = _layout.prepare_activation_storage(input, rows, output_features, out)
    block_count = rows * (output_features // _NVFP4_BLOCK_SIZE)
    _prepare_static_kernel[(triton.cdiv(block_count, _PREPARE_BLOCKS),)](
        contiguous_input,
        per_tensor_scale,
        source_global_scale if source_global_scale is not None else per_tensor_scale,
        source_bias if source_bias is not None else contiguous_input,
        qdata,
        scale,
        block_count,
        input_features=input_features,
        output_features=output_features,
        scale_column_blocks=(output_features + _layout.SCALE_COLUMN_TILE - 1)
        // _layout.SCALE_COLUMN_TILE,
        swiglu=swiglu,
        apply_source_affine=source_global_scale is not None,
        has_source_bias=source_bias is not None,
        blocks_per_program=_PREPARE_BLOCKS,
        num_warps=2,
    )
    return qdata, scale


def prepare_static(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    swiglu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a static-scale NVFP4 activation without intermediate tensors."""
    qdata, scale = _prepare_static_storage(
        input,
        per_tensor_scale,
        swiglu=swiglu,
        source_global_scale=None,
        source_bias=None,
    )
    return qdata, scale, per_tensor_scale.clone()


def prepare_static_out(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare static NVFP4 storage into reusable caller-owned buffers."""
    return _prepare_static_storage(
        input,
        per_tensor_scale,
        swiglu=False,
        source_global_scale=None,
        source_bias=None,
        out=out,
    )


def prepare_static_projected_swiglu(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    source_global_scale: torch.Tensor,
    source_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a raw projection's affine result and prepare its packed SwiGLU output."""
    return _prepare_static_storage(
        input,
        per_tensor_scale,
        swiglu=True,
        source_global_scale=source_global_scale,
        source_bias=source_bias,
    )


def apply_projection_epilogue(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    global_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    """Apply a raw NVFP4 projection's affine result into caller-owned storage."""
    elements = input.numel()
    _projection_epilogue_kernel[(triton.cdiv(elements, _EPILOGUE_BLOCK_SIZE),)](
        input,
        global_scale,
        bias if bias is not None else input,
        output,
        elements,
        features=input.shape[-1],
        has_bias=bias is not None,
        block_size=_EPILOGUE_BLOCK_SIZE,
        num_warps=4,
    )


@triton.jit
def _dequantized_input_mean_partial_kernel(
    input_ptr,
    input_scale_ptr,
    partial_ptr,
    block_lengths_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Sum one sequence block represented by packed FP4 activation storage."""
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    feature_start = feature_block * block_k
    feature_offsets = feature_start + tl.arange(0, block_k)
    rows = batch * sequence_length + sequence_offsets
    valid_rows = sequence_offsets < sequence_length
    if mask_block_lengths:
        block_lengths = tl.load(
            block_lengths_ptr + sequence_offsets // 64,
            mask=valid_rows,
            other=0,
        )
        valid_rows &= sequence_offsets % 64 < block_lengths
    packed_feature_offsets = feature_start // 2 + tl.arange(0, block_k // 2)
    valid_packed = valid_rows[:, None] & (packed_feature_offsets[None, :] * 2 < input_features)
    packed = tl.load(
        input_ptr + rows[:, None] * (input_features // 2) + packed_feature_offsets[None, :],
        mask=valid_packed,
        other=0,
    )
    values = tl.interleave(
        _decode_fp4_code(packed & 0xF),
        _decode_fp4_code(packed >> 4),
    )
    scale_columns = feature_start // _NVFP4_BLOCK_SIZE_TL + tl.arange(
        0,
        block_k // _NVFP4_BLOCK_SIZE_TL,
    )
    scale_offsets = swizzled_scale_offsets(
        rows[:, None],
        scale_columns[None, :],
        scale_column_blocks,
    )
    scales = tl.load(
        input_scale_ptr + scale_offsets,
        mask=valid_rows[:, None] & (scale_columns[None, :] * _NVFP4_BLOCK_SIZE_TL < input_features),
        other=0.0,
    ).to(tl.float32)
    scales = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (block_m, block_k // _NVFP4_BLOCK_SIZE_TL, 1)),
            (block_m, block_k // _NVFP4_BLOCK_SIZE_TL, _NVFP4_BLOCK_SIZE_TL),
        ),
        (block_m, block_k),
    )
    valid = valid_rows[:, None] & (feature_offsets[None, :] < input_features)
    values = tl.where(valid, values * scales, 0.0)
    partial_offsets = (batch * row_block_count + row_block) * input_features + feature_offsets
    tl.store(
        partial_ptr + partial_offsets,
        tl.sum(values, axis=0),
        mask=feature_offsets < input_features,
    )


@triton.jit
def _dequantized_input_mean_reduce_kernel(
    partial_ptr,
    input_per_tensor_scale_ptr,
    mean_ptr,
    valid_count_ptr,
    sequence_length,
    input_features: tl.constexpr,
    row_block_count: tl.constexpr,
    reduction_rows: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    block_k: tl.constexpr,
):
    """Reduce represented-activation partial sums into one FP32 mean per batch."""
    feature_block = tl.program_id(0)
    batch = tl.program_id(1)
    row_offsets = tl.arange(0, reduction_rows)
    feature_offsets = feature_block * block_k + tl.arange(0, block_k)
    values = tl.load(
        partial_ptr
        + (batch * row_block_count + row_offsets[:, None]) * input_features
        + feature_offsets[None, :],
        mask=(row_offsets[:, None] < row_block_count) & (feature_offsets[None, :] < input_features),
        other=0.0,
    )
    per_tensor_scale = tl.load(input_per_tensor_scale_ptr).to(tl.float32)
    valid_count = tl.load(valid_count_ptr) if mask_block_lengths else sequence_length
    mean = tl.sum(values, axis=0) * per_tensor_scale / valid_count
    tl.store(
        mean_ptr + batch * input_features + feature_offsets,
        mean,
        mask=feature_offsets < input_features,
    )


@triton.jit
def _project_input_mean_kernel(
    input_mean_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    output_ptr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Project a represented FP32 input mean through one packed NVFP4 weight."""
    output_block = tl.program_id(0)
    batch = tl.program_id(1)
    output_offsets = output_block * block_n + tl.arange(0, block_n)
    feature_offsets = tl.arange(0, block_k)
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    for feature_block in range(tl.cdiv(input_features, block_k)):
        logical_features = feature_block * block_k + feature_offsets
        valid = (output_offsets[:, None] < output_features) & (
            logical_features[None, :] < input_features
        )
        input_mean = tl.load(
            input_mean_ptr + batch * input_features + logical_features,
            mask=logical_features < input_features,
            other=0.0,
        )
        packed = tl.load(
            weight_ptr
            + output_offsets[:, None] * (input_features // 2)
            + logical_features[None, :] // 2,
            mask=valid,
            other=0,
        )
        scale_offsets = swizzled_scale_offsets(
            output_offsets[:, None],
            logical_features[None, :] // _NVFP4_BLOCK_SIZE_TL,
            scale_column_blocks,
        )
        scales = tl.load(weight_scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.float32)
        weight = _decode_fp4(packed, logical_features[None, :]) * scales
        accumulator += tl.sum(weight * input_mean[None, :], axis=1)
    if has_weight_per_tensor_scale:
        accumulator *= tl.load(weight_per_tensor_scale_ptr).to(tl.float32)
    if has_bias:
        accumulator += tl.load(
            bias_ptr + output_offsets,
            mask=output_offsets < output_features,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        output_ptr + batch * output_features + output_offsets,
        accumulator,
        mask=output_offsets < output_features,
    )


def _validate_linear_mean(  # noqa: PLR0912
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    batch: int,
    sequence_length: int,
    block_lengths: torch.Tensor | None,
) -> tuple[int, int]:
    if input_qdata.ndim != 2 or input_qdata.dtype is not torch.uint8:
        raise ValueError("NVFP4 mean input must be a two-dimensional packed UINT8 tensor")
    rows, packed_input_features = input_qdata.shape
    input_features = 2 * packed_input_features
    if batch <= 0 or sequence_length <= 0 or rows != batch * sequence_length:
        raise ValueError("NVFP4 mean batch and sequence dimensions must match its input rows")
    if input_features % _NVFP4_BLOCK_SIZE:
        raise ValueError(f"NVFP4 mean input features must be divisible by {_NVFP4_BLOCK_SIZE}")
    expected_input_scale_shape = _layout.scale_shape(rows, input_features)
    if (
        input_scale.shape != expected_input_scale_shape
        or input_scale.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError("NVFP4 mean input scale has an incompatible swizzled layout")
    if input_per_tensor_scale.shape != () or input_per_tensor_scale.dtype is not torch.float32:
        raise ValueError("NVFP4 mean input per-tensor scale must be an FP32 scalar")
    if (
        weight_qdata.ndim != 2
        or weight_qdata.dtype is not torch.uint8
        or weight_qdata.shape[1] != packed_input_features
    ):
        raise ValueError("NVFP4 mean weight must be a compatible packed UINT8 matrix")
    output_features = weight_qdata.shape[0]
    expected_weight_scale_shape = _layout.scale_shape(output_features, input_features)
    if (
        weight_scale.shape != expected_weight_scale_shape
        or weight_scale.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError("NVFP4 mean weight scale has an incompatible swizzled layout")
    if weight_per_tensor_scale is not None and (
        weight_per_tensor_scale.shape != () or weight_per_tensor_scale.dtype is not torch.float32
    ):
        raise ValueError("NVFP4 mean weight per-tensor scale must be an FP32 scalar")
    if bias is not None and (bias.shape != (output_features,) or bias.dtype is not torch.bfloat16):
        raise ValueError("NVFP4 mean bias must be one BF16 value per output feature")
    if block_lengths is not None and (
        sequence_length % 64
        or block_lengths.shape != (sequence_length // 64,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != input_qdata.device
        or not block_lengths.is_contiguous()
    ):
        raise ValueError("NVFP4 mean block lengths must be one contiguous device INT32 per K64")
    operands = [
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
    ]
    operands.extend(operand for operand in (weight_per_tensor_scale, bias) if operand is not None)
    if block_lengths is not None:
        operands.append(block_lengths)
    if input_qdata.device.type != "cuda" or any(
        operand.device != input_qdata.device for operand in operands
    ):
        raise ValueError("NVFP4 mean operands must share a CUDA device")
    if any(not operand.is_contiguous() for operand in operands):
        raise ValueError("NVFP4 mean operands must be contiguous")
    return input_features, output_features


@torch.library.custom_op("piper_kernels::nvfp4_linear_mean", mutates_args=())
def linear_mean(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    batch: int,
    sequence_length: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project the compact or valid-front padded mean represented by NVFP4 storage."""
    input_features, output_features = _validate_linear_mean(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        batch,
        sequence_length,
        block_lengths,
    )
    row_block_count = int(triton.cdiv(sequence_length, _MEAN_BLOCK_M))
    partial = torch.empty(
        (batch, row_block_count, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    input_mean = torch.empty(
        (batch, input_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    output = torch.empty(
        (batch, output_features),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    scale_column_blocks = int(triton.cdiv(input_features, 64))
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else input_scale
    valid_count = block_lengths.sum(dtype=torch.float32) if has_block_lengths else input_scale
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
        scale_column_blocks=scale_column_blocks,
        mask_block_lengths=has_block_lengths,
        block_m=_MEAN_BLOCK_M,
        block_k=_MEAN_BLOCK_K,
        num_warps=8,
    )
    _dequantized_input_mean_reduce_kernel[(triton.cdiv(input_features, _MEAN_BLOCK_K), batch)](
        partial,
        input_per_tensor_scale,
        input_mean,
        valid_count,
        sequence_length,
        input_features=input_features,
        row_block_count=row_block_count,
        reduction_rows=triton.next_power_of_2(row_block_count),
        mask_block_lengths=has_block_lengths,
        block_k=_MEAN_BLOCK_K,
        num_warps=8,
    )
    _project_input_mean_kernel[(triton.cdiv(output_features, _PROJECTION_BLOCK_N), batch)](
        input_mean,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        output,
        input_features=input_features,
        output_features=output_features,
        scale_column_blocks=scale_column_blocks,
        has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
        has_bias=bias is not None,
        block_n=_PROJECTION_BLOCK_N,
        block_k=_PROJECTION_BLOCK_K,
        num_warps=8,
    )
    return output


@linear_mean.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _linear_mean_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    batch: int,
    _sequence_length: int,
    _block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    return input_qdata.new_empty((batch, weight_qdata.shape[0]), dtype=torch.float32)


__all__ = [
    "apply_projection_epilogue",
    "dynamic_scale",
    "encode_nvfp4_blocks",
    "linear_mean",
    "pack_e2m1_pairs",
    "prepare_static",
    "prepare_static_out",
    "prepare_static_projected_swiglu",
    "swizzled_scale_offsets",
]
