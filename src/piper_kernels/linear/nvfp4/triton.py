"""Triton kernels for NVFP4 activation preparation and projection."""

# Triton's JIT launcher and constexpr options are not fully represented in its typing surface.
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels import _input_activations as input_activations
from piper_kernels._input_activations import triton as triton_input_activations
from piper_kernels._triton.primitives import (
    TRITON_FLOAT_POINTER_TYPES,
    TritonCompileFn,
    default_triton_compile,
    splitk_tile_bounds,
)
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear import _bias
from piper_kernels.linear._splitk import device_sm_count, select_split_k
from piper_kernels.linear.nvfp4._storage import prepare_activation_storage
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.nvfp4 import _layout
from piper_kernels.weights.nvfp4.triton import (
    _decode_fp4,
    _decode_fp4_code,
    encode_nvfp4_blocks,
    pack_e2m1_pairs,
    swizzled_scale_offsets,
)
from piper_kernels.weights.nvfp4.triton import dynamic_scale as _weights_dynamic_scale

_NVFP4_BLOCK_SIZE = _layout.BLOCK_SIZE
_NVFP4_QDATA_BLOCK_SIZE = _layout.QDATA_BLOCK_SIZE
_NVFP4_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_BLOCK_SIZE)
_NVFP4_QDATA_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_QDATA_BLOCK_SIZE)
_ACTIVATION_AMAX_BLOCK_SIZE = 4_096
_PREPARE_BLOCKS = 32
_BIAS_BLOCK_SIZE = 1_024
_MEAN_BLOCK_M = 256
_MEAN_BLOCK_K = 128
_PROJECTION_BLOCK_N = 64
_PROJECTION_BLOCK_K = 128


@triton.jit
def _prepare_static_kernel(
    input_ptr,
    per_tensor_scale_ptr,
    qdata_ptr,
    scale_ptr,
    global_scale_out_ptr,
    block_count,
    row_count,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    activation_fn: tl.constexpr,
    blocks_per_program: tl.constexpr,
    high_first: tl.constexpr,
):
    """Quantize static-scale activations directly into both hardware layouts."""
    block_offsets = tl.program_id(0) * blocks_per_program + tl.arange(0, blocks_per_program)
    scale_columns = block_offsets % (scale_column_blocks * 4)
    rows = block_offsets // (scale_column_blocks * 4)
    valid_blocks = (rows < row_count) & (scale_columns < output_features // _NVFP4_BLOCK_SIZE_TL)
    encoded_scale = tl.full((blocks_per_program,), 0, tl.float8e4nv)
    # Whole padding programs only write zeros, especially for short wide inputs.
    if tl.sum(valid_blocks.to(tl.int32), 0) > 0:  # pyright: ignore[reportOperatorIssue]
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
        if activation_fn == "swiglu":
            gate = tl.load(
                input_ptr + input_offsets + output_features,
                mask=valid_blocks[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            values = triton_input_activations.swiglu(values, gate)
        elif activation_fn == "gelu_tanh":
            values = triton_input_activations.gelu_tanh(
                values,
                "cuda",  # pyright: ignore[reportArgumentType]
            )

        per_tensor_scale = 1.0
        if per_tensor_scale_ptr is not None:
            per_tensor_scale = tl.load(per_tensor_scale_ptr).to(tl.float32)
        if global_scale_out_ptr is not None:  # noqa: SIM102 - specialize the optional output pointer
            if tl.program_id(0) == 0:
                tl.store(global_scale_out_ptr, per_tensor_scale)
        packed, encoded_scale = encode_nvfp4_blocks(  # pyright: ignore[reportGeneralTypeIssues]
            values,
            per_tensor_scale,
            blocks_per_program,
            high_first,
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
    # The launch covers complete scale tiles, so masked input lanes also
    # initialize row/column padding without a separate zero-fill kernel.
    tl.store(scale_ptr + scale_offsets, encoded_scale, mask=block_offsets < block_count)


@triton.jit
def _add_bias_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    elements,
    features: tl.constexpr,
    input_row_stride: tl.constexpr,
    bias_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    valid = offsets < elements
    rows = offsets // features
    columns = offsets % features
    values = tl.load(
        input_ptr + rows * input_row_stride + columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    values += tl.load(bias_ptr + columns * bias_stride, mask=valid, other=0.0).to(tl.float32)
    tl.store(output_ptr + rows * output_row_stride + columns, values, mask=valid)


def _prepare_static_storage(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor | None,
    *,
    activation_fn: str | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    high_first: bool = False,
    global_scale_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    contiguous_input = input.contiguous()
    input_features = int(contiguous_input.shape[-1])
    output_features = input_features // input_activations.input_activation_width(activation_fn)
    rows = int(contiguous_input.numel() // input_features)
    qdata, scale = prepare_activation_storage(
        input, rows, output_features, out, initialize_padding=False
    )
    block_count = scale.numel()
    with device_context(input.device):
        _prepare_static_kernel[(triton.cdiv(block_count, _PREPARE_BLOCKS),)](
            contiguous_input,
            per_tensor_scale,
            qdata,
            scale,
            global_scale_out,
            block_count,
            rows,
            input_features=input_features,
            output_features=output_features,
            scale_column_blocks=(output_features + _layout.SCALE_COLUMN_TILE - 1)
            // _layout.SCALE_COLUMN_TILE,
            activation_fn=activation_fn,
            blocks_per_program=_PREPARE_BLOCKS,
            high_first=high_first,
            num_warps=2,
        )
        return qdata, scale


def prepare_static(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor | None,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare nonempty static NVFP4 storage and an owned global scale; None means unit."""
    global_scale = torch.empty((), device=input.device, dtype=torch.float32)
    qdata, scale = _prepare_static_storage(
        input,
        per_tensor_scale,
        activation_fn=activation_fn,
        high_first=high_first,
        global_scale_out=global_scale,
    )
    return qdata, scale, global_scale


def prepare_static_out(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor],
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare static NVFP4 storage into reusable caller-owned buffers."""
    return _prepare_static_storage(
        input,
        per_tensor_scale,
        activation_fn=None,
        out=out,
        high_first=high_first,
    )


@triton.jit
def _activation_amax_partial_kernel(
    input_ptr,
    partial_ptr,
    elements,
    output_features: tl.constexpr,
    activation_fn: tl.constexpr,
    block_size: tl.constexpr,
):
    """Reduce one tile of activated values into a block-level FP32 amax."""
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size)
    valid = offsets < elements
    if activation_fn == "swiglu":
        input_offsets = (offsets // output_features) * (2 * output_features) + (
            offsets % output_features
        )
        values = tl.load(input_ptr + input_offsets, mask=valid, other=0.0).to(tl.float32)
        gate = tl.load(input_ptr + input_offsets + output_features, mask=valid, other=0.0).to(
            tl.float32
        )
        values = triton_input_activations.swiglu(values, gate)
    else:
        values = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        values = triton_input_activations.gelu_tanh(
            values,
            "cuda",  # pyright: ignore[reportArgumentType]
        )
    values = tl.where(valid, values, 0.0)
    tl.store(partial_ptr + tl.program_id(0), tl.max(tl.abs(values), axis=0))


def _validate_activated_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_fn: str | None,
) -> tuple[int, int]:
    if input.ndim == 0 or input.numel() < 1:
        raise ValueError("dynamic NVFP4 scale requires a nonempty tensor")
    if not input.is_floating_point():
        raise ValueError("dynamic NVFP4 scale requires a floating tensor")
    if input.device.type != "cuda":
        raise ValueError("NVFP4 Triton preparation requires CUDA")
    width = input_activations.input_activation_width(activation_fn)
    input_features = int(input.shape[-1])
    if input_features < 1 or input_features % width:
        raise ValueError("NVFP4 input activation requires equal feature partitions")
    output_features = input_features // width
    rows = int(input.numel() // input_features)
    return rows, output_features


def dynamic_scale(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    *,
    activation_fn: str | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce a logical or activated tensor to its exact NVFP4 global scale."""
    input_activations.validate_input_activation(activation_fn)
    if activation_fn is None:
        return _weights_dynamic_scale(input, out=out)
    rows, output_features = _validate_activated_input(input, activation_fn)
    if out is not None and (
        out.shape != ()
        or out.dtype is not torch.float32
        or out.device != input.device
        or not out.is_contiguous()
    ):
        raise ValueError("dynamic NVFP4 scale output must be a contiguous FP32 scalar")
    contiguous_input = input.contiguous()
    elements = rows * output_features
    partial_count = int(triton.cdiv(elements, _ACTIVATION_AMAX_BLOCK_SIZE))
    partial = torch.empty(partial_count, device=contiguous_input.device, dtype=torch.float32)
    with device_context(contiguous_input.device):
        _activation_amax_partial_kernel[(partial_count,)](
            contiguous_input,
            partial,
            elements,
            output_features=output_features,
            activation_fn=activation_fn,
            block_size=_ACTIVATION_AMAX_BLOCK_SIZE,
            num_warps=8,
        )
        return _weights_dynamic_scale(partial, out=out)


def prepare_dynamic(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_fn: str | None = None,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then prepare exact dynamic NVFP4 storage."""
    input_activations.validate_input_activation(activation_fn)
    rows, output_features = _validate_activated_input(input, activation_fn)
    if output_features < _NVFP4_BLOCK_SIZE or output_features % _NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"NVFP4 dynamic preparation requires output features divisible by {_NVFP4_BLOCK_SIZE}"
        )
    output_storage = (
        None
        if out is None
        else prepare_activation_storage(
            input,
            rows,
            output_features,
            out,
            initialize_padding=False,
        )
    )
    contiguous_input = input.contiguous()
    per_tensor_scale = dynamic_scale(contiguous_input, activation_fn=activation_fn)
    qdata, scale = _prepare_static_storage(
        contiguous_input,
        per_tensor_scale,
        activation_fn=activation_fn,
        out=output_storage,
        high_first=high_first,
    )
    return qdata, scale, per_tensor_scale


def add_bias_out(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bias: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Add bias in FP32 and cast into caller-owned storage, which may alias input."""
    if (
        input.ndim != 2
        or output.shape != input.shape
        or input.stride(1) != 1
        or output.stride(1) != 1
        or bias.ndim != 1
        or bias.shape[0] != input.shape[1]
    ):
        raise ValueError("NVFP4 bias addition requires matching row-major matrices and bias width")
    elements = input.numel()
    with device_context(input.device):
        _add_bias_kernel[(triton.cdiv(elements, _BIAS_BLOCK_SIZE),)](
            input,
            bias,
            output,
            elements,
            features=input.shape[-1],
            input_row_stride=input.stride(0),
            bias_stride=bias.stride(0),
            output_row_stride=output.stride(0),
            block_size=_BIAS_BLOCK_SIZE,
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
    if bias is not None:
        if bias.shape != (output_features,):
            raise ValueError("NVFP4 mean bias must contain one value per output feature")
        _bias.validate_dtype(bias, "NVFP4 mean")
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


_W4A16_GEMV_MAX_ROWS = 16
_W4A16_GEMV_BLOCK_N = 32
_W4A16_NARROW_BLOCK_N = 16
_W4A16_GEMV_BLOCK_K = 256
_W4A16_DOT_BLOCK_N = 64
_W4A16_DOT_BLOCK_K = 128
_W4A16_DEQUANT_BLOCK_N = 32
_W4A16_DEQUANT_BLOCK_K = 128
_W4A16_REDUCE_BLOCK_N = 64
_W4A16_FP32_WORKSPACE_BYTES = 32 * 1024 * 1024
_W4A16_MAX_REDUCE_ROWS = 65_535
_W4A16_MAX_SPLIT_K = 32
_W4A16_MIN_SPLITK_WEIGHT_BYTES = 2 * 1024 * 1024
_W4A16_NARROW_NO_SPLITK_WEIGHT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _W4A16Schedule:
    """Resolved NVFP4 W4A16 execution schedule for one target and workload."""

    mode: Literal["gemv", "dequant"]
    block_m: int = 1
    block_n: int = _W4A16_GEMV_BLOCK_N
    block_k: int = _W4A16_GEMV_BLOCK_K
    num_warps: int = 4
    num_stages: int | None = None
    use_dot: bool = False
    single_dot: bool = False
    use_sm120_e2m1x2: bool = False
    split_k: int = 1
    group_m: int = 0


_W4A16_DEQUANT_SCHEDULE = _W4A16Schedule(
    mode="dequant",
    block_n=_W4A16_DEQUANT_BLOCK_N,
    block_k=_W4A16_DEQUANT_BLOCK_K,
    num_warps=4,
)
_W4A16_SM120_DEQUANT_SCHEDULE = _W4A16Schedule(
    mode="dequant",
    block_n=64,
    block_k=256,
    num_warps=8,
    use_sm120_e2m1x2=True,
)
_W4A16_RDNA4_DEQUANT_SCHEDULE = _W4A16Schedule(
    mode="dequant",
    block_n=64,
    block_k=256,
    num_warps=8,
)


@triton.jit
def _decode_e4m3_u8(code):
    """Exact FP32 value of FP8 E4M3FN byte bits without backend FP8 lowering quirks."""
    sign = code >> 7
    exponent = (code >> 3) & 15
    mantissa = code & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    subnormal = mantissa.to(tl.float32) * 0.001953125
    magnitude = tl.where(exponent == 0, subnormal, normal)
    return tl.where(sign != 0, -magnitude, magnitude)


@triton.jit
def _decode_sm120_e2m1x2_f32(packed_u8, high_first: tl.constexpr):
    """Decode packed E2M1 byte pairs via SM120 hardware cvt into exact FP32 pairs."""
    lo, hi = tl.inline_asm_elementwise(
        asm=(
            "{ .reg .b8 b0; .reg .b32 p; "
            "mov.b16 {b0, _}, $2; "
            "cvt.rn.f16x2.e2m1x2 p, b0; "
            "mov.b32 {$0, $1}, p; }"
        ),
        constraints="=h,=h,h",
        args=[packed_u8.to(tl.uint16)],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    if high_first:
        return hi.to(tl.float32), lo.to(tl.float32)
    return lo.to(tl.float32), hi.to(tl.float32)


@triton.jit
def _decode_scale_sm120_e2m1x2(
    packed_u8,
    scales_16,
    high_first: tl.constexpr,
    is_fp16: tl.constexpr,
):
    """Decode and block-scale packed E2M1 byte pairs using SM120 hardware vector ops."""
    if is_fp16:
        lo, hi = tl.inline_asm_elementwise(
            asm=(
                "{ .reg .b8 b0; .reg .b32 p, s2; "
                "mov.b16 {b0, _}, $2; "
                "cvt.rn.f16x2.e2m1x2 p, b0; "
                "mov.b32 s2, {$3, $3}; "
                "mul.rn.f16x2 p, p, s2; "
                "mov.b32 {$0, $1}, p; }"
            ),
            constraints="=h,=h,h,h",
            args=[packed_u8.to(tl.uint16), scales_16],
            dtype=(tl.float16, tl.float16),
            is_pure=True,
            pack=1,
        )
    else:
        lo, hi = tl.inline_asm_elementwise(
            asm=(
                "{ .reg .b8 b0; .reg .b32 p, s2; "
                "mov.b16 {b0, _}, $2; "
                "cvt.rn.bf16x2.e2m1x2 p, b0; "
                "mov.b32 s2, {$3, $3}; "
                "mul.rn.bf16x2 p, p, s2; "
                "mov.b32 {$0, $1}, p; }"
            ),
            constraints="=h,=h,h,h",
            args=[packed_u8.to(tl.uint16), scales_16],
            dtype=(tl.bfloat16, tl.bfloat16),
            is_pure=True,
            pack=1,
        )
    if high_first:
        return hi, lo
    return lo, hi


@triton.jit
def _load_nvfp4_raw_tile(
    weight_ptr,
    weight_scale_ptr,
    n_offs_i64,
    n_valid,
    k_base,
    packed_k,
    scale_k,
    scale_column_blocks,
    block_k: tl.constexpr,
):
    """Load one swizzled NVFP4 weight tile as packed uint8 bytes and raw E4M3FN bytes."""
    block_packed_k: tl.constexpr = block_k // 2
    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    k_byte_offs = (k_base // 2) + tl.arange(0, block_packed_k)
    packed = tl.load(
        weight_ptr + n_offs_i64[:, None] * packed_k + k_byte_offs[None, :],
        mask=n_valid[:, None] & (k_byte_offs[None, :] < packed_k),
        other=0,
    )
    scale_cols = (k_base // _NVFP4_BLOCK_SIZE_TL) + tl.arange(0, block_scale_k)
    scale_offsets = swizzled_scale_offsets(
        n_offs_i64[:, None],
        scale_cols[None, :],
        scale_column_blocks,
    )
    scale_u8 = tl.load(
        weight_scale_ptr + scale_offsets,
        mask=n_valid[:, None] & (scale_cols[None, :] < scale_k),
        other=0,
    )
    return packed, scale_u8


@triton.jit
def _load_nvfp4_tile(
    weight_ptr,
    weight_scale_ptr,
    n_offs_i64,
    n_valid,
    k_base,
    packed_k,
    scale_k,
    scale_column_blocks,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
):
    """Load one swizzled NVFP4 weight tile as decoded E2M1 pairs and FP32 block scales."""
    packed, scale_u8 = _load_nvfp4_raw_tile(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_k,
    )
    w_lo = _decode_fp4_code(packed & 0xF)
    w_hi = _decode_fp4_code(packed >> 4)
    w_even = w_hi if high_first else w_lo
    w_odd = w_lo if high_first else w_hi
    scales = _decode_e4m3_u8(scale_u8.to(tl.uint32))
    return w_even, w_odd, scales


@triton.jit
def _expand_nvfp4_scales(
    scales,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    block_packed_k: tl.constexpr = block_k // 2
    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    return tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (block_n, block_scale_k, 1)),
            (block_n, block_scale_k, _NVFP4_QDATA_BLOCK_SIZE_TL),
        ),
        (block_n, block_packed_k),
    )


@triton.jit
def _split_global_scale(global_scale):
    apply_early = tl.abs(global_scale) < 1.0
    pre_scale = tl.where(apply_early, global_scale, 1.0)
    post_scale = tl.where(apply_early, 1.0, global_scale)
    return pre_scale, post_scale


@triton.jit
def _accumulate_scaled_row_blocks(
    x_vec,
    w_even,
    w_odd,
    scales,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Compute scaled per-block dot partials ``[block_n, block_k // 16]`` for one row."""
    block_packed_k: tl.constexpr = block_k // 2
    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    x_even, x_odd = tl.split(tl.reshape(x_vec.to(tl.float32), (block_packed_k, 2)))
    prod = w_even * x_even[None, :] + w_odd * x_odd[None, :]
    grouped = tl.reshape(prod, (block_n, block_scale_k, _NVFP4_QDATA_BLOCK_SIZE_TL))
    block_sum = tl.sum(grouped, axis=2)
    return tl.where(scales == 0.0, 0.0, block_sum * scales)


@triton.jit
def _accumulate_scaled_row(
    x_vec,
    w_even,
    w_odd,
    scales,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    return tl.sum(
        _accumulate_scaled_row_blocks(x_vec, w_even, w_odd, scales, block_n, block_k),
        axis=1,
    )


@triton.jit
def _store_w4a16_result(
    acc,
    epilogue_scale,
    bias_ptr,
    output_ptr,
    pid_k,
    m_offs,
    m_offs_i64,
    n_offs_i64,
    n_valid,
    rows,
    out_features,
    block_m: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    is_splitk: tl.constexpr,
):
    if block_m == 1:
        if is_splitk:
            tl.store(
                output_ptr + pid_k.to(tl.int64) * out_features + n_offs_i64,
                acc,
                mask=n_valid,
            )
            return
        if has_weight_per_tensor_scale:
            acc *= epilogue_scale
        if has_bias:
            acc += tl.load(bias_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
        tl.store(output_ptr + n_offs_i64, acc, mask=n_valid)
        return

    row_valid = (m_offs[:, None] < rows) & n_valid[None, :]
    if is_splitk:
        partial_base = (pid_k.to(tl.int64) * rows + m_offs_i64[:, None]) * out_features
        tl.store(output_ptr + partial_base + n_offs_i64[None, :], acc, mask=row_valid)
        return
    if has_weight_per_tensor_scale:
        acc *= epilogue_scale
    if has_bias:
        bias_vec = tl.load(bias_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
        acc += bias_vec[None, :]
    tl.store(
        output_ptr + m_offs_i64[:, None] * out_features + n_offs_i64[None, :],
        acc,
        mask=row_valid,
    )


@triton.jit
def _w4a16_program_coords(
    rows,
    out_features,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    group_m: tl.constexpr,
):
    """Map program_id(0) to 2D output tile coordinates with optional L2 group swizzling."""
    pid_mn = tl.program_id(0)
    if group_m > 0:
        num_pid_m = tl.cdiv(rows, block_m)
        num_pid_n = tl.cdiv(out_features, block_n)
        num_pid_in_group = group_m * num_pid_n
        group_id = pid_mn // num_pid_in_group
        first_pid_m = group_id * group_m
        group_size_m = tl.minimum(num_pid_m - first_pid_m, group_m)
        pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
        pid_n = (pid_mn % num_pid_in_group) // group_size_m
        return pid_m, pid_n
    return 0, pid_mn


@triton.jit
def _load_scaled_nvfp4_pairs(
    weight_ptr,
    weight_scale_ptr,
    n_offs_i64,
    n_valid,
    k_base,
    packed_k,
    scale_k,
    scale_column_blocks,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    out_dtype: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr,
):
    """Load, decode, and scale one NVFP4 K-tile into even/odd pairs."""
    if use_sm120_e2m1x2:
        # Every finite (E2M1, E4M3FN) product fits exactly in FP16 (max 2688, min 2^-10,
        # 6 significand bits), so f16x2 decode+scale is bit-exact for both FP16 and FP32.
        is_fp16_scale: tl.constexpr = tl.constexpr(out_dtype != tl.bfloat16)
        scale_dtype = tl.float16 if is_fp16_scale else tl.bfloat16
        packed, scale_u8 = _load_nvfp4_raw_tile(
            weight_ptr,
            weight_scale_ptr,
            n_offs_i64,
            n_valid,
            k_base,
            packed_k,
            scale_k,
            scale_column_blocks,
            block_k,
        )
        scales_16 = _expand_nvfp4_scales(
            scale_u8.to(tl.float8e4nv, bitcast=True).to(scale_dtype),
            block_n,
            block_k,
        )
        w_e, w_o = _decode_scale_sm120_e2m1x2(packed, scales_16, high_first, is_fp16_scale)
        return w_e.to(out_dtype), w_o.to(out_dtype)

    w_even, w_odd, scales = _load_nvfp4_tile(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_k,
        high_first,
    )
    scales_expanded = _expand_nvfp4_scales(scales, block_n, block_k)
    return (w_even * scales_expanded).to(out_dtype), (w_odd * scales_expanded).to(out_dtype)


@triton.jit
def _w4a16_accumulate_dot_tile(
    acc,
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    m_offs,
    m_offs_i64,
    n_offs_i64,
    n_valid,
    k_base,
    rows,
    input_features,
    packed_k,
    scale_k,
    scale_column_blocks,
    pre_scale,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    pre_scale_in_gemv: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr,
    single_dot: tl.constexpr,
):
    """Accumulate one NVFP4 K-tile via tensor-core ``tl.dot``."""
    block_packed_k: tl.constexpr = block_k // 2
    # Decode weights before loading X so RDNA3/3.5 does not hold live X tiles across nibble unpacks.
    w_e, w_o = _load_scaled_nvfp4_pairs(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_n,
        block_k,
        high_first,
        tl.constexpr(input_ptr.dtype.element_ty),
        use_sm120_e2m1x2,
    )
    k_elem_offs = k_base + tl.arange(0, block_k)
    x_tile = tl.load(
        input_ptr + m_offs_i64[:, None] * input_features + k_elem_offs[None, :],
        mask=(m_offs[:, None] < rows) & (k_elem_offs[None, :] < input_features),
        other=0.0,
    )
    if pre_scale_in_gemv:
        x_tile = (x_tile.to(tl.float32) * pre_scale).to(x_tile.dtype)

    if single_dot:
        w_full = tl.reshape(tl.join(w_e, w_o), (block_n, block_k))
        return tl.dot(x_tile, tl.trans(w_full), acc=acc)
    x_even, x_odd = tl.split(tl.reshape(x_tile, (block_m, block_packed_k, 2)))
    acc = tl.dot(x_even, tl.trans(w_e), acc=acc)
    return tl.dot(x_odd, tl.trans(w_o), acc=acc)


@triton.jit
def _w4a16_accumulate_scalar_tile(
    acc,
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    m_offs,
    n_offs_i64,
    n_valid,
    k_base,
    rows,
    input_features,
    packed_k,
    scale_k,
    scale_column_blocks,
    pre_scale,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    pre_scale_in_gemv: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr,
):
    """Accumulate one NVFP4 K-tile via FP32 scalar decode and reduction."""
    if use_sm120_e2m1x2:
        packed, scale_u8 = _load_nvfp4_raw_tile(
            weight_ptr,
            weight_scale_ptr,
            n_offs_i64,
            n_valid,
            k_base,
            packed_k,
            scale_k,
            scale_column_blocks,
            block_k,
        )
        w_even, w_odd = _decode_sm120_e2m1x2_f32(packed, high_first)
        scales = scale_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        w_even, w_odd, scales = _load_nvfp4_tile(
            weight_ptr,
            weight_scale_ptr,
            n_offs_i64,
            n_valid,
            k_base,
            packed_k,
            scale_k,
            scale_column_blocks,
            block_k,
            high_first,
        )

    k_elem_offs = k_base + tl.arange(0, block_k)
    k_elem_valid = k_elem_offs < input_features
    if block_m == 1:
        x_vec = tl.load(input_ptr + k_elem_offs, mask=k_elem_valid, other=0.0).to(tl.float32)
        if pre_scale_in_gemv:
            x_vec = x_vec * pre_scale
        acc += _accumulate_scaled_row_blocks(x_vec, w_even, w_odd, scales, block_n, block_k)
    else:
        # Triton's typing does not declare static_range iterable.
        for row_idx in tl.static_range(block_m):  # pyright: ignore[reportGeneralTypeIssues]
            if row_idx < rows:
                x_vec = tl.load(
                    input_ptr + tl.full((), row_idx, tl.int64) * input_features + k_elem_offs,
                    mask=k_elem_valid,
                    other=0.0,
                ).to(tl.float32)
                if pre_scale_in_gemv:
                    x_vec = x_vec * pre_scale
                row_acc = _accumulate_scaled_row(x_vec, w_even, w_odd, scales, block_n, block_k)
                acc += tl.where((m_offs == row_idx)[:, None], row_acc[None, :], 0.0)
    return acc


@triton.jit(do_not_specialize=["rows", "split_k"])
def w4a16_gemv_splitk_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features,
    packed_k,
    scale_column_blocks,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    is_splitk: tl.constexpr,
    use_dot: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr = False,
    single_dot: tl.constexpr = False,
    group_m: tl.constexpr = 0,
):
    """Stream swizzled NVFP4 weights with Split-K/2D tiling and accumulate in FP32."""
    tl.static_assert(use_dot or group_m == 0)
    tl.static_assert(not (block_m == 1 and use_dot))
    pid_m, pid_n = _w4a16_program_coords(rows, out_features, block_m, block_n, group_m)
    pid_k = tl.program_id(1)
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features
    m_offs = pid_m * block_m + tl.arange(0, block_m)
    m_offs_i64 = m_offs.to(tl.int64)

    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    input_features = packed_k * 2
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    # Keep FP16 tl.dot unscaled (cannot overflow FP32, would underflow FP16); BF16 pre-scales
    # against FP32 overflow, trading away deep subnormal |x * pre_scale| < 2^-133 products.
    pre_scale_in_gemv: tl.constexpr = tl.constexpr(
        has_weight_per_tensor_scale and not (use_dot and input_ptr.dtype.element_ty == tl.float16)
    )
    global_scale = (
        tl.load(weight_per_tensor_scale_ptr).to(tl.float32) if has_weight_per_tensor_scale else 1.0
    )
    pre_scale, post_scale = _split_global_scale(global_scale)
    epilogue_scale = post_scale if pre_scale_in_gemv else global_scale
    k_tile_start, k_tile_end = splitk_tile_bounds(tl.cdiv(input_features, block_k), pid_k, split_k)

    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    acc = (
        tl.zeros((block_n, block_scale_k), dtype=tl.float32)
        if block_m == 1
        else tl.zeros((block_m, block_n), dtype=tl.float32)
    )
    for k_tile in range(k_tile_start, k_tile_end):
        k_base = k_tile * block_k
        if use_dot:
            acc = _w4a16_accumulate_dot_tile(
                acc,
                input_ptr,
                weight_ptr,
                weight_scale_ptr,
                m_offs,
                m_offs_i64,
                n_offs_i64,
                n_valid,
                k_base,
                rows,
                input_features,
                packed_k,
                scale_k,
                scale_column_blocks,
                pre_scale,
                block_m,
                block_n,
                block_k,
                high_first,
                pre_scale_in_gemv,
                use_sm120_e2m1x2,
                single_dot,
            )
        else:
            acc = _w4a16_accumulate_scalar_tile(
                acc,
                input_ptr,
                weight_ptr,
                weight_scale_ptr,
                m_offs,
                n_offs_i64,
                n_valid,
                k_base,
                rows,
                input_features,
                packed_k,
                scale_k,
                scale_column_blocks,
                pre_scale,
                block_m,
                block_n,
                block_k,
                high_first,
                pre_scale_in_gemv,
                use_sm120_e2m1x2,
            )

    if block_m == 1:
        acc = tl.sum(acc, axis=1)

    _store_w4a16_result(
        acc,
        epilogue_scale,
        bias_ptr,
        output_ptr,
        pid_k,
        m_offs,
        m_offs_i64,
        n_offs_i64,
        n_valid,
        rows,
        out_features,
        block_m,
        has_weight_per_tensor_scale,
        has_bias,
        is_splitk,
    )


@triton.jit(do_not_specialize=["rows", "split_k"])
def w4a16_splitk_reduce_kernel(
    partial_ptr,
    weight_per_tensor_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features,
    split_k,
    block_n: tl.constexpr,
    has_weight_per_tensor_scale: tl.constexpr,
    has_bias: tl.constexpr,
    post_scale_only: tl.constexpr = False,
):
    """Deterministically reduce Split-K partials in fixed order and apply affine epilogue."""
    pid_n = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    acc = tl.zeros((block_n,), dtype=tl.float32)
    split_stride = tl.full((), rows, tl.int64) * out_features
    ptr = partial_ptr + row * out_features + n_offs_i64
    for _ in range(split_k):
        acc += tl.load(ptr, mask=n_valid, other=0.0)
        ptr += split_stride

    if has_weight_per_tensor_scale:
        global_scale = tl.load(weight_per_tensor_scale_ptr).to(tl.float32)
        if post_scale_only:
            _, post_scale = _split_global_scale(global_scale)
            acc *= post_scale
        else:
            acc *= global_scale
    if has_bias:
        acc += tl.load(bias_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
    tl.store(output_ptr + row * out_features + n_offs_i64, acc, mask=n_valid)


@triton.jit
def dequantize_nvfp4_swizzled_kernel(
    weight_ptr,
    weight_scale_ptr,
    output_ptr,
    out_features,
    packed_k,
    scale_column_blocks,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr = False,
):
    """Dequantize packed swizzled NVFP4 weights into a contiguous floating matrix."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    input_features = packed_k * 2
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    k_base = pid_k * block_k

    w_e, w_o = _load_scaled_nvfp4_pairs(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_n,
        block_k,
        high_first,
        tl.constexpr(output_ptr.dtype.element_ty),
        use_sm120_e2m1x2,
    )
    values = tl.reshape(tl.join(w_e, w_o), (block_n, block_k))
    k_elem_offs = k_base + tl.arange(0, block_k)
    tl.store(
        output_ptr + n_offs_i64[:, None] * input_features + k_elem_offs[None, :],
        values,
        mask=n_valid[:, None] & (k_elem_offs[None, :] < input_features),
    )


def _select_w4a16_split_k(
    num_sms: int,
    out_features: int,
    input_features: int,
    *,
    block_n: int,
    block_k: int,
    rows: int = 1,
    block_m: int = 1,
    cta_multiplier: int = 8,
    min_k_tiles: int = 8,
    min_tiles_per_split: int = 4,
    max_split_k: int = _W4A16_MAX_SPLIT_K,
    round_up: bool = True,
    power_of_two: bool = False,
    cap_workspace: bool = False,
) -> int:
    if cap_workspace:
        if rows > _W4A16_MAX_REDUCE_ROWS:
            return 1
        max_split_k = min(
            max_split_k,
            max(1, _W4A16_FP32_WORKSPACE_BYTES // max(1, rows * out_features * 4)),
        )
        if max_split_k <= 1:
            return 1
    output_tiles = int(triton.cdiv(rows, block_m)) * int(triton.cdiv(out_features, block_n))
    split_k = select_split_k(
        num_sms,
        output_tiles=output_tiles,
        k_tiles=int(triton.cdiv(input_features, block_k)),
        weight_bytes=out_features * (input_features // 2),
        min_weight_bytes=_W4A16_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=max_split_k,
        cta_multiplier=cta_multiplier,
        min_k_tiles=min_k_tiles,
        min_tiles_per_split=min_tiles_per_split,
        round_up=round_up,
    )
    if power_of_two and split_k > 1:
        return 1 << (split_k.bit_length() - 1)
    return split_k


def _sm120_w4a16_fp32_schedule(
    num_sms: int,
    rows: int,
    out_features: int,
    input_features: int,
) -> _W4A16Schedule:
    if rows > _W4A16_GEMV_MAX_ROWS:
        return _W4A16_SM120_DEQUANT_SCHEDULE
    block_m = 4 if rows <= 4 else (8 if rows <= 8 else _W4A16_GEMV_MAX_ROWS)
    split_k = _select_w4a16_split_k(
        num_sms,
        out_features,
        input_features,
        rows=rows,
        block_m=block_m,
        block_n=_W4A16_GEMV_BLOCK_N,
        block_k=_W4A16_GEMV_BLOCK_K,
        cta_multiplier=4,
        power_of_two=True,
        cap_workspace=True,
    )
    return _W4A16Schedule(
        mode="gemv",
        block_m=block_m,
        block_n=_W4A16_GEMV_BLOCK_N,
        block_k=_W4A16_GEMV_BLOCK_K,
        num_warps=4,
        num_stages=2,
        use_dot=False,
        use_sm120_e2m1x2=True,
        split_k=split_k,
    )


def _sm120_w4a16_large_m_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> _W4A16Schedule:
    # Float16 above M=256 is compute-bound: cuBLAS on pre-dequantized weights beats Triton.
    if input_dtype is torch.float16 and rows > 256:
        return _W4A16_SM120_DEQUANT_SCHEDULE
    tiles_64x128 = int(triton.cdiv(rows, 64)) * int(triton.cdiv(out_features, 128))
    group_m = 8 if rows > 64 else 0
    # Use 64x128 split_k=1 above 60% SM occupancy (5/3 ratio); hand off tall deep-K wave
    # tails below 80% full-wave fill (5/4 ratio) to dequantize + cuBLAS.
    if tiles_64x128 * 5 >= max(1, num_sms) * 3:
        waves = int(triton.cdiv(tiles_64x128, max(1, num_sms)))
        if (
            num_sms > 0
            and rows >= 1024
            and input_features >= 8192
            and tiles_64x128 * 5 < waves * num_sms * 4
        ):
            return _W4A16_SM120_DEQUANT_SCHEDULE
        return _W4A16Schedule(
            mode="gemv",
            block_m=64,
            block_n=128,
            block_k=64,
            num_warps=8,
            num_stages=3,
            use_dot=True,
            single_dot=False,
            use_sm120_e2m1x2=True,
            split_k=1,
            group_m=group_m,
        )
    split_k = _select_w4a16_split_k(
        num_sms,
        out_features,
        input_features,
        rows=rows,
        block_m=64,
        block_n=_W4A16_DOT_BLOCK_N,
        block_k=_W4A16_DOT_BLOCK_K,
        cta_multiplier=1,
        max_split_k=8,
        round_up=False,
        power_of_two=True,
        cap_workspace=True,
    )
    return _W4A16Schedule(
        mode="gemv",
        block_m=64,
        block_n=_W4A16_DOT_BLOCK_N,
        block_k=_W4A16_DOT_BLOCK_K,
        num_warps=8,
        num_stages=3,
        use_dot=True,
        single_dot=True,
        use_sm120_e2m1x2=True,
        split_k=split_k,
        group_m=group_m,
    )


def _sm120_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> _W4A16Schedule:
    if rows == 1:
        output_tiles = int(triton.cdiv(out_features, _W4A16_NARROW_BLOCK_N))
        underfilled = output_tiles < num_sms
        split_k = (
            _select_w4a16_split_k(
                num_sms,
                out_features,
                input_features,
                block_m=1,
                block_n=_W4A16_NARROW_BLOCK_N,
                block_k=512,
                cta_multiplier=6 if underfilled else 5,
                min_k_tiles=4 if underfilled else 20,
                min_tiles_per_split=2,
                power_of_two=True,
                cap_workspace=True,
            )
            if (num_sms > 0 and output_tiles < 3 * num_sms)
            else 1
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=1,
            block_n=_W4A16_NARROW_BLOCK_N,
            block_k=512,
            num_warps=4,
            num_stages=2,
            use_dot=False,
            use_sm120_e2m1x2=True,
            split_k=split_k,
        )
    if input_dtype is torch.float32:
        return _sm120_w4a16_fp32_schedule(num_sms, rows, out_features, input_features)
    if rows <= 32:
        # 16x128 uses 6x CTA oversubscription and exact split_k for low-M latency;
        # 32x128 backs off to 2x / power-of-two so larger batches avoid reduce overhead.
        small_batch = rows <= _W4A16_GEMV_MAX_ROWS
        block_m = _W4A16_GEMV_MAX_ROWS if small_batch else 32
        split_k = _select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=block_m,
            block_n=_W4A16_DOT_BLOCK_N,
            block_k=_W4A16_DOT_BLOCK_K,
            cta_multiplier=6 if small_batch else 2,
            max_split_k=16 if small_batch else 8,
            round_up=False,
            power_of_two=not small_batch,
            cap_workspace=True,
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=block_m,
            block_n=_W4A16_DOT_BLOCK_N,
            block_k=_W4A16_DOT_BLOCK_K,
            num_warps=4,
            num_stages=3,
            use_dot=True,
            single_dot=False,
            use_sm120_e2m1x2=True,
            split_k=split_k,
        )
    return _sm120_w4a16_large_m_schedule(num_sms, rows, input_dtype, out_features, input_features)


def _generic_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
    *,
    target: AcceleratorTarget | None = None,
) -> _W4A16Schedule:
    if rows > _W4A16_GEMV_MAX_ROWS:
        return _W4A16_DEQUANT_SCHEDULE
    weight_bytes = out_features * (input_features // 2)
    no_dot = target is not None and target.is_nvidia_cuda and not target.cuda_capability_at_least(8)
    if rows == 1:
        if (
            triton.cdiv(out_features, _W4A16_GEMV_BLOCK_N) <= num_sms
            and weight_bytes < _W4A16_NARROW_NO_SPLITK_WEIGHT_BYTES
        ):
            return _W4A16Schedule(
                mode="gemv",
                block_m=1,
                block_n=_W4A16_NARROW_BLOCK_N,
                block_k=_W4A16_GEMV_BLOCK_K,
                num_warps=8,
                use_dot=False,
                split_k=1,
            )
        split_k = _select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            block_n=_W4A16_GEMV_BLOCK_N,
            block_k=_W4A16_GEMV_BLOCK_K,
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=1,
            block_n=_W4A16_GEMV_BLOCK_N,
            block_k=_W4A16_GEMV_BLOCK_K,
            num_warps=4,
            use_dot=False,
            split_k=split_k,
        )
    if rows <= 4 or input_dtype is torch.float32 or no_dot:
        block_m = 4 if rows <= 4 else (8 if rows <= 8 else _W4A16_GEMV_MAX_ROWS)
        split_k = _select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            block_n=_W4A16_GEMV_BLOCK_N,
            block_k=_W4A16_GEMV_BLOCK_K,
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=block_m,
            block_n=_W4A16_GEMV_BLOCK_N,
            block_k=_W4A16_GEMV_BLOCK_K,
            num_warps=4,
            use_dot=False,
            split_k=split_k,
        )
    split_k = _select_w4a16_split_k(
        num_sms,
        out_features,
        input_features,
        block_n=_W4A16_DOT_BLOCK_N,
        block_k=_W4A16_DOT_BLOCK_K,
    )
    # 8 warps eliminates pre-existing 40-76 B num_warps=4 ptxas spills on SM86/SM89.
    dot_num_warps = 8 if (target is not None and target.is_nvidia_cuda) else 4
    return _W4A16Schedule(
        mode="gemv",
        block_m=_W4A16_GEMV_MAX_ROWS,
        block_n=_W4A16_DOT_BLOCK_N,
        block_k=_W4A16_DOT_BLOCK_K,
        num_warps=dot_num_warps,
        use_dot=True,
        split_k=split_k,
    )


def _sm8x_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
    *,
    target: AcceleratorTarget | None = None,
) -> _W4A16Schedule:
    return _generic_w4a16_schedule(
        num_sms, rows, input_dtype, out_features, input_features, target=target
    )


def _rdna4_w4a16_large_m_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> _W4A16Schedule:
    if rows <= 32:
        split_k = _select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=32,
            block_n=_W4A16_DOT_BLOCK_N,
            block_k=_W4A16_DOT_BLOCK_K,
            cta_multiplier=8,
            max_split_k=4,
            round_up=False,
            power_of_two=True,
            cap_workspace=True,
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=32,
            block_n=_W4A16_DOT_BLOCK_N,
            block_k=_W4A16_DOT_BLOCK_K,
            num_warps=8,
            use_dot=True,
            split_k=split_k,
        )
    if rows <= 64:
        split_k = _select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=64,
            block_n=_W4A16_DOT_BLOCK_N,
            block_k=_W4A16_DOT_BLOCK_K,
            cta_multiplier=3,
            max_split_k=2,
            min_tiles_per_split=16,
            round_up=False,
            power_of_two=True,
            cap_workspace=True,
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=64,
            block_n=_W4A16_DOT_BLOCK_N,
            block_k=_W4A16_DOT_BLOCK_K,
            num_warps=8,
            use_dot=True,
            split_k=split_k,
            group_m=8,
        )
    dequant_crossover = 512 if input_dtype is torch.bfloat16 else 1024
    if rows > dequant_crossover:
        return _W4A16_RDNA4_DEQUANT_SCHEDULE
    tiles_128x128 = int(triton.cdiv(rows, 128)) * int(triton.cdiv(out_features, 128))
    if (
        input_dtype is torch.bfloat16
        and rows > 128
        and num_sms > 0
        and tiles_128x128 >= 2 * num_sms
    ):
        return _W4A16Schedule(
            mode="gemv",
            block_m=128,
            block_n=128,
            block_k=64,
            num_warps=8,
            use_dot=True,
            single_dot=False,
            split_k=1,
            group_m=8,
        )
    tiles_128x64 = int(triton.cdiv(rows, 128)) * int(triton.cdiv(out_features, 64))
    if num_sms > 0 and tiles_128x64 >= 2 * num_sms:
        return _W4A16Schedule(
            mode="gemv",
            block_m=128,
            block_n=64,
            block_k=64,
            num_warps=8,
            use_dot=True,
            single_dot=True,
            split_k=1,
            group_m=8,
        )
    split_k = _select_w4a16_split_k(
        num_sms,
        out_features,
        input_features,
        rows=rows,
        block_m=64,
        block_n=_W4A16_DOT_BLOCK_N,
        block_k=_W4A16_DOT_BLOCK_K,
        cta_multiplier=2,
        max_split_k=2,
        round_up=False,
        power_of_two=True,
        cap_workspace=True,
    )
    return _W4A16Schedule(
        mode="gemv",
        block_m=64,
        block_n=_W4A16_DOT_BLOCK_N,
        block_k=_W4A16_DOT_BLOCK_K,
        num_warps=8,
        use_dot=True,
        split_k=split_k,
        group_m=8,
    )


def _rdna4_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> _W4A16Schedule:
    if input_dtype is torch.float32 and rows > 1:
        if rows > _W4A16_GEMV_MAX_ROWS:
            return _W4A16_RDNA4_DEQUANT_SCHEDULE
        block_m = 4 if rows <= 4 else (8 if rows <= 8 else _W4A16_GEMV_MAX_ROWS)
        split_k = _select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=block_m,
            block_n=_W4A16_NARROW_BLOCK_N,
            block_k=512,
            cta_multiplier=16,
            min_k_tiles=4,
            min_tiles_per_split=2,
            max_split_k=8,
            power_of_two=True,
            cap_workspace=True,
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=block_m,
            block_n=_W4A16_NARROW_BLOCK_N,
            block_k=512,
            num_warps=4,
            use_dot=False,
            split_k=split_k,
        )
    if rows <= 4:
        block_m, block_n, block_k = (
            (1, 1, 512 if input_features < 1024 else 1024) if rows == 1 else (4, 2, 512)
        )
        split_k = (
            _select_w4a16_split_k(
                num_sms,
                out_features,
                input_features,
                rows=rows,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                cta_multiplier=4,
                min_k_tiles=4,
                min_tiles_per_split=2,
                power_of_two=True,
                cap_workspace=True,
            )
            if (num_sms > 0 and int(triton.cdiv(out_features, block_n)) < 4 * num_sms)
            else 1
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=1,
            use_dot=False,
            split_k=split_k,
        )
    if rows <= _W4A16_GEMV_MAX_ROWS:
        split_k = _select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=_W4A16_GEMV_MAX_ROWS,
            block_n=32,
            block_k=_W4A16_DOT_BLOCK_K,
            cta_multiplier=4,
            max_split_k=4,
            round_up=False,
            power_of_two=True,
            cap_workspace=True,
        )
        return _W4A16Schedule(
            mode="gemv",
            block_m=_W4A16_GEMV_MAX_ROWS,
            block_n=32,
            block_k=_W4A16_DOT_BLOCK_K,
            num_warps=4,
            use_dot=True,
            split_k=split_k,
        )
    return _rdna4_w4a16_large_m_schedule(num_sms, rows, input_dtype, out_features, input_features)


def _select_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
    *,
    target: AcceleratorTarget | None = None,
) -> _W4A16Schedule:
    """Resolve the architecture-specific W4A16 schedule shared by runtime and offline compile."""
    if target is not None and target.is_cuda_capability(12):
        schedule = _sm120_w4a16_schedule(num_sms, rows, input_dtype, out_features, input_features)
    elif target is not None and target.is_cuda_capability(8):
        schedule = _sm8x_w4a16_schedule(
            num_sms, rows, input_dtype, out_features, input_features, target=target
        )
    elif target is not None and target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201"):
        schedule = _rdna4_w4a16_schedule(num_sms, rows, input_dtype, out_features, input_features)
    else:
        schedule = _generic_w4a16_schedule(
            num_sms, rows, input_dtype, out_features, input_features, target=target
        )
    if schedule.mode == "gemv" and schedule.group_m == 0 and rows > schedule.block_m:
        raise ValueError(
            f"W4A16 schedule requires group_m > 0 when rows ({rows}) > block_m ({schedule.block_m})"
        )
    return schedule


def _optional_ptr_binding(
    name: str, ptr_type: str | None
) -> tuple[dict[str, str], dict[str, object]]:
    return ({name: ptr_type}, {}) if ptr_type is not None else ({}, {name: None})


def _compile_w4a16_reduce_kernel(
    target: GPUTarget,
    *,
    bias_ty: str | None,
    out_ty: str,
    has_weight_per_tensor_scale: bool,
    has_bias: bool,
    post_scale_only: bool = False,
    compile_fn: TritonCompileFn = default_triton_compile,
) -> triton.compiler.CompiledKernel:
    scale_sig, scale_const = _optional_ptr_binding(
        "weight_per_tensor_scale_ptr", "*fp32" if has_weight_per_tensor_scale else None
    )
    bias_sig, bias_const = _optional_ptr_binding("bias_ptr", bias_ty if has_bias else None)
    return compile_fn(
        ASTSource(
            w4a16_splitk_reduce_kernel,
            {
                "partial_ptr": "*fp32",
                **scale_sig,
                **bias_sig,
                "output_ptr": out_ty,
                "rows": "i32",
                "out_features": "i32",
                "split_k": "i32",
            },
            constexprs={
                **scale_const,
                **bias_const,
                "block_n": _W4A16_REDUCE_BLOCK_N,
                "has_weight_per_tensor_scale": has_weight_per_tensor_scale,
                "has_bias": has_bias,
                "post_scale_only": post_scale_only,
            },
        ),
        target,
        {"num_warps": 4},
    )


def compile_w4a16_kernels(
    target: GPUTarget,
    *,
    rows: int,
    out_features: int,
    input_features: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype | None = None,
    group_size: int = 0,
    high_first: bool = False,
    has_weight_per_tensor_scale: bool = True,
    num_sms: int = 64,
    compile_fn: TritonCompileFn = default_triton_compile,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile the production W4A16 Triton kernels selected for one workload."""
    if rows <= 0 or out_features <= 0 or input_features <= 0 or num_sms <= 0:
        raise ValueError("Workload dimensions and SM count must be positive integers")
    if input_dtype not in TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported W4A16 input dtype: {input_dtype}")
    if bias_dtype is not None and bias_dtype not in TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported W4A16 bias dtype: {bias_dtype}")
    if group_size != 0:
        validate_group_size(group_size)

    active_dtype = (
        torch.float32 if (group_size > 0 and input_dtype is torch.float16) else input_dtype
    )
    active_ty = TRITON_FLOAT_POINTER_TYPES[active_dtype]
    out_ty = TRITON_FLOAT_POINTER_TYPES[input_dtype]
    has_bias = bias_dtype is not None
    bias_ty = None if bias_dtype is None else TRITON_FLOAT_POINTER_TYPES[bias_dtype]
    target_info = AcceleratorTarget.from_compiler_target(target)
    schedule = _select_w4a16_schedule(
        num_sms,
        rows,
        active_dtype,
        out_features,
        input_features,
        target=target_info,
    )
    if schedule.mode == "dequant":
        sm75_bf16 = (
            target_info.is_nvidia_cuda
            and not target_info.cuda_capability_at_least(8)
            and active_dtype is torch.bfloat16
        )
        compiled_large: dict[str, triton.compiler.CompiledKernel] = {
            "dequant": compile_fn(
                ASTSource(
                    dequantize_nvfp4_swizzled_kernel,
                    {
                        "weight_ptr": "*u8",
                        "weight_scale_ptr": "*u8",
                        "output_ptr": "*fp32" if sm75_bf16 else active_ty,
                        "out_features": "i32",
                        "packed_k": "i32",
                        "scale_column_blocks": "i32",
                    },
                    constexprs={
                        "block_n": schedule.block_n,
                        "block_k": schedule.block_k,
                        "high_first": high_first,
                        "use_sm120_e2m1x2": schedule.use_sm120_e2m1x2,
                    },
                ),
                target,
                {"num_warps": schedule.num_warps},
            )
        }
        if has_weight_per_tensor_scale or has_bias or active_dtype is not input_dtype:
            compiled_large["reduce"] = _compile_w4a16_reduce_kernel(
                target,
                bias_ty=bias_ty,
                out_ty=out_ty,
                has_weight_per_tensor_scale=has_weight_per_tensor_scale,
                has_bias=has_bias,
                post_scale_only=has_weight_per_tensor_scale and active_dtype is not torch.float16,
                compile_fn=compile_fn,
            )
        return compiled_large

    is_splitk = schedule.split_k > 1
    scale_sig, scale_const = _optional_ptr_binding(
        "weight_per_tensor_scale_ptr", "*fp32" if has_weight_per_tensor_scale else None
    )
    bias_sig, bias_const = _optional_ptr_binding("bias_ptr", bias_ty if has_bias else None)
    compile_opts: dict[str, object] = {"num_warps": schedule.num_warps}
    if schedule.num_stages is not None:
        compile_opts["num_stages"] = schedule.num_stages
    compiled: dict[str, triton.compiler.CompiledKernel] = {
        "gemv": compile_fn(
            ASTSource(
                w4a16_gemv_splitk_kernel,
                {
                    "input_ptr": active_ty,
                    "weight_ptr": "*u8",
                    "weight_scale_ptr": "*u8",
                    **scale_sig,
                    **bias_sig,
                    "output_ptr": "*fp32" if is_splitk else out_ty,
                    "rows": "i32",
                    "out_features": "i32",
                    "packed_k": "i32",
                    "scale_column_blocks": "i32",
                    "split_k": "i32",
                },
                constexprs={
                    **scale_const,
                    **bias_const,
                    "block_m": schedule.block_m,
                    "block_n": schedule.block_n,
                    "block_k": schedule.block_k,
                    "high_first": high_first,
                    "has_weight_per_tensor_scale": has_weight_per_tensor_scale,
                    "has_bias": has_bias,
                    "is_splitk": is_splitk,
                    "use_dot": schedule.use_dot,
                    "use_sm120_e2m1x2": schedule.use_sm120_e2m1x2,
                    "single_dot": schedule.single_dot,
                    "group_m": schedule.group_m,
                },
            ),
            target,
            compile_opts,
        )
    }
    if is_splitk:
        compiled["reduce"] = _compile_w4a16_reduce_kernel(
            target,
            bias_ty=bias_ty,
            out_ty=out_ty,
            has_weight_per_tensor_scale=has_weight_per_tensor_scale,
            has_bias=has_bias,
            post_scale_only=(
                has_weight_per_tensor_scale
                and not (schedule.use_dot and active_dtype is torch.float16)
            ),
            compile_fn=compile_fn,
        )
    return compiled


def _dequantize_nvfp4_swizzled(
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    high_first: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
    block_n: int = _W4A16_DEQUANT_BLOCK_N,
    block_k: int = _W4A16_DEQUANT_BLOCK_K,
    num_warps: int = 4,
    use_sm120_e2m1x2: bool = False,
) -> torch.Tensor:
    """Dequantize a 2-D swizzled NVFP4 weight matrix in Triton without global scale."""
    out_features, packed_k = weight_qdata.shape
    input_features = 2 * packed_k
    output = torch.empty(
        (out_features, input_features),
        device=weight_qdata.device,
        dtype=output_dtype,
    )
    if out_features == 0 or input_features == 0:
        return output
    scale_column_blocks = int(triton.cdiv(input_features, _layout.SCALE_COLUMN_TILE))
    grid = (
        triton.cdiv(out_features, block_n),
        triton.cdiv(input_features, block_k),
    )
    with device_context(weight_qdata.device):
        dequantize_nvfp4_swizzled_kernel[grid](
            weight_qdata,
            weight_scale.view(torch.uint8),
            output,
            out_features,
            packed_k,
            scale_column_blocks,
            block_n=block_n,
            block_k=block_k,
            high_first=high_first,
            use_sm120_e2m1x2=use_sm120_e2m1x2,
            num_warps=num_warps,
        )
    return output


def _linear_w4a16_large_m(
    active_2d: torch.Tensor,
    input_dtype: torch.dtype,
    input_shape: torch.Size,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    high_first: bool,
    target_info: AcceleratorTarget,
    schedule: _W4A16Schedule,
) -> torch.Tensor:
    rows, out_features = active_2d.shape[0], weight_qdata.shape[0]
    sm75_bf16 = (
        target_info.is_nvidia_cuda
        and not target_info.cuda_capability_at_least(8)
        and active_2d.dtype is torch.bfloat16
    )
    dequant_weight = _dequantize_nvfp4_swizzled(
        weight_qdata,
        weight_scale,
        high_first=high_first,
        output_dtype=torch.float32 if sm75_bf16 else active_2d.dtype,
        block_n=schedule.block_n,
        block_k=schedule.block_k,
        num_warps=schedule.num_warps,
        use_sm120_e2m1x2=schedule.use_sm120_e2m1x2,
    )
    with torch.amp.autocast(active_2d.device.type, enabled=False):
        weight_t = dequant_weight.T
        if weight_per_tensor_scale is None and bias is None and active_2d.dtype is input_dtype:
            product = (
                torch.mm(active_2d.float(), weight_t).to(input_dtype)
                if sm75_bf16
                else torch.mm(active_2d, weight_t)
            )
            return product.reshape(*input_shape[:-1], out_features)
        output = torch.empty((rows, out_features), device=active_2d.device, dtype=input_dtype)
        post_scale_only = (
            weight_per_tensor_scale is not None and active_2d.dtype is not torch.float16
        )
        pre_scale = None
        if post_scale_only:
            assert weight_per_tensor_scale is not None
            gs_f32 = weight_per_tensor_scale.to(device=active_2d.device, dtype=torch.float32)
            pre_scale = torch.where(gs_f32.abs() < 1.0, gs_f32, torch.ones_like(gs_f32))
        chunk_rows = min(
            rows,
            max(1, _W4A16_FP32_WORKSPACE_BYTES // (out_features * 4)),
            _W4A16_MAX_REDUCE_ROWS,
        )
        with device_context(active_2d.device):
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                active_chunk = active_2d[start:end]
                if pre_scale is not None:
                    active_chunk = (active_chunk.float() * pre_scale).to(active_2d.dtype)
                if active_2d.dtype is torch.float32 or sm75_bf16:
                    partial = torch.mm(
                        active_chunk.float() if sm75_bf16 else active_chunk, weight_t
                    )
                else:
                    partial = torch.mm(active_chunk, weight_t, out_dtype=torch.float32)
                chunk_len = end - start
                reduce_grid = (triton.cdiv(out_features, _W4A16_REDUCE_BLOCK_N), chunk_len)
                w4a16_splitk_reduce_kernel[reduce_grid](
                    partial,
                    weight_per_tensor_scale,
                    bias,
                    output[start:end],
                    chunk_len,
                    out_features,
                    1,
                    block_n=_W4A16_REDUCE_BLOCK_N,
                    has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
                    has_bias=bias is not None,
                    post_scale_only=post_scale_only,
                    num_warps=4,
                )
    return output.reshape(*input_shape[:-1], out_features)


def _linear_w4a16(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    high_first: bool = False,
    *,
    group_size: int = 0,
) -> torch.Tensor:
    """Apply a swizzled NVFP4 weight to a floating activation, optionally rotated."""
    out_features, packed_k = weight_qdata.shape
    input_features = 2 * packed_k
    if input.numel() == 0 or out_features == 0:
        return input.new_empty((*input.shape[:-1], out_features))

    input_2d = input.reshape(-1, input_features).contiguous()
    rows = int(input_2d.shape[0])
    if group_size > 0:
        from piper_kernels.weights.convrot.triton import rotate_input  # noqa: PLC0415

        rotated = torch.empty(
            (rows, input_features),
            device=input_2d.device,
            dtype=torch.float32 if input.dtype is torch.float16 else input.dtype,
        )
        rotate_input(input_2d, rotated, group_size, num_warps=4)
        active_2d = rotated
    else:
        active_2d = input_2d

    target_info = AcceleratorTarget.from_device(input.device)
    schedule = _select_w4a16_schedule(
        device_sm_count(input.device),
        rows,
        active_2d.dtype,
        out_features,
        input_features,
        target=target_info,
    )
    if schedule.mode == "dequant":
        return _linear_w4a16_large_m(
            active_2d,
            input.dtype,
            input.shape,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            bias,
            high_first,
            target_info,
            schedule,
        )

    output = torch.empty((rows, out_features), device=input.device, dtype=input.dtype)
    scale_column_blocks = int(triton.cdiv(input_features, _layout.SCALE_COLUMN_TILE))
    is_splitk = schedule.split_k > 1
    target = (
        torch.empty(
            (schedule.split_k, rows, out_features), device=input.device, dtype=torch.float32
        )
        if is_splitk
        else output
    )
    grid = (
        int(triton.cdiv(rows, schedule.block_m)) * int(triton.cdiv(out_features, schedule.block_n)),
        schedule.split_k,
    )
    launch_kwargs: dict[str, int] = {"num_warps": schedule.num_warps}
    if schedule.num_stages is not None:
        launch_kwargs["num_stages"] = schedule.num_stages
    with device_context(input.device):
        w4a16_gemv_splitk_kernel[grid](
            active_2d,
            weight_qdata,
            weight_scale.view(torch.uint8),
            weight_per_tensor_scale,
            bias,
            target,
            rows,
            out_features,
            packed_k,
            scale_column_blocks,
            schedule.split_k,
            block_m=schedule.block_m,
            block_n=schedule.block_n,
            block_k=schedule.block_k,
            high_first=high_first,
            has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
            has_bias=bias is not None,
            is_splitk=is_splitk,
            use_dot=schedule.use_dot,
            use_sm120_e2m1x2=schedule.use_sm120_e2m1x2,
            single_dot=schedule.single_dot,
            group_m=schedule.group_m,
            **launch_kwargs,
        )
        if is_splitk:
            reduce_grid = (triton.cdiv(out_features, _W4A16_REDUCE_BLOCK_N), rows)
            w4a16_splitk_reduce_kernel[reduce_grid](
                target,
                weight_per_tensor_scale,
                bias,
                output,
                rows,
                out_features,
                schedule.split_k,
                block_n=_W4A16_REDUCE_BLOCK_N,
                has_weight_per_tensor_scale=weight_per_tensor_scale is not None,
                has_bias=bias is not None,
                post_scale_only=(
                    weight_per_tensor_scale is not None
                    and not (schedule.use_dot and active_2d.dtype is torch.float16)
                ),
                num_warps=4,
            )
    return output.reshape(*input.shape[:-1], out_features)


__all__ = [
    "add_bias_out",
    "compile_w4a16_kernels",
    "dynamic_scale",
    "encode_nvfp4_blocks",
    "linear_mean",
    "pack_e2m1_pairs",
    "prepare_dynamic",
    "prepare_static",
    "prepare_static_out",
    "swizzled_scale_offsets",
]
