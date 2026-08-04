"""UINT8-equivalent P with feature-rotated signed-INT8 V.

This experiment avoids rotating the quadratic probability matrix.  Instead it
rotates V along the head-feature dimension, evaluates PV in the rotated basis,
and applies the inverse rotation to the output.

The primary path chooses one V scale per key row after rotation. Because that
scale lies on the PV contraction dimension, it must affect each key's
probability contribution:

    V_rot = diag(s_v) @ V_q
    P @ V_rot = (P * s_v) @ V_q.

The default evaluates the same product in log-softmax coordinates. For
``y = score + log(s_v)``:

    sum(exp(score) * s_v * V_q)   sum(exp(y) * V_q)
    -------------------------------- = ---------------------.
           sum(exp(score))           sum(exp(y) / s_v)

This produces the same normalized UINT8 probability operand without an extra
``max(P * s_v)`` reduction for every query/K tile.

The older per-feature scaling mode remains available as a control.

NVIDIA's integer MMA consumes signed INT8 operands through Triton's public dot
interface.  Nonnegative probabilities use all 256 UINT8 codes through the
affine identity

    u @ v = (u - 128) @ v + 128 * sum(v),

where ``u`` is an integer in ``[0, 255]`` and ``u - 128`` is representable as
signed INT8.  The complete ``128 * sum(v)`` term is produced during V
quantization and supplied as the integer MMA accumulator, so the attention loop
only loads one INT32 correction vector per K=64 tile.
"""

# ruff: noqa: ANN001, ANN202, ARG001, PLR0912, PLR0913, PLR0915, PLR0917

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from typing import Literal

import torch
import triton
import triton.language as tl

from piper_kernels.attention._convrot_triton import rotate_rows_in_registers
from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.uint4_pv_convrot import (
    _inverse_rotate_output_kernel,
    _prepare_qk,
)

_PV_BLOCK = 64
_P_UINT8_RANGE = tl.constexpr(255.0)
_P_ZERO_POINT = tl.constexpr(128)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)


@triton.jit
def _quantize_value_feature_convrot_int8_kernel(
    value_ptr,
    scale_ptr,
    log_scale_ptr,
    inverse_scale_ptr,
    correction_ptr,
    output_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    rotation_group: tl.constexpr,
    value_scale_floor: tl.constexpr,
    store_log_scale: tl.constexpr,
    store_value_scale: tl.constexpr,
    store_value_correction: tl.constexpr,
    output_transposed: tl.constexpr,
):
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value = rotate_rows_in_registers(value, offsets_d, block_n, rotation_group)
    scale = tl.max(tl.abs(value), axis=0) / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = _sage_backend._round_to_int8(
        value / scale[None, :],
        _V_INT8_RANGE,
    )
    scale_block = (batch * heads + head) * tl.cdiv(key_length, block_n) + key_block
    metadata_offsets = scale_block * head_dim + offsets_d
    if store_value_scale:
        tl.store(scale_ptr + metadata_offsets, scale)
    if store_value_correction:
        value_correction = _P_ZERO_POINT * tl.sum(quantized.to(tl.int32), axis=0)
        tl.store(correction_ptr + metadata_offsets, value_correction)
    _sage_backend._store_value_tile(
        output_ptr,
        quantized,
        batch,
        head,
        offsets_n,
        offsets_d,
        mask,
        stride_ob,
        stride_oh,
        stride_on,
        output_transposed,
    )


@triton.jit
def _quantize_value_feature_convrot_per_key_int8_kernel(
    value_ptr,
    scale_ptr,
    log_scale_ptr,
    inverse_scale_ptr,
    correction_ptr,
    output_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    rotation_group: tl.constexpr,
    value_scale_floor: tl.constexpr,
    store_log_scale: tl.constexpr,
    store_value_scale: tl.constexpr,
    store_value_correction: tl.constexpr,
    output_transposed: tl.constexpr,
):
    """Rotate features, then use one symmetric INT8 scale per key row."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid_keys = offsets_n < key_length
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid_keys[:, None],
        other=0.0,
    ).to(tl.float32)
    value = rotate_rows_in_registers(value, offsets_d, block_n, rotation_group)
    scale = tl.max(tl.abs(value), axis=1) / _V_INT8_RANGE + _SCALE_EPSILON
    if value_scale_floor > 0.0:
        scale = tl.maximum(scale, tl.max(scale, axis=0) * value_scale_floor)
    quantized = _sage_backend._round_to_int8(
        value / scale[:, None],
        _V_INT8_RANGE,
    )
    batch_head = batch * heads + head
    if store_value_scale:
        tl.store(
            scale_ptr + batch_head * key_length + offsets_n,
            scale,
            mask=valid_keys,
        )
    if store_log_scale:
        tl.store(
            log_scale_ptr + batch_head * key_length + offsets_n,
            tl.log2(scale),
            mask=valid_keys,
        )
        tl.store(
            inverse_scale_ptr + batch_head * key_length + offsets_n,
            1.0 / scale,
            mask=valid_keys,
        )
    if store_value_correction:
        value_correction = _P_ZERO_POINT * tl.sum(quantized.to(tl.int32), axis=0)
        correction_offsets = (
            batch_head * tl.cdiv(key_length, block_n) + key_block
        ) * head_dim + offsets_d
        tl.store(correction_ptr + correction_offsets, value_correction)
    _sage_backend._store_value_tile(
        output_ptr,
        quantized,
        batch,
        head,
        offsets_n,
        offsets_d,
        valid_keys[:, None],
        stride_ob,
        stride_oh,
        stride_on,
        output_transposed,
    )


@triton.jit
def _uint8_pv_feature_convrot_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    value_log_scale_ptr,
    value_inverse_scale_ptr,
    value_correction_ptr,
    rotated_output_ptr,
    query_length,
    key_length,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    value_scale_per_key: tl.constexpr,
    tile_probability_scale: tl.constexpr,
    log_probability_scale: tl.constexpr,
    weighted_log_denominator: tl.constexpr,
    affine_probability: tl.constexpr,
    output_rotation_group: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    value_transposed: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
    use_tensor_descriptors: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid_queries = offsets_m < query_length

    query = tl.load(
        query_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr + (batch * heads + head) * tl.cdiv(query_length, 32) + offsets_m // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + (batch * heads + head) * query_length + offsets_m,
            mask=valid_queries,
            other=0.0,
        )

    accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, (query_block + 1) * block_m)

    for start_n in tl.range(0, end_n, block_n, disable_licm=True):
        current_n = start_n + offsets_n
        batch_head = batch * heads + head
        key = _sage_backend._load_attention_key_tile(
            key_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores = tl.dot(query, key, out_dtype=tl.int32)
        if grouped_qk:
            key_scale = tl.load(
                key_scale_ptr
                + (batch * heads + head) * tl.cdiv(key_length, block_n)
                + start_n // block_n
            )
            scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
        else:
            key_scale = tl.load(
                key_scale_ptr + (batch * heads + head) * key_length + current_n,
                mask=current_n < key_length,
                other=0.0,
            )
            scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]

        valid_keys = current_n[None, :] < key_length
        if is_causal:
            valid_keys &= current_n[None, :] <= offsets_m[:, None]
        scores = tl.where(valid_queries[:, None] & valid_keys, scores, -float("inf"))

        if value_scale_per_key:
            if log_probability_scale:
                # Change softmax coordinates from z to y=z+log2(s_v):
                #   sum(exp(z) s_v Vq) / sum(exp(z))
                # = sum(exp(y) Vq) / sum(exp(y) / s_v).
                # This gives fixed-range UINT8 probabilities without the
                # separate max(P * s_v) reduction used by dynamic mode.
                key_value_log_scale = tl.load(
                    value_log_scale_ptr + (batch * heads + head) * key_length + current_n,
                    mask=current_n < key_length,
                    other=0.0,
                )
                scores += key_value_log_scale[None, :]
            else:
                key_value_scale = tl.load(
                    value_scale_ptr + (batch * heads + head) * key_length + current_n,
                    mask=current_n < key_length,
                    other=0.0,
                )

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_weight = tl.where(valid_queries, tl.exp2(running_max - next_max), 0.0)
        current_weight = tl.where(
            valid_queries,
            tl.exp2(block_max - next_max),
            0.0,
        )
        probabilities = tl.where(
            valid_queries[:, None] & valid_keys,
            tl.exp2(scores - block_max[:, None]),
            0.0,
        )
        accumulator *= old_weight[:, None]
        if value_scale_per_key and log_probability_scale and weighted_log_denominator:
            key_value_inverse_scale = tl.load(
                value_inverse_scale_ptr + (batch * heads + head) * key_length + current_n,
                mask=current_n < key_length,
                other=1.0,
            )
            denominator_contribution = tl.sum(
                probabilities * key_value_inverse_scale[None, :],
                axis=1,
            )
        else:
            denominator_contribution = tl.sum(probabilities, axis=1)
        denominator = denominator * old_weight + denominator_contribution * current_weight

        metadata_block = (batch * heads + head) * tl.cdiv(key_length, block_n) + start_n // block_n
        metadata_offsets = metadata_block * head_dim + offsets_d
        if affine_probability:
            probability_range: tl.constexpr = _P_UINT8_RANGE
            value_correction = tl.load(value_correction_ptr + metadata_offsets)
        else:
            probability_range: tl.constexpr = _V_INT8_RANGE
        if value_scale_per_key:
            if log_probability_scale:
                probability_for_dot = probabilities
                probability_quant_scale = tl.full(
                    (block_m,),
                    1.0 / probability_range,
                    dtype=tl.float32,
                )
                probability_output_scale = probability_quant_scale
            elif tile_probability_scale:
                # A single scale for the whole K tile avoids a separate
                # block_m-wide max reduction.  Dividing V's per-key scales by
                # their tile maximum keeps the UINT8 operand in [0, 255].
                tile_value_scale = tl.max(key_value_scale, axis=0) + 1e-30
                probability_for_dot = probabilities * (key_value_scale / tile_value_scale)[None, :]
                probability_quant_scale = tl.full(
                    (block_m,),
                    1.0 / probability_range,
                    dtype=tl.float32,
                )
                probability_output_scale = tl.full(
                    (block_m,),
                    tile_value_scale / probability_range,
                    dtype=tl.float32,
                )
            else:
                probability_for_dot = probabilities * key_value_scale[None, :]
                probability_quant_scale = (
                    tl.max(probability_for_dot, axis=1) / probability_range + 1e-30
                )
                probability_output_scale = probability_quant_scale
        else:
            probability_for_dot = probabilities
            probability_quant_scale = tl.full(
                (block_m,),
                1.0 / probability_range,
                dtype=tl.float32,
            )
            probability_output_scale = probability_quant_scale

        # Encode UINT8 probabilities as signed INT8 for IMMA.  With per-key V
        # scales, the quantity represented here is P[k] * scale_V[k], which
        # folds the scale on the contraction dimension into the left operand.
        # Invalid and
        # masked probabilities are zero, hence encode as -128 and are exactly
        # cancelled by the tile-wide +128 * sum(Vq) correction below.
        probability_codes = tl.minimum(
            probability_range,
            probability_for_dot / probability_quant_scale[:, None] + 0.5,
        ).to(tl.int32)
        if affine_probability:
            probability_int8 = (probability_codes - _P_ZERO_POINT).to(tl.int8)
        else:
            probability_int8 = probability_codes.to(tl.int8)
        value = _sage_backend._load_attention_value_tile(
            value_ptr,
            batch,
            head,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            value_transposed,
            use_tensor_descriptors,
            heads,
            head_dim,
            block_n,
        )
        if affine_probability:
            correction_accumulator = (
                tl.zeros(
                    (block_m, head_dim),
                    dtype=tl.int32,
                )
                + value_correction[None, :]
            )
            corrected_int32 = tl.dot(
                probability_int8,
                value,
                correction_accumulator,
                out_dtype=tl.int32,
            )
        else:
            corrected_int32 = tl.dot(
                probability_int8,
                value,
                out_dtype=tl.int32,
            )
        if value_scale_per_key:
            accumulator += (
                corrected_int32.to(tl.float32)
                * probability_output_scale[:, None]
                * current_weight[:, None]
            )
        else:
            value_scale = tl.load(value_scale_ptr + metadata_offsets)
            accumulator += (
                corrected_int32.to(tl.float32)
                * probability_output_scale[:, None]
                * value_scale[None, :]
                * current_weight[:, None]
            )
        running_max = next_max

    rotated_output = accumulator / tl.maximum(denominator, 1e-30)[:, None]
    rotated_output = rotate_rows_in_registers(
        rotated_output,
        offsets_d,
        block_m,
        output_rotation_group,
    )
    tl.store(
        rotated_output_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        rotated_output,
        mask=valid_queries[:, None],
    )


def _prepare_uint8_pv_feature_convrot_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    rotation_group: int,
    value_scale_axis: Literal["feature", "key"],
    value_scale_floor: float,
    probability_scale_mode: Literal["dynamic", "tile", "log"],
    value_transposed: bool = True,
    affine_probability: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Quantize canonical Q/K and feature-rotated block-scaled INT8 V."""
    if rotation_group not in (0, 16, 64):
        raise ValueError(f"rotation group must be 0, 16, or 64, got {rotation_group}")
    head_dim = value.shape[-1]
    if rotation_group and head_dim % rotation_group:
        raise ValueError(
            f"head dimension {head_dim} must be divisible by rotation group {rotation_group}"
        )
    if value_scale_axis not in ("feature", "key"):
        raise ValueError(f"value scale axis must be 'feature' or 'key', got {value_scale_axis!r}")
    if not 0.0 <= value_scale_floor <= 1.0:
        raise ValueError(f"value scale floor must be in [0, 1], got {value_scale_floor}")
    if value_scale_floor and value_scale_axis != "key":
        raise ValueError("value scale flooring requires per-key value scales")
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        scale,
        grouped_qk,
    )
    batch, heads, key_length, _ = value.shape
    value_blocks = (key_length + _PV_BLOCK - 1) // _PV_BLOCK
    correction_shape = (batch, heads, value_blocks, head_dim)
    scale_shape = (batch, heads, key_length) if value_scale_axis == "key" else correction_shape
    if value_scale_axis == "key" and probability_scale_mode == "log":
        value_log_scale = torch.empty(scale_shape, device=value.device, dtype=torch.float16)
        value_inverse_scale = torch.empty(scale_shape, device=value.device, dtype=torch.float16)
        value_scale = value_inverse_scale
    else:
        value_scale = torch.empty(scale_shape, device=value.device, dtype=torch.float32)
        value_log_scale = value_scale
        value_inverse_scale = value_scale
    value_correction = torch.empty(
        correction_shape if affine_probability else (1,),
        device=value.device,
        dtype=torch.int32,
    )
    value_int8_shape = (
        (batch, heads, head_dim, key_length) if value_transposed else value.shape
    )
    value_int8 = torch.empty(value_int8_shape, device=value.device, dtype=torch.int8)
    quantize_kernel = (
        _quantize_value_feature_convrot_per_key_int8_kernel
        if value_scale_axis == "key"
        else _quantize_value_feature_convrot_int8_kernel
    )
    quantize_kernel[(value_blocks, heads, batch)](
        value,
        value_scale,
        value_log_scale,
        value_inverse_scale,
        value_correction,
        value_int8,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_int8.stride(0),
        value_int8.stride(1),
        value_int8.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=_PV_BLOCK,
        rotation_group=rotation_group,
        value_scale_floor=value_scale_floor,
        store_log_scale=probability_scale_mode == "log",
        store_value_scale=probability_scale_mode != "log" or value_scale_axis == "feature",
        store_value_correction=affine_probability,
        output_transposed=value_transposed,
        num_warps=4,
    )
    return (
        query_int8,
        key_int8,
        value_int8,
        query_scale,
        key_scale,
        value_scale,
        value_log_scale,
        value_inverse_scale,
        value_correction,
    )


def _launch_uint8_pv_feature_convrot_attention(
    prepared: tuple[torch.Tensor, ...],
    rotated_output: torch.Tensor,
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    is_causal: bool,
    *,
    grouped_qk: bool,
    rotation_group: int,
    value_scale_axis: Literal["feature", "key"],
    probability_scale_mode: Literal["dynamic", "tile", "log"],
    fuse_output_rotation: bool,
    block_m: int,
    num_warps: int,
    num_stages: int,
    value_transposed: bool = True,
    weighted_log_denominator: bool = True,
    affine_probability: bool = True,
    use_tensor_descriptors: bool = False,
) -> torch.Tensor:
    """Launch prequantized attention followed by its feature inverse rotation."""
    (
        query,
        key,
        value,
        query_scale,
        key_scale,
        value_scale,
        value_log_scale,
        value_inverse_scale,
        value_correction,
    ) = prepared
    batch, heads, _, head_dim = query.shape
    if probability_scale_mode not in ("dynamic", "tile", "log"):
        raise ValueError(
            "probability scale mode must be 'dynamic', 'tile', or 'log', "
            f"got {probability_scale_mode!r}"
        )
    if probability_scale_mode != "dynamic" and value_scale_axis != "key":
        raise ValueError(f"{probability_scale_mode} probability scaling requires per-key scales")
    key_argument, value_argument = _sage_backend._make_attention_arguments(
        key,
        value,
        batch,
        heads,
        key_length,
        head_dim,
        value_transposed,
        use_tensor_descriptors,
    )
    attention_grid = (triton.cdiv(query_length, block_m), heads, batch)
    attention_output = output if rotation_group == 0 or fuse_output_rotation else rotated_output
    _uint8_pv_feature_convrot_attention_kernel[attention_grid](
        query,
        key_argument,
        value_argument,
        query_scale,
        key_scale,
        value_scale,
        value_log_scale,
        value_inverse_scale,
        value_correction,
        attention_output,
        query_length,
        key_length,
        is_causal=is_causal,
        grouped_qk=grouped_qk,
        value_scale_per_key=value_scale_axis == "key",
        tile_probability_scale=probability_scale_mode == "tile",
        log_probability_scale=probability_scale_mode == "log",
        weighted_log_denominator=weighted_log_denominator,
        affine_probability=affine_probability,
        output_rotation_group=rotation_group if fuse_output_rotation else 0,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=_PV_BLOCK,
        value_transposed=value_transposed,
        use_tensor_descriptors=use_tensor_descriptors,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if rotation_group == 0 or fuse_output_rotation:
        return output
    rows = batch * heads * query_length
    _inverse_rotate_output_kernel[(rows,)](
        rotated_output,
        output,
        head_dim=head_dim,
        rotation_group=rotation_group,
        num_warps=4,
    )
    return output


def triton_sage_attention_uint8_pv_feature_convrot(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int = 0,
    value_scale_axis: Literal["feature", "key"] = "key",
    probability_scale_mode: Literal["dynamic", "tile", "log"] | None = None,
    value_scale_floor: float = 0.0,
    fuse_output_rotation: bool = True,
    grouped_qk: bool | None = None,
    affine_probability: bool = True,
) -> torch.Tensor:
    """Run per-key-scaled feature-ConvRot V with affine UINT8 P attention.

    Log-domain scaling is algebraically equivalent to dynamic ``P * scale_v`` normalization
    while avoiding its second per-query reduction. ``probability_scale_mode="tile"`` and
    ``value_scale_floor`` remain quality/performance ablations.
    """
    if grouped_qk is None:
        grouped_qk = torch.cuda.get_device_capability(query.device)[0] == 12
    if probability_scale_mode is None:
        probability_scale_mode = "log" if value_scale_axis == "key" else "dynamic"
    prepared = _prepare_uint8_pv_feature_convrot_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
        rotation_group=rotation_group,
        value_scale_axis=value_scale_axis,
        value_scale_floor=value_scale_floor,
        probability_scale_mode=probability_scale_mode,
        affine_probability=affine_probability,
    )
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    output = torch.empty_like(query)
    rotated_output = (
        output
        if rotation_group == 0 or fuse_output_rotation
        else torch.empty(query.shape, device=query.device, dtype=torch.float32)
    )
    use_tensor_descriptors = False
    if is_causal:
        block_m = 64
        num_stages = 3
    elif value_scale_axis == "key":
        if rotation_group:
            block_m = 32 if query_length <= 1152 else 64
            num_stages = 2 if block_m == 32 else 3
        elif not affine_probability:
            candidate_block_m = _sage_backend._select_query_block(
                query,
                batch,
                heads,
                query_length,
            )
            use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
                query,
                candidate_block_m,
                head_dim,
                key_length,
                True,
            )
            block_m = candidate_block_m if use_tensor_descriptors else 64
            num_stages = 2 if use_tensor_descriptors else 3
        else:
            block_m = 64
            num_stages = 3
    else:
        block_m = _sage_backend._select_query_block(
            query,
            batch,
            heads,
            query_length,
        )
        num_stages = 3
    return _launch_uint8_pv_feature_convrot_attention(
        prepared,
        rotated_output,
        output,
        query_length,
        key_length,
        is_causal,
        grouped_qk=grouped_qk,
        rotation_group=rotation_group,
        value_scale_axis=value_scale_axis,
        probability_scale_mode=probability_scale_mode,
        fuse_output_rotation=fuse_output_rotation,
        block_m=block_m,
        num_warps=4,
        num_stages=num_stages,
        affine_probability=affine_probability,
        use_tensor_descriptors=use_tensor_descriptors,
    )


def triton_sage_attention_int8_pv_per_key_log(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
) -> torch.Tensor:
    """Run exact per-key-scaled V with nonnegative signed-INT8 P."""
    return triton_sage_attention_uint8_pv_feature_convrot(
        query,
        key,
        value,
        scale,
        is_causal,
        rotation_group=0,
        value_scale_axis="key",
        probability_scale_mode="log",
        grouped_qk=grouped_qk,
        affine_probability=False,
    )
