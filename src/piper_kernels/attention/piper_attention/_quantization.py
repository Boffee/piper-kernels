"""Shared K1024/V1024 statistics and quantization for Piper Attention."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    count_key_scales,
)

CENTER_CHUNK = 1024
_BLOCK_N = 64
_TILES_PER_CHUNK = CENTER_CHUNK // _BLOCK_N
_TL_CENTER_CHUNK = tl.constexpr(CENTER_CHUNK)
_TL_BLOCK_N = tl.constexpr(_BLOCK_N)
_TL_TILES_PER_CHUNK = tl.constexpr(_TILES_PER_CHUNK)
_P_UINT8_RANGE = tl.constexpr(255.0)
_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)
# Keep per-chunk mean restoration finite when a constant chunk has tiny residual scale.
_VALUE_MEAN_SCALE_FLOOR = tl.constexpr(2.0**-80)


@triton.jit
def quantize_value_rows(value, scale_floor=None):
    """Return dense Piper's per-token signed INT8 values and FP32 scales."""
    value_scale = tl.max(tl.abs(value), axis=1) / _INT8_RANGE + _SCALE_EPSILON
    if scale_floor is not None:
        value_scale = tl.maximum(value_scale, scale_floor)
    return qk_quantization.round_to_int8(value / value_scale[:, None]), value_scale


@triton.jit(
    do_not_specialize=["key_length", "storage_key_length", "scale_groups", "chunk_count", "heads"]
)
def _quantize_key_k1024_kernel(
    key_ptr,
    output_ptr,
    scale_ptr,
    center_ptr,
    key_length,
    storage_key_length,
    scale_groups,
    chunk_count,  # noqa: ARG001 - Triton launch signature
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    stride_cb,
    stride_ch,
    grouped_qk: tl.constexpr,
    heads,  # noqa: ARG001 - Triton launch signature
    head_dim: tl.constexpr,
):
    """Compute one K1024 chunk's rotated mean and center/quantize its keys in one CTA."""
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    chunk_start = chunk * _TL_CENTER_CHUNK
    valid_count = tl.minimum(_TL_CENTER_CHUNK, key_length - chunk_start)
    active_tiles = tl.cdiv(valid_count, _TL_BLOCK_N)
    offsets_tile = tl.arange(0, _TL_BLOCK_N)
    offsets_d = tl.arange(0, head_dim)
    key_base = key_ptr + batch * stride_kb + head * stride_kh
    out_base = output_ptr + batch * stride_ob + head * stride_oh
    scale_base = scale_ptr + batch * stride_sb + head * stride_sh

    key_sum = tl.zeros((head_dim,), dtype=tl.float32)
    for tile_idx in tl.range(0, active_tiles, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        rows = chunk_start + tile_idx * _TL_BLOCK_N + offsets_tile
        valid = rows < key_length
        values = tl.load(
            key_base + rows[:, None] * stride_kn + offsets_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        key_sum += tl.sum(values, axis=0)

    key_mean = key_sum / tl.maximum(valid_count, 1).to(tl.float32)
    rotated_mean = tl.reshape(
        qk_quantization.rotate_signed_hadamard_heads(key_mean[None, :], head_dim),
        (head_dim,),
    )
    tl.store(
        center_ptr + batch * stride_cb + head * stride_ch + chunk * head_dim + offsets_d,
        rotated_mean,
    )

    base_tile = chunk * _TL_TILES_PER_CHUNK
    for tile_idx in tl.range(0, active_tiles, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        tile = base_tile + tile_idx
        rows = tile * _TL_BLOCK_N + offsets_tile
        valid = rows < key_length
        values = tl.load(
            key_base + rows[:, None] * stride_kn + offsets_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        centered = tl.where(valid[:, None], values - key_mean[None, :], 0.0)
        rotated = qk_quantization.rotate_signed_hadamard_heads(centered, head_dim)
        if grouped_qk:
            maximum = tl.max(tl.max(tl.abs(rotated), axis=1), axis=0)
            tile_scale = maximum / _INT8_RANGE + _SCALE_EPSILON
            quantized = qk_quantization.round_to_int8(rotated / tile_scale)
            tl.store(scale_base + tile, tile_scale, mask=tile < scale_groups)
        else:
            grouped = tl.reshape(rotated, (8, 4, 2, head_dim))
            thread_max = tl.max(tl.max(tl.max(tl.abs(grouped), axis=3), axis=2), axis=0)
            thread_scale = thread_max / _INT8_RANGE + _SCALE_EPSILON
            row_scales = tl.reshape(
                tl.broadcast_to(thread_scale[None, :, None], (8, 4, 2)),
                (_TL_BLOCK_N,),
            )
            quantized = qk_quantization.round_to_int8(rotated / row_scales[:, None])
            tl.store(scale_base + rows, row_scales, mask=rows < scale_groups)
        tl.store(
            out_base + rows[:, None] * stride_on + offsets_d[None, :],
            quantized,
            mask=(rows < storage_key_length)[:, None],
        )


# Unspecializing ``stride_od`` prevents output-padding alignment from altering the
# value-sum reduction tree (keeping padded vs unpadded V means bitwise equal) and
# avoids 16-byte transpose register bloat that halves value quantization throughput.
@triton.jit(
    do_not_specialize=[
        "key_length",
        "storage_key_length",
        "meta_storage_length",
        "heads",
        "stride_od",
    ]
)
def _quantize_value_k1024_kernel(
    value_ptr,
    output_ptr,
    scale_multiplier_ptr,
    log_scale_ptr,
    value_mean_ptr,
    key_length,
    storage_key_length,
    meta_storage_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_od,
    stride_ok,
    stride_mb,
    stride_mh,
    stride_cb,
    stride_ch,
    store_log_scale: tl.constexpr,
    pack_wmma: tl.constexpr,
    heads,  # noqa: ARG001 - Triton launch signature
    head_dim: tl.constexpr,
):
    """Center and quantize one V1024 chunk per CTA without partial workspace."""
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    chunk_start = chunk * _TL_CENTER_CHUNK
    valid_count = tl.minimum(_TL_CENTER_CHUNK, key_length - chunk_start)
    active_tiles = tl.cdiv(valid_count, _TL_BLOCK_N)
    offsets_tile = tl.arange(0, _TL_BLOCK_N)
    offsets_d = tl.arange(0, head_dim)
    value_base = value_ptr + batch * stride_vb + head * stride_vh
    out_base = output_ptr + batch * stride_ob + head * stride_oh
    meta_base = batch * stride_mb + head * stride_mh

    value_sum = tl.zeros((head_dim,), dtype=tl.float32)
    for tile_idx in tl.range(0, active_tiles, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        rows = chunk_start + tile_idx * _TL_BLOCK_N + offsets_tile
        valid = rows < key_length
        values = tl.load(
            value_base + rows[:, None] * stride_vn + offsets_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        value_sum += tl.sum(values, axis=0)

    value_mean = value_sum / tl.maximum(valid_count, 1).to(tl.float32)
    tl.store(
        value_mean_ptr + batch * stride_cb + head * stride_ch + chunk * head_dim + offsets_d,
        value_mean,
    )
    scale_floor = tl.max(tl.abs(value_mean), axis=0) * _VALUE_MEAN_SCALE_FLOOR

    base_tile = chunk * _TL_TILES_PER_CHUNK
    for tile_idx in tl.range(0, active_tiles, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        tile = base_tile + tile_idx
        rows = tile * _TL_BLOCK_N + offsets_tile
        valid = rows < key_length
        values = tl.load(
            value_base + rows[:, None] * stride_vn + offsets_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        centered = tl.where(valid[:, None], values - value_mean[None, :], 0.0)
        codes, scales = quantize_value_rows(centered, scale_floor)
        meta_mask = rows < meta_storage_length
        tl.store(scale_multiplier_ptr + meta_base + rows, scales * _P_UINT8_RANGE, mask=meta_mask)
        if store_log_scale:
            tl.store(log_scale_ptr + meta_base + rows, tl.log2(scales), mask=meta_mask)
        transposed = tl.trans(codes)
        if pack_wmma:
            packed_token = (
                (offsets_tile & ~24) | ((offsets_tile & 8) << 1) | ((offsets_tile & 16) >> 1)
            )
            tl.store(
                out_base
                + tile * head_dim * _TL_BLOCK_N
                + offsets_d[:, None] * _TL_BLOCK_N
                + packed_token[None, :],
                transposed,
            )
        else:
            tl.store(
                out_base + offsets_d[:, None] * stride_od + rows[None, :] * stride_ok,
                transposed,
                mask=(rows < storage_key_length)[None, :],
            )


_quantize_value_k1024_unspecialized_kernel = triton.jit(
    do_not_specialize=[
        "key_length",
        "storage_key_length",
        "meta_storage_length",
        "heads",
        "stride_od",
    ],
)(_quantize_value_k1024_kernel.fn)


def prepare_chunk_centered_key(
    key: torch.Tensor,
    *,
    grouped: bool,
    storage_key_length: int,
    scale_storage_length: int | None = None,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K1024-centered keys and return (key_int8, key_scale, key_center)."""
    batch, heads, key_length, head_dim = key.shape
    chunk_count = int(triton.cdiv(key_length, CENTER_CHUNK))
    granularity = "per_warp" if grouped else "per_thread"
    key_shape = (batch, heads, storage_key_length, head_dim)
    scale_length = key_length if scale_storage_length is None else scale_storage_length
    scale_count = (
        count_key_scales(scale_length, granularity)
        if not grouped
        else int(triton.cdiv(scale_length, _BLOCK_N))
    )
    if out is None:
        key_int8 = torch.empty(key_shape, device=key.device, dtype=torch.int8)
        key_scale = torch.empty((batch, heads, scale_count), device=key.device, dtype=torch.float32)
        key_center = torch.empty(
            (batch, heads, chunk_count, head_dim), device=key.device, dtype=torch.float32
        )
    else:
        key_int8, key_scale, key_center = out
    with device_context(key.device):
        _quantize_key_k1024_kernel[(chunk_count, heads, batch)](
            key,
            key_int8,
            key_scale,
            key_center,
            key_length,
            storage_key_length,
            key_scale.shape[2],
            chunk_count,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key_int8.stride(0),
            key_int8.stride(1),
            key_int8.stride(2),
            key_scale.stride(0),
            key_scale.stride(1),
            key_center.stride(0),
            key_center.stride(1),
            grouped_qk=grouped,
            heads=heads,
            head_dim=head_dim,
            num_warps=4,
        )
    return key_int8, key_scale, key_center


def prepare_chunk_centered_value(
    value: torch.Tensor,
    *,
    storage_key_length: int,
    store_log_scale: bool,
    scale_storage_length: int | None = None,
    unspecialized_value_stride: bool = False,
    pack_wmma: bool = False,
    log_scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize V1024-centered values in one pass without sequence-wide reductions."""
    batch, heads, key_length, head_dim = value.shape
    chunk_count = int(triton.cdiv(key_length, CENTER_CHUNK))
    meta_length = key_length if scale_storage_length is None else scale_storage_length
    value_shape = (
        (batch, heads, storage_key_length // _BLOCK_N, head_dim, _BLOCK_N)
        if pack_wmma
        else (batch, heads, head_dim, storage_key_length)
    )
    value_int8 = torch.empty(value_shape, device=value.device, dtype=torch.int8)
    value_scale_multiplier = torch.empty(
        (batch, heads, meta_length), device=value.device, dtype=torch.float32
    )
    value_log_scale = (
        torch.empty((batch, heads, meta_length), device=value.device, dtype=log_scale_dtype)
        if store_log_scale
        else torch.empty((1,), device=value.device, dtype=log_scale_dtype)
    )
    value_mean = torch.empty(
        (batch, heads, chunk_count, head_dim), device=value.device, dtype=torch.float32
    )
    kernel = (
        _quantize_value_k1024_unspecialized_kernel
        if unspecialized_value_stride
        else _quantize_value_k1024_kernel
    )
    stride_od = head_dim * _BLOCK_N if pack_wmma else value_int8.stride(2)
    stride_ok = 1 if pack_wmma else value_int8.stride(3)
    with device_context(value.device):
        kernel[(chunk_count, heads, batch)](
            value,
            value_int8,
            value_scale_multiplier,
            value_log_scale,
            value_mean,
            key_length,
            storage_key_length,
            meta_length,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value_int8.stride(0),
            value_int8.stride(1),
            stride_od,
            stride_ok,
            value_scale_multiplier.stride(0),
            value_scale_multiplier.stride(1),
            value_mean.stride(0),
            value_mean.stride(1),
            store_log_scale=store_log_scale,
            pack_wmma=pack_wmma,
            heads=heads,
            head_dim=head_dim,
            num_warps=4,
        )
    return value_int8, value_scale_multiplier, value_log_scale, value_mean
