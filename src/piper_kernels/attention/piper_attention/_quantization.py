"""Shared K1024/V1024 statistics and quantization for dense Piper Attention.

Numerical boundaries:
- Dense V has an FP32 scale per token (sparse V has one per K64 tile), so a full
  sparse route list does not reproduce dense quantization. Q/K scale granularity is
  backend-specific (Q32/K64 groups on SM12x and RDNA4; per-thread scales on SM89).
- Both K and V are centered independently per 1024-token chunk (``K1024 + V1024``)
  in FP32 for both non-causal and causal attention without sequence-wide reductions:
  per-chunk K bias is computed in FP32 inside each attention kernel from quantized Q
  and ``key_center`` to compensate centered K scores inside the online softmax
  maximum, and per-chunk V means fold into the FP32 numerator once per chunk.
- The recurrence uses signed INT8 QK, unsigned-by-signed INT8 PV, and FP32 online
  softmax. Numerators remain in UINT8 probability-code units until final division by
  255 and the unrounded FP32 softmax denominator. Preserve the all-zero scale guards
  and handle zero QK scales before applying masked infinities.
- GQA/MQA maps each query head to its KV head without repeating K/V storage. Query
  windows retain global row coordinates for causal masking. Window shapes and output
  strides can change FMA contraction and final rounding; cross-backend bitwise
  equality is not guaranteed. Preparation and attention reductions are numerical work,
  not implicit input validation; changes here must stay aligned with
  ``piper_kernels.fusions.convrot_int8_piper``.
"""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.piper.triton import pack_amd_wmma_token
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


@triton.jit
def _quantize_key_k1024_body(
    key_base,
    out_base,
    scale_base,
    center_base,
    in_chunk,
    out_chunk,
    key_length,
    storage_key_length,
    scale_groups,
    stride_kn,
    stride_on,
    grouped_qk: tl.constexpr,
    head_dim: tl.constexpr,
):
    chunk_start = in_chunk * _TL_CENTER_CHUNK
    valid_count = tl.minimum(_TL_CENTER_CHUNK, key_length - chunk_start)
    active_tiles = tl.cdiv(valid_count, _TL_BLOCK_N)
    offsets_tile = tl.arange(0, _TL_BLOCK_N)
    offsets_d = tl.arange(0, head_dim)

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
        center_base + out_chunk * head_dim + offsets_d,
        rotated_mean,
    )

    base_in_tile = in_chunk * _TL_TILES_PER_CHUNK
    base_out_tile = out_chunk * _TL_TILES_PER_CHUNK
    for tile_idx in tl.range(0, active_tiles, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        in_tile = base_in_tile + tile_idx
        out_tile = base_out_tile + tile_idx
        in_rows = in_tile * _TL_BLOCK_N + offsets_tile
        out_rows = out_tile * _TL_BLOCK_N + offsets_tile
        valid = in_rows < key_length
        values = tl.load(
            key_base + in_rows[:, None] * stride_kn + offsets_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        centered = tl.where(valid[:, None], values - key_mean[None, :], 0.0)
        rotated = qk_quantization.rotate_signed_hadamard_heads(centered, head_dim)
        if grouped_qk:
            maximum = tl.max(tl.max(tl.abs(rotated), axis=1), axis=0)
            tile_scale = maximum / _INT8_RANGE + _SCALE_EPSILON
            quantized = qk_quantization.round_to_int8(rotated / tile_scale)
            tl.store(scale_base + out_tile, tile_scale, mask=out_tile < scale_groups)
        else:
            grouped = tl.reshape(rotated, (8, 4, 2, head_dim))
            thread_max = tl.max(tl.max(tl.max(tl.abs(grouped), axis=3), axis=2), axis=0)
            thread_scale = thread_max / _INT8_RANGE + _SCALE_EPSILON
            row_scales = tl.reshape(
                tl.broadcast_to(thread_scale[None, :, None], (8, 4, 2)),
                (_TL_BLOCK_N,),
            )
            quantized = qk_quantization.round_to_int8(rotated / row_scales[:, None])
            tl.store(scale_base + out_rows, row_scales, mask=out_rows < scale_groups)
        tl.store(
            out_base + out_rows[:, None] * stride_on + offsets_d[None, :],
            quantized,
            mask=(out_rows < storage_key_length)[:, None],
        )


@triton.jit(
    do_not_specialize=[
        "key_length",
        "storage_key_length",
        "scale_groups",
        "chunk_count",
        "num_slots",
        "heads",
    ]
)
def _quantize_key_k1024_kernel(
    key_ptr,
    output_ptr,
    scale_ptr,
    center_ptr,
    slot_ids_ptr,
    key_length,
    storage_key_length,
    scale_groups,
    chunk_count,  # noqa: ARG001 - Triton launch signature
    num_slots,
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
    scatter_slots: tl.constexpr,
    heads,  # noqa: ARG001 - Triton launch signature
    head_dim: tl.constexpr,
):
    """Compute one K1024 chunk's rotated mean and center/quantize its keys in one CTA."""
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    if scatter_slots:
        slot = tl.load(slot_ids_ptr + chunk).to(tl.int64)
        if (slot >= 0) & (slot < num_slots):
            zero_chunk = tl.zeros((), dtype=tl.int64)
            _quantize_key_k1024_body(
                key_ptr + chunk * stride_kb + head * stride_kh,
                output_ptr + slot * stride_ob + head * stride_oh,
                scale_ptr + slot * stride_sb + head * stride_sh,
                center_ptr + slot * stride_cb + head * stride_ch,
                zero_chunk,
                zero_chunk,
                key_length,
                storage_key_length,
                scale_groups,
                stride_kn,
                stride_on,
                grouped_qk=grouped_qk,
                head_dim=head_dim,
            )
    else:
        _quantize_key_k1024_body(
            key_ptr + batch * stride_kb + head * stride_kh,
            output_ptr + batch * stride_ob + head * stride_oh,
            scale_ptr + batch * stride_sb + head * stride_sh,
            center_ptr + batch * stride_cb + head * stride_ch,
            chunk,
            chunk,
            key_length,
            storage_key_length,
            scale_groups,
            stride_kn,
            stride_on,
            grouped_qk=grouped_qk,
            head_dim=head_dim,
        )


@triton.jit
def _quantize_value_k1024_body(
    value_base,
    out_base,
    scale_multiplier_base,
    log_scale_base,
    value_mean_base,
    in_chunk,
    out_chunk,
    key_length,
    storage_key_length,
    meta_storage_length,
    stride_vn,
    stride_od,
    stride_ok,
    store_log_scale: tl.constexpr,
    scale_by_p_range: tl.constexpr,
    pack_wmma: tl.constexpr,
    tile_layout: tl.constexpr,
    head_dim: tl.constexpr,
):
    chunk_start = in_chunk * _TL_CENTER_CHUNK
    valid_count = tl.minimum(_TL_CENTER_CHUNK, key_length - chunk_start)
    active_tiles = tl.cdiv(valid_count, _TL_BLOCK_N)
    offsets_tile = tl.arange(0, _TL_BLOCK_N)
    offsets_d = tl.arange(0, head_dim)

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
        value_mean_base + out_chunk * head_dim + offsets_d,
        value_mean,
    )
    scale_floor = tl.max(tl.abs(value_mean), axis=0) * _VALUE_MEAN_SCALE_FLOOR

    base_in_tile = in_chunk * _TL_TILES_PER_CHUNK
    base_out_tile = out_chunk * _TL_TILES_PER_CHUNK
    for tile_idx in tl.range(0, active_tiles, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        in_tile = base_in_tile + tile_idx
        out_tile = base_out_tile + tile_idx
        in_rows = in_tile * _TL_BLOCK_N + offsets_tile
        out_rows = out_tile * _TL_BLOCK_N + offsets_tile
        valid = in_rows < key_length
        values = tl.load(
            value_base + in_rows[:, None] * stride_vn + offsets_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        centered = tl.where(valid[:, None], values - value_mean[None, :], 0.0)
        codes, scales = quantize_value_rows(centered, scale_floor)
        meta_mask = out_rows < meta_storage_length
        stored_scale = scales * _P_UINT8_RANGE if scale_by_p_range else scales
        tl.store(scale_multiplier_base + out_rows, stored_scale, mask=meta_mask)
        if store_log_scale:
            tl.store(log_scale_base + out_rows, tl.log2(scales), mask=meta_mask)
        transposed = tl.trans(codes)
        if pack_wmma or tile_layout:
            token_index = pack_amd_wmma_token(offsets_tile) if pack_wmma else offsets_tile
            tl.store(
                out_base
                + out_tile * head_dim * _TL_BLOCK_N
                + offsets_d[:, None] * _TL_BLOCK_N
                + token_index[None, :],
                transposed,
            )
        else:
            tl.store(
                out_base + offsets_d[:, None] * stride_od + out_rows[None, :] * stride_ok,
                transposed,
                mask=(out_rows < storage_key_length)[None, :],
            )


# Unspecializing ``stride_od`` prevents output-padding alignment from altering the
# value-sum reduction tree (keeping padded vs unpadded V means bitwise equal) and
# avoids 16-byte transpose register bloat that halves value quantization throughput.
@triton.jit(
    do_not_specialize=[
        "key_length",
        "storage_key_length",
        "meta_storage_length",
        "num_slots",
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
    slot_ids_ptr,
    key_length,
    storage_key_length,
    meta_storage_length,
    num_slots,
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
    scale_by_p_range: tl.constexpr,
    pack_wmma: tl.constexpr,
    tile_layout: tl.constexpr,
    scatter_slots: tl.constexpr,
    heads,  # noqa: ARG001 - Triton launch signature
    head_dim: tl.constexpr,
):
    """Center and quantize one V1024 chunk per CTA without partial workspace."""
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    if scatter_slots:
        slot = tl.load(slot_ids_ptr + chunk).to(tl.int64)
        if (slot >= 0) & (slot < num_slots):
            zero_chunk = tl.zeros((), dtype=tl.int64)
            meta_offset = slot * stride_mb + head * stride_mh
            _quantize_value_k1024_body(
                value_ptr + chunk * stride_vb + head * stride_vh,
                output_ptr + slot * stride_ob + head * stride_oh,
                scale_multiplier_ptr + meta_offset,
                log_scale_ptr + meta_offset,
                value_mean_ptr + slot * stride_cb + head * stride_ch,
                zero_chunk,
                zero_chunk,
                key_length,
                storage_key_length,
                meta_storage_length,
                stride_vn,
                stride_od,
                stride_ok,
                store_log_scale=store_log_scale,
                scale_by_p_range=scale_by_p_range,
                pack_wmma=pack_wmma,
                tile_layout=tile_layout,
                head_dim=head_dim,
            )
    else:
        meta_offset = batch * stride_mb + head * stride_mh
        _quantize_value_k1024_body(
            value_ptr + batch * stride_vb + head * stride_vh,
            output_ptr + batch * stride_ob + head * stride_oh,
            scale_multiplier_ptr + meta_offset,
            log_scale_ptr + meta_offset,
            value_mean_ptr + batch * stride_cb + head * stride_ch,
            chunk,
            chunk,
            key_length,
            storage_key_length,
            meta_storage_length,
            stride_vn,
            stride_od,
            stride_ok,
            store_log_scale=store_log_scale,
            scale_by_p_range=scale_by_p_range,
            pack_wmma=pack_wmma,
            tile_layout=tile_layout,
            head_dim=head_dim,
        )


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
            key_int8,
            key_length,
            storage_key_length,
            key_scale.shape[2],
            chunk_count,
            0,
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
            scatter_slots=False,
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
    stride_od = head_dim * _BLOCK_N if pack_wmma else value_int8.stride(2)
    stride_ok = 1 if pack_wmma else value_int8.stride(3)
    with device_context(value.device):
        _quantize_value_k1024_kernel[(chunk_count, heads, batch)](
            value,
            value_int8,
            value_scale_multiplier,
            value_log_scale,
            value_mean,
            value_int8,
            key_length,
            storage_key_length,
            meta_length,
            0,
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
            scale_by_p_range=True,
            pack_wmma=pack_wmma,
            tile_layout=pack_wmma,
            scatter_slots=False,
            heads=heads,
            head_dim=head_dim,
            num_warps=4,
        )
    return value_int8, value_scale_multiplier, value_log_scale, value_mean


def freeze_kv_chunks_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_ids: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    *,
    grouped_qk: bool,
    pack_wmma: bool,
) -> None:
    """Quantize validated 1024-token chunks directly into persistent pool slots."""
    num_new, kv_heads, _, head_dim = key.shape
    num_slots = key_int8.shape[0]
    if num_new == 0 or num_slots == 0:
        return
    # Zero unused size-1 strides so Triton preserves tt.divisibility=16 on pool stores.
    k_stride0 = 0 if num_new == 1 else key.stride(0)
    k_stride1 = 0 if kv_heads == 1 else key.stride(1)
    v_stride0 = 0 if num_new == 1 else value.stride(0)
    v_stride1 = 0 if kv_heads == 1 else value.stride(1)
    key_out_s0 = 0 if num_slots == 1 else key_int8.stride(0)
    key_out_s1 = 0 if kv_heads == 1 else key_int8.stride(1)
    key_scale_s0 = 0 if num_slots == 1 else key_scale.stride(0)
    key_scale_s1 = 0 if kv_heads == 1 else key_scale.stride(1)
    key_center_s0 = 0 if num_slots == 1 else key_center.stride(0)
    key_center_s1 = 0 if kv_heads == 1 else key_center.stride(1)
    val_out_s0 = 0 if num_slots == 1 else value_int8.stride(0)
    val_out_s1 = 0 if kv_heads == 1 else value_int8.stride(1)
    val_scale_s0 = 0 if num_slots == 1 else value_scale.stride(0)
    val_scale_s1 = 0 if kv_heads == 1 else value_scale.stride(1)
    val_mean_s0 = 0 if num_slots == 1 else value_mean.stride(0)
    val_mean_s1 = 0 if kv_heads == 1 else value_mean.stride(1)

    with device_context(key.device):
        _quantize_key_k1024_kernel[(num_new, kv_heads, 1)](
            key,
            key_int8,
            key_scale,
            key_center,
            slot_ids,
            CENTER_CHUNK,
            CENTER_CHUNK,
            key_scale.shape[2],
            1,
            num_slots,
            k_stride0,
            k_stride1,
            key.stride(2),
            key_out_s0,
            key_out_s1,
            key_int8.stride(2),
            key_scale_s0,
            key_scale_s1,
            key_center_s0,
            key_center_s1,
            grouped_qk=grouped_qk,
            scatter_slots=True,
            heads=kv_heads,
            head_dim=head_dim,
            num_warps=4,
        )
        _quantize_value_k1024_kernel[(num_new, kv_heads, 1)](
            value,
            value_int8,
            value_scale,
            value_scale,
            value_mean,
            slot_ids,
            CENTER_CHUNK,
            CENTER_CHUNK,
            CENTER_CHUNK,
            num_slots,
            v_stride0,
            v_stride1,
            value.stride(2),
            val_out_s0,
            val_out_s1,
            head_dim * _BLOCK_N,
            1,
            val_scale_s0,
            val_scale_s1,
            val_mean_s0,
            val_mean_s1,
            store_log_scale=False,
            scale_by_p_range=False,
            pack_wmma=pack_wmma,
            tile_layout=True,
            scatter_slots=True,
            heads=kv_heads,
            head_dim=head_dim,
            num_warps=4,
        )


def _div16_attrs(
    arg_names: list[str], div16_names: frozenset[str]
) -> dict[tuple[int, ...], list[list[object]]]:
    return {
        (idx,): [["tt.divisibility", 16]]
        for idx, name in enumerate(arg_names)
        if name in div16_names
    }


def compile_kv_freeze_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object]:
    """Offline-compile K1024 and V1024 slot-scatter freeze kernels for ``target``."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"dtype must be float16 or bfloat16, got {dtype}")
    if head_dim not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")

    from triton.compiler import ASTSource  # noqa: PLC0415 - compile-only dependency

    from . import _backend  # noqa: PLC0415 - avoid backend/triton import cycle

    _, grouped_qk, pack_wmma = _backend.kv_chunk_layout(
        AcceleratorTarget.from_compiler_target(target)
    )
    triton_dtype = "bf16" if dtype is torch.bfloat16 else "fp16"

    key_constants: dict[str, object] = {
        "grouped_qk": grouped_qk,
        "scatter_slots": True,
        "head_dim": head_dim,
    }
    key_signature: dict[str, str] = {
        name: "i32" for name in _quantize_key_k1024_kernel.arg_names if name not in key_constants
    }
    key_signature.update(
        key_ptr=f"*{triton_dtype}",
        output_ptr="*i8",
        scale_ptr="*fp32",
        center_ptr="*fp32",
        slot_ids_ptr="*i32",
    )
    key_attrs = _div16_attrs(
        _quantize_key_k1024_kernel.arg_names,
        frozenset(
            {
                "output_ptr",
                "scale_ptr",
                "center_ptr",
                "stride_ob",
                "stride_oh",
                "stride_on",
                "stride_sh",
                "stride_ch",
            }
        ),
    )

    value_constants: dict[str, object] = {
        "stride_ok": 1,
        "store_log_scale": False,
        "scale_by_p_range": False,
        "pack_wmma": pack_wmma,
        "tile_layout": True,
        "scatter_slots": True,
        "head_dim": head_dim,
    }
    value_signature: dict[str, str] = {
        name: "i32"
        for name in _quantize_value_k1024_kernel.arg_names
        if name not in value_constants
    }
    value_signature.update(
        value_ptr=f"*{triton_dtype}",
        output_ptr="*i8",
        scale_multiplier_ptr="*fp32",
        log_scale_ptr="*fp32",
        value_mean_ptr="*fp32",
        slot_ids_ptr="*i32",
    )
    value_attrs = _div16_attrs(
        _quantize_value_k1024_kernel.arg_names,
        frozenset(
            {
                "output_ptr",
                "scale_multiplier_ptr",
                "log_scale_ptr",
                "value_mean_ptr",
                "stride_ob",
                "stride_oh",
                "stride_mh",
                "stride_ch",
            }
        ),
    )

    options: dict[str, object] = {"num_warps": 4}
    compiler = triton.compile if compile_fn is None else compile_fn
    return (
        compiler(
            ASTSource(
                _quantize_key_k1024_kernel,
                key_signature,
                constexprs=key_constants,
                attrs=key_attrs,
            ),
            target,
            options,
        ),
        compiler(
            ASTSource(
                _quantize_value_k1024_kernel,
                value_signature,
                constexprs=value_constants,
                attrs=value_attrs,
            ),
            target,
            options,
        ),
    )
