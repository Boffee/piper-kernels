"""GPU Split-KV decode and quantized chunked prefill Triton kernels for Piper KV pools."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.piper.triton import (
    advance_tile_max_safe,
    max_propagate_nan,
    pack_amd_wmma_token,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear._splitk import device_sm_count

from . import _backend, _quantization
from ._quantization import _div16_attrs

_CENTER_CHUNK = 1024
_VALUE_TILE_TOKENS = 64
_KEY_SCALES_GROUPED = _CENTER_CHUNK // _VALUE_TILE_TOKENS
_BLOCK_QH = 16
_MAX_DECODE_SPLITS = 32
_FAST_MAX_DECODE_SPLITS = 64
_TARGET_ACTIVE_CTAS = 64
_SPLIT_NUM_STAGES = 1
_REDUCE_NUM_WARPS = 4
_PREFILL_MIN_SEQ_Q = 64


def _split_num_warps(head_dim: int) -> int:
    return 16 if head_dim == 128 else 8


@dataclass(frozen=True, slots=True)
class _DecodeSchedule:
    """Target-specific kernel and Split-KV schedule for ``piper_attention_decode``."""

    block_n: int
    num_warps: int
    num_stages: int
    loop_num_stages: int
    frozen_qk_mode: str
    tail_qk_precision: str
    pv_precision: str
    max_splits: int
    target_ctas_per_sm: int
    tiles_per_chunk_cap: int
    max_tail_splits: int
    compiler_options: tuple[tuple[str, object], ...] = ()


def _generic_decode_schedule(head_dim: int) -> _DecodeSchedule:
    return _DecodeSchedule(
        block_n=16,
        num_warps=_split_num_warps(head_dim),
        num_stages=_SPLIT_NUM_STAGES,
        loop_num_stages=_SPLIT_NUM_STAGES,
        frozen_qk_mode="ieee",
        tail_qk_precision="ieee",
        pv_precision="ieee",
        max_splits=_MAX_DECODE_SPLITS,
        target_ctas_per_sm=0,
        tiles_per_chunk_cap=8,
        max_tail_splits=2,
    )


def _sm120_decode_schedule() -> _DecodeSchedule:
    return _DecodeSchedule(
        block_n=64,
        num_warps=4,
        num_stages=2,
        loop_num_stages=2,
        frozen_qk_mode="tf32x2",
        tail_qk_precision="tf32x3",
        pv_precision="tf32",
        max_splits=_FAST_MAX_DECODE_SPLITS,
        target_ctas_per_sm=3,
        tiles_per_chunk_cap=_CENTER_CHUNK // 64,
        max_tail_splits=4,
    )


def _sm8x_decode_schedule(head_dim: int) -> _DecodeSchedule:
    return _generic_decode_schedule(head_dim)


def _rdna4_decode_schedule(head_dim: int) -> _DecodeSchedule:
    return _generic_decode_schedule(head_dim)


def _select_decode_schedule(
    target: AcceleratorTarget | None,
    head_dim: int,
) -> _DecodeSchedule:
    """Return the Split-KV kernel and split schedule for ``target`` and ``head_dim``."""
    if target is not None:
        if target.is_cuda_capability(12):
            return _sm120_decode_schedule()
        if target.is_cuda_capability(8):
            return _sm8x_decode_schedule(head_dim)
        if target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201"):
            return _rdna4_decode_schedule(head_dim)
    return _generic_decode_schedule(head_dim)


_TL_CENTER_CHUNK = tl.constexpr(_CENTER_CHUNK)
_TL_VALUE_TILE_TOKENS = tl.constexpr(_VALUE_TILE_TOKENS)
_TL_KEY_SCALES_GROUPED = tl.constexpr(_KEY_SCALES_GROUPED)
_LOG2_E = tl.constexpr(1.4426950408889634)
_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _split_tf32_pair(value):
    """Split finite FP32 values into rounded TF32 high bits plus exact residual."""
    bits = value.to(tl.uint32, bitcast=True)
    hi = (bits & 0xFFFFE000).to(tl.float32, bitcast=True)
    return hi, value - hi


@triton.jit
def _frozen_qk_dot(
    k_base,
    rows,
    q_lo,
    q_hi,
    q_lo_h,
    q_lo_l,
    q_hi_h,
    q_hi_l,
    offsets_d,
    offsets_d64,
    head_dim: tl.constexpr,
    qk_mode: tl.constexpr,
):
    """Compute INT8 key-tile QK^T using either split TF32 pairs or single-mode MMA."""
    if head_dim == 128:
        if qk_mode == "tf32x2":
            k_lo = tl.trans(tl.load(k_base + rows[:, None] * head_dim + offsets_d64[None, :])).to(
                tl.float32
            )
            qk = tl.dot(q_lo_h, k_lo, input_precision="tf32")
            qk = tl.dot(q_lo_l, k_lo, acc=qk, input_precision="tf32")
            k_hi = tl.trans(
                tl.load(k_base + rows[:, None] * head_dim + (offsets_d64[None, :] + 64))
            ).to(tl.float32)
            qk = tl.dot(q_hi_h, k_hi, acc=qk, input_precision="tf32")
            qk = tl.dot(q_hi_l, k_hi, acc=qk, input_precision="tf32")
        else:
            k_lo = tl.load(k_base + rows[:, None] * head_dim + offsets_d64[None, :])
            qk = tl.dot(q_lo, tl.trans(k_lo.to(tl.float32)), input_precision=qk_mode)
            k_hi = tl.load(k_base + rows[:, None] * head_dim + (offsets_d64[None, :] + 64))
            qk = tl.dot(q_hi, tl.trans(k_hi.to(tl.float32)), acc=qk, input_precision=qk_mode)
    else:
        k_q = tl.load(k_base + rows[:, None] * head_dim + offsets_d[None, :])
        if qk_mode == "tf32x2":
            kt = tl.trans(k_q.to(tl.float32))
            qk = tl.dot(q_lo_h, kt, input_precision="tf32")
            qk = tl.dot(q_lo_l, kt, acc=qk, input_precision="tf32")
        else:
            qk = tl.dot(q_lo, tl.trans(k_q.to(tl.float32)), input_precision=qk_mode)
    return qk


@triton.jit
def _compute_chunk_bias(
    key_center_ptr,
    slot64,
    kv_head64,
    stride_kc_s,
    q_lo,
    q_hi,
    offsets_d,
    offsets_d64,
    scaled_log2,
    head_dim: tl.constexpr,
):
    kc_base = key_center_ptr + slot64 * stride_kc_s + kv_head64 * head_dim
    if head_dim == 128:
        kc_lo = tl.load(kc_base + offsets_d64)
        kc_hi = tl.load(kc_base + (offsets_d64 + 64))
        bias = (
            tl.sum(q_lo * kc_lo[None, :], axis=1) + tl.sum(q_hi * kc_hi[None, :], axis=1)
        ) * scaled_log2
    else:
        k_center = tl.load(kc_base + offsets_d)
        bias = tl.sum(q_lo * k_center[None, :], axis=1) * scaled_log2
    return bias


@triton.jit(
    do_not_specialize=[
        "kv_heads",
        "gqa_group",
        "q_tiles",
        "seq_q",
        "num_slots",
        "max_chunks",
        "int8_splits",
    ]
)
def _piper_decode_frozen_split_kernel(  # noqa: PLR0915
    query_ptr,
    key_int8_ptr,
    key_scale_ptr,
    key_center_ptr,
    value_int8_ptr,
    value_scale_ptr,
    value_mean_ptr,
    chunk_to_slot_ptr,
    num_chunks_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    partial_bias_ptr,
    scale,
    kv_heads,
    gqa_group,
    q_tiles,
    seq_q,
    num_slots,
    max_chunks,
    int8_splits,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_ki_s,
    stride_ks_s,
    stride_kc_s,
    stride_vi_s,
    stride_vs_s,
    stride_vm_s,
    stride_cs_b,
    stride_cs_c,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pop,
    stride_plb,
    stride_plh,
    stride_pls,
    has_chunk_map: tl.constexpr,
    grouped_qk: tl.constexpr,
    pack_wmma: tl.constexpr,
    head_dim: tl.constexpr,
    block_qh: tl.constexpr,
    block_n: tl.constexpr,
    qk_mode: tl.constexpr,
    pv_precision: tl.constexpr,
    loop_num_stages: tl.constexpr,
):
    """Evaluate one Split-KV slice of frozen INT8 chunks using two-term winner framing."""
    pid_q = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_bh = pid_q // q_tiles
    q_tile = pid_q % q_tiles

    b = pid_bh // kv_heads
    kv_head = pid_bh % kv_heads
    b64 = b.to(tl.int64)
    kv_head64 = kv_head.to(tl.int64)

    offsets_d = tl.arange(0, head_dim)
    m_i = tl.full((block_qh,), -float("inf"), dtype=tl.float32)
    bias_i = tl.zeros((block_qh,), dtype=tl.float32)
    l_i = tl.zeros((block_qh,), dtype=tl.float32)
    acc = tl.zeros((block_qh, head_dim), dtype=tl.float32)

    subtiles_per_chunk: tl.constexpr = _TL_CENTER_CHUNK // block_n
    subtiles_per_k64: tl.constexpr = _TL_VALUE_TILE_TOKENS // block_n

    nc = tl.load(num_chunks_ptr + b64)
    active_chunks = tl.maximum(0, tl.minimum(nc, max_chunks)).to(tl.int64)
    total_subtiles = active_chunks * subtiles_per_chunk
    split64 = split_id.to(tl.int64)
    subtiles_per_split = tl.cdiv(total_subtiles, int8_splits.to(tl.int64))
    subtile_start = split64 * subtiles_per_split
    subtile_end = tl.minimum(total_subtiles, subtile_start + subtiles_per_split)

    if subtile_start < subtile_end:
        first_chunk = (subtile_start // subtiles_per_chunk).to(tl.int32)
        last_chunk = ((subtile_end - 1) // subtiles_per_chunk).to(tl.int32)
        first_inner = (subtile_start % subtiles_per_chunk).to(tl.int32)
        last_inner = ((subtile_end - 1) % subtiles_per_chunk + 1).to(tl.int32)

        qh_offs0 = q_tile * block_qh + tl.arange(0, block_qh)
        qh_valid0 = qh_offs0 < gqa_group * seq_q
        q_idx0 = qh_offs0 % seq_q
        q_head0 = kv_head * gqa_group + qh_offs0 // seq_q
        q_ptrs = (
            query_ptr
            + b64 * stride_qb
            + q_head0.to(tl.int64)[:, None] * stride_qh
            + q_idx0.to(tl.int64)[:, None] * stride_qs
            + offsets_d[None, :]
        )
        q_f32 = tl.load(q_ptrs, mask=qh_valid0[:, None], other=0.0).to(tl.float32)
        q_rot = qk_quantization.rotate_signed_hadamard_heads(q_f32, head_dim)
        offsets_d64 = tl.arange(0, 64)
        if head_dim == 128:
            q_lo, q_hi = tl.split(tl.permute(tl.reshape(q_rot, (block_qh, 2, 64)), (0, 2, 1)))
        else:
            q_lo = q_rot
            q_hi = q_rot
        if qk_mode == "tf32x2":
            q_lo_h, q_lo_l = _split_tf32_pair(q_lo)
            q_hi_h, q_hi_l = _split_tf32_pair(q_hi)
        else:
            q_lo_h = q_lo
            q_lo_l = q_lo
            q_hi_h = q_hi
            q_hi_l = q_hi

        offsets_n = tl.arange(0, block_n)
        scaled_log2 = scale * _LOG2_E
        stride_kv_head = tl.constexpr(_TL_CENTER_CHUNK * head_dim)
        stride_ks_head = tl.constexpr(_TL_KEY_SCALES_GROUPED if grouped_qk else _TL_CENTER_CHUNK)
        stride_vi_t = tl.constexpr(head_dim * _TL_VALUE_TILE_TOKENS)

        for chunk_idx in tl.range(first_chunk, last_chunk + 1, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            if has_chunk_map:
                slot = tl.load(
                    chunk_to_slot_ptr + b64 * stride_cs_b + chunk_idx.to(tl.int64) * stride_cs_c
                )
            else:
                slot = chunk_idx

            if slot >= 0 and slot < num_slots:
                slot64 = slot.to(tl.int64)
                chunk_bias = _compute_chunk_bias(
                    key_center_ptr,
                    slot64,
                    kv_head64,
                    stride_kc_s,
                    q_lo,
                    q_hi,
                    offsets_d,
                    offsets_d64,
                    scaled_log2,
                    head_dim=head_dim,
                )
                chunk_l = tl.zeros((block_qh,), dtype=tl.float32)
                chunk_has_block = tl.zeros((block_qh,), dtype=tl.int1)

                k_base = key_int8_ptr + slot64 * stride_ki_s + kv_head64 * stride_kv_head
                ks_base = key_scale_ptr + slot64 * stride_ks_s + kv_head64 * stride_ks_head
                v_base = value_int8_ptr + slot64 * stride_vi_s + kv_head64 * stride_kv_head
                vs_base = value_scale_ptr + slot64 * stride_vs_s + kv_head64 * _TL_CENTER_CHUNK

                inner_start = tl.where(chunk_idx == first_chunk, first_inner, 0)
                inner_end = tl.where(chunk_idx == last_chunk, last_inner, subtiles_per_chunk)
                for local_subtile in tl.range(
                    inner_start, inner_end, num_stages=loop_num_stages, disable_licm=True
                ):  # pyright: ignore[reportGeneralTypeIssues]
                    t64 = local_subtile // subtiles_per_k64
                    sub4 = local_subtile % subtiles_per_k64
                    rows = local_subtile * block_n + offsets_n
                    qk = _frozen_qk_dot(
                        k_base,
                        rows,
                        q_lo,
                        q_hi,
                        q_lo_h,
                        q_lo_l,
                        q_hi_h,
                        q_hi_l,
                        offsets_d,
                        offsets_d64,
                        head_dim=head_dim,
                        qk_mode=qk_mode,
                    )
                    if grouped_qk:
                        k_scale = tl.load(ks_base + t64)
                        scores = qk * (k_scale * scaled_log2)
                    else:
                        k_scale = tl.load(ks_base + rows)
                        scores = qk * (k_scale[None, :] * scaled_log2)

                    m_block = tl.reduce(scores, 1, max_propagate_nan)
                    m_i, bias_i, old_weight, cur_weight, has_block, safe_m_block = (
                        advance_tile_max_safe(m_i, bias_i, m_block, chunk_bias)
                    )
                    chunk_has_block = chunk_has_block | has_block
                    p = tl.where(
                        has_block[:, None],
                        tl.math.exp2(scores - safe_m_block[:, None]) * cur_weight[:, None],
                        0.0,
                    )
                    row_sum = tl.sum(p, axis=1)
                    l_i = l_i * old_weight + row_sum
                    chunk_l = chunk_l * old_weight + row_sum
                    acc = acc * old_weight[:, None]

                    v_scale = tl.load(vs_base + rows)
                    p_scaled = tl.where(has_block[:, None], p * v_scale[None, :], 0.0)
                    offs_tok = sub4 * block_n + offsets_n
                    v_cols = pack_amd_wmma_token(offs_tok) if pack_wmma else offs_tok
                    v_q = tl.load(
                        v_base
                        + t64 * stride_vi_t
                        + offsets_d[:, None] * _TL_VALUE_TILE_TOKENS
                        + v_cols[None, :]
                    )
                    acc = tl.dot(
                        p_scaled,
                        tl.trans(v_q.to(tl.float32)),
                        acc=acc,
                        input_precision=pv_precision,
                    )

                v_mean = tl.load(
                    value_mean_ptr + slot64 * stride_vm_s + kv_head64 * head_dim + offsets_d
                )
                acc += tl.where(chunk_has_block[:, None], chunk_l[:, None] * v_mean[None, :], 0.0)

    has_any = m_i != -float("inf")
    safe_l = tl.where(has_any, l_i, 1.0)
    out_split = tl.where(has_any[:, None], acc / safe_l[:, None], 0.0)
    lse_split = tl.where(has_any, m_i + tl.math.log2(safe_l), -float("inf"))
    bias_split = tl.where(has_any, bias_i, 0.0)

    qh_offs = q_tile * block_qh + tl.arange(0, block_qh)
    qh_valid = qh_offs < gqa_group * seq_q
    q_idx = qh_offs % seq_q
    q_head = kv_head * gqa_group + qh_offs // seq_q
    po_ptrs = (
        partial_out_ptr
        + b64 * stride_pob
        + q_head.to(tl.int64)[:, None] * stride_poh
        + q_idx.to(tl.int64)[:, None] * stride_pos
        + split64 * stride_pop
        + offsets_d[None, :]
    )
    tl.store(po_ptrs, out_split, mask=qh_valid[:, None])

    pl_offsets = (
        b64 * stride_plb
        + q_head.to(tl.int64) * stride_plh
        + q_idx.to(tl.int64) * stride_pls
        + split64
    )
    tl.store(partial_lse_ptr + pl_offsets, lse_split, mask=qh_valid)
    tl.store(partial_bias_ptr + pl_offsets, bias_split, mask=qh_valid)


@triton.jit(
    do_not_specialize=[
        "kv_heads",
        "gqa_group",
        "q_tiles",
        "seq_q",
        "tail_cap",
        "int8_splits",
        "tail_splits",
    ]
)
def _piper_decode_tail_split_kernel(  # noqa: PLR0915
    query_ptr,
    tail_key_ptr,
    tail_value_ptr,
    tail_len_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    partial_bias_ptr,
    scale,
    kv_heads,
    gqa_group,
    q_tiles,
    seq_q,
    tail_cap,
    int8_splits,
    tail_splits,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_tk_b,
    stride_tk_h,
    stride_tk_n,
    stride_tv_b,
    stride_tv_h,
    stride_tv_n,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pop,
    stride_plb,
    stride_plh,
    stride_pls,
    is_causal: tl.constexpr,
    head_dim: tl.constexpr,
    block_qh: tl.constexpr,
    block_n: tl.constexpr,
    qk_precision: tl.constexpr,
    pv_precision: tl.constexpr,
):
    """Evaluate one Split-KV slice of the live BF16/FP16 tail for a GQA head group."""
    pid_q = tl.program_id(0)
    tail_split_id = tl.program_id(1)
    pid_bh = pid_q // q_tiles
    q_tile = pid_q % q_tiles

    b = pid_bh // kv_heads
    kv_head = pid_bh % kv_heads
    b64 = b.to(tl.int64)
    kv_head64 = kv_head.to(tl.int64)

    offsets_d = tl.arange(0, head_dim)
    m_i = tl.full((block_qh,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((block_qh,), dtype=tl.float32)
    acc = tl.zeros((block_qh, head_dim), dtype=tl.float32)

    tl_raw = tl.load(tail_len_ptr + b64)
    active_tail = tl.maximum(0, tl.minimum(tl_raw, tail_cap)).to(tl.int64)
    total_tiles = tl.cdiv(active_tail, block_n)
    tiles_per_split = tl.cdiv(total_tiles, tail_splits.to(tl.int64))
    tile_start = tail_split_id.to(tl.int64) * tiles_per_split
    tile_end = tl.minimum(total_tiles, tile_start + tiles_per_split)

    if tile_start < tile_end:
        qh_offs0 = q_tile * block_qh + tl.arange(0, block_qh)
        qh_valid0 = qh_offs0 < gqa_group * seq_q
        q_idx0 = qh_offs0 % seq_q
        q_head0 = kv_head * gqa_group + qh_offs0 // seq_q
        q_ptrs = (
            query_ptr
            + b64 * stride_qb
            + q_head0.to(tl.int64)[:, None] * stride_qh
            + q_idx0.to(tl.int64)[:, None] * stride_qs
            + offsets_d[None, :]
        )
        q_f32 = tl.load(q_ptrs, mask=qh_valid0[:, None], other=0.0).to(tl.float32)
        offsets_d64 = tl.arange(0, 64)
        if head_dim == 128:
            q_lo, q_hi = tl.split(tl.permute(tl.reshape(q_f32, (block_qh, 2, 64)), (0, 2, 1)))
        else:
            q_lo = q_f32
            q_hi = q_f32

        offsets_n = tl.arange(0, block_n)
        tk_base = tail_key_ptr + b64 * stride_tk_b + kv_head64 * stride_tk_h
        tv_base = tail_value_ptr + b64 * stride_tv_b + kv_head64 * stride_tv_h
        scaled_log2 = scale * _LOG2_E
        active_tail32 = active_tail.to(tl.int32)
        cutoff = (active_tail32 - seq_q) + q_idx0

        for tile_idx in tl.range(tile_start.to(tl.int32), tile_end.to(tl.int32), disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            rows32 = tile_idx * block_n + offsets_n
            valid_n = rows32 < active_tail32
            rows = rows32.to(tl.int64)
            if head_dim == 128:
                k_lo = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + offsets_d64[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                scores = tl.dot(q_lo, tl.trans(k_lo.to(tl.float32)), input_precision=qk_precision)
                k_hi = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + (offsets_d64[None, :] + 64),
                    mask=valid_n[:, None],
                    other=0.0,
                )
                scores = tl.dot(
                    q_hi, tl.trans(k_hi.to(tl.float32)), acc=scores, input_precision=qk_precision
                )
            else:
                k_tile = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + offsets_d[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                scores = tl.dot(
                    q_f32, tl.trans(k_tile.to(tl.float32)), input_precision=qk_precision
                )
            scores *= scaled_log2
            if is_causal:
                valid_mask = valid_n[None, :] & (rows32[None, :] <= cutoff[:, None])
            else:
                valid_mask = valid_n[None, :]

            masked_scores = tl.where(valid_mask, scores, -float("inf"))
            m_block = tl.reduce(masked_scores, 1, max_propagate_nan)
            has_block = m_block != -float("inf")  # pyright: ignore[reportOperatorIssue]
            has_prev = m_i != -float("inf")
            m_new = tl.maximum(m_i, m_block, propagate_nan=tl.PropagateNan.ALL)
            has_any_new = m_new != -float("inf")
            m_safe = tl.where(has_any_new, m_new, 0.0)
            alpha = tl.where(
                has_prev,
                tl.math.exp2(tl.where(has_prev, m_i - m_safe, 0.0)),
                0.0,
            )
            p = tl.where(
                valid_mask & has_block[:, None],  # pyright: ignore[reportIndexIssue]
                tl.math.exp2(tl.where(valid_mask, masked_scores - m_safe[:, None], 0.0)),
                0.0,
            )
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_tile = tl.load(
                tv_base + rows[:, None] * stride_tv_n + offsets_d[None, :],
                mask=valid_n[:, None],
                other=0.0,
            )
            acc = tl.dot(p, v_tile.to(tl.float32), acc=acc, input_precision=pv_precision)
            m_i = m_new

    has_any = m_i != -float("inf")
    safe_l = tl.where(has_any, l_i, 1.0)
    out_split = tl.where(has_any[:, None], acc / safe_l[:, None], 0.0)
    lse_split = tl.where(has_any, m_i + tl.math.log2(safe_l), -float("inf"))

    split64 = (int8_splits + tail_split_id).to(tl.int64)
    qh_offs = q_tile * block_qh + tl.arange(0, block_qh)
    qh_valid = qh_offs < gqa_group * seq_q
    q_idx = qh_offs % seq_q
    q_head = kv_head * gqa_group + qh_offs // seq_q
    po_ptrs = (
        partial_out_ptr
        + b64 * stride_pob
        + q_head.to(tl.int64)[:, None] * stride_poh
        + q_idx.to(tl.int64)[:, None] * stride_pos
        + split64 * stride_pop
        + offsets_d[None, :]
    )
    tl.store(po_ptrs, out_split, mask=qh_valid[:, None])

    pl_offsets = (
        b64 * stride_plb
        + q_head.to(tl.int64) * stride_plh
        + q_idx.to(tl.int64) * stride_pls
        + split64
    )
    tl.store(partial_lse_ptr + pl_offsets, lse_split, mask=qh_valid)
    tl.store(partial_bias_ptr + pl_offsets, 0.0, mask=qh_valid)


@triton.jit(do_not_specialize=["q_heads", "seq_q", "num_splits", "pl_split_stride"])
def _piper_decode_splitkv_reduce_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    partial_bias_ptr,
    output_ptr,
    lse_ptr,
    q_heads,
    seq_q,
    num_splits,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pop,
    stride_plb,
    stride_plh,
    stride_pls,
    pl_split_stride,
    stride_ob,
    stride_oh,
    stride_os,
    stride_lb,
    stride_lh,
    head_dim: tl.constexpr,
):
    """Merge partial Split-KV outputs and two-term base-2 LSE pairs into natural-log LSE."""
    pid = tl.program_id(0)
    q_idx = pid % seq_q
    pid_bh = pid // seq_q
    b = pid_bh // q_heads
    q_head = pid_bh % q_heads
    b64 = b.to(tl.int64)
    qh64 = q_head.to(tl.int64)
    qi64 = q_idx.to(tl.int64)

    offsets_d = tl.arange(0, head_dim).to(tl.int64)
    pl_offset = b64 * stride_plb + qh64 * stride_plh + qi64 * stride_pls
    po_base = partial_out_ptr + b64 * stride_pob + qh64 * stride_poh + qi64 * stride_pos
    split_lse_stride = pl_split_stride.to(tl.int64)

    m_win = -float("inf")
    bias_win = 0.0
    denom = 0.0
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    for split_idx in tl.range(0, num_splits, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        s64 = split_idx.to(tl.int64)
        lse_off = pl_offset + s64 * split_lse_stride
        m_s = tl.load(partial_lse_ptr + lse_off)
        b_s = tl.load(partial_bias_ptr + lse_off)
        m_win, bias_win, old_w, cur_w, has_s, safe_m_s = advance_tile_max_safe(
            m_win, bias_win, m_s, b_s
        )

        out_s = tl.load(po_base + s64 * stride_pop + offsets_d)
        has_nan_s = safe_m_s != safe_m_s  # noqa: PLR0124
        safe_out_s = tl.where(has_nan_s, float("nan"), tl.where(has_s, out_s, 0.0))
        old_num = denom * old_w
        denom = old_num + cur_w
        has_denom = denom != 0.0
        safe_dn = tl.where(has_denom, denom, 1.0)
        old_alpha = tl.where(has_denom, old_num / safe_dn, 0.0)
        cur_alpha = tl.where(has_denom, cur_w / safe_dn, 0.0)
        acc = acc * old_alpha + cur_alpha * safe_out_s

    has_any = m_win != -float("inf")
    safe_denom = tl.where(has_any, denom, 1.0)
    final_out = tl.where(has_any, acc, 0.0)
    final_lse = tl.where(
        has_any, (m_win * _LN2 + bias_win * _LN2) + tl.math.log2(safe_denom) * _LN2, -float("inf")
    )

    out_ptrs = output_ptr + b64 * stride_ob + qh64 * stride_oh + qi64 * stride_os + offsets_d
    tl.store(out_ptrs, final_out.to(output_ptr.dtype.element_ty))
    tl.store(lse_ptr + b64 * stride_lb + qh64 * stride_lh + qi64, final_lse)


@triton.jit
def _merge_two_attention_waves_kernel(
    out_a_ptr,
    lse_a_ptr,
    out_b_ptr,
    lse_b_ptr,
    head_dim: tl.constexpr,
):
    """Merge contiguous attention wave ``B`` into wave ``A`` in-place via normalized LSE weights."""
    row = tl.program_id(0).to(tl.int64)
    la = tl.load(lse_a_ptr + row)
    lb = tl.load(lse_b_ptr + row)
    nan_a = la != la  # noqa: PLR0124
    nan_b = lb != lb  # noqa: PLR0124
    pos_inf_a = la == float("inf")
    pos_inf_b = lb == float("inf")
    has_nan = nan_a | nan_b | (pos_inf_a & pos_inf_b)
    valid_a = (la != -float("inf")) & (~nan_a)
    valid_b = (lb != -float("inf")) & (~nan_b)
    a_wins = valid_a & ((~valid_b) | (pos_inf_a & (~pos_inf_b))) & (~has_nan)
    if not a_wins:
        offsets_d = tl.arange(0, head_dim)
        out_a_ptrs = out_a_ptr + row * head_dim + offsets_d
        out_b_ptrs = out_b_ptr + row * head_dim + offsets_d
        b_wins = valid_b & ((~valid_a) | (pos_inf_b & (~pos_inf_a))) & (~has_nan)
        both_valid = valid_a & valid_b & (~pos_inf_a) & (~pos_inf_b)
        if has_nan:
            nan_vec = tl.full((head_dim,), float("nan"), dtype=tl.float32)
            tl.store(out_a_ptrs, nan_vec.to(out_a_ptr.dtype.element_ty))
            tl.store(lse_a_ptr + row, float("nan"))
        elif b_wins:
            ob = tl.load(out_b_ptrs)
            tl.store(out_a_ptrs, ob.to(out_a_ptr.dtype.element_ty))
            tl.store(lse_a_ptr + row, lb)
        elif both_valid:
            max_l = tl.maximum(la, lb)
            wa = tl.math.exp(la - max_l)
            wb = tl.math.exp(lb - max_l)
            denom = wa + wb
            na = wa / denom
            nb = wb / denom
            oa = tl.load(out_a_ptrs).to(tl.float32)
            ob = tl.load(out_b_ptrs).to(tl.float32)
            merged_out = oa * na + ob * nb
            merged_lse = max_l + tl.math.log(denom)
            tl.store(out_a_ptrs, merged_out.to(out_a_ptr.dtype.element_ty))
            tl.store(lse_a_ptr + row, merged_lse)
        else:
            zero_vec = tl.zeros((head_dim,), dtype=tl.float32)
            tl.store(out_a_ptrs, zero_vec.to(out_a_ptr.dtype.element_ty))
            tl.store(lse_a_ptr + row, -float("inf"))


def _merge_two_attention_waves(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> None:
    total_rows = out_a.shape[0] * out_a.shape[1] * out_a.shape[2]
    if total_rows == 0:
        return
    head_dim = out_a.shape[3]
    with device_context(out_a.device):
        _merge_two_attention_waves_kernel[(total_rows,)](
            out_a,
            lse_a,
            out_b,
            lse_b,
            head_dim=head_dim,
            num_warps=_REDUCE_NUM_WARPS,
        )


def _validate_prefill_backend(
    prefill_backend: Literal["amd", "nvidia"],
    *,
    grouped_qk: bool,
    pack_wmma: bool,
) -> None:
    if prefill_backend not in ("amd", "nvidia"):
        raise ValueError(f"prefill_backend must be 'amd' or 'nvidia', got {prefill_backend!r}")
    if pack_wmma != (prefill_backend == "amd"):
        raise ValueError(
            f"pack_wmma ({pack_wmma}) must match prefill_backend ({prefill_backend!r})"
        )
    if prefill_backend == "amd" and not grouped_qk:
        raise ValueError("AMD chunked prefill requires grouped_qk=True")


def _pool_slot_strides(
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    batch: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """Return signed pool views, slot-map flag/cap, and zeroed slot/batch strides."""
    num_slots = key_int8.shape[0]
    has_chunk_map = chunk_to_slot is not None
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    cs_tensor = chunk_to_slot if chunk_to_slot is not None else num_chunks
    key_i8_s = key_int8.view(torch.int8)
    val_i8_s = value_int8.view(torch.int8)
    return (
        key_i8_s,
        val_i8_s,
        cs_tensor,
        has_chunk_map,
        max_chunks,
        0 if num_slots <= 1 else key_i8_s.stride(0),
        0 if num_slots <= 1 else key_scale.stride(0),
        0 if num_slots <= 1 else key_center.stride(0),
        0 if num_slots <= 1 else val_i8_s.stride(0),
        0 if num_slots <= 1 else value_scale.stride(0),
        0 if num_slots <= 1 else value_mean.stride(0),
        0 if (not has_chunk_map or batch <= 1) else cs_tensor.stride(0),
        1 if not has_chunk_map else cs_tensor.stride(1),
    )


def _launch_frozen_pool_prefill(
    query_int8: torch.Tensor,
    query_scale: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    seq_q: int,
    grouped_qk: bool,
    prefill_backend: Literal["amd", "nvidia"],
    active_chunks_hint: int,
    target: AcceleratorTarget | None,
) -> None:
    """Assemble pool views and strides once and dispatch the backend frozen-pool kernel."""
    batch, q_heads, _, head_dim = output.shape
    num_slots, kv_heads, _, _ = key_int8.shape
    query_storage = query_int8.shape[2]
    (
        key_i8_s,
        val_i8_s,
        cs_tensor,
        has_chunk_map,
        max_chunks,
        stride_ki_s,
        stride_ks_s,
        stride_kc_s,
        stride_vi_s,
        stride_vs_s,
        stride_vm_s,
        stride_cs_b,
        stride_cs_c,
    ) = _pool_slot_strides(
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        chunk_to_slot,
        num_chunks,
        batch=batch,
    )
    gqa_group = q_heads // kv_heads

    frozen_splits = 1
    out_target = output
    lse_target = lse
    partial_out: torch.Tensor | None = None
    partial_lse: torch.Tensor | None = None
    partial_bias: torch.Tensor | None = None
    bias_target: torch.Tensor | None = None
    if prefill_backend == "nvidia":
        sm_count = device_sm_count(query_int8.device)
        base_blocks = max(1, batch * q_heads * triton.cdiv(seq_q, 64))
        hint_chunks = max_chunks if active_chunks_hint <= 0 else min(max_chunks, active_chunks_hint)
        if base_blocks < sm_count and hint_chunks >= 4:
            frozen_splits = min(8, hint_chunks // 2, max(1, (sm_count * 2) // base_blocks))
    if frozen_splits > 1:
        partial_out = torch.empty(
            (batch, frozen_splits, q_heads, seq_q, head_dim),
            device=output.device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            (batch, frozen_splits, q_heads, seq_q),
            device=output.device,
            dtype=torch.float32,
        )
        partial_bias = torch.empty_like(partial_lse)
        out_target = partial_out.view(batch * frozen_splits, q_heads, seq_q, head_dim)
        lse_target = partial_lse.view(batch * frozen_splits, q_heads, seq_q)
        bias_target = partial_bias.view(batch * frozen_splits, q_heads, seq_q)

    common_args: tuple[object, ...] = (
        query_int8,
        query_scale,
        key_i8_s,
        key_scale,
        key_center,
        val_i8_s,
        value_scale,
        value_mean,
        cs_tensor,
        num_chunks,
        out_target,
        lse_target,
        seq_q,
        query_storage,
        q_heads,
        num_slots,
        max_chunks,
        stride_ki_s,
        stride_ks_s,
        stride_kc_s,
        stride_vi_s,
        stride_vs_s,
        stride_vm_s,
        stride_cs_b,
        stride_cs_c,
        out_target.stride(0),
        out_target.stride(1),
        out_target.stride(2),
        lse_target.stride(0),
        lse_target.stride(1),
    )
    if prefill_backend == "amd":
        from ._amd import gluon as amd_gluon  # noqa: PLC0415

        amd_gluon.launch_frozen_pool_kernel(
            common_args,
            device=query_int8.device,
            batch=batch,
            q_heads=q_heads,
            seq_q=seq_q,
            query_storage=query_storage,
            gqa_group=gqa_group,
            head_dim=head_dim,
            has_chunk_map=has_chunk_map,
        )
    else:
        from ._nvidia import triton as nvidia_triton  # noqa: PLC0415

        nvidia_triton.launch_prefill_kernel(
            common_args,
            device=query_int8.device,
            batch=batch,
            q_heads=q_heads,
            seq_q=seq_q,
            gqa_group=gqa_group,
            head_dim=head_dim,
            grouped_qk=grouped_qk,
            has_chunk_map=has_chunk_map,
            is_tail=False,
            is_causal=False,
            split_chunks=frozen_splits,
            partial_bias=bias_target,
            target=target,
        )
        if partial_out is not None and partial_lse is not None and partial_bias is not None:
            po_view = partial_out.permute(0, 2, 3, 1, 4)
            pl_view = partial_lse.permute(0, 2, 3, 1)
            pb_view = partial_bias.permute(0, 2, 3, 1)
            with device_context(output.device):
                _piper_decode_splitkv_reduce_kernel[(batch * q_heads * seq_q,)](
                    po_view,
                    pl_view,
                    pb_view,
                    output,
                    lse,
                    q_heads,
                    seq_q,
                    frozen_splits,
                    po_view.stride(0),
                    po_view.stride(1),
                    po_view.stride(2),
                    po_view.stride(3),
                    pl_view.stride(0),
                    pl_view.stride(1),
                    pl_view.stride(2),
                    pl_view.stride(3),
                    output.stride(0),
                    output.stride(1),
                    output.stride(2),
                    lse.stride(0),
                    lse.stride(1),
                    head_dim=head_dim,
                    num_warps=_REDUCE_NUM_WARPS,
                )


def _run_chunked_prefill(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float,
    is_causal: bool,
    grouped_qk: bool,
    pack_wmma: bool,
    active_chunks_hint: int,
    prefill_backend: Literal["amd", "nvidia"],
    target: AcceleratorTarget | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run quantized chunked prefill over finalized pool slots plus optional live tail."""
    batch, q_heads, seq_q, head_dim = query.shape
    num_slots = key_int8.shape[0]
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    tail_cap = 0 if tail_key is None else tail_key.shape[2]
    has_chunks = max_chunks > 0 and num_slots > 0
    has_tail = tail_cap > 0

    output = torch.empty((batch, q_heads, seq_q, head_dim), device=query.device, dtype=query.dtype)
    lse = torch.empty((batch, q_heads, seq_q), device=query.device, dtype=torch.float32)
    if batch == 0:
        return output, lse
    if not has_chunks and not has_tail:
        output.zero_()
        lse.fill_(-float("inf"))
        return output, lse

    storage_q = triton.cdiv(seq_q, 64) * 64
    if prefill_backend == "amd":
        from piper_kernels.attention.kernels.piper._amd import (  # noqa: PLC0415
            quantization as amd_quantization,
        )

        query_int8, query_scale = amd_quantization.prepare_query(
            query, scale, storage_sequence_length=storage_q
        )
    else:
        with device_context(query.device):
            query_int8, query_scale = qk_quantization.prepare_query(
                query, scale, grouped=grouped_qk, storage_query_length=storage_q
            )

    if has_chunks:
        _launch_frozen_pool_prefill(
            query_int8,
            query_scale,
            key_int8,
            key_scale,
            key_center,
            value_int8,
            value_scale,
            value_mean,
            chunk_to_slot,
            num_chunks,
            output,
            lse,
            seq_q=seq_q,
            grouped_qk=grouped_qk,
            prefill_backend=prefill_backend,
            active_chunks_hint=active_chunks_hint,
            target=target,
        )

    if has_tail:
        assert tail_key is not None
        assert tail_value is not None
        assert tail_len is not None
        tail_storage = triton.cdiv(tail_cap, 64) * 64
        tail_k_int8, tail_k_scale, tail_k_center = _quantization.prepare_chunk_centered_key(
            tail_key,
            grouped=grouped_qk,
            storage_key_length=tail_storage,
            scale_storage_length=tail_storage,
            key_lengths=tail_len,
        )
        tail_v_int8, tail_v_scale_or_mult, tail_v_log_scale, tail_v_mean = (
            _quantization.prepare_chunk_centered_value(
                tail_value,
                storage_key_length=tail_storage,
                store_log_scale=pack_wmma,
                scale_storage_length=tail_storage,
                pack_wmma=pack_wmma,
                log_scale_dtype=torch.float32,
                tile_layout=True,
                scale_by_p_range=pack_wmma,
                key_lengths=tail_len,
            )
        )
        out_tail = torch.empty_like(output) if has_chunks else output
        lse_tail = torch.empty_like(lse) if has_chunks else lse
        if prefill_backend == "amd":
            from ._amd import gluon as amd_gluon  # noqa: PLC0415

            amd_gluon.launch_tail_prefill(
                query_int8,
                query_scale,
                tail_k_int8,
                tail_k_scale,
                tail_k_center,
                tail_v_int8,
                tail_v_scale_or_mult,
                tail_v_log_scale,
                tail_v_mean,
                tail_len,
                out_tail,
                lse_tail,
                seq_q=seq_q,
                tail_cap=tail_cap,
                is_causal=is_causal,
            )
        else:
            from ._nvidia import triton as nvidia_triton  # noqa: PLC0415

            nvidia_triton.launch_tail_prefill(
                query_int8,
                query_scale,
                tail_k_int8,
                tail_k_scale,
                tail_k_center,
                tail_v_int8,
                tail_v_scale_or_mult,
                tail_v_mean,
                tail_len,
                out_tail,
                lse_tail,
                seq_q=seq_q,
                tail_cap=tail_cap,
                is_causal=is_causal,
                grouped_qk=grouped_qk,
                target=target,
            )
        if has_chunks:
            _merge_two_attention_waves(output, lse, out_tail, lse_tail)

    return output, lse


def _select_split_counts(
    *,
    batch: int,
    kv_heads: int,
    gqa_group: int,
    seq_q: int,
    max_chunks: int,
    num_slots: int,
    tail_cap: int,
    active_chunks_hint: int,
    sm_count: int,
    schedule: _DecodeSchedule,
) -> tuple[int, int]:
    has_chunks = max_chunks > 0 and num_slots > 0
    has_tail = tail_cap > 0
    q_tiles = triton.cdiv(gqa_group * seq_q, _BLOCK_QH)
    base_blocks = max(1, batch * kv_heads * q_tiles)

    tail_splits = 0
    if has_tail:
        if (
            schedule.max_tail_splits > 2
            and tail_cap >= 128
            and base_blocks <= max(16, sm_count // 4)
        ):
            tail_splits = min(schedule.max_tail_splits, triton.cdiv(tail_cap, schedule.block_n))
        elif schedule.max_tail_splits <= 2 and tail_cap >= 128 and base_blocks <= 16:
            tail_splits = schedule.max_tail_splits
        else:
            tail_splits = 1

    int8_splits = 0
    if has_chunks:
        hint_chunks = max_chunks if active_chunks_hint <= 0 else min(max_chunks, active_chunks_hint)
        target_ctas = max(_TARGET_ACTIVE_CTAS, sm_count * schedule.target_ctas_per_sm)
        max_int8 = schedule.max_splits - tail_splits
        target_splits = max(1, target_ctas // base_blocks)
        int8_splits = min(
            max_int8,
            max(1, min(hint_chunks * schedule.tiles_per_chunk_cap, target_splits)),
        )

    return int8_splits, tail_splits


def piper_attention_decode_triton(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float,
    is_causal: bool,
    grouped_qk: bool,
    pack_wmma: bool,
    active_chunks_hint: int,
    prefill_backend: Literal["amd", "nvidia"] | None = None,
    target: AcceleratorTarget | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute GQA-grouped Split-KV decode or quantized chunked prefill on GPU."""
    if prefill_backend is not None:
        _validate_prefill_backend(prefill_backend, grouped_qk=grouped_qk, pack_wmma=pack_wmma)
    batch, q_heads, seq_q, head_dim = query.shape
    if seq_q >= _PREFILL_MIN_SEQ_Q and prefill_backend is not None:
        return _run_chunked_prefill(
            query,
            key_int8,
            key_scale,
            key_center,
            value_int8,
            value_scale,
            value_mean,
            chunk_to_slot,
            num_chunks,
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            scale=scale,
            is_causal=is_causal,
            grouped_qk=grouped_qk,
            pack_wmma=pack_wmma,
            active_chunks_hint=active_chunks_hint,
            prefill_backend=prefill_backend,
            target=target,
        )
    num_slots, kv_heads, _, _ = key_int8.shape
    gqa_group = q_heads // kv_heads

    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    tail_cap = 0 if tail_key is None else tail_key.shape[2]
    resolved_target = AcceleratorTarget.from_device(query.device) if target is None else target
    schedule = _select_decode_schedule(resolved_target, head_dim)
    compiler_options = dict(schedule.compiler_options)
    sm_count = device_sm_count(query.device)

    int8_splits, tail_splits = _select_split_counts(
        batch=batch,
        kv_heads=kv_heads,
        gqa_group=gqa_group,
        seq_q=seq_q,
        max_chunks=max_chunks,
        num_slots=num_slots,
        tail_cap=tail_cap,
        active_chunks_hint=active_chunks_hint,
        sm_count=sm_count,
        schedule=schedule,
    )
    num_splits = int8_splits + tail_splits
    output = torch.empty((batch, q_heads, seq_q, head_dim), device=query.device, dtype=query.dtype)
    lse = torch.empty((batch, q_heads, seq_q), device=query.device, dtype=torch.float32)

    if batch == 0:
        return output, lse
    if num_splits == 0:
        output.zero_()
        lse.fill_(-float("inf"))
        return output, lse

    partial_out = torch.empty(
        (batch, q_heads, seq_q, num_splits, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (batch, q_heads, seq_q, num_splits),
        device=query.device,
        dtype=torch.float32,
    )
    partial_bias = torch.empty_like(partial_lse)

    q_tiles = triton.cdiv(gqa_group * seq_q, _BLOCK_QH)
    stride_qb = 0 if batch <= 1 else query.stride(0)

    with device_context(query.device):
        if int8_splits > 0:
            (
                key_i8_s,
                val_i8_s,
                cs_tensor,
                has_chunk_map,
                _,
                stride_ki_s,
                stride_ks_s,
                stride_kc_s,
                stride_vi_s,
                stride_vs_s,
                stride_vm_s,
                stride_cs_b,
                stride_cs_c,
            ) = _pool_slot_strides(
                key_int8,
                key_scale,
                key_center,
                value_int8,
                value_scale,
                value_mean,
                chunk_to_slot,
                num_chunks,
                batch=batch,
            )

            _piper_decode_frozen_split_kernel[(batch * kv_heads * q_tiles, int8_splits)](
                query,
                key_i8_s,
                key_scale,
                key_center,
                val_i8_s,
                value_scale,
                value_mean,
                cs_tensor,
                num_chunks,
                partial_out,
                partial_lse,
                partial_bias,
                scale,
                kv_heads,
                gqa_group,
                q_tiles,
                seq_q,
                num_slots,
                max_chunks,
                int8_splits,
                stride_qb,
                query.stride(1),
                query.stride(2),
                stride_ki_s,
                stride_ks_s,
                stride_kc_s,
                stride_vi_s,
                stride_vs_s,
                stride_vm_s,
                stride_cs_b,
                stride_cs_c,
                partial_out.stride(0),
                partial_out.stride(1),
                partial_out.stride(2),
                partial_out.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                has_chunk_map=has_chunk_map,
                grouped_qk=grouped_qk,
                pack_wmma=pack_wmma,
                head_dim=head_dim,
                block_qh=_BLOCK_QH,
                block_n=schedule.block_n,
                qk_mode=schedule.frozen_qk_mode,
                pv_precision=schedule.pv_precision,
                loop_num_stages=schedule.loop_num_stages,
                num_warps=schedule.num_warps,
                num_stages=schedule.num_stages,
                **compiler_options,
            )

        if tail_splits > 0:
            assert tail_key is not None
            assert tail_value is not None
            assert tail_len is not None
            stride_tk_b = 0 if batch <= 1 else tail_key.stride(0)
            stride_tv_b = 0 if batch <= 1 else tail_value.stride(0)

            _piper_decode_tail_split_kernel[(batch * kv_heads * q_tiles, tail_splits)](
                query,
                tail_key,
                tail_value,
                tail_len,
                partial_out,
                partial_lse,
                partial_bias,
                scale,
                kv_heads,
                gqa_group,
                q_tiles,
                seq_q,
                tail_cap,
                int8_splits,
                tail_splits,
                stride_qb,
                query.stride(1),
                query.stride(2),
                stride_tk_b,
                tail_key.stride(1),
                tail_key.stride(2),
                stride_tv_b,
                tail_value.stride(1),
                tail_value.stride(2),
                partial_out.stride(0),
                partial_out.stride(1),
                partial_out.stride(2),
                partial_out.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                is_causal=is_causal,
                head_dim=head_dim,
                block_qh=_BLOCK_QH,
                block_n=schedule.block_n,
                qk_precision=schedule.tail_qk_precision,
                pv_precision=schedule.pv_precision,
                num_warps=schedule.num_warps,
                num_stages=schedule.num_stages,
                **compiler_options,
            )

        _piper_decode_splitkv_reduce_kernel[(batch * q_heads * seq_q,)](
            partial_out,
            partial_lse,
            partial_bias,
            output,
            lse,
            q_heads,
            seq_q,
            num_splits,
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            1,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            head_dim=head_dim,
            num_warps=_REDUCE_NUM_WARPS,
        )

    return output, lse


def compile_kv_decode_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    contiguous_inputs: bool = True,
    wide_pool: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object, object]:
    """Compile frozen-split, tail-split, and reduce decode kernels offline."""
    triton_dtype = _quantization.validate_kv_compile_args(dtype, head_dim)
    accel = AcceleratorTarget.from_compiler_target(target)
    _, grouped_qk, pack_wmma = _backend.kv_chunk_layout(accel)
    schedule = _select_decode_schedule(accel, head_dim)

    frozen_overrides: dict[str, str] = {
        "query_ptr": f"*{triton_dtype}",
        "key_int8_ptr": "*i8",
        "key_scale_ptr": "*fp32",
        "key_center_ptr": "*fp32",
        "value_int8_ptr": "*i8",
        "value_scale_ptr": "*fp32",
        "value_mean_ptr": "*fp32",
        "chunk_to_slot_ptr": "*i32",
        "num_chunks_ptr": "*i32",
        "partial_out_ptr": "*fp32",
        "partial_lse_ptr": "*fp32",
        "partial_bias_ptr": "*fp32",
        "scale": "fp32",
    }
    if wide_pool:
        frozen_overrides.update(dict.fromkeys(_quantization.POOL_SLOT_STRIDE_NAMES, "i64"))
    frozen_noncontig = frozenset(
        {
            "key_int8_ptr",
            "key_scale_ptr",
            "key_center_ptr",
            "value_int8_ptr",
            "value_scale_ptr",
            "value_mean_ptr",
            "partial_out_ptr",
            "stride_ki_s",
            "stride_vi_s",
            "stride_pob",
            "stride_poh",
            "stride_pos",
            "stride_pop",
        }
    )
    frozen_source = _quantization.build_kernel_source(
        _piper_decode_frozen_split_kernel,
        constants={
            "has_chunk_map": has_chunk_map,
            "grouped_qk": grouped_qk,
            "pack_wmma": pack_wmma,
            "head_dim": head_dim,
            "block_qh": _BLOCK_QH,
            "block_n": schedule.block_n,
            "qk_mode": schedule.frozen_qk_mode,
            "pv_precision": schedule.pv_precision,
            "loop_num_stages": schedule.loop_num_stages,
        },
        overrides=frozen_overrides,
        attrs=(
            None
            if contiguous_inputs
            else _div16_attrs(_piper_decode_frozen_split_kernel.arg_names, frozen_noncontig)
        ),
        all_strides_div16=True,
        pointer_range_32=not wide_pool,
    )

    tail_noncontig = frozenset(
        {"partial_out_ptr", "stride_pob", "stride_poh", "stride_pos", "stride_pop"}
    )
    tail_source = _quantization.build_kernel_source(
        _piper_decode_tail_split_kernel,
        constants={
            "is_causal": is_causal,
            "head_dim": head_dim,
            "block_qh": _BLOCK_QH,
            "block_n": schedule.block_n,
            "qk_precision": schedule.tail_qk_precision,
            "pv_precision": schedule.pv_precision,
        },
        overrides={
            "query_ptr": f"*{triton_dtype}",
            "tail_key_ptr": f"*{triton_dtype}",
            "tail_value_ptr": f"*{triton_dtype}",
            "tail_len_ptr": "*i32",
            "partial_out_ptr": "*fp32",
            "partial_lse_ptr": "*fp32",
            "partial_bias_ptr": "*fp32",
            "scale": "fp32",
        },
        attrs=(
            None
            if contiguous_inputs
            else _div16_attrs(_piper_decode_tail_split_kernel.arg_names, tail_noncontig)
        ),
        all_strides_div16=True,
    )

    reduce_noncontig = frozenset(
        {
            "partial_out_ptr",
            "output_ptr",
            "stride_pob",
            "stride_poh",
            "stride_pos",
            "stride_pop",
            "stride_ob",
            "stride_oh",
            "stride_os",
        }
    )
    reduce_source = _quantization.build_kernel_source(
        _piper_decode_splitkv_reduce_kernel,
        constants={"head_dim": head_dim},
        overrides={
            "partial_out_ptr": "*fp32",
            "partial_lse_ptr": "*fp32",
            "partial_bias_ptr": "*fp32",
            "output_ptr": f"*{triton_dtype}",
            "lse_ptr": "*fp32",
        },
        attrs=(
            None
            if contiguous_inputs
            else _div16_attrs(_piper_decode_splitkv_reduce_kernel.arg_names, reduce_noncontig)
        ),
        all_strides_div16=True,
    )

    compiler = triton.compile if compile_fn is None else compile_fn
    split_options: dict[str, object] = {
        "num_warps": schedule.num_warps,
        "num_stages": schedule.num_stages,
        **dict(schedule.compiler_options),
    }
    return (
        compiler(frozen_source, target, split_options),
        compiler(tail_source, target, split_options),
        compiler(reduce_source, target, {"num_warps": _REDUCE_NUM_WARPS}),
    )


def compile_kv_prefill_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    wide_pool: bool = False,
    has_chunk_splits: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, ...]:
    """Offline-compile two-wave merge and native chunked prefill kernels for ``target``."""
    triton_dtype = _quantization.validate_kv_compile_args(dtype, head_dim)
    compiler = triton.compile if compile_fn is None else compile_fn
    merge_source = _quantization.build_kernel_source(
        _merge_two_attention_waves_kernel,
        constants={"head_dim": head_dim},
        overrides={
            "out_a_ptr": f"*{triton_dtype}",
            "lse_a_ptr": "*fp32",
            "out_b_ptr": f"*{triton_dtype}",
            "lse_b_ptr": "*fp32",
        },
    )
    merge_compiled = compiler(merge_source, target, {"num_warps": _REDUCE_NUM_WARPS})

    accel = AcceleratorTarget.from_compiler_target(target)
    backend = _backend.prefill_backend(accel)
    if backend == "amd":
        from ._amd import gluon as amd_gluon  # noqa: PLC0415

        return (
            merge_compiled,
            *amd_gluon.compile_amd_prefill_kernels(
                target,
                dtype=dtype,
                head_dim=head_dim,
                has_chunk_map=has_chunk_map,
                is_causal=is_causal,
                wide_pool=wide_pool,
                compile_fn=compile_fn,
            ),
        )
    if backend == "nvidia":
        from ._nvidia import triton as nvidia_triton  # noqa: PLC0415

        return (
            merge_compiled,
            *nvidia_triton.compile_nvidia_prefill_kernels(
                target,
                dtype=dtype,
                head_dim=head_dim,
                has_chunk_map=has_chunk_map,
                is_causal=is_causal,
                wide_pool=wide_pool,
                has_chunk_splits=has_chunk_splits,
                compile_fn=compile_fn,
            ),
        )
    return (merge_compiled,)


__all__ = [
    "compile_kv_decode_kernels",
    "compile_kv_prefill_kernels",
    "piper_attention_decode_triton",
]
