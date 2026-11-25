"""Portable GQA-grouped Split-KV Triton decode kernels for chunked Piper KV pools."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

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

from . import _backend
from ._quantization import _div16_attrs

_CENTER_CHUNK = 1024
_VALUE_TILE_TOKENS = 64
_KEY_SCALES_GROUPED = _CENTER_CHUNK // _VALUE_TILE_TOKENS
_BLOCK_N = 16
_SUBTILES_PER_K64 = 4
_SUBTILES_PER_CHUNK = _CENTER_CHUNK // _BLOCK_N
_BLOCK_QH = 16
_MAX_DECODE_SPLITS = 32
_TARGET_ACTIVE_CTAS = 64
_SPLIT_NUM_STAGES = 1
_REDUCE_NUM_WARPS = 4


def _split_num_warps(head_dim: int) -> int:
    return 16 if head_dim == 128 else 8


_TL_BLOCK_N = tl.constexpr(_BLOCK_N)
_TL_SUBTILES_PER_K64 = tl.constexpr(_SUBTILES_PER_K64)
_TL_SUBTILES_PER_CHUNK = tl.constexpr(_SUBTILES_PER_CHUNK)
_TL_CENTER_CHUNK = tl.constexpr(_CENTER_CHUNK)
_TL_VALUE_TILE_TOKENS = tl.constexpr(_VALUE_TILE_TOKENS)
_TL_KEY_SCALES_GROUPED = tl.constexpr(_KEY_SCALES_GROUPED)
_LOG2_E = tl.constexpr(1.4426950408889634)
_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _is_nan(x):
    return x != x  # noqa: PLR0124


@triton.jit
def _max_propagate_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _two_term_softmax_step(m_prev, bias_prev, m_cur, bias_cur):
    """Advance two-term base-2 winner state (m, bias) with exact NaN/same-bias handling."""
    has_nan = _is_nan(m_cur + bias_cur)
    m_clean = tl.where(has_nan, float("nan"), m_cur)
    has_cur = ((m_clean != -float("inf")) & (bias_cur != -float("inf"))) | has_nan
    has_prev = m_prev != -float("inf")
    bias_diff = tl.where(bias_cur == bias_prev, 0.0, bias_cur - bias_prev)
    raw_delta = (m_clean - m_prev) + bias_diff
    delta = tl.where(has_cur, tl.where(has_prev, raw_delta, float("inf")), -float("inf"))
    cur_wins = (delta > 0.0) | has_nan
    m_next = tl.where(cur_wins, m_clean, m_prev)
    bias_next = tl.where(cur_wins, bias_cur, bias_prev)
    decay = tl.math.exp2(-tl.abs(delta))
    old_weight = tl.where(has_prev, tl.where(cur_wins, decay, 1.0), 0.0)
    cur_weight = tl.where(has_cur, tl.where(cur_wins, 1.0, decay), 0.0)
    safe_m_cur = tl.where(has_cur, m_clean, 0.0)
    return m_next, bias_next, old_weight, cur_weight, has_cur, has_nan, safe_m_cur


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

    nc = tl.load(num_chunks_ptr + b64)
    active_chunks = tl.maximum(0, tl.minimum(nc, max_chunks)).to(tl.int64)
    total_subtiles = active_chunks * _TL_SUBTILES_PER_CHUNK
    split64 = split_id.to(tl.int64)
    subtiles_per_split = tl.cdiv(total_subtiles, int8_splits.to(tl.int64))
    subtile_start = split64 * subtiles_per_split
    subtile_end = tl.minimum(total_subtiles, subtile_start + subtiles_per_split)

    if subtile_start < subtile_end:
        first_chunk = (subtile_start // _TL_SUBTILES_PER_CHUNK).to(tl.int32)
        last_chunk = ((subtile_end - 1) // _TL_SUBTILES_PER_CHUNK).to(tl.int32)
        first_inner = (subtile_start % _TL_SUBTILES_PER_CHUNK).to(tl.int32)
        last_inner = ((subtile_end - 1) % _TL_SUBTILES_PER_CHUNK + 1).to(tl.int32)

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

        offsets_n = tl.arange(0, _TL_BLOCK_N)
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
                inner_end = tl.where(chunk_idx == last_chunk, last_inner, _TL_SUBTILES_PER_CHUNK)
                for local_subtile in tl.range(inner_start, inner_end, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
                    t64 = local_subtile // _TL_SUBTILES_PER_K64
                    sub4 = local_subtile % _TL_SUBTILES_PER_K64
                    rows = local_subtile * _TL_BLOCK_N + offsets_n
                    if head_dim == 128:
                        k_lo = tl.load(k_base + rows[:, None] * head_dim + offsets_d64[None, :])
                        qk = tl.dot(q_lo, tl.trans(k_lo.to(tl.float32)), input_precision="ieee")
                        k_hi = tl.load(
                            k_base + rows[:, None] * head_dim + (offsets_d64[None, :] + 64)
                        )
                        qk = tl.dot(
                            q_hi, tl.trans(k_hi.to(tl.float32)), acc=qk, input_precision="ieee"
                        )
                    else:
                        k_q = tl.load(k_base + rows[:, None] * head_dim + offsets_d[None, :])
                        qk = tl.dot(q_lo, tl.trans(k_q.to(tl.float32)), input_precision="ieee")
                    if grouped_qk:
                        k_scale = tl.load(ks_base + t64)
                        scores = qk * (k_scale * scaled_log2)
                    else:
                        k_scale = tl.load(ks_base + rows)
                        scores = qk * (k_scale[None, :] * scaled_log2)

                    m_block = tl.reduce(scores, 1, _max_propagate_nan)
                    m_i, bias_i, old_weight, cur_weight, has_block, _, safe_m_block = (
                        _two_term_softmax_step(m_i, bias_i, m_block, chunk_bias)
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
                    offs_tok = sub4 * _TL_BLOCK_N + offsets_n
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
                        input_precision="ieee",
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
    total_tiles = tl.cdiv(active_tail, _TL_BLOCK_N)
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

        offsets_n = tl.arange(0, _TL_BLOCK_N)
        tk_base = tail_key_ptr + b64 * stride_tk_b + kv_head64 * stride_tk_h
        tv_base = tail_value_ptr + b64 * stride_tv_b + kv_head64 * stride_tv_h
        scaled_log2 = scale * _LOG2_E
        active_tail32 = active_tail.to(tl.int32)
        cutoff = (active_tail32 - seq_q) + q_idx0

        for tile_idx in tl.range(tile_start.to(tl.int32), tile_end.to(tl.int32), disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            rows32 = tile_idx * _TL_BLOCK_N + offsets_n
            valid_n = rows32 < active_tail32
            rows = rows32.to(tl.int64)
            if head_dim == 128:
                k_lo = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + offsets_d64[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                scores = tl.dot(q_lo, tl.trans(k_lo.to(tl.float32)), input_precision="ieee")
                k_hi = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + (offsets_d64[None, :] + 64),
                    mask=valid_n[:, None],
                    other=0.0,
                )
                scores = tl.dot(
                    q_hi, tl.trans(k_hi.to(tl.float32)), acc=scores, input_precision="ieee"
                )
            else:
                k_tile = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + offsets_d[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                scores = tl.dot(q_f32, tl.trans(k_tile.to(tl.float32)), input_precision="ieee")
            scores *= scaled_log2
            if is_causal:
                valid_mask = valid_n[None, :] & (rows32[None, :] <= cutoff[:, None])
            else:
                valid_mask = valid_n[None, :]

            masked_scores = tl.where(valid_mask, scores, -float("inf"))
            m_block = tl.reduce(masked_scores, 1, _max_propagate_nan)
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
            acc = tl.dot(p, v_tile.to(tl.float32), acc=acc, input_precision="ieee")
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


@triton.jit(do_not_specialize=["q_heads", "seq_q", "num_splits"])
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

    m_win = -float("inf")
    bias_win = 0.0
    denom = 0.0
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    for split_idx in tl.range(0, num_splits, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        s64 = split_idx.to(tl.int64)
        m_s = tl.load(partial_lse_ptr + pl_offset + s64)
        b_s = tl.load(partial_bias_ptr + pl_offset + s64)
        m_win, bias_win, old_w, cur_w, has_s, has_nan_s, _ = _two_term_softmax_step(
            m_win, bias_win, m_s, b_s
        )

        out_s = tl.load(po_base + s64 * stride_pop + offsets_d)
        safe_out_s = tl.where(has_nan_s, float("nan"), tl.where(has_s, out_s, 0.0))
        acc = acc * old_w + cur_w * safe_out_s
        denom = denom * old_w + cur_w

    has_any = m_win != -float("inf")
    safe_denom = tl.where(has_any, denom, 1.0)
    final_out = tl.where(has_any, acc / safe_denom, 0.0)
    final_lse = tl.where(
        has_any, (m_win * _LN2 + bias_win * _LN2) + tl.math.log2(safe_denom) * _LN2, -float("inf")
    )

    out_ptrs = output_ptr + b64 * stride_ob + qh64 * stride_oh + qi64 * stride_os + offsets_d
    tl.store(out_ptrs, final_out.to(output_ptr.dtype.element_ty))
    tl.store(lse_ptr + b64 * stride_lb + qh64 * stride_lh + qi64, final_lse)


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
) -> tuple[int, int]:
    has_chunks = max_chunks > 0 and num_slots > 0
    has_tail = tail_cap > 0
    q_tiles = triton.cdiv(gqa_group * seq_q, _BLOCK_QH)
    base_blocks = max(1, batch * kv_heads * q_tiles)

    tail_splits = 0
    if has_tail:
        tail_splits = 2 if (tail_cap >= 128 and base_blocks <= 16) else 1

    int8_splits = 0
    if has_chunks:
        hint_chunks = max_chunks if active_chunks_hint <= 0 else min(max_chunks, active_chunks_hint)
        max_int8 = _MAX_DECODE_SPLITS - tail_splits
        target_splits = max(1, _TARGET_ACTIVE_CTAS // base_blocks)
        int8_splits = min(max_int8, max(1, min(hint_chunks * 8, target_splits)))

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute GQA-grouped Split-KV decode and deterministic LSE reduction on GPU."""
    batch, q_heads, seq_q, head_dim = query.shape
    num_slots, kv_heads, _, _ = key_int8.shape
    gqa_group = q_heads // kv_heads

    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    tail_cap = 0 if tail_key is None else tail_key.shape[2]

    int8_splits, tail_splits = _select_split_counts(
        batch=batch,
        kv_heads=kv_heads,
        gqa_group=gqa_group,
        seq_q=seq_q,
        max_chunks=max_chunks,
        num_slots=num_slots,
        tail_cap=tail_cap,
        active_chunks_hint=active_chunks_hint,
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
            has_chunk_map = chunk_to_slot is not None
            cs_tensor = chunk_to_slot if chunk_to_slot is not None else num_chunks
            key_i8_s = key_int8.view(torch.int8)
            val_i8_s = value_int8.view(torch.int8)

            stride_ki_s = 0 if num_slots <= 1 else key_i8_s.stride(0)
            stride_ks_s = 0 if num_slots <= 1 else key_scale.stride(0)
            stride_kc_s = 0 if num_slots <= 1 else key_center.stride(0)
            stride_vi_s = 0 if num_slots <= 1 else val_i8_s.stride(0)
            stride_vs_s = 0 if num_slots <= 1 else value_scale.stride(0)
            stride_vm_s = 0 if num_slots <= 1 else value_mean.stride(0)
            stride_cs_b = 0 if (not has_chunk_map or batch <= 1) else cs_tensor.stride(0)
            stride_cs_c = 1 if not has_chunk_map else cs_tensor.stride(1)

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
                num_warps=_split_num_warps(head_dim),
                num_stages=_SPLIT_NUM_STAGES,
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
                num_warps=_split_num_warps(head_dim),
                num_stages=_SPLIT_NUM_STAGES,
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
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            head_dim=head_dim,
            num_warps=_REDUCE_NUM_WARPS,
        )

    return output, lse


def _live_contiguous_attrs(
    arg_names: list[str],
    signature: dict[str, str],
) -> dict[tuple[int, ...], object]:
    return {
        (idx,): (
            [["tt.divisibility", 16], ["tt.pointer_range", 32]]
            if name.endswith("_ptr")
            else [["tt.divisibility", 16]]
        )
        for idx, name in enumerate(arg_names)
        if name in signature and (name.endswith("_ptr") or name.startswith("stride_"))
    }


def compile_kv_decode_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    contiguous_inputs: bool = True,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object, object]:
    """Compile frozen-split, tail-split, and reduce decode kernels offline."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"dtype must be float16 or bfloat16, got {dtype}")
    if head_dim not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")

    from triton.compiler import ASTSource  # noqa: PLC0415 - compile-only dependency

    _, grouped_qk, pack_wmma = _backend.kv_chunk_layout(
        AcceleratorTarget.from_compiler_target(target)
    )
    triton_dtype = "bf16" if dtype is torch.bfloat16 else "fp16"

    frozen_constants: dict[str, object] = {
        "has_chunk_map": has_chunk_map,
        "grouped_qk": grouped_qk,
        "pack_wmma": pack_wmma,
        "head_dim": head_dim,
        "block_qh": _BLOCK_QH,
    }
    frozen_signature: dict[str, str] = {
        name: "i32"
        for name in _piper_decode_frozen_split_kernel.arg_names
        if name not in frozen_constants
    }
    frozen_signature.update(
        query_ptr=f"*{triton_dtype}",
        key_int8_ptr="*i8",
        key_scale_ptr="*fp32",
        key_center_ptr="*fp32",
        value_int8_ptr="*i8",
        value_scale_ptr="*fp32",
        value_mean_ptr="*fp32",
        chunk_to_slot_ptr="*i32",
        num_chunks_ptr="*i32",
        partial_out_ptr="*fp32",
        partial_lse_ptr="*fp32",
        partial_bias_ptr="*fp32",
        scale="fp32",
    )
    frozen_attrs = (
        _live_contiguous_attrs(_piper_decode_frozen_split_kernel.arg_names, frozen_signature)
        if contiguous_inputs
        else _div16_attrs(
            _piper_decode_frozen_split_kernel.arg_names,
            frozenset(
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
            ),
        )
    )

    tail_constants: dict[str, object] = {
        "is_causal": is_causal,
        "head_dim": head_dim,
        "block_qh": _BLOCK_QH,
    }
    tail_signature: dict[str, str] = {
        name: "i32"
        for name in _piper_decode_tail_split_kernel.arg_names
        if name not in tail_constants
    }
    tail_signature.update(
        query_ptr=f"*{triton_dtype}",
        tail_key_ptr=f"*{triton_dtype}",
        tail_value_ptr=f"*{triton_dtype}",
        tail_len_ptr="*i32",
        partial_out_ptr="*fp32",
        partial_lse_ptr="*fp32",
        partial_bias_ptr="*fp32",
        scale="fp32",
    )
    tail_attrs = (
        _live_contiguous_attrs(_piper_decode_tail_split_kernel.arg_names, tail_signature)
        if contiguous_inputs
        else _div16_attrs(
            _piper_decode_tail_split_kernel.arg_names,
            frozenset(
                {
                    "partial_out_ptr",
                    "stride_pob",
                    "stride_poh",
                    "stride_pos",
                    "stride_pop",
                }
            ),
        )
    )

    reduce_constants: dict[str, object] = {
        "head_dim": head_dim,
    }
    reduce_signature: dict[str, str] = {
        name: "i32"
        for name in _piper_decode_splitkv_reduce_kernel.arg_names
        if name not in reduce_constants
    }
    reduce_signature.update(
        partial_out_ptr="*fp32",
        partial_lse_ptr="*fp32",
        partial_bias_ptr="*fp32",
        output_ptr=f"*{triton_dtype}",
        lse_ptr="*fp32",
    )
    reduce_attrs = (
        _live_contiguous_attrs(_piper_decode_splitkv_reduce_kernel.arg_names, reduce_signature)
        if contiguous_inputs
        else _div16_attrs(
            _piper_decode_splitkv_reduce_kernel.arg_names,
            frozenset(
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
            ),
        )
    )

    compiler = triton.compile if compile_fn is None else compile_fn
    split_options: dict[str, object] = {
        "num_warps": _split_num_warps(head_dim),
        "num_stages": _SPLIT_NUM_STAGES,
    }
    return (
        compiler(
            ASTSource(
                _piper_decode_frozen_split_kernel,
                frozen_signature,
                constexprs=frozen_constants,
                attrs=frozen_attrs,
            ),
            target,
            split_options,
        ),
        compiler(
            ASTSource(
                _piper_decode_tail_split_kernel,
                tail_signature,
                constexprs=tail_constants,
                attrs=tail_attrs,
            ),
            target,
            split_options,
        ),
        compiler(
            ASTSource(
                _piper_decode_splitkv_reduce_kernel,
                reduce_signature,
                constexprs=reduce_constants,
                attrs=reduce_attrs,
            ),
            target,
            {"num_warps": _REDUCE_NUM_WARPS},
        ),
    )


__all__ = [
    "compile_kv_decode_kernels",
    "piper_attention_decode_triton",
]
