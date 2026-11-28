"""Quantized chunked prefill and Split-KV reduction orchestration for Piper KV pools."""

# Triton JIT kernels do not use Python runtime type annotations or narrow signatures.
# ruff: noqa: ANN001, ANN202, PLR0124, PLR0913, PLR0917, SIM102, SIM114
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import triton
import triton.language as tl

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.piper.triton import max_propagate_nan
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear._splitk import device_sm_count

from . import _quantization
from ._amd import policy as _amd_policy
from ._decode_schedule import (
    BLOCK_QH,
    PREFILL_MIN_SEQ_Q,
    REDUCE_NUM_WARPS,
    DecodeSchedule,
    default_decode_schedule,
    select_split_counts,
)
from ._nvidia import policy as _nvidia_policy

_LN2 = tl.constexpr(0.6931471805599453)
_LOG2_E = tl.constexpr(1.4426950408889634)
_MAX_MERGE_WAVES = 8
_MERGE_STRIDE_FIELDS: tuple[str, ...] = tuple(
    field
    for wave_idx in range(_MAX_MERGE_WAVES)
    for field in (
        f"so0_{wave_idx}",
        f"so1_{wave_idx}",
        f"so2_{wave_idx}",
        f"so3_{wave_idx}",
        f"sod_{wave_idx}",
        f"sl0_{wave_idx}",
        f"sl1_{wave_idx}",
        f"sl2_{wave_idx}",
        f"sl3_{wave_idx}",
    )
)


def select_decode_schedule(
    target: AcceleratorTarget | None,
    head_dim: int,
) -> DecodeSchedule:
    """Return the Split-KV kernel and split schedule for ``target`` and ``head_dim``."""
    if target is not None and target.is_nvidia_cuda:
        return _nvidia_policy.select_decode_schedule(target, head_dim)
    if target is not None and target.is_amd_hip:
        return _amd_policy.select_decode_schedule(target, head_dim)
    return default_decode_schedule(head_dim)


@triton.jit
def _advance_split_max_safe(m_win, bias_win, ld_win, m_s, b_s, ld_s):
    """Update 3-term winner-frame max/bias/log-denom state across partial splits."""
    has_nan = ((m_s + b_s) + ld_s) != ((m_s + b_s) + ld_s)
    m_clean = tl.where(has_nan, float("nan"), m_s)
    has_split = ((m_clean != -float("inf")) & (b_s != -float("inf"))) | has_nan
    has_running = m_win != -float("inf")
    bias_diff = tl.where(b_s == bias_win, 0.0, b_s - bias_win)
    max_diff = tl.where((m_clean == m_win) & (m_clean != float("inf")), 0.0, m_clean - m_win)
    ld_diff = tl.where(ld_s == ld_win, 0.0, ld_s - ld_win)
    raw_delta = ld_diff + (max_diff + bias_diff)
    delta = tl.where(has_split, tl.where(has_running, raw_delta, float("inf")), -float("inf"))
    split_wins = (delta > 0.0) | has_nan
    next_m = tl.where(split_wins, m_clean, m_win)
    next_bias = tl.where(split_wins, b_s, bias_win)
    next_ld = tl.where(split_wins, ld_s, ld_win)
    decay = tl.exp2(-tl.abs(delta))
    old_w = tl.where(has_running, tl.where(split_wins, decay, 1.0), 0.0)
    cur_w = tl.where(has_split, tl.where(split_wins, 1.0, decay), 0.0)
    return next_m, next_bias, next_ld, old_w, cur_w, has_split, has_nan


@triton.jit
def _reduce_split_chunk_weights(m_s, b_s, ld_s, s_mask, s_arange):
    """Compute exact 3-term softmax weights and chunk summary over a vector of splits."""
    tot_s = (m_s + b_s) + ld_s
    is_nan_s = s_mask & (tot_s != tot_s)
    chunk_nan = tl.max(is_nan_s.to(tl.int32), axis=0)
    valid_s = s_mask & (m_s != -float("inf")) & (b_s != -float("inf")) & (~is_nan_s)
    has_valid = tl.max(valid_s.to(tl.int32), axis=0) != 0

    ref_idx = tl.argmax(tl.where(valid_s, tot_s, -float("inf")), axis=0)
    is_ref = s_arange == ref_idx
    m_ref = tl.sum(tl.where(is_ref, m_s, 0.0), axis=0)
    b_ref = tl.sum(tl.where(is_ref, b_s, 0.0), axis=0)
    ld_ref = tl.sum(tl.where(is_ref, ld_s, 0.0), axis=0)

    b_diff = tl.where(is_ref | (b_s == b_ref), 0.0, b_s - b_ref)
    m_diff = tl.where(is_ref | ((m_s == m_ref) & (m_s != float("inf"))), 0.0, m_s - m_ref)
    ld_diff = tl.where(is_ref | (ld_s == ld_ref), 0.0, ld_s - ld_ref)
    raw_delta = ld_diff + (m_diff + b_diff)
    delta_nan = valid_s & (raw_delta != raw_delta)
    chunk_nan = chunk_nan | tl.max(delta_nan.to(tl.int32), axis=0)

    delta_clean = tl.where(valid_s & (~delta_nan), raw_delta, -float("inf"))
    max_delta = tl.max(delta_clean, axis=0)
    safe_max = tl.where(
        has_valid & (max_delta != -float("inf")) & (max_delta != float("inf")),
        max_delta,
        0.0,
    )
    w_s = tl.where(valid_s & (~delta_nan), tl.math.exp2(delta_clean - safe_max), 0.0)
    sum_w = tl.sum(w_s, axis=0)
    safe_sum = tl.where(has_valid & (sum_w != 0.0), sum_w, 1.0)
    chunk_ld = tl.where(has_valid, ld_ref + safe_max + tl.math.log2(safe_sum), 0.0)
    return w_s, valid_s, safe_sum, m_ref, b_ref, chunk_ld, has_valid, chunk_nan


@triton.jit
def _normalize_running_weights(denom, old_w, cur_w):
    """Update running denominator and return normalized weights preventing intermediate overflow."""
    old_num = denom * old_w
    new_denom = old_num + cur_w
    has_denom = new_denom != 0.0
    safe_dn = tl.where(has_denom, new_denom, 1.0)
    old_alpha = tl.where(has_denom, old_num / safe_dn, 0.0)
    cur_alpha = tl.where(has_denom, cur_w / safe_dn, 0.0)
    return new_denom, old_alpha, cur_alpha


@triton.jit
def _load_split_state(partial_lse_ptr, partial_bias_ptr, partial_log_denom_ptr, lse_off, mask):
    """Load split LSE, center bias, and log-denominator tuple with safe inactive defaults."""
    m_s = tl.load(partial_lse_ptr + lse_off, mask=mask, other=-float("inf"))
    b_s = tl.load(partial_bias_ptr + lse_off, mask=mask, other=0.0)
    ld_s = tl.load(partial_log_denom_ptr + lse_off, mask=mask, other=0.0)
    return m_s, b_s, ld_s


@triton.jit
def _finalize_splitkv_lse(has_any, denom, m_win, bias_win, ld_win):
    """Compute final natural-log LSE from three-term winner-frame state."""
    safe_denom = tl.where(has_any & (denom != 0.0), denom, 1.0)
    return tl.where(
        has_any,
        (m_win * _LN2 + bias_win * _LN2) + (ld_win + tl.math.log2(safe_denom)) * _LN2,
        -float("inf"),
    )


@triton.jit(do_not_specialize=["q_heads", "seq_q", "num_splits", "pl_split_stride"])
def _piper_decode_splitkv_reduce_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    partial_bias_ptr,
    partial_log_denom_ptr,
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
    block_r: tl.constexpr = 8,
    block_s: tl.constexpr = 1,
):
    """Merge partial Split-KV outputs and three-term base-2 state into natural-log LSE."""
    grid_rows = seq_q if block_r == 1 else tl.cdiv(seq_q, block_r)
    pid_s = tl.program_id(0) % grid_rows
    pid_bh = tl.program_id(0) // grid_rows
    b64, qh64 = (pid_bh // q_heads).to(tl.int64), (pid_bh % q_heads).to(tl.int64)
    offsets_d = tl.arange(0, head_dim).to(tl.int64)
    split_lse_stride = pl_split_stride.to(tl.int64)
    q_offs = pid_s if block_s > 1 else pid_s * block_r + tl.arange(0, block_r)
    qi64 = q_offs.to(tl.int64)
    pl_base = b64 * stride_plb + qh64 * stride_plh + qi64 * stride_pls
    po_head = partial_out_ptr + b64 * stride_pob + qh64 * stride_poh

    if block_s > 1:
        po_base = po_head + qi64 * stride_pos
        m_win, bias_win, ld_win, denom, any_nan = -float("inf"), 0.0, 0.0, 0.0, 0
        acc = tl.zeros((head_dim,), dtype=tl.float32)
        s_arange = tl.arange(0, block_s)
        for s_start in tl.range(0, num_splits, block_s):
            s_offs = s_start + s_arange
            s_mask = s_offs < num_splits
            s64 = s_offs.to(tl.int64)
            m_s, b_s, ld_s = _load_split_state(
                partial_lse_ptr,
                partial_bias_ptr,
                partial_log_denom_ptr,
                pl_base + s64 * split_lse_stride,
                s_mask,
            )
            w_s, valid_s, safe_sum, m_ref, b_ref, chunk_ld, has_valid, chunk_nan = (
                _reduce_split_chunk_weights(m_s, b_s, ld_s, s_mask, s_arange)
            )
            any_nan = any_nan | chunk_nan
            out_s = tl.load(
                po_base + s64[:, None] * stride_pop + offsets_d[None, :],
                mask=s_mask[:, None],
                other=0.0,
            )
            chunk_out = tl.sum(
                tl.where(valid_s[:, None], out_s * (w_s / safe_sum)[:, None], 0.0), axis=0
            )
            m_win, bias_win, ld_win, old_w, cur_w, has_c, has_nan_c = _advance_split_max_safe(
                m_win,
                bias_win,
                ld_win,
                tl.where(any_nan != 0, float("nan"), tl.where(has_valid, m_ref, -float("inf"))),
                tl.where(has_valid, b_ref, 0.0),
                tl.where(has_valid, chunk_ld, 0.0),
            )
            denom, old_alpha, cur_alpha = _normalize_running_weights(denom, old_w, cur_w)
            acc = acc * old_alpha + cur_alpha * tl.where(
                has_nan_c, float("nan"), tl.where(has_c, chunk_out, 0.0)
            )
        has_any = (m_win != -float("inf")) | (any_nan != 0)
        final_out = tl.where(any_nan != 0, float("nan"), tl.where(has_any, acc, 0.0))
        tl.store(
            output_ptr + b64 * stride_ob + qh64 * stride_oh + qi64 * stride_os + offsets_d,
            final_out.to(output_ptr.dtype.element_ty),
        )
        tl.store(
            lse_ptr + b64 * stride_lb + qh64 * stride_lh + qi64,
            tl.where(
                any_nan != 0,
                float("nan"),
                _finalize_splitkv_lse(has_any, denom, m_win, bias_win, ld_win),
            ),
        )
    else:
        q_mask = q_offs < seq_q
        po_base = po_head + qi64[:, None] * stride_pos + offsets_d[None, :]
        m_win = tl.full((block_r,), -float("inf"), dtype=tl.float32)
        bias_win = tl.zeros((block_r,), dtype=tl.float32)
        ld_win = tl.zeros((block_r,), dtype=tl.float32)
        denom = tl.zeros((block_r,), dtype=tl.float32)
        acc = tl.zeros((block_r, head_dim), dtype=tl.float32)
        for split_idx in tl.range(0, num_splits):
            s64 = split_idx.to(tl.int64)
            m_s, b_s, ld_s = _load_split_state(
                partial_lse_ptr,
                partial_bias_ptr,
                partial_log_denom_ptr,
                pl_base + s64 * split_lse_stride,
                q_mask,
            )
            m_win, bias_win, ld_win, old_w, cur_w, has_s, has_nan_s = _advance_split_max_safe(
                m_win, bias_win, ld_win, m_s, b_s, ld_s
            )
            out_s = tl.load(po_base + s64 * stride_pop, mask=q_mask[:, None], other=0.0)
            denom, old_alpha, cur_alpha = _normalize_running_weights(denom, old_w, cur_w)
            acc = acc * old_alpha[:, None] + cur_alpha[:, None] * tl.where(
                has_nan_s[:, None], float("nan"), tl.where(has_s[:, None], out_s, 0.0)
            )
        has_any = m_win != -float("inf")
        tl.store(
            output_ptr
            + b64 * stride_ob
            + qh64 * stride_oh
            + qi64[:, None] * stride_os
            + offsets_d[None, :],
            tl.where(has_any[:, None], acc, 0.0).to(output_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )
        tl.store(
            lse_ptr + b64 * stride_lb + qh64 * stride_lh + qi64,
            _finalize_splitkv_lse(has_any, denom, m_win, bias_win, ld_win),
            mask=q_mask,
        )


@triton.jit
def _load_wave_lse(
    lse_ptr,
    row,
    i0,
    i1,
    i2,
    i3,
    sl0,
    sl1,
    sl2,
    sl3,
    is_contiguous: tl.constexpr,
):
    if is_contiguous:
        return tl.load(lse_ptr + row).to(tl.float32)
    else:
        off = i0 * sl0 + i1 * sl1 + i2 * sl2 + i3 * sl3
        return tl.load(lse_ptr + off).to(tl.float32)


@triton.jit
def _load_wave_out(
    out_ptr,
    row,
    offs_d,
    head_dim,
    i0,
    i1,
    i2,
    i3,
    so0,
    so1,
    so2,
    so3,
    sod,
    d_mask,
    is_contiguous: tl.constexpr,
):
    if is_contiguous:
        return tl.load(out_ptr + row * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    else:
        off = i0 * so0 + i1 * so1 + i2 * so2 + i3 * so3 + offs_d * sod
        return tl.load(out_ptr + off, mask=d_mask, other=0.0).to(tl.float32)


@triton.jit
def _store_wave_fill(
    out_dst_ptr,
    row,
    head_dim,
    val,
    block_d: tl.constexpr,
    exact_d: tl.constexpr,
):
    for d_start in range(0, 1 if exact_d else head_dim, block_d):
        offs_d = d_start + tl.arange(0, block_d)
        d_mask = True if exact_d else (offs_d < head_dim)
        vec = tl.full((block_d,), val, dtype=tl.float32)
        tl.store(
            out_dst_ptr + row * head_dim + offs_d,
            vec.to(out_dst_ptr.dtype.element_ty),
            mask=d_mask,
        )


@triton.jit
def _store_wave_copy(
    out_dst_ptr,
    out_src_ptr,
    row,
    head_dim,
    i0,
    i1,
    i2,
    i3,
    so0,
    so1,
    so2,
    so3,
    sod,
    block_d: tl.constexpr,
    exact_d: tl.constexpr,
    is_contiguous: tl.constexpr,
):
    for d_start in range(0, 1 if exact_d else head_dim, block_d):
        offs_d = d_start + tl.arange(0, block_d)
        d_mask = True if exact_d else (offs_d < head_dim)
        vec = _load_wave_out(
            out_src_ptr,
            row,
            offs_d,
            head_dim,
            i0,
            i1,
            i2,
            i3,
            so0,
            so1,
            so2,
            so3,
            sod,
            d_mask,
            is_contiguous,
        )
        tl.store(
            out_dst_ptr + row * head_dim + offs_d,
            vec.to(out_dst_ptr.dtype.element_ty),
            mask=d_mask,
        )


@triton.jit(do_not_specialize=["dim1", "dim2", "dim3", "head_dim"])
def _merge_two_attention_waves_kernel(  # noqa: PLR0912, PLR0915
    out_dst_ptr,
    lse_dst_ptr,
    out_0_ptr,
    lse_0_ptr,
    out_1_ptr,
    lse_1_ptr,
    out_2_ptr,
    lse_2_ptr,
    out_3_ptr,
    lse_3_ptr,
    out_4_ptr,
    lse_4_ptr,
    out_5_ptr,
    lse_5_ptr,
    out_6_ptr,
    lse_6_ptr,
    out_7_ptr,
    lse_7_ptr,
    dim1,
    dim2,
    dim3,
    head_dim,
    so0_0,
    so1_0,
    so2_0,
    so3_0,
    sod_0,
    sl0_0,
    sl1_0,
    sl2_0,
    sl3_0,
    so0_1,
    so1_1,
    so2_1,
    so3_1,
    sod_1,
    sl0_1,
    sl1_1,
    sl2_1,
    sl3_1,
    so0_2,
    so1_2,
    so2_2,
    so3_2,
    sod_2,
    sl0_2,
    sl1_2,
    sl2_2,
    sl3_2,
    so0_3,
    so1_3,
    so2_3,
    so3_3,
    sod_3,
    sl0_3,
    sl1_3,
    sl2_3,
    sl3_3,
    so0_4,
    so1_4,
    so2_4,
    so3_4,
    sod_4,
    sl0_4,
    sl1_4,
    sl2_4,
    sl3_4,
    so0_5,
    so1_5,
    so2_5,
    so3_5,
    sod_5,
    sl0_5,
    sl1_5,
    sl2_5,
    sl3_5,
    so0_6,
    so1_6,
    so2_6,
    so3_6,
    sod_6,
    sl0_6,
    sl1_6,
    sl2_6,
    sl3_6,
    so0_7,
    so1_7,
    so2_7,
    so3_7,
    sod_7,
    sl0_7,
    sl1_7,
    sl2_7,
    sl3_7,
    block_d: tl.constexpr,
    exact_d: tl.constexpr,
    num_waves: tl.constexpr,
    is_contiguous: tl.constexpr,
    in_place_0: tl.constexpr,
    single_pos_inf_wins: tl.constexpr = False,
):
    """Merge 1..8 attention waves and natural-log LSE values into ``(out_dst, lse_dst)``."""
    row = tl.program_id(0).to(tl.int64)
    if not is_contiguous:
        rem = row
        i3 = rem % dim3
        rem = rem // dim3
        i2 = rem % dim2
        rem = rem // dim2
        i1 = rem % dim1
        i0 = rem // dim1
    else:
        i0 = tl.zeros((), dtype=tl.int64)
        i1 = tl.zeros((), dtype=tl.int64)
        i2 = tl.zeros((), dtype=tl.int64)
        i3 = tl.zeros((), dtype=tl.int64)

    has_nan = tl.full((), False, dtype=tl.int1)
    has_any = tl.full((), False, dtype=tl.int1)
    max_l = tl.full((), -float("inf"), dtype=tl.float32)

    l0 = _load_wave_lse(lse_0_ptr, row, i0, i1, i2, i3, sl0_0, sl1_0, sl2_0, sl3_0, is_contiguous)
    pos_inf_0 = l0 == float("inf")
    nan0 = (l0 != l0) | (tl.full((), False, dtype=tl.int1) if single_pos_inf_wins else pos_inf_0)
    v0 = (l0 != -float("inf")) & (~nan0)
    has_nan = has_nan | nan0
    has_any = has_any | v0
    max_l = tl.maximum(max_l, tl.where(v0 & (~pos_inf_0), l0, -float("inf")))

    if num_waves >= 2:
        l1 = _load_wave_lse(
            lse_1_ptr, row, i0, i1, i2, i3, sl0_1, sl1_1, sl2_1, sl3_1, is_contiguous
        )
        pos_inf_1 = l1 == float("inf")
        nan1 = (l1 != l1) | (
            tl.full((), False, dtype=tl.int1) if single_pos_inf_wins else pos_inf_1
        )
        v1 = (l1 != -float("inf")) & (~nan1)
        has_nan = has_nan | nan1
        has_any = has_any | v1
        max_l = tl.maximum(max_l, tl.where(v1 & (~pos_inf_1), l1, -float("inf")))
        if single_pos_inf_wins and num_waves == 2:
            has_nan = has_nan | (pos_inf_0 & pos_inf_1)
    else:
        l1 = tl.zeros((), dtype=tl.float32)
        pos_inf_1 = tl.full((), False, dtype=tl.int1)
        v1 = tl.full((), False, dtype=tl.int1)

    if num_waves >= 3:
        l2 = _load_wave_lse(
            lse_2_ptr, row, i0, i1, i2, i3, sl0_2, sl1_2, sl2_2, sl3_2, is_contiguous
        )
        nan2 = (l2 != l2) | (l2 == float("inf"))
        v2 = (l2 != -float("inf")) & (~nan2)
        has_nan = has_nan | nan2
        has_any = has_any | v2
        max_l = tl.maximum(max_l, tl.where(v2, l2, -float("inf")))
    else:
        l2 = tl.zeros((), dtype=tl.float32)
        v2 = tl.full((), False, dtype=tl.int1)

    if num_waves >= 4:
        l3 = _load_wave_lse(
            lse_3_ptr, row, i0, i1, i2, i3, sl0_3, sl1_3, sl2_3, sl3_3, is_contiguous
        )
        nan3 = (l3 != l3) | (l3 == float("inf"))
        v3 = (l3 != -float("inf")) & (~nan3)
        has_nan = has_nan | nan3
        has_any = has_any | v3
        max_l = tl.maximum(max_l, tl.where(v3, l3, -float("inf")))
    else:
        l3 = tl.zeros((), dtype=tl.float32)
        v3 = tl.full((), False, dtype=tl.int1)

    if num_waves >= 5:
        l4 = _load_wave_lse(
            lse_4_ptr, row, i0, i1, i2, i3, sl0_4, sl1_4, sl2_4, sl3_4, is_contiguous
        )
        nan4 = (l4 != l4) | (l4 == float("inf"))
        v4 = (l4 != -float("inf")) & (~nan4)
        has_nan = has_nan | nan4
        has_any = has_any | v4
        max_l = tl.maximum(max_l, tl.where(v4, l4, -float("inf")))
    else:
        l4 = tl.zeros((), dtype=tl.float32)
        v4 = tl.full((), False, dtype=tl.int1)

    if num_waves >= 6:
        l5 = _load_wave_lse(
            lse_5_ptr, row, i0, i1, i2, i3, sl0_5, sl1_5, sl2_5, sl3_5, is_contiguous
        )
        nan5 = (l5 != l5) | (l5 == float("inf"))
        v5 = (l5 != -float("inf")) & (~nan5)
        has_nan = has_nan | nan5
        has_any = has_any | v5
        max_l = tl.maximum(max_l, tl.where(v5, l5, -float("inf")))
    else:
        l5 = tl.zeros((), dtype=tl.float32)
        v5 = tl.full((), False, dtype=tl.int1)

    if num_waves >= 7:
        l6 = _load_wave_lse(
            lse_6_ptr, row, i0, i1, i2, i3, sl0_6, sl1_6, sl2_6, sl3_6, is_contiguous
        )
        nan6 = (l6 != l6) | (l6 == float("inf"))
        v6 = (l6 != -float("inf")) & (~nan6)
        has_nan = has_nan | nan6
        has_any = has_any | v6
        max_l = tl.maximum(max_l, tl.where(v6, l6, -float("inf")))
    else:
        l6 = tl.zeros((), dtype=tl.float32)
        v6 = tl.full((), False, dtype=tl.int1)

    if num_waves >= 8:
        l7 = _load_wave_lse(
            lse_7_ptr, row, i0, i1, i2, i3, sl0_7, sl1_7, sl2_7, sl3_7, is_contiguous
        )
        nan7 = (l7 != l7) | (l7 == float("inf"))
        v7 = (l7 != -float("inf")) & (~nan7)
        has_nan = has_nan | nan7
        has_any = has_any | v7
        max_l = tl.maximum(max_l, tl.where(v7, l7, -float("inf")))
    else:
        l7 = tl.zeros((), dtype=tl.float32)
        v7 = tl.full((), False, dtype=tl.int1)

    if single_pos_inf_wins and (num_waves == 2):
        two_a_wins = (v0 & (~v1)) | (pos_inf_0 & (~pos_inf_1))
        two_b_wins = ((~v0) & v1) | (pos_inf_1 & (~pos_inf_0))
    elif num_waves == 2:
        two_a_wins = v0 & (~v1)
        two_b_wins = (~v0) & v1
    else:
        two_a_wins = tl.full((), False, dtype=tl.int1)
        two_b_wins = tl.full((), False, dtype=tl.int1)
    skip_inplace = (two_a_wins & (~has_nan)) if (in_place_0 and num_waves == 2) else False
    if not skip_inplace:
        if has_nan:
            tl.store(lse_dst_ptr + row, float("nan"))
            _store_wave_fill(out_dst_ptr, row, head_dim, float("nan"), block_d, exact_d)
        elif not has_any:
            tl.store(lse_dst_ptr + row, -float("inf"))
            _store_wave_fill(out_dst_ptr, row, head_dim, 0.0, block_d, exact_d)
        elif num_waves == 1:
            tl.store(lse_dst_ptr + row, l0)
            _store_wave_copy(
                out_dst_ptr,
                out_0_ptr,
                row,
                head_dim,
                i0,
                i1,
                i2,
                i3,
                so0_0,
                so1_0,
                so2_0,
                so3_0,
                sod_0,
                block_d,
                exact_d,
                is_contiguous,
            )
        elif two_a_wins:
            tl.store(lse_dst_ptr + row, l0)
            _store_wave_copy(
                out_dst_ptr,
                out_0_ptr,
                row,
                head_dim,
                i0,
                i1,
                i2,
                i3,
                so0_0,
                so1_0,
                so2_0,
                so3_0,
                sod_0,
                block_d,
                exact_d,
                is_contiguous,
            )
        elif two_b_wins:
            tl.store(lse_dst_ptr + row, l1)
            _store_wave_copy(
                out_dst_ptr,
                out_1_ptr,
                row,
                head_dim,
                i0,
                i1,
                i2,
                i3,
                so0_1,
                so1_1,
                so2_1,
                so3_1,
                sod_1,
                block_d,
                exact_d,
                is_contiguous,
            )
        else:
            w0 = tl.where(v0, tl.math.exp(l0 - max_l), 0.0)
            denom = w0
            w1 = tl.where(v1, tl.math.exp(l1 - max_l), 0.0) if num_waves >= 2 else 0.0
            if num_waves >= 2:
                denom = denom + w1
            w2 = tl.where(v2, tl.math.exp(l2 - max_l), 0.0) if num_waves >= 3 else 0.0
            if num_waves >= 3:
                denom = denom + w2
            w3 = tl.where(v3, tl.math.exp(l3 - max_l), 0.0) if num_waves >= 4 else 0.0
            if num_waves >= 4:
                denom = denom + w3
            w4 = tl.where(v4, tl.math.exp(l4 - max_l), 0.0) if num_waves >= 5 else 0.0
            if num_waves >= 5:
                denom = denom + w4
            w5 = tl.where(v5, tl.math.exp(l5 - max_l), 0.0) if num_waves >= 6 else 0.0
            if num_waves >= 6:
                denom = denom + w5
            w6 = tl.where(v6, tl.math.exp(l6 - max_l), 0.0) if num_waves >= 7 else 0.0
            if num_waves >= 7:
                denom = denom + w6
            w7 = tl.where(v7, tl.math.exp(l7 - max_l), 0.0) if num_waves >= 8 else 0.0
            if num_waves >= 8:
                denom = denom + w7

            merged_lse = max_l + tl.math.log(denom)
            tl.store(lse_dst_ptr + row, merged_lse)

            nw0 = w0 / denom
            nw1 = w1 / denom if num_waves >= 2 else 0.0
            nw2 = w2 / denom if num_waves >= 3 else 0.0
            nw3 = w3 / denom if num_waves >= 4 else 0.0
            nw4 = w4 / denom if num_waves >= 5 else 0.0
            nw5 = w5 / denom if num_waves >= 6 else 0.0
            nw6 = w6 / denom if num_waves >= 7 else 0.0
            nw7 = w7 / denom if num_waves >= 8 else 0.0

            for d_start in range(0, 1 if exact_d else head_dim, block_d):
                offs_d = d_start + tl.arange(0, block_d)
                d_mask = True if exact_d else (offs_d < head_dim)
                acc = tl.zeros((block_d,), dtype=tl.float32)
                if v0:
                    acc += nw0 * _load_wave_out(
                        out_0_ptr,
                        row,
                        offs_d,
                        head_dim,
                        i0,
                        i1,
                        i2,
                        i3,
                        so0_0,
                        so1_0,
                        so2_0,
                        so3_0,
                        sod_0,
                        d_mask,
                        is_contiguous,
                    )
                if num_waves >= 2:
                    if v1:
                        acc += nw1 * _load_wave_out(
                            out_1_ptr,
                            row,
                            offs_d,
                            head_dim,
                            i0,
                            i1,
                            i2,
                            i3,
                            so0_1,
                            so1_1,
                            so2_1,
                            so3_1,
                            sod_1,
                            d_mask,
                            is_contiguous,
                        )
                if num_waves >= 3:
                    if v2:
                        acc += nw2 * _load_wave_out(
                            out_2_ptr,
                            row,
                            offs_d,
                            head_dim,
                            i0,
                            i1,
                            i2,
                            i3,
                            so0_2,
                            so1_2,
                            so2_2,
                            so3_2,
                            sod_2,
                            d_mask,
                            is_contiguous,
                        )
                if num_waves >= 4:
                    if v3:
                        acc += nw3 * _load_wave_out(
                            out_3_ptr,
                            row,
                            offs_d,
                            head_dim,
                            i0,
                            i1,
                            i2,
                            i3,
                            so0_3,
                            so1_3,
                            so2_3,
                            so3_3,
                            sod_3,
                            d_mask,
                            is_contiguous,
                        )
                if num_waves >= 5:
                    if v4:
                        acc += nw4 * _load_wave_out(
                            out_4_ptr,
                            row,
                            offs_d,
                            head_dim,
                            i0,
                            i1,
                            i2,
                            i3,
                            so0_4,
                            so1_4,
                            so2_4,
                            so3_4,
                            sod_4,
                            d_mask,
                            is_contiguous,
                        )
                if num_waves >= 6:
                    if v5:
                        acc += nw5 * _load_wave_out(
                            out_5_ptr,
                            row,
                            offs_d,
                            head_dim,
                            i0,
                            i1,
                            i2,
                            i3,
                            so0_5,
                            so1_5,
                            so2_5,
                            so3_5,
                            sod_5,
                            d_mask,
                            is_contiguous,
                        )
                if num_waves >= 7:
                    if v6:
                        acc += nw6 * _load_wave_out(
                            out_6_ptr,
                            row,
                            offs_d,
                            head_dim,
                            i0,
                            i1,
                            i2,
                            i3,
                            so0_6,
                            so1_6,
                            so2_6,
                            so3_6,
                            sod_6,
                            d_mask,
                            is_contiguous,
                        )
                if num_waves >= 8:
                    if v7:
                        acc += nw7 * _load_wave_out(
                            out_7_ptr,
                            row,
                            offs_d,
                            head_dim,
                            i0,
                            i1,
                            i2,
                            i3,
                            so0_7,
                            so1_7,
                            so2_7,
                            so3_7,
                            sod_7,
                            d_mask,
                            is_contiguous,
                        )
                tl.store(
                    out_dst_ptr + row * head_dim + offs_d,
                    acc.to(out_dst_ptr.dtype.element_ty),
                    mask=d_mask,
                )


def _coalesce_4d_strides(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
    leading_dims: tuple[int, ...],
) -> tuple[tuple[int, int, int], list[tuple[int, ...]]] | None:
    """Coalesce leading dimensions to <= 4 axes across all strided waves, or return None."""
    dims = list(leading_dims)
    out_strides = [list(out.stride()[:-1]) for out in outputs]
    lse_strides = [list(lse.stride()) for lse in lses]
    idx = len(dims) - 2
    while len(dims) > 4 and idx >= 0:
        can_merge = all(
            dims[idx] == 1 or dims[idx + 1] == 1 or s[idx] == s[idx + 1] * dims[idx + 1]
            for s in (*out_strides, *lse_strides)
        )
        if can_merge:
            merged_dim = dims[idx] * dims[idx + 1]
            for s in (*out_strides, *lse_strides):
                kept = (
                    s[idx + 1] if dims[idx] == 1 else (s[idx] if dims[idx + 1] == 1 else s[idx + 1])
                )
                s[idx : idx + 2] = [kept]
            dims[idx : idx + 2] = [merged_dim]
            if idx >= len(dims) - 1:
                idx -= 1
        else:
            idx -= 1
    if len(dims) > 4:
        return None
    pad = 4 - len(dims)
    padded_dims = ([1] * pad) + dims
    per_wave_strides: list[tuple[int, ...]] = []
    for out, os, ls in zip(outputs, out_strides, lse_strides, strict=True):
        padded_os = ([0] * pad) + os
        padded_ls = ([0] * pad) + ls
        per_wave_strides.append((*padded_os, out.stride(-1), *padded_ls))
    return (padded_dims[1], padded_dims[2], padded_dims[3]), per_wave_strides


def can_launch_merge_attention_lse(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
) -> bool:
    """Return whether ``outputs`` and ``lses`` can run in the fused Triton merge kernel."""
    leading_dims = tuple(outputs[0].shape[:-1])
    if len(leading_dims) <= 4 or all(t.is_contiguous() for t in (*outputs, *lses)):
        return True
    return _coalesce_4d_strides(outputs, lses, leading_dims) is not None


def launch_merge_attention_lse_chunk(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
    out_dst: torch.Tensor,
    lse_dst: torch.Tensor,
    *,
    in_place_0: bool = False,
    single_pos_inf_wins: bool = False,
) -> None:
    """Merge 1..8 waves into contiguous ``(out_dst, lse_dst)`` in one Triton launch."""
    num_waves = len(outputs)
    assert 1 <= num_waves <= _MAX_MERGE_WAVES
    head_dim = out_dst.shape[-1]
    leading_dims = tuple(out_dst.shape[:-1])
    total_rows = 1
    for dim in leading_dims:
        total_rows *= dim
    if total_rows == 0 or head_dim == 0:
        return

    block_d = min(256, max(16, triton.next_power_of_2(head_dim)))
    exact_d = head_dim == block_d
    is_contiguous = all(t.is_contiguous() for t in (*outputs, *lses))
    if is_contiguous:
        dim1 = 1
        dim2 = 1
        dim3 = 1
        stride_args: list[object] = [None] * (_MAX_MERGE_WAVES * 9)
    else:
        coalesced = _coalesce_4d_strides(outputs, lses, leading_dims)
        assert coalesced is not None
        (dim1, dim2, dim3), per_wave = coalesced
        stride_args = []
        for idx in range(_MAX_MERGE_WAVES):
            if idx < num_waves:
                stride_args.extend(per_wave[idx])
            else:
                stride_args.extend([None] * 9)

    ptr_args: list[object] = []
    for idx in range(_MAX_MERGE_WAVES):
        if idx < num_waves:
            ptr_args.extend((outputs[idx], lses[idx]))
        else:
            ptr_args.extend((None, None))

    with device_context(out_dst.device):
        _merge_two_attention_waves_kernel[(total_rows,)](
            out_dst,
            lse_dst,
            *ptr_args,
            dim1,
            dim2,
            dim3,
            head_dim,
            *stride_args,
            block_d=block_d,
            exact_d=exact_d,
            num_waves=num_waves,
            is_contiguous=is_contiguous,
            in_place_0=in_place_0,
            single_pos_inf_wins=single_pos_inf_wins,
            num_warps=REDUCE_NUM_WARPS,
        )


def _merge_two_attention_waves(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> None:
    launch_merge_attention_lse_chunk(
        (out_a, out_b),
        (lse_a, lse_b),
        out_a,
        lse_a,
        in_place_0=True,
        single_pos_inf_wins=True,
    )


def validate_prefill_backend(
    prefill_backend: Literal["amd", "nvidia"],
    *,
    grouped_qk: bool,
    pack_wmma: bool,
) -> None:
    """Validate prefill backend compatibility with grouped QK scales and WMMA packing."""
    if prefill_backend not in ("amd", "nvidia"):
        raise ValueError(f"prefill_backend must be 'amd' or 'nvidia', got {prefill_backend!r}")
    if pack_wmma != (prefill_backend == "amd"):
        raise ValueError(
            f"pack_wmma ({pack_wmma}) must match prefill_backend ({prefill_backend!r})"
        )
    if prefill_backend == "amd" and not grouped_qk:
        raise ValueError("AMD chunked prefill requires grouped_qk=True")


def pool_slot_strides(
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


def _launch_splitkv_reduce(
    po_view: torch.Tensor,
    pl_view: torch.Tensor,
    pb_view: torch.Tensor,
    pld_view: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    batch: int,
    q_heads: int,
    seq_q: int,
    num_splits: int,
    head_dim: int,
    use_lds_only_fence: bool = False,
) -> None:
    """Reduce Split-KV partial tensors ordered as ``(B, H, S, split[, D])``."""
    if seq_q <= 8:
        block_r = 1
        block_s = 16
        grid_rows = seq_q
    else:
        block_r = 8
        block_s = 1
        grid_rows = triton.cdiv(seq_q, block_r)
    extra_opts = amd_compiler_workarounds.lds_only_fence_options(block_s > 1 and use_lds_only_fence)
    with device_context(output.device):
        _piper_decode_splitkv_reduce_kernel[(batch * q_heads * grid_rows,)](
            po_view,
            pl_view,
            pb_view,
            pld_view,
            output,
            lse,
            q_heads,
            seq_q,
            num_splits,
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
            block_r=block_r,
            block_s=block_s,
            num_warps=REDUCE_NUM_WARPS,
            **extra_opts,
        )


@triton.jit
def _split_bf16_pair(value):
    """Split finite FP32 values into rounded BF16 high bits plus exact residual."""
    hi = value.to(tl.bfloat16)
    return hi, (value - hi.to(tl.float32)).to(tl.bfloat16)


@triton.jit(
    do_not_specialize=[
        "kv_heads",
        "gqa_group",
        "q_tiles",
        "seq_q",
        "tail_cap",
        "int8_splits",
        "tail_splits",
        "stride_plp",
        "window_left",
        "window_right",
    ]
)
def _piper_decode_tail_split_kernel(  # noqa: PLR0912, PLR0915
    query_ptr,
    tail_key_ptr,
    tail_value_ptr,
    tail_len_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    partial_bias_ptr,
    partial_log_denom_ptr,
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
    stride_plp,
    window_mode: tl.constexpr,
    head_dim: tl.constexpr,
    block_qh: tl.constexpr,
    block_n: tl.constexpr,
    qk_precision: tl.constexpr,
    pv_precision: tl.constexpr,
    window_left=-1,
    window_right=-1,
):
    """Evaluate one Split-KV slice of the live BF16/FP16 tail for a GQA head group."""
    is_causal_mode: tl.constexpr = window_mode == "causal"
    is_windowed_mode: tl.constexpr = window_mode == "windowed"

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
    active_tail32 = active_tail.to(tl.int32)
    total_tiles = tl.cdiv(active_tail32, block_n)
    if is_windowed_mode:
        eff_wl_t = tl.minimum(window_left.to(tl.int64), active_tail)
        eff_wr_t = tl.minimum(window_right.to(tl.int64), active_tail + seq_q.to(tl.int64))
        q_tile_first = q_tile * block_qh
        q_tile_last = tl.minimum((q_tile + 1) * block_qh, gqa_group * seq_q) - 1
        same_q_head = (q_tile_first // seq_q) == (q_tile_last // seq_q)
        q_idx_min = tl.where(same_q_head, q_tile_first % seq_q, 0)
        q_idx_max = tl.where(same_q_head, q_tile_last % seq_q, seq_q - 1)
        q_tail_min = (active_tail32 - seq_q) + q_idx_min
        q_tail_max = (active_tail32 - seq_q) + q_idx_max
        k_min = (
            tl.minimum(
                active_tail,
                tl.maximum(0, q_tail_min.to(tl.int64) - eff_wl_t),
            ).to(tl.int32)
            if window_left >= 0
            else tl.zeros((), dtype=tl.int32)
        )
        if window_right >= 0:
            k_max_excl = tl.minimum(
                active_tail,
                tl.maximum(0, q_tail_max.to(tl.int64) + eff_wr_t + 1),
            ).to(tl.int32)
        else:
            k_max_excl = active_tail32
        first_vis_tile = tl.minimum(total_tiles, k_min // block_n)
        last_vis_tile = tl.maximum(
            first_vis_tile, tl.minimum(total_tiles, tl.cdiv(k_max_excl, block_n))
        )
    else:
        first_vis_tile = tl.zeros((), dtype=tl.int32)
        last_vis_tile = total_tiles

    visible_tiles = last_vis_tile - first_vis_tile
    tiles_per_split = tl.cdiv(visible_tiles, tail_splits)
    tile_start = first_vis_tile + tail_split_id * tiles_per_split
    tile_end = tl.minimum(last_vis_tile, tile_start + tiles_per_split)

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
        q_raw = tl.load(q_ptrs, mask=qh_valid0[:, None], other=0.0)
        q_dtype = tail_key_ptr.dtype.element_ty if qk_precision == "native" else tl.float32
        q_qk = q_raw.to(q_dtype)
        offsets_d64 = tl.arange(0, 64)
        if head_dim == 128:
            q_lo, q_hi = tl.split(tl.permute(tl.reshape(q_qk, (block_qh, 2, 64)), (0, 2, 1)))
        else:
            q_lo = q_qk
            q_hi = q_qk

        offsets_n = tl.arange(0, block_n)
        tk_base = tail_key_ptr + b64 * stride_tk_b + kv_head64 * stride_tk_h
        tv_base = tail_value_ptr + b64 * stride_tv_b + kv_head64 * stride_tv_h
        scaled_log2 = scale * _LOG2_E
        if is_causal_mode:
            q_tail_pos = (active_tail32 - seq_q) + q_idx0
            tail_win_lo = tl.zeros((block_qh,), dtype=tl.int32)
            tail_win_hi = q_tail_pos
        elif is_windowed_mode:
            eff_wl_tw = tl.minimum(window_left.to(tl.int64), active_tail)
            eff_wr_tw = tl.minimum(window_right.to(tl.int64), active_tail + seq_q.to(tl.int64))
            q_tail_pos64 = (active_tail32 - seq_q).to(tl.int64) + q_idx0.to(tl.int64)
            tail_win_lo = (
                tl.minimum(
                    active_tail,
                    tl.maximum(0, q_tail_pos64 - eff_wl_tw),
                ).to(tl.int32)
                if window_left >= 0
                else tl.zeros((block_qh,), dtype=tl.int32)
            )
            tail_win_hi = (
                tl.minimum(
                    active_tail,
                    tl.maximum(-1, q_tail_pos64 + eff_wr_tw),
                ).to(tl.int32)
                if window_right >= 0
                else tl.full((block_qh,), active_tail32, dtype=tl.int32)
            )
        else:
            tail_win_lo = tl.zeros((block_qh,), dtype=tl.int32)
            tail_win_hi = tl.zeros((block_qh,), dtype=tl.int32)

        for tile_idx in tl.range(tile_start, tile_end, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
            rows32 = tile_idx * block_n + offsets_n
            valid_n = rows32 < active_tail32
            rows = rows32.to(tl.int64)
            if head_dim == 128:
                k_lo = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + offsets_d64[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                k_hi = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + (offsets_d64[None, :] + 64),
                    mask=valid_n[:, None],
                    other=0.0,
                )
                if qk_precision == "native":
                    scores = tl.dot(q_lo, tl.trans(k_lo.to(q_lo.dtype)))
                    scores = tl.dot(q_hi, tl.trans(k_hi.to(q_hi.dtype)), acc=scores)
                else:
                    scores = tl.dot(
                        q_lo, tl.trans(k_lo.to(tl.float32)), input_precision=qk_precision
                    )
                    scores = tl.dot(
                        q_hi,
                        tl.trans(k_hi.to(tl.float32)),
                        acc=scores,
                        input_precision=qk_precision,
                    )
            else:
                k_tile = tl.load(
                    tk_base + rows[:, None] * stride_tk_n + offsets_d[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                if qk_precision == "native":
                    scores = tl.dot(q_lo, tl.trans(k_tile.to(q_lo.dtype)))
                else:
                    scores = tl.dot(
                        q_lo, tl.trans(k_tile.to(tl.float32)), input_precision=qk_precision
                    )
            scores *= scaled_log2
            valid_mask = tl.broadcast_to(valid_n[None, :], (block_qh, block_n))
            if is_causal_mode:
                valid_mask = valid_mask & (rows32[None, :] <= tail_win_hi[:, None])
            elif is_windowed_mode:
                valid_mask = valid_mask & (
                    (rows32[None, :] >= tail_win_lo[:, None])
                    & (rows32[None, :] <= tail_win_hi[:, None])
                )

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
            if pv_precision == "bf16x2":
                if v_tile.dtype == tl.float16:
                    acc = tl.dot(p.to(tl.float16), v_tile, acc=acc)
                else:
                    p_hi, p_lo = _split_bf16_pair(p)
                    v_native = v_tile.to(tl.bfloat16)
                    acc = tl.dot(p_hi, v_native, acc=acc)
                    acc = tl.dot(p_lo, v_native, acc=acc)
            else:
                acc = tl.dot(p, v_tile.to(tl.float32), acc=acc, input_precision=pv_precision)
            m_i = m_new

    has_any = m_i != -float("inf")
    safe_l = tl.where(has_any & (l_i != 0.0), l_i, 1.0)
    out_split = tl.where(has_any[:, None], acc / safe_l[:, None], 0.0)
    lse_split = tl.where(has_any, m_i, -float("inf"))
    log_denom_split = tl.where(has_any, tl.math.log2(safe_l), 0.0)

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
        + split64 * stride_plp.to(tl.int64)
    )
    tl.store(partial_lse_ptr + pl_offsets, lse_split, mask=qh_valid)
    tl.store(partial_bias_ptr + pl_offsets, 0.0, mask=qh_valid)
    tl.store(partial_log_denom_ptr + pl_offsets, log_denom_split, mask=qh_valid)


def launch_direct_tail_split_decode(
    query: torch.Tensor,
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    tail_len: torch.Tensor,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    partial_bias: torch.Tensor,
    partial_log_denom: torch.Tensor,
    *,
    scale: float,
    kv_heads: int,
    gqa_group: int,
    q_tiles: int,
    seq_q: int,
    tail_cap: int,
    int8_splits: int,
    tail_splits: int,
    is_causal: bool,
    head_dim: int,
    schedule: DecodeSchedule,
    window_mode: Literal["none", "causal", "windowed"] | None = None,
    window_left: int = -1,
    window_right: int = -1,
) -> None:
    """Launch Split-KV direct BF16/FP16 tail attention into ``partial_*[split >= int8_splits]``."""
    resolved_mode: Literal["none", "causal", "windowed"] = (
        ("causal" if is_causal else "none") if window_mode is None else window_mode
    )
    batch = query.shape[0]
    stride_qb = 0 if batch <= 1 else query.stride(0)
    stride_tk_b = 0 if batch <= 1 else tail_key.stride(0)
    stride_tv_b = 0 if batch <= 1 else tail_value.stride(0)
    compiler_options = dict(schedule.compiler_options)
    with device_context(query.device):
        _piper_decode_tail_split_kernel[(batch * kv_heads * q_tiles, tail_splits)](
            query,
            tail_key,
            tail_value,
            tail_len,
            partial_out,
            partial_lse,
            partial_bias,
            partial_log_denom,
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
            partial_lse.stride(3),
            window_mode=resolved_mode,
            head_dim=head_dim,
            block_qh=BLOCK_QH,
            block_n=schedule.block_n,
            qk_precision=schedule.tail_qk_precision,
            pv_precision=schedule.pv_precision,
            window_left=window_left,
            window_right=window_right,
            num_warps=schedule.num_warps,
            num_stages=schedule.num_stages,
            **compiler_options,
        )


def _select_frozen_prefill_splits(
    device: torch.device,
    *,
    prefill_backend: Literal["amd", "nvidia"],
    batch: int,
    q_heads: int,
    seq_q: int,
    head_dim: int,
    max_chunks: int,
    active_chunks_hint: int,
    allow_subchunk_single: bool = False,
    logical_seq_q: int = 0,
    window_left: int = -1,
) -> int:
    sm_count = device_sm_count(device)
    base_blocks = max(1, batch * q_heads * triton.cdiv(seq_q, 64))
    hint_chunks = max_chunks if active_chunks_hint <= 0 else min(max_chunks, active_chunks_hint)
    eff_seq_q = logical_seq_q if logical_seq_q > 0 else seq_q
    if window_left >= 0:
        hint_chunks = min(
            hint_chunks, triton.cdiv(eff_seq_q + window_left, _quantization.CENTER_CHUNK)
        )
    policy = _amd_policy if prefill_backend == "amd" else _nvidia_policy
    return policy.select_frozen_prefill_splits(
        base_blocks,
        hint_chunks,
        sm_count,
        batch * q_heads * seq_q * head_dim * 4,
        allow_subchunk_single=allow_subchunk_single,
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
    per_row_query_scale: bool = False,
    allow_subchunk_single: bool = False,
    frozen_splits: int | None = None,
    total_splits: int | None = None,
    partial_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    reduce_heads_seq: tuple[int, int] | None = None,
    logical_seq_q: int = 0,
    tail_len: torch.Tensor | None = None,
    tail_cap: int = 0,
    window_mode: Literal["none", "causal", "windowed"] = "none",
    window_left: int = -1,
    window_right: int = -1,
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
    ) = pool_slot_strides(
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

    resolved_splits = (
        _select_frozen_prefill_splits(
            query_int8.device,
            prefill_backend=prefill_backend,
            batch=batch,
            q_heads=q_heads,
            seq_q=seq_q,
            head_dim=head_dim,
            max_chunks=max_chunks,
            active_chunks_hint=active_chunks_hint,
            allow_subchunk_single=allow_subchunk_single,
            logical_seq_q=logical_seq_q,
            window_left=window_left,
        )
        if frozen_splits is None
        else frozen_splits
    )
    resolved_total = resolved_splits if total_splits is None else total_splits
    has_splits = partial_targets is not None or resolved_splits > 1

    out_target = output
    lse_target = lse
    partial_out: torch.Tensor | None = None
    partial_lse: torch.Tensor | None = None
    partial_bias: torch.Tensor | None = None
    partial_log_denom: torch.Tensor | None = None
    bias_target: torch.Tensor | None = None
    log_denom_target: torch.Tensor | None = None
    if partial_targets is not None:
        out_target, lse_target, bias_target, log_denom_target = partial_targets
    elif resolved_splits > 1:
        partial_out = torch.empty(
            (batch, resolved_splits, q_heads, seq_q, head_dim),
            device=output.device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            (batch, resolved_splits, q_heads, seq_q),
            device=output.device,
            dtype=torch.float32,
        )
        partial_bias = torch.empty_like(partial_lse)
        partial_log_denom = torch.empty_like(partial_lse)
        out_target = partial_out.view(batch * resolved_splits, q_heads, seq_q, head_dim)
        lse_target = partial_lse.view(batch * resolved_splits, q_heads, seq_q)
        bias_target = partial_bias.view(batch * resolved_splits, q_heads, seq_q)
        log_denom_target = partial_log_denom.view(batch * resolved_splits, q_heads, seq_q)

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
            split_chunks=resolved_splits,
            total_splits=resolved_total,
            has_chunk_splits=has_splits,
            per_row_query_scale=per_row_query_scale,
            partial_bias=bias_target,
            partial_log_denom=log_denom_target,
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_left=window_left,
            window_right=window_right,
            logical_seq_q=logical_seq_q,
            window_mode=window_mode,
        )
    else:
        from ._nvidia import prefill as nvidia_prefill  # noqa: PLC0415

        nvidia_prefill.launch_prefill_kernel(
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
            split_chunks=resolved_splits,
            total_splits=resolved_total,
            has_chunk_splits=has_splits,
            per_row_query_scale=per_row_query_scale,
            partial_bias=bias_target,
            partial_log_denom=log_denom_target,
            target=target,
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_mode=window_mode,
            window_left=window_left,
            window_right=window_right,
            logical_seq_q=logical_seq_q,
        )
    if (
        partial_out is not None
        and partial_lse is not None
        and partial_bias is not None
        and partial_log_denom is not None
    ):
        reduce_heads, reduce_seq_q = (
            (q_heads, seq_q) if reduce_heads_seq is None else reduce_heads_seq
        )
        reduce_shape = (batch, resolved_splits, reduce_heads, reduce_seq_q)
        po_view = partial_out.view(*reduce_shape, head_dim).permute(0, 2, 3, 1, 4)
        pl_view, pb_view, pld_view = (
            t.view(reduce_shape).permute(0, 2, 3, 1)
            for t in (partial_lse, partial_bias, partial_log_denom)
        )
        _launch_splitkv_reduce(
            po_view,
            pl_view,
            pb_view,
            pld_view,
            output.view(batch, reduce_heads, reduce_seq_q, head_dim),
            lse.view(batch, reduce_heads, reduce_seq_q),
            batch=batch,
            q_heads=reduce_heads,
            seq_q=reduce_seq_q,
            num_splits=resolved_splits,
            head_dim=head_dim,
            use_lds_only_fence=(prefill_backend == "amd"),
        )


def _quantize_query(
    query: torch.Tensor,
    scale: float,
    *,
    prefill_backend: Literal["amd", "nvidia"],
    grouped_qk: bool,
    per_row: bool,
    storage_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prefill_backend == "amd":
        from piper_kernels.attention.kernels.piper._amd import (  # noqa: PLC0415
            quantization as amd_quantization,
        )

        return amd_quantization.prepare_query(
            query,
            scale,
            per_row=per_row,
            storage_sequence_length=storage_length,
        )
    with device_context(query.device):
        return qk_quantization.prepare_query(
            query,
            scale,
            grouped=grouped_qk and not per_row,
            per_row=per_row,
            storage_query_length=storage_length,
        )


def _resolve_kv_window_modes(
    *,
    is_causal: bool,
    window_left: int,
    window_right: int,
) -> tuple[
    Literal["none", "causal", "windowed"],
    int,
    int,
    Literal["none", "causal", "windowed"],
    int,
    int,
]:
    """Resolve frozen-pool and live-tail window modes and bounds from decode/prefill args."""
    has_explicit_window = window_left >= 0 or window_right >= 0
    eff_left = window_left
    eff_right = 0 if is_causal else window_right
    frozen_left = eff_left if has_explicit_window else -1
    frozen_right = eff_right if has_explicit_window else -1
    frozen_mode: Literal["none", "causal", "windowed"] = (
        "none"
        if not has_explicit_window
        else ("causal" if (frozen_left < 0 and frozen_right == 0) else "windowed")
    )
    tail_left = eff_left if has_explicit_window else -1
    tail_right = eff_right if has_explicit_window else (0 if is_causal else -1)
    tail_mode: Literal["none", "causal", "windowed"] = (
        "none"
        if (tail_left < 0 and tail_right < 0)
        else ("causal" if (tail_left < 0 and tail_right == 0) else "windowed")
    )
    return frozen_mode, frozen_left, frozen_right, tail_mode, tail_left, tail_right


def _run_small_q_decode(
    query: torch.Tensor,
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
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float,
    is_causal: bool,
    grouped_qk: bool,
    active_chunks_hint: int,
    prefill_backend: Literal["amd", "nvidia"],
    target: AcceleratorTarget | None,
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute native GQA-packed INT8 pool decode plus optional direct BF16/FP16 tail."""
    batch, q_heads, seq_q, head_dim = query.shape
    num_slots, kv_heads, _, _ = key_int8.shape
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    tail_cap = 0 if tail_key is None else tail_key.shape[2]
    has_chunks = max_chunks > 0 and num_slots > 0
    has_tail = tail_cap > 0
    gqa_group = q_heads // kv_heads
    packed_m = gqa_group * seq_q
    storage_m = triton.cdiv(packed_m, 64) * 64
    frozen_mode, frozen_left, frozen_right, tail_mode, tail_left, tail_right = (
        _resolve_kv_window_modes(
            is_causal=is_causal,
            window_left=window_left,
            window_right=window_right,
        )
    )

    if has_chunks and not has_tail:
        query_int8, query_scale = _quantize_query(
            query.reshape(batch, kv_heads, packed_m, head_dim),
            scale,
            prefill_backend=prefill_backend,
            grouped_qk=grouped_qk,
            per_row=True,
            storage_length=storage_m,
        )
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
            output.view(batch, kv_heads, packed_m, head_dim),
            lse.view(batch, kv_heads, packed_m),
            seq_q=packed_m,
            grouped_qk=grouped_qk,
            prefill_backend=prefill_backend,
            active_chunks_hint=active_chunks_hint,
            target=target,
            per_row_query_scale=True,
            allow_subchunk_single=True,
            reduce_heads_seq=(q_heads, seq_q),
            logical_seq_q=seq_q,
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_mode=frozen_mode,
            window_left=frozen_left,
            window_right=frozen_right,
        )
        return output, lse

    resolved_target = AcceleratorTarget.from_device(query.device) if target is None else target
    schedule = select_decode_schedule(resolved_target, head_dim)
    sm_count = device_sm_count(query.device)
    _, tail_splits = select_split_counts(
        batch=batch,
        kv_heads=kv_heads,
        gqa_group=gqa_group,
        seq_q=seq_q,
        max_chunks=0,
        num_slots=0,
        tail_cap=tail_cap,
        active_chunks_hint=0,
        sm_count=sm_count,
        schedule=schedule,
        window_left=tail_left,
    )
    frozen_splits = (
        _select_frozen_prefill_splits(
            query.device,
            prefill_backend=prefill_backend,
            batch=batch,
            q_heads=kv_heads,
            seq_q=packed_m,
            head_dim=head_dim,
            max_chunks=max_chunks,
            active_chunks_hint=active_chunks_hint,
            allow_subchunk_single=True,
            logical_seq_q=seq_q,
            window_left=frozen_left,
        )
        if has_chunks
        else 0
    )
    total_splits = frozen_splits + tail_splits
    partial_out = torch.empty(
        (batch, total_splits, q_heads, seq_q, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (batch, total_splits, q_heads, seq_q),
        device=query.device,
        dtype=torch.float32,
    )
    partial_bias = torch.empty_like(partial_lse)
    partial_log_denom = torch.empty_like(partial_lse)

    if has_chunks:
        query_int8, query_scale = _quantize_query(
            query.reshape(batch, kv_heads, packed_m, head_dim),
            scale,
            prefill_backend=prefill_backend,
            grouped_qk=grouped_qk,
            per_row=True,
            storage_length=storage_m,
        )
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
            output.view(batch, kv_heads, packed_m, head_dim),
            lse.view(batch, kv_heads, packed_m),
            seq_q=packed_m,
            grouped_qk=grouped_qk,
            prefill_backend=prefill_backend,
            active_chunks_hint=active_chunks_hint,
            target=resolved_target,
            per_row_query_scale=True,
            allow_subchunk_single=True,
            frozen_splits=frozen_splits,
            total_splits=total_splits,
            partial_targets=(
                partial_out.view(batch * total_splits, kv_heads, packed_m, head_dim),
                partial_lse.view(batch * total_splits, kv_heads, packed_m),
                partial_bias.view(batch * total_splits, kv_heads, packed_m),
                partial_log_denom.view(batch * total_splits, kv_heads, packed_m),
            ),
            logical_seq_q=seq_q,
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_mode=frozen_mode,
            window_left=frozen_left,
            window_right=frozen_right,
        )

    po_view = partial_out.permute(0, 2, 3, 1, 4)
    pl_view = partial_lse.permute(0, 2, 3, 1)
    pb_view = partial_bias.permute(0, 2, 3, 1)
    pld_view = partial_log_denom.permute(0, 2, 3, 1)
    q_tiles = triton.cdiv(packed_m, BLOCK_QH)
    assert tail_key is not None
    assert tail_value is not None
    assert tail_len is not None
    launch_direct_tail_split_decode(
        query,
        tail_key,
        tail_value,
        tail_len,
        po_view,
        pl_view,
        pb_view,
        pld_view,
        scale=scale,
        kv_heads=kv_heads,
        gqa_group=gqa_group,
        q_tiles=q_tiles,
        seq_q=seq_q,
        tail_cap=tail_cap,
        int8_splits=frozen_splits,
        tail_splits=tail_splits,
        is_causal=is_causal,
        head_dim=head_dim,
        schedule=schedule,
        window_mode=tail_mode,
        window_left=tail_left,
        window_right=tail_right,
    )
    _launch_splitkv_reduce(
        po_view,
        pl_view,
        pb_view,
        pld_view,
        output,
        lse,
        batch=batch,
        q_heads=q_heads,
        seq_q=seq_q,
        num_splits=total_splits,
        head_dim=head_dim,
        use_lds_only_fence=(prefill_backend == "amd"),
    )
    return output, lse


def run_chunked_prefill(
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
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run native INT8-Q/UINT8-P frozen pool execution plus direct or quantized live tail."""
    validate_prefill_backend(prefill_backend, grouped_qk=grouped_qk, pack_wmma=pack_wmma)
    batch, q_heads, seq_q, head_dim = query.shape
    num_slots = key_int8.shape[0]
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    tail_cap = 0 if tail_key is None else tail_key.shape[2]
    has_chunks = max_chunks > 0 and num_slots > 0
    has_tail = tail_cap > 0

    frozen_mode, frozen_left, frozen_right, tail_mode, tail_left, tail_right = (
        _resolve_kv_window_modes(
            is_causal=is_causal,
            window_left=window_left,
            window_right=window_right,
        )
    )

    output = torch.empty((batch, q_heads, seq_q, head_dim), device=query.device, dtype=query.dtype)
    lse = torch.empty((batch, q_heads, seq_q), device=query.device, dtype=torch.float32)
    if batch == 0:
        return output, lse
    if not has_chunks and not has_tail:
        output.zero_()
        lse.fill_(-float("inf"))
        return output, lse

    if seq_q < PREFILL_MIN_SEQ_Q:
        return _run_small_q_decode(
            query,
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
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            scale=scale,
            is_causal=is_causal,
            grouped_qk=grouped_qk,
            active_chunks_hint=active_chunks_hint,
            prefill_backend=prefill_backend,
            target=target,
            window_left=window_left,
            window_right=window_right,
        )

    storage_q = triton.cdiv(seq_q, 64) * 64
    query_int8, query_scale = _quantize_query(
        query,
        scale,
        prefill_backend=prefill_backend,
        grouped_qk=grouped_qk,
        per_row=False,
        storage_length=storage_q,
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
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_mode=frozen_mode,
            window_left=frozen_left,
            window_right=frozen_right,
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
                window_mode=tail_mode,
                window_left=tail_left,
                window_right=tail_right,
            )
        else:
            from ._nvidia import prefill as nvidia_prefill  # noqa: PLC0415

            nvidia_prefill.launch_tail_prefill(
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
                window_mode=tail_mode,
                window_left=tail_left,
                window_right=tail_right,
            )
        if has_chunks:
            _merge_two_attention_waves(output, lse, out_tail, lse_tail)

    return output, lse
