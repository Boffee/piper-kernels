"""Quantized chunked prefill and Split-KV reduction orchestration for Piper KV pools."""

# Triton JIT kernels do not use Python runtime type annotations or narrow signatures.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

from typing import Literal

import torch
import triton
import triton.language as tl

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear._splitk import device_sm_count

from . import _quantization
from ._decode_schedule import (
    BLOCK_QH,
    PREFILL_MIN_SEQ_Q,
    REDUCE_NUM_WARPS,
    select_split_counts,
)

_LN2 = tl.constexpr(0.6931471805599453)


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
    is_nan_s = s_mask & (tot_s != tot_s)  # noqa: PLR0124
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
    delta_nan = valid_s & (raw_delta != raw_delta)  # noqa: PLR0124
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
            num_warps=REDUCE_NUM_WARPS,
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
) -> int:
    sm_count = device_sm_count(device)
    base_blocks = max(1, batch * q_heads * triton.cdiv(seq_q, 64))
    hint_chunks = max_chunks if active_chunks_hint <= 0 else min(max_chunks, active_chunks_hint)
    if hint_chunks <= 0:
        return 1
    if prefill_backend == "amd":
        from ._amd import policy as amd_policy  # noqa: PLC0415

        return amd_policy.select_frozen_prefill_splits(
            base_blocks,
            hint_chunks,
            sm_count,
            batch * q_heads * seq_q * head_dim * 4,
            allow_subchunk_single=allow_subchunk_single,
        )
    total_tiles = hint_chunks * 16
    mem_cap = max(1, (64 * 1024 * 1024) // max(1, batch * q_heads * seq_q * head_dim * 4))
    if allow_subchunk_single:
        if base_blocks < sm_count * 3 and total_tiles >= 8:
            return min(32, total_tiles // 4, max(1, (sm_count * 3) // base_blocks), mem_cap)
        return 1
    if prefill_backend == "nvidia" and base_blocks < sm_count and hint_chunks >= 2:
        return min(16, total_tiles // 4, max(1, (sm_count * 2) // base_blocks), mem_cap)
    return 1


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
        )
    if (
        partial_out is not None
        and partial_lse is not None
        and partial_bias is not None
        and partial_log_denom is not None
    ):
        _launch_splitkv_reduce(
            partial_out.permute(0, 2, 3, 1, 4),
            partial_lse.permute(0, 2, 3, 1),
            partial_bias.permute(0, 2, 3, 1),
            partial_log_denom.permute(0, 2, 3, 1),
            output,
            lse,
            batch=batch,
            q_heads=q_heads,
            seq_q=seq_q,
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
        )
        return output, lse

    from .triton import (  # noqa: PLC0415
        launch_direct_tail_split_decode,
        select_decode_schedule,
    )

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run native INT8-Q/UINT8-P frozen pool execution plus direct or quantized live tail."""
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
            )
        if has_chunks:
            _merge_two_attention_waves(output, lse, out_tail, lse_tail)

    return output, lse
