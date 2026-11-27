"""Quantized chunked prefill and Split-KV reduction orchestration for Piper KV pools."""

# Triton JIT kernels do not use Python runtime type annotations or narrow signatures.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

from typing import Literal

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear._splitk import device_sm_count

from . import _quantization
from ._decode_schedule import REDUCE_NUM_WARPS

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
):
    """Merge partial Split-KV outputs and three-term base-2 state into natural-log LSE."""
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
    ld_win = 0.0
    denom = 0.0
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    for split_idx in tl.range(0, num_splits, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        s64 = split_idx.to(tl.int64)
        lse_off = pl_offset + s64 * split_lse_stride
        m_s = tl.load(partial_lse_ptr + lse_off)
        b_s = tl.load(partial_bias_ptr + lse_off)
        ld_s = tl.load(partial_log_denom_ptr + lse_off)
        m_win, bias_win, ld_win, old_w, cur_w, has_s, has_nan_s = _advance_split_max_safe(
            m_win, bias_win, ld_win, m_s, b_s, ld_s
        )

        out_s = tl.load(po_base + s64 * stride_pop + offsets_d)
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
        has_any,
        (m_win * _LN2 + bias_win * _LN2) + (ld_win + tl.math.log2(safe_denom)) * _LN2,
        -float("inf"),
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

    frozen_splits = 1
    out_target = output
    lse_target = lse
    partial_out: torch.Tensor | None = None
    partial_lse: torch.Tensor | None = None
    partial_bias: torch.Tensor | None = None
    partial_log_denom: torch.Tensor | None = None
    bias_target: torch.Tensor | None = None
    log_denom_target: torch.Tensor | None = None
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
        partial_log_denom = torch.empty_like(partial_lse)
        out_target = partial_out.view(batch * frozen_splits, q_heads, seq_q, head_dim)
        lse_target = partial_lse.view(batch * frozen_splits, q_heads, seq_q)
        bias_target = partial_bias.view(batch * frozen_splits, q_heads, seq_q)
        log_denom_target = partial_log_denom.view(batch * frozen_splits, q_heads, seq_q)

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
            split_chunks=frozen_splits,
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
            po_view = partial_out.permute(0, 2, 3, 1, 4)
            pl_view = partial_lse.permute(0, 2, 3, 1)
            pb_view = partial_bias.permute(0, 2, 3, 1)
            pld_view = partial_log_denom.permute(0, 2, 3, 1)
            with device_context(output.device):
                _piper_decode_splitkv_reduce_kernel[(batch * q_heads * seq_q,)](
                    po_view,
                    pl_view,
                    pb_view,
                    pld_view,
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
                    num_warps=REDUCE_NUM_WARPS,
                )


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
