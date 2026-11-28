"""GPU Split-KV decode and quantized chunked prefill Triton kernels for Piper KV pools."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from collections.abc import Callable
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

from ._decode_schedule import (
    BLOCK_QH,
    CENTER_CHUNK,
    KEY_SCALES_GROUPED,
    VALUE_TILE_TOKENS,
    select_split_counts,
)
from ._kv_prefill import (
    _launch_splitkv_reduce,
    launch_direct_tail_split_decode,
    pool_slot_strides,
    run_chunked_prefill,
    select_decode_schedule,
)

_TL_CENTER_CHUNK = tl.constexpr(CENTER_CHUNK)
_TL_VALUE_TILE_TOKENS = tl.constexpr(VALUE_TILE_TOKENS)
_TL_KEY_SCALES_GROUPED = tl.constexpr(KEY_SCALES_GROUPED)
_LOG2_E = tl.constexpr(1.4426950408889634)


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
        "tail_cap",
        "window_left",
        "window_right",
    ]
)
def _piper_decode_frozen_split_kernel(  # noqa: PLR0912, PLR0915
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
    partial_log_denom_ptr,
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
    tail_len_ptr=None,
    tail_cap=0,
    window_left=-1,
    window_right=-1,
    has_tail: tl.constexpr = False,
    window_mode: tl.constexpr = "none",
):
    """Evaluate one Split-KV slice of frozen INT8 chunks using two-term winner framing."""
    is_causal_mode: tl.constexpr = window_mode == "causal"
    is_windowed_mode: tl.constexpr = window_mode == "windowed"
    has_window: tl.constexpr = is_causal_mode or is_windowed_mode

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
    active_chunks = tl.maximum(0, tl.minimum(nc, max_chunks))
    if has_window:
        active_tail = (
            tl.maximum(0, tl.minimum(tl.load(tail_len_ptr + b64), tail_cap)).to(tl.int64)
            if has_tail
            else tl.zeros((), dtype=tl.int64)
        )
        frozen_tokens = active_chunks.to(tl.int64) * _TL_CENTER_CHUNK
        total_kv = frozen_tokens + active_tail
        eff_wl = tl.minimum(window_left.to(tl.int64), total_kv)
        eff_wr = tl.minimum(window_right.to(tl.int64), total_kv + seq_q.to(tl.int64))
        q_tile_first = q_tile * block_qh
        q_tile_last = tl.minimum((q_tile + 1) * block_qh, gqa_group * seq_q) - 1
        same_q_head = (q_tile_first // seq_q) == (q_tile_last // seq_q)
        q_idx_min = tl.where(same_q_head, q_tile_first % seq_q, 0).to(tl.int64)
        q_idx_max = tl.where(same_q_head, q_tile_last % seq_q, seq_q - 1).to(tl.int64)
        q_min = (total_kv - seq_q.to(tl.int64)) + q_idx_min
        q_max = (total_kv - seq_q.to(tl.int64)) + q_idx_max
        k_min = (
            tl.maximum(0, q_min - eff_wl)
            if (is_windowed_mode and window_left >= 0)
            else tl.zeros((), dtype=tl.int64)
        )
        if is_causal_mode:
            k_max_excl = tl.minimum(frozen_tokens, tl.maximum(0, q_max + 1))
        elif window_right >= 0:
            k_max_excl = tl.minimum(frozen_tokens, tl.maximum(0, q_max + eff_wr + 1))
        else:
            k_max_excl = frozen_tokens
        first_vis_chunk = tl.minimum(active_chunks.to(tl.int64), k_min // _TL_CENTER_CHUNK)
        last_vis_chunk = tl.maximum(
            first_vis_chunk,
            tl.minimum(active_chunks.to(tl.int64), tl.cdiv(k_max_excl, _TL_CENTER_CHUNK)),
        )
        first_vis_subtile = tl.maximum(
            first_vis_chunk * subtiles_per_chunk,
            tl.minimum(last_vis_chunk * subtiles_per_chunk, k_min // block_n),
        ).to(tl.int32)
        last_vis_subtile = tl.maximum(
            first_vis_subtile.to(tl.int64),
            tl.minimum(last_vis_chunk * subtiles_per_chunk, tl.cdiv(k_max_excl, block_n)),
        ).to(tl.int32)
    else:
        first_vis_subtile = tl.zeros((), dtype=tl.int32)
        last_vis_subtile = active_chunks * subtiles_per_chunk
        frozen_tokens = tl.zeros((), dtype=tl.int64)
        total_kv = tl.zeros((), dtype=tl.int64)

    visible_subtiles = last_vis_subtile - first_vis_subtile
    subtiles_per_split = tl.cdiv(visible_subtiles, int8_splits)
    subtile_start = first_vis_subtile + split_id * subtiles_per_split
    subtile_end = tl.minimum(last_vis_subtile, subtile_start + subtiles_per_split)

    if subtile_start < subtile_end:
        first_chunk = subtile_start // subtiles_per_chunk
        last_chunk = (subtile_end - 1) // subtiles_per_chunk
        first_inner = subtile_start % subtiles_per_chunk
        last_inner = (subtile_end - 1) % subtiles_per_chunk + 1

        qh_offs0 = q_tile * block_qh + tl.arange(0, block_qh)
        q_idx0 = qh_offs0 % seq_q
        q_head0 = kv_head * gqa_group + qh_offs0 // seq_q
        q_ptrs = (
            query_ptr
            + b64 * stride_qb
            + q_head0.to(tl.int64)[:, None] * stride_qh
            + q_idx0.to(tl.int64)[:, None] * stride_qs
            + offsets_d[None, :]
        )
        q_f32 = tl.load(q_ptrs, mask=(qh_offs0 < gqa_group * seq_q)[:, None], other=0.0).to(
            tl.float32
        )
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
        q_base_pos = total_kv - seq_q.to(tl.int64)
        if is_causal_mode:
            win_lo_base = tl.zeros((), dtype=tl.int64)
            win_hi_base = q_base_pos
        elif is_windowed_mode:
            eff_wl_f = tl.minimum(window_left.to(tl.int64), total_kv)
            eff_wr_f = tl.minimum(window_right.to(tl.int64), total_kv + seq_q.to(tl.int64))
            win_lo_base = q_base_pos - eff_wl_f if window_left >= 0 else -seq_q.to(tl.int64) - 1
            win_hi_base = (
                q_base_pos + eff_wr_f if window_right >= 0 else frozen_tokens + _TL_CENTER_CHUNK
            )
        else:
            win_lo_base = tl.zeros((), dtype=tl.int64)
            win_hi_base = tl.zeros((), dtype=tl.int64)

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
                if has_window:
                    chunk_base64 = chunk_idx.to(tl.int64) * _TL_CENTER_CHUNK
                    q_idx_cur = (q_tile * block_qh + tl.arange(0, block_qh)) % seq_q
                    chunk_win_lo = tl.maximum(
                        0,
                        tl.minimum(
                            _TL_CENTER_CHUNK,
                            (win_lo_base + q_idx_cur.to(tl.int64)) - chunk_base64,
                        ),
                    ).to(tl.int32)
                    chunk_win_hi = tl.maximum(
                        -1,
                        tl.minimum(
                            _TL_CENTER_CHUNK,
                            (win_hi_base + q_idx_cur.to(tl.int64)) - chunk_base64,
                        ),
                    ).to(tl.int32)
                    chunk_win_packed = ((chunk_win_lo + 1) << 16) | (chunk_win_hi + 1)
                else:
                    chunk_win_packed = tl.zeros((block_qh,), dtype=tl.int32)
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

                    if has_window:
                        qh_val_cur = (
                            q_tile * block_qh + tl.arange(0, block_qh)
                        ) < gqa_group * seq_q
                        valid_mask = tl.broadcast_to(qh_val_cur[:, None], (block_qh, block_n))
                        win_hi = (chunk_win_packed & 0xFFFF) - 1
                        if is_causal_mode:
                            valid_mask = valid_mask & (rows[None, :] <= win_hi[:, None])
                        elif is_windowed_mode:
                            win_lo = (chunk_win_packed >> 16) - 1
                            valid_mask = (
                                valid_mask
                                & (rows[None, :] >= win_lo[:, None])
                                & (rows[None, :] <= win_hi[:, None])
                            )
                        scores = tl.where(valid_mask, scores, -float("inf"))

                    m_block = tl.reduce(scores, 1, max_propagate_nan)
                    subtile_k_bias = (
                        tl.where(m_block != -float("inf"), chunk_bias, 0.0)
                        if has_window
                        else chunk_bias
                    )
                    m_i, bias_i, old_weight, cur_weight, has_block, safe_m_block = (
                        advance_tile_max_safe(
                            m_i,
                            bias_i,
                            m_block,
                            subtile_k_bias,
                            ((q_tile * block_qh + tl.arange(0, block_qh)) < gqa_group * seq_q)
                            if has_window
                            else None,
                        )
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
    safe_l = tl.where(has_any & (l_i != 0.0), l_i, 1.0)
    out_split = tl.where(has_any[:, None], acc / safe_l[:, None], 0.0)
    lse_split = tl.where(has_any, m_i, -float("inf"))
    bias_split = tl.where(has_any, bias_i, 0.0)
    log_denom_split = tl.where(has_any, tl.math.log2(safe_l), 0.0)

    qh_offs = q_tile * block_qh + tl.arange(0, block_qh)
    qh_valid = qh_offs < gqa_group * seq_q
    q_idx = qh_offs % seq_q
    q_head = kv_head * gqa_group + qh_offs // seq_q
    split64 = split_id.to(tl.int64)
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
    tl.store(partial_log_denom_ptr + pl_offsets, log_denom_split, mask=qh_valid)


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
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute native INT8-Q/UINT8-P or fallback Split-KV decode and prefill on GPU."""
    if prefill_backend is not None:
        return run_chunked_prefill(
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
            window_left=window_left,
            window_right=window_right,
        )
    batch, q_heads, seq_q, head_dim = query.shape
    num_slots, kv_heads, _, _ = key_int8.shape
    gqa_group = q_heads // kv_heads

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

    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    tail_cap = 0 if tail_key is None else tail_key.shape[2]
    resolved_target = AcceleratorTarget.from_device(query.device) if target is None else target
    schedule = select_decode_schedule(resolved_target, head_dim)
    compiler_options = dict(schedule.compiler_options)
    sm_count = device_sm_count(query.device)

    int8_splits, tail_splits = select_split_counts(
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
        window_left=frozen_left,
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
    partial_log_denom = torch.empty_like(partial_lse)

    q_tiles = triton.cdiv(gqa_group * seq_q, BLOCK_QH)
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
                partial_log_denom,
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
                block_qh=BLOCK_QH,
                block_n=schedule.block_n,
                qk_mode=schedule.frozen_qk_mode,
                pv_precision=schedule.pv_precision,
                # Free a pipeline buffer stage when windowed D=128 spills at loop=2.
                loop_num_stages=(
                    1
                    if (frozen_mode == "windowed" and head_dim == 128)
                    else schedule.loop_num_stages
                ),
                tail_len_ptr=tail_len,
                tail_cap=tail_cap,
                window_left=frozen_left,
                window_right=frozen_right,
                has_tail=tail_len is not None,
                window_mode=frozen_mode,
                num_warps=schedule.num_warps,
                num_stages=schedule.num_stages,
                **compiler_options,
            )

        if tail_splits > 0:
            assert tail_key is not None
            assert tail_value is not None
            assert tail_len is not None
            launch_direct_tail_split_decode(
                query,
                tail_key,
                tail_value,
                tail_len,
                partial_out,
                partial_lse,
                partial_bias,
                partial_log_denom,
                scale=scale,
                kv_heads=kv_heads,
                gqa_group=gqa_group,
                q_tiles=q_tiles,
                seq_q=seq_q,
                tail_cap=tail_cap,
                int8_splits=int8_splits,
                tail_splits=tail_splits,
                is_causal=is_causal,
                head_dim=head_dim,
                schedule=schedule,
                window_mode=tail_mode,
                window_left=tail_left,
                window_right=tail_right,
            )

        _launch_splitkv_reduce(
            partial_out,
            partial_lse,
            partial_bias,
            partial_log_denom,
            output,
            lse,
            batch=batch,
            q_heads=q_heads,
            seq_q=seq_q,
            num_splits=num_splits,
            head_dim=head_dim,
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
    reduce_vectorize_splits: bool = False,
    window_mode: Literal["none", "causal", "windowed"] | None = None,
    has_tail: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object, object]:
    """Compile frozen-split, tail-split, and reduce decode kernels offline."""
    from . import _kv_compile  # noqa: PLC0415

    return _kv_compile.compile_kv_decode_kernels(
        target,
        dtype=dtype,
        head_dim=head_dim,
        has_chunk_map=has_chunk_map,
        is_causal=is_causal,
        contiguous_inputs=contiguous_inputs,
        wide_pool=wide_pool,
        reduce_vectorize_splits=reduce_vectorize_splits,
        window_mode=window_mode,
        has_tail=has_tail,
        compile_fn=compile_fn,
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
    per_row_query_scale: bool = False,
    window_mode: Literal["none", "causal", "windowed"] | None = None,
    has_tail: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, ...]:
    """Offline-compile two-wave merge and native chunked prefill kernels for ``target``."""
    from . import _kv_compile  # noqa: PLC0415

    return _kv_compile.compile_kv_prefill_kernels(
        target,
        dtype=dtype,
        head_dim=head_dim,
        has_chunk_map=has_chunk_map,
        is_causal=is_causal,
        wide_pool=wide_pool,
        has_chunk_splits=has_chunk_splits,
        per_row_query_scale=per_row_query_scale,
        window_mode=window_mode,
        has_tail=has_tail,
        compile_fn=compile_fn,
    )


def compile_merge_attention_lse_kernel(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    num_waves: int = 2,
    is_contiguous: bool = True,
    in_place_0: bool = True,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> object:
    """Offline-compile the unified 1..8-wave LSE merge kernel for ``target``."""
    from . import _kv_compile  # noqa: PLC0415

    return _kv_compile.compile_merge_attention_lse_kernel(
        target,
        dtype=dtype,
        head_dim=head_dim,
        num_waves=num_waves,
        is_contiguous=is_contiguous,
        in_place_0=in_place_0,
        compile_fn=compile_fn,
    )


__all__ = [
    "compile_kv_decode_kernels",
    "compile_kv_prefill_kernels",
    "compile_merge_attention_lse_kernel",
    "piper_attention_decode_triton",
]
