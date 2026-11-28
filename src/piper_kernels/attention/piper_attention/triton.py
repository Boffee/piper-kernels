"""GPU Split-KV decode, direct live-tail, and split reduction Triton kernels for Piper KV pools."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# ruff: noqa: PLR0124
# pyright: reportCallIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.piper.triton import (
    advance_tile_max_safe,
    clip_kv_window_tiles,
    max_propagate_nan,
    pack_amd_wmma_token,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear._splitk import device_sm_count

from ._amd import policy as _amd_policy
from ._decode_schedule import (
    BLOCK_QH,
    CENTER_CHUNK,
    KEY_SCALES_GROUPED,
    REDUCE_NUM_WARPS,
    SPLIT_NUM_STAGES,
    VALUE_TILE_TOKENS,
    DecodeSchedule,
    default_decode_schedule,
    select_split_counts,
)
from ._nvidia import policy as _nvidia_policy
from ._validation import (
    PoolSlotLayout,
    WindowMode,
    resolve_kv_window_modes,
    resolve_window_mode,
)

_TL_CENTER_CHUNK = tl.constexpr(CENTER_CHUNK)
_TL_VALUE_TILE_TOKENS = tl.constexpr(VALUE_TILE_TOKENS)
_TL_KEY_SCALES_GROUPED = tl.constexpr(KEY_SCALES_GROUPED)
_TL_SPLIT_NUM_STAGES = tl.constexpr(SPLIT_NUM_STAGES)
_LN2 = tl.constexpr(0.6931471805599453)
_LOG2_E = tl.constexpr(1.4426950408889634)


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
) -> PoolSlotLayout:
    """Return signed pool views, slot-map flag/cap, and zeroed slot/batch strides."""
    num_slots = key_int8.shape[0]
    has_chunk_map = chunk_to_slot is not None
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    cs_tensor = chunk_to_slot if chunk_to_slot is not None else num_chunks
    key_i8_s = key_int8.view(torch.int8)
    val_i8_s = value_int8.view(torch.int8)
    return PoolSlotLayout(
        key_int8=key_i8_s,
        value_int8=val_i8_s,
        chunk_to_slot=cs_tensor,
        has_chunk_map=has_chunk_map,
        max_chunks=max_chunks,
        stride_ki_s=0 if num_slots <= 1 else key_i8_s.stride(0),
        stride_ks_s=0 if num_slots <= 1 else key_scale.stride(0),
        stride_kc_s=0 if num_slots <= 1 else key_center.stride(0),
        stride_vi_s=0 if num_slots <= 1 else val_i8_s.stride(0),
        stride_vs_s=0 if num_slots <= 1 else value_scale.stride(0),
        stride_vm_s=0 if num_slots <= 1 else value_mean.stride(0),
        stride_cs_b=0 if (not has_chunk_map or batch <= 1) else cs_tensor.stride(0),
        stride_cs_c=1 if not has_chunk_map else cs_tensor.stride(1),
    )


@triton.jit
def _frozen_qk_dot(
    k_base,
    rows,
    q_lo,
    q_hi,
    offsets_d,
    offsets_d64,
    head_dim: tl.constexpr,
):
    """Compute INT8 key-tile QK^T in IEEE FP32 for non-native Split-KV fallback targets."""
    if head_dim == 128:
        k_lo = tl.load(k_base + rows[:, None] * head_dim + offsets_d64[None, :])
        qk = tl.dot(q_lo, tl.trans(k_lo.to(tl.float32)), input_precision="ieee")
        k_hi = tl.load(k_base + rows[:, None] * head_dim + (offsets_d64[None, :] + 64))
        qk = tl.dot(q_hi, tl.trans(k_hi.to(tl.float32)), acc=qk, input_precision="ieee")
    else:
        k_q = tl.load(k_base + rows[:, None] * head_dim + offsets_d[None, :])
        qk = tl.dot(q_lo, tl.trans(k_q.to(tl.float32)), input_precision="ieee")
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
    pv_precision: tl.constexpr,
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
        q_base_pos = total_kv - seq_q.to(tl.int64)
        first_vis_subtile, last_vis_subtile, eff_wl, eff_wr = clip_kv_window_tiles(
            q_tile,
            gqa_group * seq_q,
            seq_q,
            q_base_pos,
            frozen_tokens,
            total_kv,
            window_left,
            window_right,
            block_m=block_qh,
            block_n=block_n,
            packed_gqa=True,
            window_mode=window_mode,
        )
    else:
        first_vis_subtile = tl.zeros((), dtype=tl.int32)
        last_vis_subtile = active_chunks * subtiles_per_chunk
        frozen_tokens = tl.zeros((), dtype=tl.int64)
        q_base_pos = tl.zeros((), dtype=tl.int64)
        eff_wl = tl.zeros((), dtype=tl.int64)
        eff_wr = tl.zeros((), dtype=tl.int64)

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

        offsets_n = tl.arange(0, block_n)
        scaled_log2 = scale * _LOG2_E
        stride_kv_head = tl.constexpr(_TL_CENTER_CHUNK * head_dim)
        stride_ks_head = tl.constexpr(_TL_KEY_SCALES_GROUPED if grouped_qk else _TL_CENTER_CHUNK)
        stride_vi_t = tl.constexpr(head_dim * _TL_VALUE_TILE_TOKENS)
        if is_causal_mode:
            win_lo_base = tl.zeros((), dtype=tl.int64)
            win_hi_base = q_base_pos
        elif is_windowed_mode:
            win_lo_base = q_base_pos - eff_wl if window_left >= 0 else -seq_q.to(tl.int64) - 1
            win_hi_base = (
                q_base_pos + eff_wr if window_right >= 0 else frozen_tokens + _TL_CENTER_CHUNK
            )
        else:
            win_lo_base = tl.zeros((), dtype=tl.int64)
            win_hi_base = tl.zeros((), dtype=tl.int64)

        for chunk_idx in tl.range(first_chunk, last_chunk + 1, disable_licm=True):
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
                    inner_start, inner_end, num_stages=_TL_SPLIT_NUM_STAGES, disable_licm=True
                ):
                    t64 = local_subtile // subtiles_per_k64
                    sub4 = local_subtile % subtiles_per_k64
                    rows = local_subtile * block_n + offsets_n
                    qk = _frozen_qk_dot(
                        k_base,
                        rows,
                        q_lo,
                        q_hi,
                        offsets_d,
                        offsets_d64,
                        head_dim=head_dim,
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
    if is_windowed_mode:
        first_vis_tile, last_vis_tile, eff_wl, eff_wr = clip_kv_window_tiles(
            q_tile,
            gqa_group * seq_q,
            seq_q,
            active_tail - seq_q.to(tl.int64),
            active_tail,
            active_tail,
            window_left,
            window_right,
            block_m=block_qh,
            block_n=block_n,
            packed_gqa=True,
            window_mode=window_mode,
        )
    else:
        first_vis_tile = tl.zeros((), dtype=tl.int32)
        last_vis_tile = tl.cdiv(active_tail32, block_n)
        eff_wl = tl.zeros((), dtype=tl.int64)
        eff_wr = tl.zeros((), dtype=tl.int64)

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
            q_tail_pos64 = (active_tail32 - seq_q).to(tl.int64) + q_idx0.to(tl.int64)
            tail_win_lo = (
                tl.minimum(
                    active_tail,
                    tl.maximum(0, q_tail_pos64 - eff_wl),
                ).to(tl.int32)
                if window_left >= 0
                else tl.zeros((block_qh,), dtype=tl.int32)
            )
            tail_win_hi = (
                tl.minimum(
                    active_tail,
                    tl.maximum(-1, q_tail_pos64 + eff_wr),
                ).to(tl.int32)
                if window_right >= 0
                else tl.full((block_qh,), active_tail32, dtype=tl.int32)
            )
        else:
            tail_win_lo = tl.zeros((block_qh,), dtype=tl.int32)
            tail_win_hi = tl.zeros((block_qh,), dtype=tl.int32)

        for tile_idx in tl.range(tile_start, tile_end, disable_licm=True):
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
            has_block = m_block != -float("inf")
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
                valid_mask & has_block[:, None],
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
    window_mode: WindowMode | None = None,
    window_left: int = -1,
    window_right: int = -1,
) -> None:
    """Launch Split-KV direct BF16/FP16 tail attention into ``partial_*[split >= int8_splits]``."""
    resolved_mode = resolve_window_mode(is_causal, window_mode)
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


def launch_splitkv_reduce(
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
    target: AcceleratorTarget | None = None,
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute fallback GPU Split-KV decode over frozen INT8 chunks and optional live tail."""
    batch, q_heads, seq_q, head_dim = query.shape
    num_slots, kv_heads, _, _ = key_int8.shape
    gqa_group = q_heads // kv_heads
    windows = resolve_kv_window_modes(
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
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
        window_left=windows.frozen_left,
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
            pool_layout = pool_slot_strides(
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
                pool_layout.key_int8,
                key_scale,
                key_center,
                pool_layout.value_int8,
                value_scale,
                value_mean,
                pool_layout.chunk_to_slot,
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
                pool_layout.stride_ki_s,
                pool_layout.stride_ks_s,
                pool_layout.stride_kc_s,
                pool_layout.stride_vi_s,
                pool_layout.stride_vs_s,
                pool_layout.stride_vm_s,
                pool_layout.stride_cs_b,
                pool_layout.stride_cs_c,
                partial_out.stride(0),
                partial_out.stride(1),
                partial_out.stride(2),
                partial_out.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                has_chunk_map=pool_layout.has_chunk_map,
                grouped_qk=grouped_qk,
                pack_wmma=pack_wmma,
                head_dim=head_dim,
                block_qh=BLOCK_QH,
                block_n=schedule.block_n,
                pv_precision=schedule.pv_precision,
                tail_len_ptr=tail_len,
                tail_cap=tail_cap,
                window_left=windows.frozen_left,
                window_right=windows.frozen_right,
                has_tail=tail_len is not None,
                window_mode=windows.frozen_mode,
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
                window_mode=windows.tail_mode,
                window_left=windows.tail_left,
                window_right=windows.tail_right,
            )

        launch_splitkv_reduce(
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


__all__ = ["piper_attention_decode_triton"]
