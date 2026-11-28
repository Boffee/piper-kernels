"""NVIDIA Triton chunked prefill kernel and launchers."""

# Triton JIT kernels do not use Python runtime type annotations or narrow signatures.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false

import math

import torch
import triton
import triton.language as tl

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.piper.triton import (
    advance_tile_max_safe,
    clip_kv_window_tiles,
    conservative_value_log_scale_bound,
    max_propagate_nan,
)

from .. import _quantization
from .._validation import PrefillKernelOperands, WindowMode, resolve_window_mode
from . import policy as _policy
from .triton import (
    _accumulate_pv_tile,
    _fold_chunk_value_mean,
    _quantize_probabilities_u8,
    _safe_softmax_probabilities,
    _store_pv_output,
)

_TL_CENTER_CHUNK = tl.constexpr(_quantization.CENTER_CHUNK)
_TL_TILES_PER_CHUNK = tl.constexpr(_quantization._TILES_PER_CHUNK)
_P_UINT8_RANGE = tl.constexpr(255.0)
_LN2 = tl.constexpr(math.log(2.0))
_LOG2_SCALE_EPSILON = tl.constexpr(-23.253496664211536)


@triton.jit(
    do_not_specialize=[
        "seq_q",
        "query_storage",
        "heads",
        "num_slots",
        "max_chunks",
        "stride_ki_s",
        "stride_ks_s",
        "stride_kc_s",
        "stride_vi_s",
        "stride_vs_s",
        "stride_vm_s",
        "stride_cs_b",
        "stride_cs_c",
        "stride_ob",
        "stride_oh",
        "stride_om",
        "stride_lb",
        "stride_lh",
        "split_chunks",
        "total_splits",
        "tail_cap",
        "window_left",
        "window_right",
        "logical_seq_q",
    ]
)
def _piper_chunked_prefill_kernel(  # noqa: PLR0912, PLR0915
    query_ptr,
    query_scale_ptr,
    key_int8_ptr,
    key_scale_ptr,
    key_center_ptr,
    value_int8_ptr,
    value_scale_ptr,
    value_mean_ptr,
    chunk_to_slot_ptr,
    num_chunks_ptr,
    output_ptr,
    lse_ptr,
    seq_q,
    query_storage,
    heads,
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
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    split_chunks,
    partial_bias_ptr,
    partial_log_denom_ptr,
    total_splits,
    query_heads_per_kv_head: tl.constexpr,
    head_dim: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    has_chunk_map: tl.constexpr,
    is_tail: tl.constexpr,
    window_mode: tl.constexpr,
    block_m: tl.constexpr,
    loop_num_stages: tl.constexpr,
    has_chunk_splits: tl.constexpr = False,
    per_row_query_scale: tl.constexpr = False,
    tail_len_ptr=None,
    tail_cap=0,
    window_left=-1,
    window_right=-1,
    logical_seq_q=0,
    has_tail: tl.constexpr = False,
):
    """Zero-spill NVIDIA prefill over PiperKVChunkPool (is_tail=False) or quantized live tail."""
    tl.static_assert(not (is_tail and has_chunk_splits))
    is_causal_mode: tl.constexpr = window_mode == "causal"
    is_windowed_mode: tl.constexpr = window_mode == "windowed"
    has_window: tl.constexpr = is_causal_mode or is_windowed_mode

    query_block = tl.program_id(0)
    if is_tail and is_causal_mode:
        query_block = tl.num_programs(0) - 1 - query_block
    head = tl.program_id(1)
    if has_chunk_splits:
        batch_split = tl.program_id(2)
        split_id = batch_split % split_chunks
        batch = batch_split // split_chunks
    else:
        batch = tl.program_id(2)
        split_id = tl.zeros((), dtype=tl.int32)
    batch_head = batch * heads + head
    kv_head = head // query_heads_per_kv_head
    kv_heads = heads // query_heads_per_kv_head
    kv_batch_head = (batch * kv_heads + kv_head).to(tl.int64)
    output_rows = query_block * block_m + tl.arange(0, block_m)
    valid_queries = output_rows < seq_q
    offsets_n = tl.arange(0, 64)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    offsets_vd = tl.arange(0, half_head_dim)

    query = tl.load(
        query_ptr
        + ((batch_head.to(tl.int64) * query_storage + output_rows[:, None]) * head_dim)
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    if grouped_qk and not per_row_query_scale:
        query_scale = tl.load(
            query_scale_ptr
            + batch_head.to(tl.int64) * tl.cdiv(query_storage, 32)
            + output_rows // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + batch_head.to(tl.int64) * query_storage + output_rows,
            mask=valid_queries,
            other=0.0,
        )

    if split_pv_head_dim:
        numerator = (
            tl.zeros((block_m, half_head_dim), dtype=tl.float32),
            tl.zeros((block_m, half_head_dim), dtype=tl.float32),
        )
    else:
        numerator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    running_bias = tl.zeros((block_m,), dtype=tl.float32)

    active_count = tl.minimum(tl.maximum(tl.load(num_chunks_ptr + batch), 0), max_chunks)
    eff_logical_seq_q = tl.maximum(logical_seq_q, 1)
    eff_seq_q = eff_logical_seq_q if per_row_query_scale else seq_q
    row_q_idx = (
        ((output_rows % eff_logical_seq_q) if per_row_query_scale else output_rows)
        if has_window
        else tl.zeros((), dtype=tl.int32)
    )
    if is_tail:
        # For tail prefill: num_chunks_ptr holds tail_len, max_chunks is tail_cap,
        # and num_slots holds key_storage.
        active_len = active_count
        global_query_start = active_len.to(tl.int64) - seq_q.to(tl.int64)
        tile_start, tile_end, eff_wl, eff_wr = clip_kv_window_tiles(
            query_block,
            seq_q,
            seq_q,
            global_query_start,
            active_len,
            active_len,
            window_left,
            window_right,
            block_m=block_m,
            block_n=64,
            packed_gqa=False,
            window_mode=window_mode,
        )
        has_work = tile_start < tile_end
        chunk_start = tl.where(has_work, tile_start // _TL_TILES_PER_CHUNK, 0)
        chunk_end = tl.where(has_work, (tile_end - 1) // _TL_TILES_PER_CHUNK + 1, 0)
        if is_windowed_mode:
            win_lo_base = (
                global_query_start - eff_wl if window_left >= 0 else -seq_q.to(tl.int64) - 1
            )
            win_hi_base = (
                global_query_start + eff_wr if window_right >= 0 else active_len.to(tl.int64) + 1024
            )
        else:
            win_lo_base = tl.zeros((), dtype=tl.int64)
            win_hi_base = tl.zeros((), dtype=tl.int64)
        key_storage64 = num_slots.to(tl.int64)
        key_chunks64 = tl.cdiv(max_chunks, _TL_CENTER_CHUNK).to(tl.int64)
    else:
        active_tail = (
            tl.minimum(tl.maximum(tl.load(tail_len_ptr + batch), 0), tail_cap).to(tl.int64)
            if has_tail
            else tl.zeros((), dtype=tl.int64)
        )
        frozen_tokens = active_count.to(tl.int64) * 1024
        total_kv = frozen_tokens + active_tail
        global_query_start = (
            total_kv - eff_seq_q.to(tl.int64) if has_window else tl.zeros((), dtype=tl.int64)
        )
        vis_t_start, vis_t_end, eff_wl, eff_wr = clip_kv_window_tiles(
            query_block,
            seq_q,
            eff_seq_q,
            global_query_start,
            frozen_tokens,
            total_kv,
            window_left,
            window_right,
            block_m=block_m,
            block_n=64,
            packed_gqa=per_row_query_scale,
            window_mode=window_mode,
        )
        if has_chunk_splits:
            vis_tiles = vis_t_end - vis_t_start
            tiles_per_split = tl.cdiv(vis_tiles, split_chunks)
            tile_start = tl.minimum(vis_t_end, vis_t_start + split_id * tiles_per_split)
            tile_end = tl.minimum(vis_t_end, tile_start + tiles_per_split)
        else:
            tile_start = vis_t_start
            tile_end = vis_t_end
        has_work = tile_start < tile_end
        chunk_start = tl.where(has_work, tile_start // _TL_TILES_PER_CHUNK, 0)
        chunk_end = tl.where(has_work, (tile_end - 1) // _TL_TILES_PER_CHUNK + 1, 0)
        if is_causal_mode:
            win_lo_base = tl.zeros((), dtype=tl.int64)
            win_hi_base = global_query_start
        elif is_windowed_mode:
            win_lo_base = (
                global_query_start - eff_wl if window_left >= 0 else -eff_seq_q.to(tl.int64) - 1
            )
            win_hi_base = global_query_start + eff_wr if window_right >= 0 else frozen_tokens + 1024
        else:
            win_lo_base = tl.zeros((), dtype=tl.int64)
            win_hi_base = tl.zeros((), dtype=tl.int64)
        key_storage64 = tl.zeros((), dtype=tl.int64)
        key_chunks64 = tl.zeros((), dtype=tl.int64)

    kv_stride_ki: tl.constexpr = 1024 * head_dim
    kv_stride_ks = tl.constexpr(16 if grouped_qk else 1024)
    for chunk in tl.range(chunk_start, chunk_end, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
        if is_tail:
            slot = tl.zeros((), dtype=tl.int32)
            chunk_valid = True
        elif has_chunk_map:
            slot = tl.load(
                chunk_to_slot_ptr
                + batch.to(tl.int64) * stride_cs_b
                + chunk.to(tl.int64) * stride_cs_c
            )
            chunk_valid = (slot >= 0) & (slot < num_slots)
        else:
            slot = chunk
            chunk_valid = (slot >= 0) & (slot < num_slots)

        if chunk_valid:
            if is_tail:
                chunk64 = chunk.to(tl.int64)
                chunk_row_offset = chunk64 * 1024
                k_chunk_ptr = (
                    key_int8_ptr + (kv_batch_head * key_storage64 + chunk_row_offset) * head_dim
                )
                ks_chunk_ptr = (
                    key_scale_ptr
                    + kv_batch_head * (key_storage64 // 64 if grouped_qk else key_storage64)
                    + (chunk64 * 16 if grouped_qk else chunk_row_offset)
                )
                kc_chunk_ptr = key_center_ptr + (kv_batch_head * key_chunks64 + chunk64) * head_dim
                v_chunk_ptr = (
                    value_int8_ptr + (kv_batch_head * key_storage64 + chunk_row_offset) * head_dim
                )
                vs_chunk_ptr = value_scale_ptr + kv_batch_head * key_storage64 + chunk_row_offset
                vm_chunk_ptr = value_mean_ptr + (kv_batch_head * key_chunks64 + chunk64) * head_dim
            else:
                slot64 = slot.to(tl.int64)
                kv_head64 = kv_head.to(tl.int64)
                k_chunk_ptr = key_int8_ptr + slot64 * stride_ki_s + kv_head64 * kv_stride_ki
                ks_chunk_ptr = key_scale_ptr + slot64 * stride_ks_s + kv_head64 * kv_stride_ks
                kc_chunk_ptr = key_center_ptr + slot64 * stride_kc_s + kv_head64 * head_dim
                v_chunk_ptr = value_int8_ptr + slot64 * stride_vi_s + kv_head64 * kv_stride_ki
                vs_chunk_ptr = value_scale_ptr + slot64 * stride_vs_s + kv_head64 * 1024
                vm_chunk_ptr = value_mean_ptr + slot64 * stride_vm_s + kv_head64 * head_dim

            if split_pv_head_dim:
                query_even, query_odd = tl.split(tl.reshape(query, (block_m, half_head_dim, 2)))
                kc = tl.load(kc_chunk_ptr + offsets_d)
                kc_even, kc_odd = tl.split(tl.reshape(kc, (half_head_dim, 2)))
                q_scale = query_scale[:, None]
                k_acc = (query_even.to(tl.float32) * q_scale) * kc_even[None, :]
                k_acc += (query_odd.to(tl.float32) * q_scale) * kc_odd[None, :]
                k_bias = tl.sum(k_acc, axis=1)
            else:
                kc = tl.load(kc_chunk_ptr + offsets_d)
                k_bias = tl.sum((query.to(tl.float32) * query_scale[:, None]) * kc[None, :], axis=1)
            chunk_active = valid_queries & (k_bias != -float("inf"))
            chunk_denom = tl.zeros((block_m,), dtype=tl.float32)
            if is_tail or has_chunk_splits or has_window:
                t_begin = tl.maximum(tile_start - chunk * _TL_TILES_PER_CHUNK, 0)
                t_end = tl.minimum(tile_end - chunk * _TL_TILES_PER_CHUNK, _TL_TILES_PER_CHUNK)
            else:
                t_begin = 0
                t_end = _TL_TILES_PER_CHUNK
            if is_windowed_mode:
                chunk_base64 = chunk.to(tl.int64) * 1024
                chunk_tail_lim = (
                    tl.maximum(0, tl.minimum(1024, active_count.to(tl.int64) - chunk_base64)).to(
                        tl.int32
                    )
                    if is_tail
                    else tl.full((), 1024, dtype=tl.int32)
                )
                c_lo = tl.maximum(
                    -(eff_seq_q.to(tl.int64)) - 2, tl.minimum(1025, win_lo_base - chunk_base64)
                ).to(tl.int32)
                c_hi = tl.maximum(
                    -(eff_seq_q.to(tl.int64)) - 2, tl.minimum(1025, win_hi_base - chunk_base64)
                ).to(tl.int32)
            else:
                chunk_tail_lim = tl.full((), 1024, dtype=tl.int32)
                c_lo = tl.zeros((), dtype=tl.int32)
                c_hi = tl.full((), 1025, dtype=tl.int32)

            for tile in tl.range(t_begin, t_end, num_stages=loop_num_stages, disable_licm=True):  # pyright: ignore[reportGeneralTypeIssues]
                start_n = tile * 64
                current_n = start_n + offsets_n
                key = tl.trans(
                    tl.load(k_chunk_ptr + current_n[:, None] * head_dim + offsets_d[None, :])
                )
                integer_scores = tl.dot(query, key, out_dtype=tl.int32)
                if grouped_qk:
                    key_scale = tl.load(ks_chunk_ptr + tile)
                    scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
                else:
                    key_scale = tl.load(ks_chunk_ptr + current_n)
                    scores = integer_scores.to(tl.float32) * (
                        query_scale[:, None] * key_scale[None, :]
                    )
                if is_windowed_mode:
                    if is_tail:
                        valid_keys = tl.broadcast_to(
                            current_n[None, :] < chunk_tail_lim, (block_m, 64)
                        )
                    else:
                        valid_keys = tl.full((block_m, 64), True, dtype=tl.int1)
                    valid_keys = (
                        valid_keys
                        & (current_n[None, :] >= (c_lo + row_q_idx)[:, None])
                        & (current_n[None, :] <= (c_hi + row_q_idx)[:, None])
                    )
                    scores = tl.where(chunk_active[:, None] & valid_keys, scores, -float("inf"))
                elif is_tail or is_causal_mode:
                    global_n = chunk.to(tl.int64) * 1024 + current_n.to(tl.int64)
                    if is_tail:
                        valid_keys = tl.broadcast_to(
                            global_n[None, :] < active_count.to(tl.int64), (block_m, 64)
                        )
                    else:
                        valid_keys = tl.full((block_m, 64), True, dtype=tl.int1)
                    if is_causal_mode:
                        valid_keys = valid_keys & (
                            global_n[None, :]
                            <= (global_query_start + row_q_idx.to(tl.int64))[:, None]
                        )
                    scores = tl.where(chunk_active[:, None] & valid_keys, scores, -float("inf"))
                raw_vs = tl.load(vs_chunk_ptr + current_n)
                value_scale_multiplier = raw_vs * _P_UINT8_RANGE
                value_log_scale = tl.where(
                    raw_vs == 0.0,
                    _LOG2_SCALE_EPSILON,
                    conservative_value_log_scale_bound(value_scale_multiplier),
                )
                if is_tail or has_window:
                    shifted_scores = tl.where(
                        scores != -float("inf"), scores + value_log_scale[None, :], -float("inf")
                    )
                    block_max = tl.where(
                        chunk_active, tl.reduce(shifted_scores, 1, max_propagate_nan), -float("inf")
                    )
                else:
                    shifted_scores = tl.where(
                        chunk_active[:, None], scores + value_log_scale[None, :], scores
                    )
                    block_max = tl.where(
                        valid_queries,
                        tl.reduce(shifted_scores, 1, max_propagate_nan),
                        -float("inf"),
                    )
                tile_k_bias = (
                    tl.where(block_max != -float("inf"), k_bias, 0.0)
                    if (is_tail or has_window)
                    else k_bias
                )
                next_max, next_bias, old_weight, current_weight, has_block, safe_block_max = (
                    advance_tile_max_safe(
                        running_max,
                        running_bias,
                        block_max,
                        tile_k_bias,
                        valid_queries,
                    )
                )
                probabilities = _safe_softmax_probabilities(
                    scores,
                    safe_block_max,
                    value_log_scale,
                )
                row_sum = tl.where(has_block, tl.sum(probabilities, axis=1) * current_weight, 0.0)
                denominator = denominator * old_weight + row_sum
                chunk_denom = chunk_denom * old_weight + row_sum
                probability_uint8 = _quantize_probabilities_u8(
                    probabilities,
                    value_scale_multiplier,
                    use_packed_probability_conversion=tl.constexpr(True),
                    active_mask=(chunk_active & has_block)[:, None],
                )

                v_tile_base = v_chunk_ptr + tile * (head_dim * 64)
                if split_pv_head_dim:
                    v_lo = tl.trans(
                        tl.load(v_tile_base + offsets_vd[:, None] * 64 + offsets_n[None, :])
                    )
                    v_hi = tl.trans(
                        tl.load(
                            v_tile_base
                            + (half_head_dim + offsets_vd[:, None]) * 64
                            + offsets_n[None, :]
                        )
                    )
                    v_tile = (v_lo, v_hi)
                else:
                    v_tile = tl.trans(
                        tl.load(v_tile_base + offsets_d[:, None] * 64 + offsets_n[None, :])
                    )
                numerator = _accumulate_pv_tile(
                    numerator,
                    probability_uint8,
                    v_tile,
                    old_weight,
                    current_weight,
                    split_pv_head_dim=split_pv_head_dim,
                )
                running_max = next_max
                running_bias = next_bias

            numerator = _fold_chunk_value_mean(
                numerator,
                chunk_denom,
                vm_chunk_ptr,
                offsets_d,
                offsets_vd,
                split_pv_head_dim=split_pv_head_dim,
                head_dim=head_dim,
                guard_zero=tl.constexpr(True),
                active_mask=None if (is_tail or has_window) else chunk_active,
            )

    has_any = running_max != -float("inf")
    safe_denom = tl.where(has_any & (denominator != 0.0), denominator, 1.0)
    denominator_code_units = safe_denom[:, None] * _P_UINT8_RANGE
    out_batch64 = (
        (
            (tl.program_id(2) // split_chunks).to(tl.int64) * total_splits.to(tl.int64)
            + (tl.program_id(2) % split_chunks).to(tl.int64)
        )
        if has_chunk_splits
        else batch.to(tl.int64)
    )
    lse_off = out_batch64 * stride_lb + head.to(tl.int64) * stride_lh + output_rows
    if has_chunk_splits:
        lse = tl.where(has_any, running_max, -float("inf"))
        bias = tl.where(has_any, running_bias, 0.0)
        log_denom = tl.where(has_any, tl.math.log2(safe_denom), 0.0)
        tl.store(lse_ptr + lse_off, lse, mask=valid_queries)
        tl.store(partial_bias_ptr + lse_off, bias, mask=valid_queries)
        tl.store(partial_log_denom_ptr + lse_off, log_denom, mask=valid_queries)
    else:
        lse = tl.where(
            has_any,
            ((running_max + running_bias) + tl.math.log2(safe_denom)) * _LN2,
            -float("inf"),
        )
        tl.store(lse_ptr + lse_off, lse, mask=valid_queries)
    output_base = (
        output_ptr
        + out_batch64 * stride_ob
        + head.to(tl.int64) * stride_oh
        + output_rows[:, None].to(tl.int64) * stride_om
    )
    _store_pv_output(
        output_base,
        numerator,
        denominator_code_units,
        offsets_d,
        offsets_vd,
        valid_queries,
        has_any,
        split_pv_head_dim=split_pv_head_dim,
        head_dim=head_dim,
        mask_empty=tl.constexpr(True),
    )


def launch_prefill_kernel(
    operands: PrefillKernelOperands,
    *,
    batch: int,
    gqa_group: int,
    head_dim: int,
    grouped_qk: bool,
    has_chunk_map: bool,
    is_tail: bool,
    is_causal: bool,
    split_chunks: int = 1,
    total_splits: int | None = None,
    has_chunk_splits: bool | None = None,
    per_row_query_scale: bool = False,
    partial_bias: torch.Tensor | None = None,
    partial_log_denom: torch.Tensor | None = None,
    target: AcceleratorTarget | None = None,
    tail_len: torch.Tensor | None = None,
    tail_cap: int = 0,
    window_mode: WindowMode | None = None,
    window_left: int = -1,
    window_right: int = -1,
    logical_seq_q: int = 0,
) -> None:
    """Launch NVIDIA unified chunked prefill kernel with typed pre-assembled operands."""
    device = operands.query_int8.device
    resolved_target = AcceleratorTarget.from_device(device) if target is None else target
    resolved_total = split_chunks if total_splits is None else total_splits
    resolved_chunk_splits = (resolved_total > 1) if has_chunk_splits is None else has_chunk_splits
    if resolved_chunk_splits and (partial_bias is None or partial_log_denom is None):
        raise ValueError("partial_bias and partial_log_denom are required when has_chunk_splits")
    resolved_mode = resolve_window_mode(is_causal, window_mode)
    block_m, num_warps, split_pv, num_stages, loop_num_stages = _policy.select_prefill_config(
        resolved_target,
        head_dim=head_dim,
        is_tail=is_tail,
        has_chunk_map=has_chunk_map,
        has_chunk_splits=resolved_chunk_splits,
        window_mode=resolved_mode,
    )
    bias_target = partial_bias if partial_bias is not None else operands.lse
    log_denom_target = partial_log_denom if partial_log_denom is not None else operands.lse
    with device_context(device):
        install_uint8_int8_dot_hook()
        _piper_chunked_prefill_kernel[
            (triton.cdiv(operands.seq_q, block_m), operands.q_heads, batch * split_chunks)
        ](
            *operands.kernel_args(),
            split_chunks,
            bias_target,
            log_denom_target,
            resolved_total,
            query_heads_per_kv_head=gqa_group,
            head_dim=head_dim,
            grouped_qk=grouped_qk,
            split_pv_head_dim=split_pv,
            has_chunk_map=has_chunk_map,
            is_tail=is_tail,
            window_mode=resolved_mode,
            block_m=block_m,
            loop_num_stages=loop_num_stages,
            has_chunk_splits=resolved_chunk_splits,
            per_row_query_scale=per_row_query_scale,
            tail_len_ptr=tail_len,
            tail_cap=tail_cap,
            window_left=window_left,
            window_right=window_right,
            logical_seq_q=logical_seq_q,
            has_tail=tail_len is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def launch_tail_prefill(
    query_int8: torch.Tensor,
    query_scale: torch.Tensor,
    tail_key_int8: torch.Tensor,
    tail_key_scale: torch.Tensor,
    tail_key_center: torch.Tensor,
    tail_value_int8: torch.Tensor,
    tail_value_scale: torch.Tensor,
    tail_value_mean: torch.Tensor,
    tail_len: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    seq_q: int,
    tail_cap: int,
    is_causal: bool,
    grouped_qk: bool,
    target: AcceleratorTarget | None = None,
    window_mode: WindowMode | None = None,
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch NVIDIA zero-spill offset-window quantized tail prefill returning output and LSE."""
    batch, q_heads, _, head_dim = output.shape
    tail_operands = PrefillKernelOperands(
        query_int8=query_int8,
        query_scale=query_scale,
        key_int8=tail_key_int8,
        key_scale=tail_key_scale,
        key_center=tail_key_center,
        value_int8=tail_value_int8,
        value_scale=tail_value_scale,
        value_mean=tail_value_mean,
        chunk_to_slot=tail_len,
        num_chunks=tail_len,
        output=output,
        lse=lse,
        seq_q=seq_q,
        query_storage=query_int8.shape[2],
        q_heads=q_heads,
        num_slots=tail_key_int8.shape[2],
        max_chunks=tail_cap,
    )
    launch_prefill_kernel(
        tail_operands,
        batch=batch,
        gqa_group=q_heads // tail_key_int8.shape[1],
        head_dim=head_dim,
        grouped_qk=grouped_qk,
        has_chunk_map=False,
        is_tail=True,
        is_causal=is_causal,
        target=target,
        window_mode=window_mode,
        window_left=window_left,
        window_right=window_right,
    )
    return output, lse
