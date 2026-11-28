"""Shared MoE Triton routing, reduction, activation kernels, and schedule routers."""

# Triton's JIT launcher, constexpr arguments, and wide kernel signatures are not
# fully represented by Python typing rules.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from piper_kernels._input_activations.triton import gelu_tanh, swiglu
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.nvfp4._w4a16_triton import _split_global_scale
from piper_kernels.weights.convrot.triton import rotate_hadamard_groups, rotate_input

from . import _schedule
from ._amd import policy as _amd_policy
from ._nvidia import policy as _nvidia_policy


def select_nvfp4_m1_schedule(target: AcceleratorTarget) -> _schedule.Nvfp4M1Schedule:
    """Resolve the NVFP4 ``rows == 1`` decode schedule for ``target``."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.select_nvfp4_m1_schedule(target)
    if target.is_amd_hip:
        return _amd_policy.select_nvfp4_m1_schedule(target)
    return _schedule.default_nvfp4_m1_schedule()


def select_nvfp4_batched_schedule(
    target: AcceleratorTarget,
    *,
    rows: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> _schedule.Nvfp4BatchedSchedule:
    """Resolve the NVFP4 batched decode schedule for ``target``."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.select_nvfp4_batched_schedule(
            target, rows=rows, active_dtype=active_dtype, logical_dtype=logical_dtype
        )
    if target.is_amd_hip:
        return _amd_policy.select_nvfp4_batched_schedule(
            target, rows=rows, active_dtype=active_dtype, logical_dtype=logical_dtype
        )
    return _schedule.default_nvfp4_batched_schedule(
        rows=rows, active_dtype=active_dtype, logical_dtype=logical_dtype
    )


def select_nvfp4_prefill_schedule(
    target: AcceleratorTarget,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> _schedule.Nvfp4PrefillSchedule:
    """Resolve the NVFP4 grouped prefill schedule for ``target``."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.select_nvfp4_prefill_schedule(
            target,
            chunk_rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            active_dtype=active_dtype,
            logical_dtype=logical_dtype,
        )
    if target.is_amd_hip:
        return _amd_policy.select_nvfp4_prefill_schedule(
            target,
            chunk_rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            active_dtype=active_dtype,
            logical_dtype=logical_dtype,
        )
    return _schedule.default_nvfp4_prefill_schedule(
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        active_dtype=active_dtype,
        logical_dtype=logical_dtype,
    )


def select_int8_launch_schedule(target: AcceleratorTarget) -> _schedule.Int8LaunchSchedule:
    """Resolve the INT8 decode launch schedule for ``target``."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.select_int8_launch_schedule(target)
    if target.is_amd_hip:
        return _amd_policy.select_int8_launch_schedule(target)
    return _schedule.default_int8_launch_schedule()


def select_int8_prefill_schedule(
    target: AcceleratorTarget,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
) -> _schedule.Int8PrefillSchedule:
    """Resolve the INT8 grouped prefill schedule for ``target``."""
    return _schedule.build_int8_prefill_schedule(
        select_int8_launch_schedule(target),
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
    )


def select_trellis_schedule(
    target: AcceleratorTarget,
    *,
    batched: bool,
    num_pairs: int,
    expected_slots: int,
) -> _schedule.TrellisMoeSchedule:
    """Resolve the Trellis GEMV schedule for ``target``."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.select_trellis_schedule(
            target, batched=batched, num_pairs=num_pairs, expected_slots=expected_slots
        )
    if target.is_amd_hip:
        return _amd_policy.select_trellis_schedule(
            target, batched=batched, num_pairs=num_pairs, expected_slots=expected_slots
        )
    return _schedule.default_trellis_schedule(batched=batched)


def supports_trellis_int8_prefill(target: AcceleratorTarget) -> bool:
    """Return whether ``target`` permits routing Trellis prefill through an INT8 backend."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.supports_trellis_int8_prefill(target)
    return True


@triton.jit
def _swap_halves_col(offs_n_i64, out_features, swap_halves: tl.constexpr):
    """Map column offsets to swap upper/lower SwiGLU halves when gate precedes up."""
    if not swap_halves:
        return offs_n_i64
    half_n = out_features // 2
    return tl.where(offs_n_i64 < half_n, offs_n_i64 + half_n, offs_n_i64 - half_n)


@triton.jit
def _resolve_slot(
    topk_ids_ptr,
    expert_to_slot_ptr,
    k_idx,
    num_experts,
    num_slots,
    has_slot_map: tl.constexpr,
):
    """Resolve logical expert ID for ``k_idx`` to a slot index, or ``-1`` on miss."""
    expert_id = tl.load(topk_ids_ptr + k_idx).to(tl.int64)
    if has_slot_map:
        in_range = (expert_id >= 0) & (expert_id < num_experts)
        safe_expert = tl.where(in_range, expert_id, 0)
        slot = tl.load(expert_to_slot_ptr + safe_expert).to(tl.int64)
        slot = tl.where(in_range, slot, -1)
    else:
        slot = expert_id
    return tl.where((slot >= 0) & (slot < num_slots), slot, -1)


@triton.jit
def _segmented_count_combine(slot_a, cnt_a, slot_b, cnt_b):
    """Segmented inclusive-scan combiner for counting positions within each sorted slot run."""
    same = slot_a == slot_b
    return slot_b, tl.where(same, cnt_a + cnt_b, cnt_b)


@triton.jit
def moe_bucket_tokens_kernel(
    topk_ids_ptr,
    expert_to_slot_ptr,
    tile_slots_ptr,
    tile_pair_ids_ptr,
    num_experts,
    num_slots,
    num_pairs,
    max_tiles,
    block_p: tl.constexpr,
    block_m: tl.constexpr,
    has_slot_map: tl.constexpr,
):
    """Group ``[0, num_pairs)`` hit assignments into fixed-width ``block_m`` slot tiles."""
    p_offs = tl.arange(0, block_p)
    valid = p_offs < num_pairs
    safe_p = tl.where(valid, p_offs, 0)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        safe_p,
        num_experts,
        num_slots,
        has_slot_map,
    )
    sort_slot = tl.where(valid & (slot >= 0), slot, num_slots)

    keys = (sort_slot << 32) | p_offs.to(tl.int64)
    sorted_keys = tl.sort(keys)
    # Triton's sort stub returns a tensor-or-tuple union.
    sorted_slot = (sorted_keys >> 32).to(tl.int32)  # pyright: ignore[reportOperatorIssue]
    sorted_pair = (sorted_keys & 0xFFFFFFFF).to(tl.int32)  # pyright: ignore[reportOperatorIssue]
    is_hit = sorted_slot < num_slots

    _, run_pos_1based = tl.associative_scan(
        (sorted_slot, tl.where(is_hit, 1, 0).to(tl.int32)),
        axis=0,
        combine_fn=_segmented_count_combine,
    )
    pos_in_run = run_pos_1based - 1
    lane_in_tile = pos_in_run % block_m
    is_tile_start = is_hit & (lane_in_tile == 0)
    tile_id = tl.cumsum(is_tile_start.to(tl.int32), axis=0) - 1
    num_active_tiles = tl.sum(is_tile_start.to(tl.int32))

    # Active tiles fill [0, num_active_tiles); inactive tail fills [num_active_tiles, max_tiles).
    tl.store(tile_slots_ptr + tile_id.to(tl.int64), sorted_slot, mask=is_tile_start)
    tl.store(
        tile_slots_ptr + p_offs.to(tl.int64),
        tl.full((block_p,), -1, tl.int32),
        mask=(p_offs >= num_active_tiles) & (p_offs < max_tiles),
    )
    tl.store(
        tile_pair_ids_ptr + tile_id.to(tl.int64) * block_m + lane_in_tile.to(tl.int64),
        sorted_pair,
        mask=is_hit,
    )


@triton.jit
def moe_m1_activate_rotate_kernel(
    mid_ptr,
    act_ptr,
    logical_ref_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    intermediate_size,
    num_experts,
    num_slots,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    activation_fn: tl.constexpr,
    gate_first: tl.constexpr,
    accelerator_backend: tl.constexpr,
    has_slot_map: tl.constexpr,
):
    """Apply SwiGLU or GeLU-tanh and optional ConvRot transform for active slots."""
    n_tiles = tl.cdiv(intermediate_size, block_size)
    pid_0 = tl.program_id(0)
    tile_idx = pid_0 % n_tiles
    k_idx = pid_0 // n_tiles
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        return

    offs = tile_idx * block_size + tl.arange(0, block_size)
    valid = offs < intermediate_size
    up_gate_stride = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    k_base = k_idx.to(tl.int64) * up_gate_stride

    if activation_fn == "swiglu":
        first = tl.load(mid_ptr + k_base + offs, mask=valid, other=0.0).to(tl.float32)
        second = tl.load(mid_ptr + k_base + intermediate_size + offs, mask=valid, other=0.0).to(
            tl.float32
        )
        act = swiglu(second, first) if gate_first else swiglu(first, second)
    else:
        vals = tl.load(mid_ptr + k_base + offs, mask=valid, other=0.0).to(tl.float32)
        act = gelu_tanh(vals, accelerator_backend)

    # Round activation outputs to logical_dtype before optional FP32 ConvRot rotation.
    act = act.to(logical_ref_ptr.dtype.element_ty).to(tl.float32)
    if group_size > 0:
        act = rotate_hadamard_groups(act, block_size, group_size, normalize=True)
    tl.store(
        act_ptr + k_idx.to(tl.int64) * intermediate_size + offs,
        act.to(act_ptr.dtype.element_ty),
        mask=valid,
    )


@triton.jit
def _reduce_splitk_slot_value(
    partial_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    pair_idx,
    slot,
    offs_n_i64,
    n_valid,
    out_features,
    split_stride,
    split_k,
    block_n: tl.constexpr,
    is_int8: tl.constexpr,
    has_global_scale: tl.constexpr,
    post_scale_only: tl.constexpr,
    has_row_scale: tl.constexpr = False,
    top_k=1,
    scale_by_token: tl.constexpr = False,
):
    base_ptr = partial_ptr + pair_idx * out_features + offs_n_i64
    if is_int8:
        acc_i32 = tl.zeros((block_n,), dtype=tl.int32)
        ptr = base_ptr
        for _ in range(split_k):
            acc_i32 += tl.load(ptr, mask=n_valid, other=0)
            ptr += split_stride
        in_scale = tl.load(input_scale_ptr + (pair_idx // top_k if scale_by_token else pair_idx))
        w_scale = tl.load(
            weight_scale_ptr + slot * out_features + offs_n_i64,
            mask=n_valid,
            other=0.0,
        )
        return acc_i32.to(tl.float32) * in_scale * w_scale
    expert_val = tl.zeros((block_n,), dtype=tl.float32)
    ptr = base_ptr
    for _ in range(split_k):
        expert_val += tl.load(ptr, mask=n_valid, other=0.0)
        ptr += split_stride
    if has_row_scale:
        w_scale = tl.load(
            weight_scale_ptr + slot * out_features + offs_n_i64,
            mask=n_valid,
            other=0.0,
        ).to(tl.float32)
        return expert_val * w_scale
    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        if post_scale_only:
            _, post_scale = _split_global_scale(global_scale)
            return expert_val * post_scale
        return expert_val * global_scale
    return expert_val


@triton.jit(do_not_specialize=["split_k"])
def moe_splitk_reduce_kernel(
    partial_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    output_ptr,
    out_features,
    num_experts,
    num_slots,
    num_pairs,
    top_k,
    split_k,
    block_n: tl.constexpr,
    is_int8: tl.constexpr,
    has_slot_map: tl.constexpr,
    has_global_scale: tl.constexpr = False,
    post_scale_only: tl.constexpr = True,
    has_row_scale: tl.constexpr = False,
    swap_halves: tl.constexpr = False,
):
    """Deterministically reduce Split-K up/gate partials across active slots."""
    n_tiles = tl.cdiv(out_features, block_n)
    pid_0 = tl.program_id(0)
    pid_n = pid_0 % n_tiles
    k_idx = pid_0 // n_tiles
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    n_valid = offs_n < out_features
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        if is_int8:
            zeros = tl.zeros((block_n,), dtype=tl.float32).to(output_ptr.dtype.element_ty)
            tl.store(
                output_ptr + k_idx.to(tl.int64) * out_features + offs_n_i64,
                zeros,
                mask=n_valid,
            )
        return

    split_stride = tl.full((), num_pairs, tl.int64) * out_features
    reduced = _reduce_splitk_slot_value(
        partial_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        weight_global_scale_ptr,
        k_idx.to(tl.int64),
        slot,
        offs_n_i64,
        n_valid,
        out_features,
        split_stride,
        split_k,
        block_n,
        is_int8,
        has_global_scale,
        post_scale_only,
        has_row_scale,
        top_k=top_k,
        scale_by_token=is_int8,
    )
    out_val = reduced.to(output_ptr.dtype.element_ty) if is_int8 else reduced
    col_out = _swap_halves_col(offs_n_i64, out_features, swap_halves)
    tl.store(output_ptr + k_idx.to(tl.int64) * out_features + col_out, out_val, mask=n_valid)


@triton.jit(do_not_specialize=["split_k"])
def moe_m1_weighted_reduce_kernel(
    partial_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    expert_to_slot_ptr,
    output_ptr,
    out_features,
    out_row_stride,
    num_experts,
    num_slots,
    num_pairs,
    top_k,
    split_k,
    block_n: tl.constexpr,
    is_int8: tl.constexpr,
    has_slot_map: tl.constexpr,
    accumulate: tl.constexpr,
    has_global_scale: tl.constexpr = False,
    post_scale_only: tl.constexpr = True,
    use_compact_map: tl.constexpr = False,
    has_row_scale: tl.constexpr = False,
):
    """Deterministically combine active expert ``down`` partials across ``(N, rows)``."""
    pid_n = tl.program_id(0)
    row_idx = tl.program_id(1).to(tl.int64)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    n_valid = offs_n < out_features
    split_stride = tl.full((), num_pairs, tl.int64) * out_features

    total = tl.zeros((block_n,), dtype=tl.float32)
    had_hit = tl.full((), 0, tl.int32)
    pair_base = row_idx * top_k
    for k_idx in range(top_k):
        pair_idx = pair_base + k_idx
        if use_compact_map:
            compact_idx = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)
            if compact_idx >= 0:
                had_hit = tl.full((), 1, tl.int32)
                expert_val = tl.load(
                    partial_ptr + compact_idx * out_features + offs_n_i64,
                    mask=n_valid,
                    other=0.0,
                )
                router_weight = tl.load(topk_weights_ptr + pair_idx).to(tl.float32)
                total += expert_val * router_weight
        else:
            slot = _resolve_slot(
                topk_ids_ptr,
                expert_to_slot_ptr,
                pair_idx,
                num_experts,
                num_slots,
                has_slot_map,
            )
            if slot >= 0:
                had_hit = tl.full((), 1, tl.int32)
                expert_val = _reduce_splitk_slot_value(
                    partial_ptr,
                    input_scale_ptr,
                    weight_scale_ptr,
                    weight_global_scale_ptr,
                    pair_idx,
                    slot,
                    offs_n_i64,
                    n_valid,
                    out_features,
                    split_stride,
                    split_k,
                    block_n,
                    is_int8,
                    has_global_scale,
                    post_scale_only,
                    has_row_scale,
                )
                router_weight = tl.load(topk_weights_ptr + pair_idx).to(tl.float32)
                total += expert_val * router_weight

    out_row_ptr = output_ptr + row_idx * out_row_stride.to(tl.int64) + offs_n_i64
    if accumulate:
        if had_hit != 0:
            existing = tl.load(out_row_ptr, mask=n_valid, other=0.0).to(tl.float32)
            tl.store(
                out_row_ptr,
                (existing + total).to(output_ptr.dtype.element_ty),
                mask=n_valid,
            )
    else:
        tl.store(out_row_ptr, total.to(output_ptr.dtype.element_ty), mask=n_valid)


@triton.jit
def _swizzle_grouped_pid(pid, max_tiles, num_pid_n, group_m: tl.constexpr):
    if group_m > 0:
        num_pid_in_group = group_m * num_pid_n
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * group_m
        actual_group_m = tl.minimum(max_tiles - first_pid_m, group_m)
        tile_idx = first_pid_m + pid_in_group % actual_group_m
        pid_n = pid_in_group // actual_group_m
    else:
        tile_idx = pid // num_pid_n
        pid_n = pid % num_pid_n
    return tile_idx, pid_n


@triton.jit
def moe_prefill_bucket_count_kernel(
    topk_ids_ptr,
    expert_to_slot_ptr,
    slot_counts_ptr,
    pair_slots_ptr,
    pair_pos_ptr,
    num_experts,
    num_slots,
    num_pairs,
    block_p: tl.constexpr,
    has_slot_map: tl.constexpr,
):
    """Resolve slot IDs and atomically count per-slot hit positions without CPU syncs."""
    p_offs = tl.program_id(0).to(tl.int64) * block_p + tl.arange(0, block_p)
    valid = p_offs < num_pairs
    safe_p = tl.where(valid, p_offs, 0)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        safe_p,
        num_experts,
        num_slots,
        has_slot_map,
    )
    slot = tl.where(valid, slot, -1)
    tl.store(pair_slots_ptr + p_offs, slot.to(tl.int32), mask=valid)
    is_hit = valid & (slot >= 0)
    safe_slot = tl.where(is_hit, slot, 0)
    pos = tl.atomic_add(slot_counts_ptr + safe_slot, 1, mask=is_hit)
    tl.store(pair_pos_ptr + p_offs, pos, mask=is_hit)


@triton.jit
def moe_prefill_bucket_scatter_kernel(
    pair_slots_ptr,
    pair_pos_ptr,
    slot_counts_ptr,
    slot_offsets_ptr,
    slot_tile_base_ptr,
    compact_to_pair_ptr,
    pair_to_compact_ptr,
    tile_slots_ptr,
    tile_m_start_ptr,
    tile_m_len_ptr,
    num_pairs,
    block_p: tl.constexpr,
    block_m: tl.constexpr,
):
    """Scatter active ``(row, k)`` pairs into compacted rows and ``block_m`` tiles."""
    p_offs = tl.program_id(0).to(tl.int64) * block_p + tl.arange(0, block_p)
    valid = p_offs < num_pairs
    slot = tl.load(pair_slots_ptr + p_offs, mask=valid, other=-1).to(tl.int64)
    is_hit = valid & (slot >= 0)
    safe_slot = tl.where(is_hit, slot, 0)
    pos = tl.load(pair_pos_ptr + p_offs, mask=is_hit, other=0).to(tl.int64)
    slot_off = tl.load(slot_offsets_ptr + safe_slot, mask=is_hit, other=0).to(tl.int64)
    compact_idx = slot_off + pos
    tl.store(compact_to_pair_ptr + compact_idx, p_offs.to(tl.int32), mask=is_hit)
    tl.store(pair_to_compact_ptr + p_offs, compact_idx.to(tl.int32), mask=is_hit)

    is_tile_start = is_hit & ((pos % block_m) == 0)
    tile_base = tl.load(slot_tile_base_ptr + safe_slot, mask=is_tile_start, other=0).to(tl.int64)
    slot_cnt = tl.load(slot_counts_ptr + safe_slot, mask=is_tile_start, other=0).to(tl.int64)
    tile_id = tile_base + (pos // block_m)
    tile_len = tl.minimum(slot_cnt - pos, block_m).to(tl.int32)
    tl.store(tile_slots_ptr + tile_id, slot.to(tl.int32), mask=is_tile_start)
    tl.store(tile_m_start_ptr + tile_id, compact_idx.to(tl.int32), mask=is_tile_start)
    tl.store(tile_m_len_ptr + tile_id, tile_len, mask=is_tile_start)


@triton.jit
def moe_prefill_gather_kernel(
    input_ptr,
    input_scale_ptr,
    compact_to_pair_ptr,
    out_ptr,
    out_scale_ptr,
    hidden_size,
    top_k,
    block_k: tl.constexpr,
    has_scale: tl.constexpr,
):
    """Gather routed input rows (and optional INT8 scales) into contiguous slot-compacted order."""
    pid_k = tl.program_id(0)
    compact_idx = tl.program_id(1).to(tl.int64)
    pair_id = tl.load(compact_to_pair_ptr + compact_idx).to(tl.int64)
    if pair_id < 0:
        return
    src_row = pair_id // top_k
    k_offs = pid_k * block_k + tl.arange(0, block_k)
    k_valid = k_offs < hidden_size
    stride = hidden_size.to(tl.int64)
    vals = tl.load(input_ptr + src_row * stride + k_offs, mask=k_valid)
    tl.store(out_ptr + compact_idx * stride + k_offs, vals, mask=k_valid)
    if has_scale and pid_k == 0:
        in_scale = tl.load(input_scale_ptr + src_row)
        tl.store(out_scale_ptr + compact_idx, in_scale)


def launch_moe_prefill_bucket(
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
    block_m: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
]:
    """Group arbitrary prefill pairs into compacted slot order and ``block_m`` tile descriptors."""
    device = topk_ids_flat.device
    num_pairs = topk_ids_flat.shape[0]
    max_tiles = _schedule.max_bucket_tiles(num_pairs, num_slots, block_m)
    has_slot_map = expert_to_slot is not None
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots

    slot_counts = torch.zeros((num_slots,), device=device, dtype=torch.int32)
    pair_slots = torch.empty((num_pairs,), device=device, dtype=torch.int32)
    pair_pos = torch.empty((num_pairs,), device=device, dtype=torch.int32)
    tile_slots = torch.full((max_tiles,), -1, device=device, dtype=torch.int32)
    tile_m_start = torch.zeros((max_tiles,), device=device, dtype=torch.int32)
    tile_m_len = torch.zeros((max_tiles,), device=device, dtype=torch.int32)
    compact_to_pair = torch.full((num_pairs,), -1, device=device, dtype=torch.int32)
    pair_to_compact = torch.full((num_pairs,), -1, device=device, dtype=torch.int32)
    grid = (triton.cdiv(num_pairs, _schedule.PREFILL_BUCKET_BLOCK_P),)

    with device_context(device):
        moe_prefill_bucket_count_kernel[grid](
            topk_ids_flat,
            expert_to_slot,
            slot_counts,
            pair_slots,
            pair_pos,
            num_experts,
            num_slots,
            num_pairs,
            block_p=_schedule.PREFILL_BUCKET_BLOCK_P,
            has_slot_map=has_slot_map,
            num_warps=4,
        )
        slot_offsets = torch.cumsum(slot_counts, dim=0, dtype=torch.int32) - slot_counts
        tiles_per_slot = (slot_counts + (block_m - 1)) // block_m
        slot_tile_base = torch.cumsum(tiles_per_slot, dim=0, dtype=torch.int32) - tiles_per_slot
        moe_prefill_bucket_scatter_kernel[grid](
            pair_slots,
            pair_pos,
            slot_counts,
            slot_offsets,
            slot_tile_base,
            compact_to_pair,
            pair_to_compact,
            tile_slots,
            tile_m_start,
            tile_m_len,
            num_pairs,
            block_p=_schedule.PREFILL_BUCKET_BLOCK_P,
            block_m=block_m,
            num_warps=4,
        )
    return (
        tile_slots,
        tile_m_start,
        tile_m_len,
        compact_to_pair,
        pair_to_compact,
        max_tiles,
        slot_counts,
    )


def launch_moe_prefill_gather(
    input_chunk: torch.Tensor,
    input_scale_chunk: torch.Tensor | None,
    compact_to_pair: torch.Tensor,
    out_compact: torch.Tensor,
    out_scale_compact: torch.Tensor | None,
    hidden_size: int,
    top_k: int,
    *,
    has_scale: bool,
) -> None:
    num_pairs = compact_to_pair.shape[0]
    with device_context(input_chunk.device):
        moe_prefill_gather_kernel[
            (triton.cdiv(hidden_size, _schedule.PREFILL_GATHER_BLOCK_K), num_pairs)
        ](
            input_chunk,
            input_scale_chunk,
            compact_to_pair,
            out_compact,
            out_scale_compact,
            hidden_size,
            top_k,
            block_k=_schedule.PREFILL_GATHER_BLOCK_K,
            has_scale=has_scale,
            num_warps=4,
        )


def launch_moe_prefill_weighted_reduce(
    down_out: torch.Tensor,
    pair_to_compact: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    out_chunk: torch.Tensor,
    hidden_size: int,
    top_k: int,
    *,
    accumulate: bool,
) -> None:
    chunk_rows = out_chunk.shape[0]
    with device_context(down_out.device):
        moe_m1_weighted_reduce_kernel[
            (triton.cdiv(hidden_size, _schedule.WEIGHTED_REDUCE_BLOCK_N), chunk_rows)
        ](
            down_out,
            None,
            None,
            None,
            pair_to_compact,
            topk_weights_flat,
            None,
            out_chunk,
            hidden_size,
            out_chunk.stride(0),
            1,
            1,
            pair_to_compact.shape[0],
            top_k,
            1,
            block_n=_schedule.WEIGHTED_REDUCE_BLOCK_N,
            is_int8=False,
            has_slot_map=False,
            accumulate=accumulate,
            has_global_scale=False,
            post_scale_only=True,
            use_compact_map=True,
            num_warps=4,
        )


def launch_moe_bucket(
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Group ``topk_ids_flat`` hits into fixed ``_schedule.BATCH_BLOCK_M`` slot tiles on GPU."""
    device = topk_ids_flat.device
    num_pairs = topk_ids_flat.shape[0]
    max_tiles = _schedule.max_bucket_tiles(num_pairs, num_slots, _schedule.BATCH_BLOCK_M)
    block_p = triton.next_power_of_2(max(16, num_pairs))
    has_slot_map = expert_to_slot is not None
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots
    tile_slots = torch.empty((max_tiles,), device=device, dtype=torch.int32)
    tile_pair_ids = torch.full(
        (max_tiles, _schedule.BATCH_BLOCK_M), -1, device=device, dtype=torch.int32
    )
    with device_context(device):
        moe_bucket_tokens_kernel[(1,)](
            topk_ids_flat,
            expert_to_slot,
            tile_slots,
            tile_pair_ids,
            num_experts,
            num_slots,
            num_pairs,
            max_tiles,
            block_p=block_p,
            block_m=_schedule.BATCH_BLOCK_M,
            has_slot_map=has_slot_map,
            num_warps=4,
        )
    return tile_slots, tile_pair_ids, max_tiles


@dataclass(frozen=True, slots=True)
class A16ChunkRouting:
    topk_ids_flat: torch.Tensor
    expert_to_slot: torch.Tensor | None
    tile_slots: torch.Tensor | None
    tile_pair_ids: torch.Tensor | None
    num_experts: int
    num_pairs: int
    top_k: int
    has_slot_map: bool


def run_a16_chunk(
    active_x_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    out_chunk: torch.Tensor,
    *,
    num_slots: int,
    hidden_size: int,
    intermediate_size: int,
    up_gate_out: int,
    expert_to_slot: torch.Tensor | None,
    target: AcceleratorTarget,
    batched: bool,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    split_k_up: int,
    split_k_down: int,
    up_row_scale: torch.Tensor | None,
    down_row_scale: torch.Tensor | None,
    up_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    post_scale_only: bool,
    launch_proj: Callable[[bool, torch.Tensor, A16ChunkRouting, torch.Tensor, int], None],
) -> None:
    device = active_x_chunk.device
    chunk_rows = active_x_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    topk_ids_flat = topk_ids_chunk.reshape(-1)
    topk_weights_flat = topk_weights_chunk.reshape(-1)
    has_slot_map = expert_to_slot is not None
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots
    tile_slots, tile_pair_ids = (
        launch_moe_bucket(topk_ids_flat, expert_to_slot, num_slots)[:2] if batched else (None, None)
    )
    routing = A16ChunkRouting(
        topk_ids_flat=topk_ids_flat,
        expert_to_slot=expert_to_slot,
        tile_slots=tile_slots,
        tile_pair_ids=tile_pair_ids,
        num_experts=num_experts,
        num_pairs=num_pairs,
        top_k=top_k,
        has_slot_map=has_slot_map,
    )
    has_row_scale = up_row_scale is not None

    mid = torch.empty((num_pairs, up_gate_out), device=device, dtype=torch.float32)
    up_target = (
        torch.empty((split_k_up, num_pairs, up_gate_out), device=device, dtype=torch.float32)
        if split_k_up > 1
        else mid
    )
    act_dtype = torch.float32 if group_size > 0 else out_chunk.dtype
    act = torch.empty((num_pairs, intermediate_size), device=device, dtype=act_dtype)
    down_partial = torch.empty(
        (split_k_down, num_pairs, hidden_size), device=device, dtype=torch.float32
    )

    with device_context(device):
        launch_proj(False, active_x_chunk, routing, up_target, split_k_up)
        if split_k_up > 1:
            moe_splitk_reduce_kernel[
                (triton.cdiv(up_gate_out, _schedule.NVFP4_REDUCE_BLOCK_N) * num_pairs,)
            ](
                up_target,
                None,
                up_row_scale,
                up_global_scale,
                topk_ids_flat,
                expert_to_slot,
                mid,
                up_gate_out,
                num_experts,
                num_slots,
                num_pairs,
                top_k,
                split_k_up,
                block_n=_schedule.NVFP4_REDUCE_BLOCK_N,
                is_int8=False,
                has_slot_map=has_slot_map,
                has_global_scale=up_global_scale is not None,
                post_scale_only=post_scale_only,
                has_row_scale=has_row_scale,
                swap_halves=False,
                num_warps=4,
            )
        moe_m1_activate_rotate_kernel[
            (triton.cdiv(intermediate_size, _schedule.ACTIVATE_BLOCK_SIZE) * num_pairs,)
        ](
            mid,
            act,
            out_chunk,
            topk_ids_flat,
            expert_to_slot,
            intermediate_size,
            num_experts,
            num_slots,
            block_size=_schedule.ACTIVATE_BLOCK_SIZE,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            accelerator_backend=target.backend,
            has_slot_map=has_slot_map,
            num_warps=4,
        )
        launch_proj(True, act, routing, down_partial, split_k_down)
        moe_m1_weighted_reduce_kernel[
            (triton.cdiv(hidden_size, _schedule.WEIGHTED_REDUCE_BLOCK_N), chunk_rows)
        ](
            down_partial,
            None,
            down_row_scale,
            down_global_scale,
            topk_ids_flat,
            topk_weights_flat,
            expert_to_slot,
            out_chunk,
            hidden_size,
            out_chunk.stride(0),
            num_experts,
            num_slots,
            num_pairs,
            top_k,
            split_k_down,
            block_n=_schedule.WEIGHTED_REDUCE_BLOCK_N,
            is_int8=False,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            has_global_scale=down_global_scale is not None,
            post_scale_only=post_scale_only,
            has_row_scale=has_row_scale,
            num_warps=4,
        )


def prepare_a16_input_gpu(
    input_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    if prepared_data is not None:
        return prepared_data.reshape(input_2d.shape[0], input_2d.shape[1]).contiguous()
    input_c = input_2d.contiguous()
    if group_size == 0:
        return input_c
    rows, hidden_size = input_c.shape
    rotated = torch.empty((rows, hidden_size), device=input_c.device, dtype=torch.float32)
    rotate_input(input_c, rotated, group_size, num_warps=4)
    return rotated


def iter_a16_chunks(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    out_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    group_size: int,
    chunk_rows: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    rows, hidden_size = input_2d.shape
    prep_full = None if prepared_data is None else prepared_data.reshape(rows, hidden_size)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        active_chunk = prepare_a16_input_gpu(
            input_2d[start:end],
            None if prep_full is None else prep_full[start:end],
            group_size,
        )
        yield active_chunk, ids_2d[start:end], weights_2d[start:end], out_2d[start:end]
