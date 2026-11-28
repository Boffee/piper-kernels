"""Shared MoE Triton routing, reduction, and activation kernels."""

# Triton's JIT launcher, constexpr arguments, and wide kernel signatures are not
# fully represented by Python typing rules.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from piper_kernels._input_activations.triton import gelu_tanh, swiglu
from piper_kernels.linear.nvfp4._w4a16_triton import _split_global_scale
from piper_kernels.weights.convrot.triton import rotate_hadamard_groups

_CUDA_MAX_GRID_X = 0x7FFF_FFFF
_HIP_MAX_WORKITEMS_X = 0xFFFF_FFFF
# Conservative wave64 bound covers both wave32 (RDNA) and wave64 (CDNA) HIP targets.
_HIP_MAX_WARP_SIZE = 64


def moe_2d_tile_grid(
    n_tiles: int,
    item_count: int,
    *,
    num_warps: int,
    is_amd_hip: bool,
    split_k: int = 1,
) -> tuple[int, int, int]:
    """Pack ``(n_tiles, item_count)`` across ``(grid_x, grid_y, split_k)`` safely."""
    if n_tiles <= 0 or item_count <= 0:
        return (0, 0, split_k)
    max_grid_x = (
        _HIP_MAX_WORKITEMS_X // (num_warps * _HIP_MAX_WARP_SIZE) if is_amd_hip else _CUDA_MAX_GRID_X
    )
    if n_tiles * item_count <= max_grid_x:
        return (n_tiles * item_count, 1, split_k)
    items_per_x = max(1, max_grid_x // n_tiles)
    return (n_tiles * items_per_x, triton.cdiv(item_count, items_per_x), split_k)


@triton.jit
def _unpack_2d_tile_pid(n_tiles):
    """Unpack ``(pid_n, item_idx)`` from ``(n_tiles * items_per_x, grid_y[, split_k])``."""
    pid_0 = tl.program_id(0)
    pid_n = pid_0 % n_tiles
    item_idx = pid_0 // n_tiles
    if tl.num_programs(1) > 1:
        items_per_x = tl.num_programs(0) // n_tiles
        item_idx = tl.program_id(1) * items_per_x + item_idx
    return pid_n, item_idx


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


@triton.jit(do_not_specialize=["num_pairs", "gate_bound", "up_bound"])
def moe_m1_activate_rotate_kernel(
    mid_ptr,
    act_ptr,
    logical_ref_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    intermediate_size,
    num_experts,
    num_slots,
    num_pairs,
    gate_bound,
    up_bound,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    is_gated: tl.constexpr,
    gate_first: tl.constexpr,
    has_gate_bound: tl.constexpr,
    has_up_bound: tl.constexpr,
    accelerator_backend: tl.constexpr,
    has_slot_map: tl.constexpr,
):
    """Apply SwiGLU (optionally bounded) or GeLU-tanh and optional ConvRot transform."""
    n_tiles = tl.cdiv(intermediate_size, block_size)
    tile_idx, k_idx = _unpack_2d_tile_pid(n_tiles)
    if k_idx >= num_pairs:
        return
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
    up_gate_stride = intermediate_size * (2 if is_gated else 1)
    k_base = k_idx.to(tl.int64) * up_gate_stride

    if is_gated:
        first = tl.load(mid_ptr + k_base + offs, mask=valid, other=0.0).to(tl.float32)
        second = tl.load(mid_ptr + k_base + intermediate_size + offs, mask=valid, other=0.0).to(
            tl.float32
        )
        gate = first if gate_first else second
        up = second if gate_first else first
        if has_gate_bound:
            gate = tl.minimum(gate, gate_bound)
        if has_up_bound:
            up = tl.clamp(up, -up_bound, up_bound)
        act = swiglu(up, gate)
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
):
    """Deterministically reduce Split-K up/gate partials across active slots."""
    n_tiles = tl.cdiv(out_features, block_n)
    pid_n, k_idx = _unpack_2d_tile_pid(n_tiles)
    if k_idx >= num_pairs:
        return
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
    tl.store(output_ptr + k_idx.to(tl.int64) * out_features + offs_n_i64, out_val, mask=n_valid)


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
