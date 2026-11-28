"""ConvRot INT8 W8A8 MoE Triton kernels and leaf kernel launchers."""

# Triton's JIT launcher, constexpr arguments, and wide kernel signatures are not
# fully represented by Python typing rules.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.primitives import splitk_tile_bounds
from piper_kernels._triton.runtime import device_context
from piper_kernels.linear.convrot.int8.triton import _accumulate_int8_tile

from . import _schedule
from ._routing_triton import (
    _resolve_slot,
    _swap_halves_col,
    _swizzle_grouped_pid,
    _unpack_2d_tile_pid,
    moe_2d_tile_grid,
)


@triton.jit(do_not_specialize=["split_k"])
def int8_moe_m1_matmul_kernel(
    input_ptr,
    weight_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    output_ptr,
    out_features,
    in_features,
    num_experts,
    num_slots,
    top_k,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_slot_map: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    dequant_output: tl.constexpr,
    swap_halves: tl.constexpr,
    aligned_nk: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    """Single-token ConvRot INT8 projection across ``(N * top_k_x, grid_y, split_k)``."""
    n_tiles = tl.cdiv(out_features, block_n)
    pid_n, k_idx = _unpack_2d_tile_pid(n_tiles)
    if k_idx >= top_k:
        return
    pid_k = tl.program_id(2)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        if not is_splitk:
            zeros = tl.zeros((block_n,), dtype=tl.float32).to(output_ptr.dtype.element_ty)
            tl.store(
                output_ptr + k_idx.to(tl.int64) * out_features + offs_n_i64,
                zeros,
                mask=offs_n < out_features,
            )
        return

    offs_m = tl.arange(0, block_m)

    k_start, k_end = splitk_tile_bounds(tl.cdiv(in_features, block_k), pid_k, split_k)

    slot_input_ptr = input_ptr + (k_idx.to(tl.int64) * in_features if per_expert_input else 0)
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * in_features
    acc = _accumulate_int8_tile(
        slot_input_ptr,
        slot_weight_ptr,
        offs_m,
        offs_n,
        1,
        out_features,
        in_features,
        k_start,
        k_end,
        block_m,
        block_n,
        block_k,
        False,
        aligned_nk,
        accumulator_seed,
    )
    centered = acc - accumulator_seed

    if dequant_output:
        in_scale = tl.load(input_scale_ptr + (k_idx if per_expert_input else 0))
        w_scale = tl.load(
            weight_scale_ptr + slot * out_features + offs_n_i64,
            mask=offs_n < out_features,
            other=0.0,
        )
        out_vals = (centered.to(tl.float32) * in_scale * w_scale[None, :]).to(
            output_ptr.dtype.element_ty
        )
    else:
        out_vals = centered

    col_out = _swap_halves_col(offs_n_i64, out_features, swap_halves)

    row_idx = pid_k.to(tl.int64) * top_k + k_idx if is_splitk else k_idx.to(tl.int64)
    out_ptrs = (
        output_ptr + row_idx * out_features + tl.broadcast_to(col_out[None, :], (block_m, block_n))
    )
    row0_mask = (offs_m[:, None] == 0) & (offs_n[None, :] < out_features)
    tl.store(out_ptrs, out_vals, mask=row0_mask)


@triton.jit(do_not_specialize=["split_k"])
def int8_moe_batched_matmul_kernel(
    input_ptr,
    weight_ptr,
    tile_slots_ptr,
    tile_pair_ids_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    output_ptr,
    out_features,
    in_features,
    num_pairs,
    top_k,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    dequant_output: tl.constexpr,
    swap_halves: tl.constexpr,
    aligned_nk: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    """Slot-batched ConvRot INT8 projection across ``(N, bucket_tile, split_k)``."""
    pid_n = tl.program_id(0)
    tile_idx = tl.program_id(1)
    pid_k = tl.program_id(2)
    slot = tl.load(tile_slots_ptr + tile_idx).to(tl.int64)
    if slot < 0:
        return

    offs_m = tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    pair_ids = tl.load(tile_pair_ids_ptr + tile_idx.to(tl.int64) * block_m + offs_m).to(tl.int64)
    row_valid = pair_ids >= 0
    src_rows = tl.where(row_valid, pair_ids if per_expert_input else pair_ids // top_k, 0)

    k_start, k_end = splitk_tile_bounds(tl.cdiv(in_features, block_k), pid_k, split_k)

    offsets_k = tl.arange(0, block_k)
    offsets_k_i64 = offsets_k.to(tl.int64)
    k_stride = tl.full((), in_features, tl.int64)
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * in_features
    weight_pointers = (
        slot_weight_ptr
        + offs_n_i64[None, :] * k_stride
        + (tl.full((), k_start, tl.int64) * block_k + offsets_k_i64)[:, None]
    )
    src_row_offsets = src_rows * k_stride
    accumulator = tl.full((block_m, block_n), accumulator_seed, tl.int32)

    for k_offset in range(k_start, k_end):
        k_base = tl.full((), k_offset, tl.int64) * block_k
        in_ptrs = input_ptr + src_row_offsets[:, None] + (k_base + offsets_k_i64)[None, :]
        if aligned_nk:
            input_values = tl.load(in_ptrs, mask=row_valid[:, None], other=0)
            weight = tl.load(weight_pointers)
        else:
            remaining_k = in_features - k_offset * block_k
            input_values = tl.load(
                in_ptrs,
                mask=row_valid[:, None] & (offsets_k[None, :] < remaining_k),
                other=0,
            )
            weight = tl.load(
                weight_pointers,
                mask=(offs_n[None, :] < out_features) & (offsets_k[:, None] < remaining_k),
                other=0,
            )
        if accumulator_seed:
            accumulator = tl.dot(input_values, weight, accumulator)
        else:
            accumulator += tl.dot(input_values, weight)
        weight_pointers += block_k

    centered = accumulator - accumulator_seed
    if dequant_output:
        in_scales = tl.load(input_scale_ptr + src_rows, mask=row_valid, other=0.0)
        w_scale = tl.load(
            weight_scale_ptr + slot * out_features + offs_n_i64,
            mask=offs_n < out_features,
            other=0.0,
        )
        out_vals = (centered.to(tl.float32) * in_scales[:, None] * w_scale[None, :]).to(
            output_ptr.dtype.element_ty
        )
    else:
        out_vals = centered

    col_out = _swap_halves_col(offs_n_i64, out_features, swap_halves)

    dst_rows = tl.where(
        row_valid, pid_k.to(tl.int64) * num_pairs + pair_ids if is_splitk else pair_ids, 0
    )
    tl.store(
        output_ptr + dst_rows[:, None] * out_features + col_out[None, :],
        out_vals,
        mask=row_valid[:, None] & (offs_n[None, :] < out_features),
    )


@triton.jit(do_not_specialize=["slot_start", "slot_end", "n_start", "n_rows"])
def int8_moe_grouped_gemm_kernel(
    input_ptr,
    weight_ptr,
    tile_slots_ptr,
    tile_m_start_ptr,
    tile_m_len_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    output_ptr,
    max_tiles,
    out_features,
    in_features,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    swap_halves: tl.constexpr,
    aligned_nk: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
    slot_start=0,
    slot_end=0,
    n_start=0,
    n_rows=0,
    slot_window: tl.constexpr = False,
):
    """Grouped ConvRot INT8 W8A8 GEMM over contiguous slot-compacted tiles."""
    active_n_rows = n_rows if slot_window else out_features
    n_base = n_start if slot_window else 0
    num_pid_n = tl.cdiv(active_n_rows, block_n)
    tile_idx, pid_n = _swizzle_grouped_pid(tl.program_id(0), max_tiles, num_pid_n, group_m)
    slot = tl.load(tile_slots_ptr + tile_idx).to(tl.int64)
    if slot < 0:
        return
    if slot_window:  # noqa: SIM102 - keep constexpr guard outer so LLVM strips bounds checks
        if slot < slot_start or slot >= slot_end:
            return
    m_start = tl.load(tile_m_start_ptr + tile_idx).to(tl.int64)
    tile_len = tl.load(tile_m_len_ptr + tile_idx)

    offs_m = tl.arange(0, block_m)
    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_local = pid_n * block_n + tl.arange(0, block_n)
    offs_n_global_i64 = (n_base + offs_n_local).to(tl.int64)
    m_valid = offs_m < tile_len
    n_valid = offs_n_local < active_n_rows
    k_stride = in_features.to(tl.int64)
    slot_input_ptr = input_ptr + m_start * k_stride
    weight_slot = (slot - slot_start) if slot_window else slot
    slot_weight_ptr = weight_ptr + weight_slot * tl.full((), active_n_rows, tl.int64) * k_stride

    acc = _accumulate_int8_tile(
        slot_input_ptr,
        slot_weight_ptr,
        offs_m,
        offs_n_local,
        tile_len,
        active_n_rows,
        in_features,
        0,
        tl.cdiv(in_features, block_k),
        block_m,
        block_n,
        block_k,
        False,
        aligned_nk,
        accumulator_seed,
    )
    centered = acc - accumulator_seed
    in_scales = tl.load(input_scale_ptr + m_start + offs_m_i64, mask=m_valid, other=0.0)
    w_scale = tl.load(
        weight_scale_ptr + slot * out_features + offs_n_global_i64,
        mask=n_valid,
        other=0.0,
    )
    out_vals = (centered.to(tl.float32) * in_scales[:, None] * w_scale[None, :]).to(
        output_ptr.dtype.element_ty
    )

    col_out = _swap_halves_col(offs_n_global_i64, out_features, swap_halves)

    tl.store(
        output_ptr + (m_start + offs_m_i64)[:, None] * out_features + col_out[None, :],
        out_vals,
        mask=m_valid[:, None] & n_valid[None, :],
    )


def launch_int8_proj(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    in_scale: torch.Tensor,
    w_scale: torch.Tensor,
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    tile_slots: torch.Tensor | None,
    tile_pair_ids: torch.Tensor | None,
    output: torch.Tensor,
    *,
    out_tiles: int,
    out_features: int,
    in_features: int,
    num_experts: int,
    num_slots: int,
    num_pairs: int,
    top_k: int,
    split_k: int,
    block_m: int,
    block_k: int,
    num_stages: int,
    compiler_options: dict[str, object],
    has_slot_map: bool,
    per_expert_input: bool,
    is_splitk: bool,
    swap_halves: bool,
    aligned_nk: bool,
    accumulator_seed: int,
    is_amd_hip: bool = False,
) -> None:
    with device_context(x_q.device):
        if tile_slots is not None:
            assert tile_pair_ids is not None
            int8_moe_batched_matmul_kernel[(out_tiles, tile_slots.shape[0], split_k)](
                x_q,
                w_q,
                tile_slots,
                tile_pair_ids,
                in_scale,
                w_scale,
                output,
                out_features,
                in_features,
                num_pairs,
                top_k,
                split_k,
                block_m=block_m,
                block_n=_schedule.INT8_GEMM_BLOCK_N,
                block_k=block_k,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                dequant_output=not is_splitk,
                swap_halves=swap_halves,
                aligned_nk=aligned_nk,
                accumulator_seed=accumulator_seed,
                **compiler_options,
                num_stages=num_stages,
                num_warps=4,
            )
        else:
            grid = moe_2d_tile_grid(
                out_tiles,
                top_k,
                num_warps=4,
                is_amd_hip=is_amd_hip,
                split_k=split_k,
            )
            int8_moe_m1_matmul_kernel[grid](
                x_q,
                w_q,
                topk_ids_flat,
                expert_to_slot,
                in_scale,
                w_scale,
                output,
                out_features,
                in_features,
                num_experts,
                num_slots,
                top_k,
                split_k,
                block_m=block_m,
                block_n=_schedule.INT8_GEMM_BLOCK_N,
                block_k=block_k,
                has_slot_map=has_slot_map,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                dequant_output=not is_splitk,
                swap_halves=swap_halves,
                aligned_nk=aligned_nk,
                accumulator_seed=accumulator_seed,
                **compiler_options,
                num_stages=num_stages,
                num_warps=4,
            )


def launch_int8_grouped_gemm(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    tile_slots: torch.Tensor,
    tile_m_start: torch.Tensor,
    tile_m_len: torch.Tensor,
    in_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    *,
    max_tiles: int,
    out_features: int,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    compiler_options: dict[str, object],
    swap_halves: bool,
    aligned_nk: bool,
    accumulator_seed: int,
    slot_start: int = 0,
    slot_end: int = 0,
    n_start: int = 0,
    n_rows: int = 0,
    slot_window: bool = False,
) -> None:
    active_n_rows = n_rows if slot_window else out_features
    grid = (max_tiles * triton.cdiv(active_n_rows, block_n),)
    with device_context(x_q.device):
        int8_moe_grouped_gemm_kernel[grid](
            x_q,
            w_q,
            tile_slots,
            tile_m_start,
            tile_m_len,
            in_scale,
            w_scale,
            output,
            max_tiles,
            out_features,
            in_features,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=_schedule.PREFILL_GROUP_M,
            swap_halves=swap_halves,
            aligned_nk=aligned_nk,
            accumulator_seed=accumulator_seed,
            slot_start=slot_start,
            slot_end=slot_end,
            n_start=n_start,
            n_rows=n_rows,
            slot_window=slot_window,
            **compiler_options,
            num_stages=num_stages,
            num_warps=num_warps,
        )
