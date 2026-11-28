"""ConvRot INT8 W8A8 MoE Triton kernels and host dispatcher."""

# Triton's JIT launcher, constexpr arguments, and wide kernel signatures are not
# fully represented by Python typing rules.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.primitives import splitk_tile_bounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget, rdna4_int8_accumulator_seed
from piper_kernels.linear._splitk import device_sm_count
from piper_kernels.linear.convrot.int8.triton import _accumulate_int8_tile, splitk_accumulator_seed
from piper_kernels.trellis.triton import compute_row_offsets_2d, decode_slots_int8_out
from piper_kernels.weights.convrot.int8._quantization import dynamic_quantize_rows

from . import _routing_triton, _schedule
from ._routing_triton import (
    _resolve_slot,
    _swap_halves_col,
    _swizzle_grouped_pid,
    moe_m1_weighted_reduce_kernel,
    moe_splitk_reduce_kernel,
)

if TYPE_CHECKING:
    from piper_kernels.linear.convrot.int8._backend import LinearBackend, LinearExecutionPlan


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
    """Single-token slot-indirect ConvRot INT8 projection across ``(N * top_k, split_k)``."""
    n_tiles = tl.cdiv(out_features, block_n)
    pid_0 = tl.program_id(0)
    pid_n = pid_0 % n_tiles
    k_idx = pid_0 // n_tiles
    pid_k = tl.program_id(1)
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
            int8_moe_m1_matmul_kernel[(out_tiles * top_k, split_k)](
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


def run_int8_chunk(
    input_q_chunk: torch.Tensor,
    input_s_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    out_chunk: torch.Tensor,
    *,
    target: AcceleratorTarget,
    int8_backend: LinearBackend,
    down_prep_plan: LinearExecutionPlan,
    batched: bool,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    swap_halves: bool,
    block_k: int,
    num_stages: int,
    compiler_options: dict[str, object],
    split_k_up: int,
    split_k_down: int,
    up_seed: int,
    down_seed: int,
    up_aligned_nk: bool,
    down_aligned_nk: bool,
    up_tiles: int,
    down_tiles: int,
) -> None:
    device = input_q_chunk.device
    chunk_rows = input_q_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    num_slots, hidden_size, intermediate_size = down_qdata.shape
    up_gate_out = up_gate_qdata.shape[1]
    topk_ids_flat = topk_ids_chunk.reshape(-1)
    topk_weights_flat = topk_weights_chunk.reshape(-1)
    has_slot_map = expert_to_slot is not None
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots

    if batched:
        tile_slots, tile_pair_ids, _ = _routing_triton.launch_moe_bucket(
            topk_ids_flat, expert_to_slot, num_slots
        )
        block_m = _schedule.BATCH_BLOCK_M
    else:
        tile_slots = tile_pair_ids = None
        block_m = _schedule.INT8_GEMM_BLOCK_M

    # Batched INT8 skips missed rows when split_k_up == 1; zero mid before preparing down.
    mid = (
        torch.zeros((num_pairs, up_gate_out), device=device, dtype=out_chunk.dtype)
        if batched and split_k_up == 1
        else torch.empty((num_pairs, up_gate_out), device=device, dtype=out_chunk.dtype)
    )
    up_target = (
        torch.empty((split_k_up, num_pairs, up_gate_out), device=device, dtype=torch.int32)
        if split_k_up > 1
        else mid
    )

    with device_context(device):
        launch_int8_proj(
            input_q_chunk,
            up_gate_qdata,
            input_s_chunk,
            up_gate_scale,
            topk_ids_flat,
            expert_to_slot,
            tile_slots,
            tile_pair_ids,
            up_target,
            out_tiles=up_tiles,
            out_features=up_gate_out,
            in_features=hidden_size,
            num_experts=num_experts,
            num_slots=num_slots,
            num_pairs=num_pairs,
            top_k=top_k,
            split_k=split_k_up,
            block_m=block_m,
            block_k=block_k,
            num_stages=num_stages,
            compiler_options=compiler_options,
            has_slot_map=has_slot_map,
            per_expert_input=False,
            is_splitk=split_k_up > 1,
            swap_halves=(swap_halves and split_k_up == 1),
            aligned_nk=up_aligned_nk,
            accumulator_seed=up_seed,
        )
        if split_k_up > 1:
            moe_splitk_reduce_kernel[
                (triton.cdiv(up_gate_out, _schedule.INT8_REDUCE_BLOCK_N) * num_pairs,)
            ](
                up_target,
                input_s_chunk,
                up_gate_scale,
                None,
                topk_ids_flat,
                expert_to_slot,
                mid,
                up_gate_out,
                num_experts,
                num_slots,
                num_pairs,
                top_k,
                split_k_up,
                block_n=_schedule.INT8_REDUCE_BLOCK_N,
                is_int8=True,
                has_slot_map=has_slot_map,
                has_global_scale=False,
                post_scale_only=True,
                has_row_scale=False,
                swap_halves=swap_halves,
                num_warps=4,
            )

    act_q, act_s = int8_backend.prepare_input_with_plan(
        mid,
        intermediate_size,
        group_size,
        activation_fn=activation_fn,
        execution_plan=down_prep_plan,
        target=target,
    )
    down_partial = torch.empty(
        (split_k_down, num_pairs, hidden_size), device=device, dtype=torch.int32
    )

    with device_context(device):
        launch_int8_proj(
            act_q,
            down_qdata,
            act_s,
            down_scale,
            topk_ids_flat,
            expert_to_slot,
            tile_slots,
            tile_pair_ids,
            down_partial,
            out_tiles=down_tiles,
            out_features=hidden_size,
            in_features=intermediate_size,
            num_experts=num_experts,
            num_slots=num_slots,
            num_pairs=num_pairs,
            top_k=top_k,
            split_k=split_k_down,
            block_m=block_m,
            block_k=block_k,
            num_stages=num_stages,
            compiler_options=compiler_options,
            has_slot_map=has_slot_map,
            per_expert_input=True,
            is_splitk=True,
            swap_halves=False,
            aligned_nk=down_aligned_nk,
            accumulator_seed=down_seed,
        )
        moe_m1_weighted_reduce_kernel[
            (triton.cdiv(hidden_size, _schedule.WEIGHTED_REDUCE_BLOCK_N), chunk_rows)
        ](
            down_partial,
            act_s,
            down_scale,
            None,
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
            is_int8=True,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            has_global_scale=False,
            post_scale_only=True,
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


def run_prefill_int8_proj(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    tile_slots: torch.Tensor,
    tile_m_start: torch.Tensor,
    tile_m_len: torch.Tensor,
    output: torch.Tensor,
    *,
    block_scales: torch.Tensor | None,
    row_widths: torch.Tensor | None,
    slot_offsets: torch.Tensor | None,
    block_size: int,
    slot_counts: torch.Tensor,
    num_slots: int,
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
    accumulator_seed: int,
) -> None:
    if block_scales is None:
        launch_int8_grouped_gemm(
            x_q,
            qdata,
            tile_slots,
            tile_m_start,
            tile_m_len,
            x_scale,
            scale,
            output,
            max_tiles=max_tiles,
            out_features=out_features,
            in_features=in_features,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            compiler_options=compiler_options,
            swap_halves=swap_halves,
            aligned_nk=(out_features % block_n == 0 and in_features % block_k == 0),
            accumulator_seed=accumulator_seed,
        )
        return

    slot_decoded_bytes = out_features * in_features
    if slot_decoded_bytes <= _schedule.TRELLIS_SCRATCH_BYTES:
        max_chunk_slots = min(
            num_slots, _schedule.TRELLIS_SCRATCH_BYTES // max(1, slot_decoded_bytes)
        )
        n_chunk = out_features
    else:
        max_chunk_slots = 1
        raw_chunk_rows = max(1, _schedule.TRELLIS_SCRATCH_BYTES // max(1, in_features))
        n_chunk = (
            (raw_chunk_rows // block_n) * block_n if raw_chunk_rows >= block_n else raw_chunk_rows
        )

    scratch = torch.empty(
        (max_chunk_slots, n_chunk, in_features), device=x_q.device, dtype=torch.int8
    )
    block_scales_u8 = block_scales.view(torch.uint8)
    for slot_start in range(0, num_slots, max_chunk_slots):
        slot_end = min(slot_start + max_chunk_slots, num_slots)
        window_slots = slot_end - slot_start
        window_counts = slot_counts[slot_start:slot_end]
        window_slot_offsets = None if slot_offsets is None else slot_offsets[slot_start:slot_end]
        window_widths = None if row_widths is None else row_widths[slot_start:slot_end]
        window_offsets = (
            None
            if window_widths is None
            else compute_row_offsets_2d(window_widths, dtype=torch.int32)
        )
        for n_start in range(0, out_features, n_chunk):
            n_end = min(n_start + n_chunk, out_features)
            n_rows = n_end - n_start
            q_view = scratch[:window_slots, :n_rows]
            if row_widths is None:
                chunk_streams = (
                    qdata[slot_start:slot_end, n_start:n_end].contiguous()
                    if n_rows < out_features
                    else qdata[slot_start:slot_end]
                )
            elif slot_offsets is None:
                chunk_streams = qdata[slot_start:slot_end]
            else:
                chunk_streams = qdata
            decode_slots_int8_out(
                chunk_streams,
                block_scales_u8[slot_start:slot_end, n_start:n_end],
                q_view,
                block_size=block_size,
                row_widths=None if window_widths is None else window_widths[:, n_start:n_end],
                slot_offsets=window_slot_offsets,
                active_counts=window_counts,
                row_offsets=None if window_offsets is None else window_offsets[:, n_start:n_end],
            )
            launch_int8_grouped_gemm(
                x_q,
                q_view,
                tile_slots,
                tile_m_start,
                tile_m_len,
                x_scale,
                scale,
                output,
                max_tiles=max_tiles,
                out_features=out_features,
                in_features=in_features,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                compiler_options=compiler_options,
                swap_halves=swap_halves,
                aligned_nk=(n_rows % block_n == 0 and in_features % block_k == 0),
                accumulator_seed=accumulator_seed,
                slot_start=slot_start,
                slot_end=slot_end,
                n_start=n_start,
                n_rows=n_rows,
                slot_window=True,
            )


def run_int8_prefill_chunk(
    input_q_chunk: torch.Tensor,
    input_s_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    out_chunk: torch.Tensor,
    *,
    target: AcceleratorTarget,
    int8_backend: LinearBackend,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    swap_halves: bool,
    active_slots_hint: int,
    up_gate_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_scales: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    down_block_size: int = 0,
) -> None:
    device = input_q_chunk.device
    chunk_rows = input_q_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    if down_block_scales is not None:
        assert up_gate_block_scales is not None
        num_slots, hidden_size, down_blocks = down_block_scales.shape
        intermediate_size = down_blocks * down_block_size
        up_gate_out = up_gate_block_scales.shape[1]
    else:
        num_slots, hidden_size, intermediate_size = down_qdata.shape
        up_gate_out = up_gate_qdata.shape[1]
    prefill_schedule = _routing_triton.select_int8_prefill_schedule(
        target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
    )
    block_m = prefill_schedule.block_m
    block_n = prefill_schedule.block_n
    block_k = prefill_schedule.block_k
    num_warps = prefill_schedule.num_warps
    num_stages = prefill_schedule.num_stages
    compiler_options = amd_compiler_workarounds.lds_only_fence_options(
        prefill_schedule.lds_only_fences
    )
    (
        tile_slots,
        tile_m_start,
        tile_m_len,
        compact_to_pair,
        pair_to_compact,
        max_tiles,
        slot_counts,
    ) = _routing_triton.launch_moe_prefill_bucket(
        topk_ids_chunk.reshape(-1), expert_to_slot, num_slots, block_m
    )
    expected_tiles = _schedule.expected_bucket_tiles(
        num_pairs, num_slots, active_slots_hint, block_m=block_m, batched=True
    )
    up_seed = rdna4_int8_accumulator_seed(
        target, hidden_size, expected_tiles * triton.cdiv(up_gate_out, block_n)
    )
    down_seed = rdna4_int8_accumulator_seed(
        target, intermediate_size, expected_tiles * triton.cdiv(hidden_size, block_n)
    )
    x_compact = torch.empty((num_pairs, hidden_size), device=device, dtype=torch.int8)
    x_scale_compact = torch.empty((num_pairs,), device=device, dtype=torch.float32)
    _routing_triton.launch_moe_prefill_gather(
        input_q_chunk,
        input_s_chunk,
        compact_to_pair,
        x_compact,
        x_scale_compact,
        hidden_size,
        top_k,
        has_scale=True,
    )
    # Zero mid so inactive compacted tail rows produce finite zeroed activations.
    mid = torch.zeros((num_pairs, up_gate_out), device=device, dtype=out_chunk.dtype)
    run_prefill_int8_proj(
        x_compact,
        x_scale_compact,
        up_gate_qdata,
        up_gate_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        mid,
        block_scales=up_gate_block_scales,
        row_widths=up_gate_row_widths,
        slot_offsets=up_gate_slot_offsets,
        block_size=up_gate_block_size,
        slot_counts=slot_counts,
        num_slots=num_slots,
        max_tiles=max_tiles,
        out_features=up_gate_out,
        in_features=hidden_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        compiler_options=compiler_options,
        swap_halves=swap_halves,
        accumulator_seed=up_seed,
    )
    del x_compact, x_scale_compact
    down_prep_plan = int8_backend.policy.select_execution_plan(
        target, in_features=intermediate_size, rows=num_pairs, out_features=None
    )
    act_q, act_s = int8_backend.prepare_input_with_plan(
        mid,
        intermediate_size,
        group_size,
        activation_fn=activation_fn,
        execution_plan=down_prep_plan,
        target=target,
    )
    del mid
    down_out = torch.empty((num_pairs, hidden_size), device=device, dtype=torch.float32)
    run_prefill_int8_proj(
        act_q,
        act_s,
        down_qdata,
        down_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        down_out,
        block_scales=down_block_scales,
        row_widths=down_row_widths,
        slot_offsets=down_slot_offsets,
        block_size=down_block_size,
        slot_counts=slot_counts,
        num_slots=num_slots,
        max_tiles=max_tiles,
        out_features=hidden_size,
        in_features=intermediate_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        compiler_options=compiler_options,
        swap_halves=False,
        accumulator_seed=down_seed,
    )
    del act_q, act_s
    _routing_triton.launch_moe_prefill_weighted_reduce(
        down_out,
        pair_to_compact,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        hidden_size,
        top_k,
        accumulate=accumulate,
    )


def prepare_int8_input_gpu(
    input_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    group_size: int,
    *,
    target: AcceleratorTarget,
    int8_backend: LinearBackend,
    up_prep_plan: LinearExecutionPlan | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prepared_data is not None:
        rows, hidden_size = input_2d.shape
        prep_2d = prepared_data.reshape(rows, hidden_size).contiguous()
        if prepared_scale is not None:
            return prep_2d, prepared_scale.reshape(rows).contiguous()
        inp_q, inp_s = dynamic_quantize_rows(prep_2d)
        return inp_q, inp_s.squeeze(-1).contiguous()
    assert up_prep_plan is not None
    return int8_backend.prepare_input_with_plan(
        input_2d.contiguous(),
        input_2d.shape[1],
        group_size,
        activation_fn=None,
        execution_plan=up_prep_plan,
        target=target,
    )


def dispatch_int8_prefill_chunks(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    resolved_target: AcceleratorTarget,
    int8_backend: LinearBackend,
    chunk_rows: int,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    active_slots_hint: int,
    up_gate_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_scales: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    down_block_size: int = 0,
) -> None:
    rows, hidden_size = input_2d.shape
    swap_halves = activation_fn == "swiglu" and gate_first
    prep_q_full = None if prepared_data is None else prepared_data.reshape(rows, hidden_size)
    prep_s_full = None if prepared_scale is None else prepared_scale.reshape(rows)
    up_prep_plan = (
        None
        if prepared_data is not None
        else int8_backend.policy.select_execution_plan(resolved_target, in_features=hidden_size)
    )
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        inp_q_chunk, inp_s_chunk = prepare_int8_input_gpu(
            input_2d[start:end],
            None if prep_q_full is None else prep_q_full[start:end],
            None if prep_s_full is None else prep_s_full[start:end],
            group_size,
            target=resolved_target,
            int8_backend=int8_backend,
            up_prep_plan=up_prep_plan,
        )
        run_int8_prefill_chunk(
            inp_q_chunk,
            inp_s_chunk,
            ids_2d[start:end],
            weights_2d[start:end],
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            out_2d[start:end],
            target=resolved_target,
            int8_backend=int8_backend,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            swap_halves=swap_halves,
            active_slots_hint=active_slots_hint,
            up_gate_block_scales=up_gate_block_scales,
            up_gate_row_widths=up_gate_row_widths,
            up_gate_slot_offsets=up_gate_slot_offsets,
            up_gate_block_size=up_gate_block_size,
            down_block_scales=down_block_scales,
            down_row_widths=down_row_widths,
            down_slot_offsets=down_slot_offsets,
            down_block_size=down_block_size,
        )


def dispatch_int8_moe(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    resolved_target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    active_slots_hint: int,
) -> None:
    from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

    device = input_2d.device
    rows, hidden_size = input_2d.shape
    num_slots, _, intermediate_size = down_qdata.shape
    up_gate_out = up_gate_qdata.shape[1]
    top_k = ids_2d.shape[1]
    int8_backend = int8_backend_mod.select_linear_backend_for_target(resolved_target)
    assert int8_backend is not None
    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=True,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=out_2d.element_size(),
        group_size=group_size,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill":
        dispatch_int8_prefill_chunks(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            out_2d,
            resolved_target=resolved_target,
            int8_backend=int8_backend,
            chunk_rows=chunk_rows,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            active_slots_hint=active_slots_hint,
        )
        return

    swap_halves = activation_fn == "swiglu" and gate_first
    up_prep_plan = (
        None
        if prepared_data is not None
        else int8_backend.policy.select_execution_plan(resolved_target, in_features=hidden_size)
    )
    batched = regime == "batched"
    expected_tiles = _schedule.expected_bucket_tiles(
        chunk_rows * top_k, num_slots, active_slots_hint, batched=batched
    )
    down_prep_plan = int8_backend.policy.select_execution_plan(
        resolved_target,
        in_features=intermediate_size,
        rows=chunk_rows * top_k,
        out_features=None,
    )
    launch_schedule = _routing_triton.select_int8_launch_schedule(resolved_target)
    block_k = launch_schedule.block_k
    num_stages = launch_schedule.num_stages
    compiler_options = amd_compiler_workarounds.lds_only_fence_options(
        launch_schedule.lds_only_fences
    )
    split_k_up = _schedule.select_int8_split_k(
        device_sm_count(device), expected_tiles, up_gate_out, hidden_size, block_k
    )
    split_k_down = _schedule.select_int8_split_k(
        device_sm_count(device), expected_tiles, hidden_size, intermediate_size, block_k
    )
    up_seed = splitk_accumulator_seed(
        resolved_target,
        rows=expected_tiles * _schedule.INT8_GEMM_BLOCK_M,
        out_features=up_gate_out,
        in_features=hidden_size,
        block_m=_schedule.INT8_GEMM_BLOCK_M,
        block_n=_schedule.INT8_GEMM_BLOCK_N,
        block_k=block_k,
        paired=False,
        split_k=split_k_up,
    )
    down_seed = splitk_accumulator_seed(
        resolved_target,
        rows=expected_tiles * _schedule.INT8_GEMM_BLOCK_M,
        out_features=hidden_size,
        in_features=intermediate_size,
        block_m=_schedule.INT8_GEMM_BLOCK_M,
        block_n=_schedule.INT8_GEMM_BLOCK_N,
        block_k=block_k,
        paired=False,
        split_k=split_k_down,
    )
    up_aligned_nk = up_gate_out % _schedule.INT8_GEMM_BLOCK_N == 0 and hidden_size % block_k == 0
    down_aligned_nk = (
        hidden_size % _schedule.INT8_GEMM_BLOCK_N == 0 and intermediate_size % block_k == 0
    )
    up_tiles = triton.cdiv(up_gate_out, _schedule.INT8_GEMM_BLOCK_N)
    down_tiles = triton.cdiv(hidden_size, _schedule.INT8_GEMM_BLOCK_N)

    inp_q_2d, inp_s_1d = prepare_int8_input_gpu(
        input_2d,
        prepared_data,
        prepared_scale,
        group_size,
        target=resolved_target,
        int8_backend=int8_backend,
        up_prep_plan=up_prep_plan,
    )
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        run_int8_chunk(
            inp_q_2d[start:end],
            inp_s_1d[start:end],
            ids_2d[start:end],
            weights_2d[start:end],
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            out_2d[start:end],
            target=resolved_target,
            int8_backend=int8_backend,
            down_prep_plan=down_prep_plan,
            batched=batched,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            swap_halves=swap_halves,
            block_k=block_k,
            num_stages=num_stages,
            compiler_options=compiler_options,
            split_k_up=split_k_up,
            split_k_down=split_k_down,
            up_seed=up_seed,
            down_seed=down_seed,
            up_aligned_nk=up_aligned_nk,
            down_aligned_nk=down_aligned_nk,
            up_tiles=up_tiles,
            down_tiles=down_tiles,
        )
