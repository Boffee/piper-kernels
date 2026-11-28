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
from piper_kernels.weights.convrot.int8.triton import _quantize_int8, int8_scale_from_max

from . import _schedule
from ._routing_triton import (
    ChunkRouting,
    PrefillBuckets,
    _load_activate_rotate_tile,
    _resolve_slot,
    _swizzle_grouped_pid,
    _unpack_2d_tile_pid,
    moe_2d_tile_grid,
)
from .reference import _ActivationConfig


@triton.jit(do_not_specialize=["num_pairs", "gate_bound", "up_bound"])
def int8_moe_activate_quant_kernel(
    mid_ptr,
    routing_ids_ptr,
    expert_to_slot_ptr,
    q_out_ptr,
    scale_out_ptr,
    intermediate_size,
    num_experts,
    num_slots,
    num_pairs,
    gate_bound,
    up_bound,
    block_k: tl.constexpr,
    group_size: tl.constexpr,
    single_tile: tl.constexpr,
    is_gated: tl.constexpr,
    gate_first: tl.constexpr,
    has_gate_bound: tl.constexpr,
    has_up_bound: tl.constexpr,
    has_slot_map: tl.constexpr,
    use_compact_map: tl.constexpr,
    accelerator_backend: tl.constexpr,
):
    """Apply MoE activation (with optional bounds), ConvRot Hadamard rotation, and INT8 quant."""
    pair = tl.program_id(0).to(tl.int64)
    if pair >= num_pairs:
        return
    if use_compact_map:
        orig_pair = tl.load(routing_ids_ptr + pair)
        is_inactive = orig_pair < 0
    else:
        slot = _resolve_slot(
            routing_ids_ptr,
            expert_to_slot_ptr,
            pair,
            num_experts,
            num_slots,
            has_slot_map,
        )
        is_inactive = slot < 0
    if is_inactive:
        tl.store(scale_out_ptr + pair, tl.full((), 1e-30, tl.float32))
        return

    reciprocal_scale: tl.constexpr = accelerator_backend == "hip"
    inter_i64 = intermediate_size.to(tl.int64)
    if single_tile:
        k_offs = tl.arange(0, block_k)
        k_valid = k_offs < intermediate_size
        vals = _load_activate_rotate_tile(
            mid_ptr,
            pair,
            k_offs,
            k_valid,
            intermediate_size,
            gate_bound,
            up_bound,
            block_k=block_k,
            group_size=group_size,
            is_gated=is_gated,
            gate_first=gate_first,
            has_gate_bound=has_gate_bound,
            has_up_bound=has_up_bound,
            accelerator_backend=accelerator_backend,
        )
        row_amax = tl.max(tl.abs(vals), axis=0)
        row_scale = tl.maximum(
            int8_scale_from_max(row_amax, reciprocal_scale=reciprocal_scale),
            1e-30,
        )
        tl.store(scale_out_ptr + pair, row_scale)
        quantized = _quantize_int8(vals, row_scale, accelerator_backend)
        tl.store(q_out_ptr + pair * inter_i64 + k_offs.to(tl.int64), quantized, mask=k_valid)
        return

    row_amax = tl.full((), 0.0, dtype=tl.float32)
    for k_base in range(0, intermediate_size, block_k):
        k_offs = k_base + tl.arange(0, block_k)
        k_valid = k_offs < intermediate_size
        vals = _load_activate_rotate_tile(
            mid_ptr,
            pair,
            k_offs,
            k_valid,
            intermediate_size,
            gate_bound,
            up_bound,
            block_k=block_k,
            group_size=group_size,
            is_gated=is_gated,
            gate_first=gate_first,
            has_gate_bound=has_gate_bound,
            has_up_bound=has_up_bound,
            accelerator_backend=accelerator_backend,
        )
        row_amax = tl.maximum(row_amax, tl.max(tl.abs(vals), axis=0))

    row_scale = tl.maximum(
        int8_scale_from_max(row_amax, reciprocal_scale=reciprocal_scale),
        1e-30,
    )
    tl.store(scale_out_ptr + pair, row_scale)
    for k_base in range(0, intermediate_size, block_k):
        k_offs = k_base + tl.arange(0, block_k)
        k_valid = k_offs < intermediate_size
        vals = _load_activate_rotate_tile(
            mid_ptr,
            pair,
            k_offs,
            k_valid,
            intermediate_size,
            gate_bound,
            up_bound,
            block_k=block_k,
            group_size=group_size,
            is_gated=is_gated,
            gate_first=gate_first,
            has_gate_bound=has_gate_bound,
            has_up_bound=has_up_bound,
            accelerator_backend=accelerator_backend,
        )
        quantized = _quantize_int8(vals, row_scale, accelerator_backend)
        tl.store(q_out_ptr + pair * inter_i64 + k_offs.to(tl.int64), quantized, mask=k_valid)


def launch_int8_activate_quant(
    mid: torch.Tensor,
    routing_ids: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    q_out: torch.Tensor,
    scale_out: torch.Tensor,
    *,
    intermediate_size: int,
    group_size: int,
    num_experts: int,
    num_slots: int,
    activation: _ActivationConfig,
    has_slot_map: bool,
    use_compact_map: bool,
    accelerator_backend: str,
) -> None:
    num_pairs = mid.shape[0]
    if num_pairs == 0:
        return
    single_tile, block_k, num_warps = _schedule.select_int8_activate_schedule(intermediate_size)
    with device_context(mid.device):
        int8_moe_activate_quant_kernel[(num_pairs,)](
            mid,
            routing_ids,
            expert_to_slot,
            q_out,
            scale_out,
            intermediate_size,
            num_experts,
            num_slots,
            num_pairs,
            activation.gate_bound_value,
            activation.up_bound_value,
            block_k=block_k,
            group_size=group_size,
            single_tile=single_tile,
            is_gated=activation.is_gated,
            gate_first=activation.gate_first,
            has_gate_bound=activation.gate_bound is not None,
            has_up_bound=activation.up_bound is not None,
            has_slot_map=has_slot_map,
            use_compact_map=use_compact_map,
            accelerator_backend=accelerator_backend,
            num_warps=num_warps,
        )


@triton.jit
def _dequant_int8_epilogue(
    acc,
    input_scale_ptr,
    weight_scale_ptr,
    in_scale_offs,
    in_scale_mask,
    slot,
    out_features,
    offs_n_i64,
    n_valid,
    output_ptr,
    accumulator_seed: tl.constexpr,
    dequant_output: tl.constexpr = True,
    scalar_in_scale: tl.constexpr = False,
):
    """Remove accumulator seed and optionally dequantize INT32 accumulators to output dtype."""
    centered = acc - accumulator_seed
    if not dequant_output:
        return centered
    if scalar_in_scale:
        in_scale = tl.load(input_scale_ptr + in_scale_offs)
    else:
        in_scale = tl.load(input_scale_ptr + in_scale_offs, mask=in_scale_mask, other=0.0)[:, None]
    w_scale = tl.load(
        weight_scale_ptr + slot * out_features + offs_n_i64,
        mask=n_valid,
        other=0.0,
    )
    return (centered.to(tl.float32) * in_scale * w_scale[None, :]).to(output_ptr.dtype.element_ty)


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
    out_vals = _dequant_int8_epilogue(
        acc,
        input_scale_ptr,
        weight_scale_ptr,
        k_idx if per_expert_input else 0,
        None,
        slot,
        out_features,
        offs_n_i64,
        offs_n < out_features,
        output_ptr,
        accumulator_seed=accumulator_seed,
        dequant_output=dequant_output,
        scalar_in_scale=True,
    )

    row_idx = pid_k.to(tl.int64) * top_k + k_idx if is_splitk else k_idx.to(tl.int64)
    out_ptrs = (
        output_ptr
        + row_idx * out_features
        + tl.broadcast_to(offs_n_i64[None, :], (block_m, block_n))
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

    out_vals = _dequant_int8_epilogue(
        accumulator,
        input_scale_ptr,
        weight_scale_ptr,
        src_rows,
        row_valid,
        slot,
        out_features,
        offs_n_i64,
        offs_n < out_features,
        output_ptr,
        accumulator_seed=accumulator_seed,
        dequant_output=dequant_output,
    )

    dst_rows = tl.where(
        row_valid, pid_k.to(tl.int64) * num_pairs + pair_ids if is_splitk else pair_ids, 0
    )
    tl.store(
        output_ptr + dst_rows[:, None] * out_features + offs_n_i64[None, :],
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
    out_vals = _dequant_int8_epilogue(
        acc,
        input_scale_ptr,
        weight_scale_ptr,
        m_start + offs_m_i64,
        m_valid,
        slot,
        out_features,
        offs_n_global_i64,
        n_valid,
        output_ptr,
        accumulator_seed=accumulator_seed,
    )

    tl.store(
        output_ptr + (m_start + offs_m_i64)[:, None] * out_features + offs_n_global_i64[None, :],
        out_vals,
        mask=m_valid[:, None] & n_valid[None, :],
    )


def launch_int8_proj(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    in_scale: torch.Tensor,
    w_scale: torch.Tensor,
    routing: ChunkRouting,
    output: torch.Tensor,
    *,
    out_tiles: int,
    out_features: int,
    in_features: int,
    num_slots: int,
    split_k: int,
    block_m: int,
    schedule: _schedule.Int8LaunchSchedule,
    compiler_options: dict[str, object],
    per_expert_input: bool,
    is_splitk: bool,
    aligned_nk: bool,
    accumulator_seed: int,
    is_amd_hip: bool = False,
) -> None:
    with device_context(x_q.device):
        if routing.tile_slots is not None:
            assert routing.tile_pair_ids is not None
            int8_moe_batched_matmul_kernel[(out_tiles, routing.tile_slots.shape[0], split_k)](
                x_q,
                w_q,
                routing.tile_slots,
                routing.tile_pair_ids,
                in_scale,
                w_scale,
                output,
                out_features,
                in_features,
                routing.num_pairs,
                routing.top_k,
                split_k,
                block_m=block_m,
                block_n=_schedule.INT8_GEMM_BLOCK_N,
                block_k=schedule.block_k,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                dequant_output=not is_splitk,
                aligned_nk=aligned_nk,
                accumulator_seed=accumulator_seed,
                **compiler_options,
                num_stages=schedule.num_stages,
                num_warps=4,
            )
        else:
            grid = moe_2d_tile_grid(
                out_tiles,
                routing.top_k,
                num_warps=4,
                is_amd_hip=is_amd_hip,
                split_k=split_k,
            )
            int8_moe_m1_matmul_kernel[grid](
                x_q,
                w_q,
                routing.topk_ids_flat,
                routing.expert_to_slot,
                in_scale,
                w_scale,
                output,
                out_features,
                in_features,
                routing.num_experts,
                num_slots,
                routing.top_k,
                split_k,
                block_m=block_m,
                block_n=_schedule.INT8_GEMM_BLOCK_N,
                block_k=schedule.block_k,
                has_slot_map=routing.has_slot_map,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                dequant_output=not is_splitk,
                aligned_nk=aligned_nk,
                accumulator_seed=accumulator_seed,
                **compiler_options,
                num_stages=schedule.num_stages,
                num_warps=4,
            )


def launch_int8_grouped_gemm(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    buckets: PrefillBuckets,
    in_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    schedule: _schedule.Int8PrefillSchedule,
    compiler_options: dict[str, object],
    aligned_nk: bool,
    accumulator_seed: int,
    slot_start: int = 0,
    slot_end: int = 0,
    n_start: int = 0,
    n_rows: int = 0,
    slot_window: bool = False,
) -> None:
    active_n_rows = n_rows if slot_window else out_features
    grid = (buckets.max_tiles * triton.cdiv(active_n_rows, schedule.block_n),)
    with device_context(x_q.device):
        int8_moe_grouped_gemm_kernel[grid](
            x_q,
            w_q,
            buckets.tile_slots,
            buckets.tile_m_start,
            buckets.tile_m_len,
            in_scale,
            w_scale,
            output,
            buckets.max_tiles,
            out_features,
            in_features,
            block_m=schedule.block_m,
            block_n=schedule.block_n,
            block_k=schedule.block_k,
            group_m=_schedule.PREFILL_GROUP_M,
            aligned_nk=aligned_nk,
            accumulator_seed=accumulator_seed,
            slot_start=slot_start,
            slot_end=slot_end,
            n_start=n_start,
            n_rows=n_rows,
            slot_window=slot_window,
            **compiler_options,
            num_stages=schedule.num_stages,
            num_warps=schedule.num_warps,
        )
