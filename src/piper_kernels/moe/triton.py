"""Triton kernels and GPU dispatcher for quantized MoE decode and prefill."""

# This host orchestrator launches Triton routing kernels and mirrors their
# runtime tensor argument lists.
# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch
import triton

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.primitives import TritonCompileFn
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget, rdna4_int8_accumulator_seed
from piper_kernels.linear._splitk import device_sm_count
from piper_kernels.linear.convrot.int8.triton import splitk_accumulator_seed
from piper_kernels.trellis.triton import compute_row_offsets_2d, decode_slots_int8_out
from piper_kernels.weights.convrot.int8._quantization import dynamic_quantize_rows
from piper_kernels.weights.convrot.triton import rotate_input
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

from . import _int8_triton, _nvfp4_triton, _routing_triton, _schedule, _trellis_triton
from ._int8_triton import (
    int8_moe_batched_matmul_kernel,
    int8_moe_grouped_gemm_kernel,
    int8_moe_m1_matmul_kernel,
)
from ._nvfp4_triton import (
    nvfp4_moe_batched_gemv_kernel,
    nvfp4_moe_grouped_gemm_kernel,
    nvfp4_moe_m1_gemv_kernel,
)
from ._routing_triton import (
    A16ChunkRouting,
    moe_bucket_tokens_kernel,
    moe_m1_activate_rotate_kernel,
    moe_m1_weighted_reduce_kernel,
    moe_prefill_bucket_count_kernel,
    moe_prefill_bucket_scatter_kernel,
    moe_prefill_gather_kernel,
    moe_splitk_reduce_kernel,
)
from ._trellis_triton import trellis_moe_gemv_kernel

if TYPE_CHECKING:
    from triton.backends.compiler import GPUTarget

    from piper_kernels.linear.convrot.int8._backend import LinearBackend, LinearExecutionPlan


def _launch_moe_prefill_bucket(
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


def _launch_moe_prefill_gather(
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


def _launch_moe_prefill_weighted_reduce(
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


def _launch_moe_bucket(
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


def _run_a16_chunk(
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
        _launch_moe_bucket(topk_ids_flat, expert_to_slot, num_slots)[:2]
        if batched
        else (None, None)
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
            reduce_grid = _routing_triton.moe_2d_tile_grid(
                triton.cdiv(up_gate_out, _schedule.NVFP4_REDUCE_BLOCK_N),
                num_pairs,
                num_warps=4,
                is_amd_hip=target.is_amd_hip,
            )
            moe_splitk_reduce_kernel[reduce_grid](
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
        activate_grid = _routing_triton.moe_2d_tile_grid(
            triton.cdiv(intermediate_size, _schedule.ACTIVATE_BLOCK_SIZE),
            num_pairs,
            num_warps=4,
            is_amd_hip=target.is_amd_hip,
        )
        moe_m1_activate_rotate_kernel[activate_grid](
            mid,
            act,
            out_chunk,
            topk_ids_flat,
            expert_to_slot,
            intermediate_size,
            num_experts,
            num_slots,
            num_pairs,
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


def _prepare_a16_input_gpu(
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


def _iter_a16_chunks(
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
        active_chunk = _prepare_a16_input_gpu(
            input_2d[start:end],
            None if prep_full is None else prep_full[start:end],
            group_size,
        )
        yield active_chunk, ids_2d[start:end], weights_2d[start:end], out_2d[start:end]


def _run_nvfp4_prefill_chunk(
    active_x_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    out_chunk: torch.Tensor,
    *,
    target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
    up_scale_cols: int,
    up_scale_stride: int,
    down_scale_cols: int,
    down_scale_stride: int,
) -> None:
    device = active_x_chunk.device
    chunk_rows = active_x_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    num_slots, hidden_size, packed_inter = down_qdata.shape
    intermediate_size = 2 * packed_inter
    up_gate_out = up_gate_qdata.shape[1]
    prefill_schedule = _routing_triton.select_nvfp4_prefill_schedule(
        target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        active_dtype=active_x_chunk.dtype,
        logical_dtype=out_chunk.dtype,
    )
    cast_bf16 = prefill_schedule.cast_fp32_to_bf16
    force_fp32_dot = prefill_schedule.force_fp32_dot
    half_scales = prefill_schedule.half_scales
    block_m = prefill_schedule.block_m
    block_n = prefill_schedule.block_n
    block_k = prefill_schedule.block_k
    num_warps = prefill_schedule.num_warps
    single_dot = prefill_schedule.single_dot
    use_sm120_e2m1x2 = prefill_schedule.use_sm120_e2m1x2
    tile_slots, tile_m_start, tile_m_len, compact_to_pair, pair_to_compact, max_tiles, _ = (
        _launch_moe_prefill_bucket(topk_ids_chunk.reshape(-1), expert_to_slot, num_slots, block_m)
    )
    x_compact = torch.empty((num_pairs, hidden_size), device=device, dtype=active_x_chunk.dtype)
    mid = torch.empty((num_pairs, up_gate_out), device=device, dtype=torch.float32)

    _launch_moe_prefill_gather(
        active_x_chunk,
        None,
        compact_to_pair,
        x_compact,
        None,
        hidden_size,
        top_k,
        has_scale=False,
    )
    _nvfp4_triton.launch_nvfp4_grouped_gemm(
        x_compact,
        up_gate_qdata,
        up_gate_scale,
        up_gate_global_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        mid,
        max_tiles=max_tiles,
        out_features=up_gate_out,
        in_features=hidden_size,
        scale_cols=up_scale_cols,
        scale_stride=up_scale_stride,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        high_first=high_first,
        has_global_scale=up_gate_global_scale is not None,
        cast_fp32_to_bf16=cast_bf16,
        force_fp32_dot=force_fp32_dot,
        half_scales=half_scales,
        use_sm120_e2m1x2=use_sm120_e2m1x2,
        single_dot=single_dot,
    )
    del x_compact
    act_dtype = torch.float32 if group_size > 0 else out_chunk.dtype
    act = torch.empty((num_pairs, intermediate_size), device=device, dtype=act_dtype)
    with device_context(device):
        activate_grid = _routing_triton.moe_2d_tile_grid(
            triton.cdiv(intermediate_size, _schedule.ACTIVATE_BLOCK_SIZE),
            num_pairs,
            num_warps=4,
            is_amd_hip=target.is_amd_hip,
        )
        moe_m1_activate_rotate_kernel[activate_grid](
            mid,
            act,
            out_chunk,
            compact_to_pair,
            None,
            intermediate_size,
            num_pairs,
            num_pairs,
            num_pairs,
            block_size=_schedule.ACTIVATE_BLOCK_SIZE,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            accelerator_backend=target.backend,
            has_slot_map=False,
            num_warps=4,
        )
    del mid
    down_out = torch.empty((num_pairs, hidden_size), device=device, dtype=torch.float32)
    _nvfp4_triton.launch_nvfp4_grouped_gemm(
        act,
        down_qdata,
        down_scale,
        down_global_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        down_out,
        max_tiles=max_tiles,
        out_features=hidden_size,
        in_features=intermediate_size,
        scale_cols=down_scale_cols,
        scale_stride=down_scale_stride,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        high_first=high_first,
        has_global_scale=down_global_scale is not None,
        cast_fp32_to_bf16=cast_bf16,
        force_fp32_dot=force_fp32_dot,
        half_scales=half_scales,
        use_sm120_e2m1x2=use_sm120_e2m1x2,
        single_dot=single_dot,
    )
    del act
    _launch_moe_prefill_weighted_reduce(
        down_out,
        pair_to_compact,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        hidden_size,
        top_k,
        accumulate=accumulate,
    )


def _dispatch_nvfp4_moe(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    resolved_target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
) -> None:
    device = input_2d.device
    rows, hidden_size = input_2d.shape
    num_slots, _, packed_inter = down_qdata.shape
    intermediate_size = 2 * packed_inter
    up_gate_out = up_gate_qdata.shape[1]
    top_k = ids_2d.shape[1]
    up_scale_cols = int(triton.cdiv(hidden_size, nvfp4_layout.SCALE_COLUMN_TILE))
    up_scale_stride = up_gate_scale.shape[1] * up_gate_scale.shape[2]
    down_scale_cols = int(triton.cdiv(intermediate_size, nvfp4_layout.SCALE_COLUMN_TILE))
    down_scale_stride = down_scale.shape[1] * down_scale.shape[2]

    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=False,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=input_2d.element_size(),
        group_size=group_size,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill":
        for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
            input_2d, ids_2d, weights_2d, out_2d, prepared_data, group_size, chunk_rows
        ):
            _run_nvfp4_prefill_chunk(
                active_chunk,
                ids_chunk,
                weights_chunk,
                up_gate_qdata,
                up_gate_scale,
                down_qdata,
                down_scale,
                up_gate_global_scale,
                down_global_scale,
                expert_to_slot,
                out_chunk,
                target=resolved_target,
                accumulate=accumulate,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                high_first=high_first,
                active_slots_hint=active_slots_hint,
                up_scale_cols=up_scale_cols,
                up_scale_stride=up_scale_stride,
                down_scale_cols=down_scale_cols,
                down_scale_stride=down_scale_stride,
            )
        return

    batched = regime == "batched"
    split_k_up = split_k_down = 1
    m1_schedule = _routing_triton.select_nvfp4_m1_schedule(resolved_target)
    active_schedule: _schedule.Nvfp4M1Schedule | _schedule.Nvfp4BatchedSchedule = m1_schedule
    post_scale_only = True
    num_sms = device_sm_count(device)
    if not batched:
        active_slots = _schedule.expected_bucket_tiles(
            top_k, num_slots, active_slots_hint, batched=False
        )
        split_k_up = _schedule.select_nvfp4_split_k(
            num_sms,
            active_slots,
            up_gate_out,
            hidden_size,
            block_n=m1_schedule.block_n,
            block_k=m1_schedule.block_k,
            cta_multiplier=m1_schedule.splitk_cta_multiplier,
        )
        split_k_down = _schedule.select_nvfp4_split_k(
            num_sms,
            active_slots,
            hidden_size,
            intermediate_size,
            block_n=m1_schedule.block_n,
            block_k=m1_schedule.block_k,
            cta_multiplier=m1_schedule.splitk_cta_multiplier,
        )

    for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
        input_2d, ids_2d, weights_2d, out_2d, prepared_data, group_size, chunk_rows
    ):
        if batched:
            chunk_n = active_chunk.shape[0]
            batched_schedule = _routing_triton.select_nvfp4_batched_schedule(
                resolved_target,
                rows=chunk_n,
                active_dtype=active_chunk.dtype,
                logical_dtype=out_chunk.dtype,
            )
            active_schedule = batched_schedule
            post_scale_only = batched_schedule.post_scale_only
            tiles = _schedule.expected_bucket_tiles(
                chunk_n * top_k, num_slots, active_slots_hint, batched=True
            )
            split_k_up = _schedule.select_nvfp4_split_k(
                num_sms,
                tiles,
                up_gate_out,
                hidden_size,
                block_n=batched_schedule.block_n,
                block_k=batched_schedule.block_k,
            )
            split_k_down = _schedule.select_nvfp4_split_k(
                num_sms,
                tiles,
                hidden_size,
                intermediate_size,
                block_n=batched_schedule.block_n,
                block_k=batched_schedule.block_k,
            )

        def _launch_nvfp4_step(
            per_expert_input: bool,
            x: torch.Tensor,
            routing: A16ChunkRouting,
            output: torch.Tensor,
            split_k: int,
            *,
            sched: _schedule.Nvfp4M1Schedule | _schedule.Nvfp4BatchedSchedule = active_schedule,
        ) -> None:
            if not per_expert_input:
                q, s, gs = up_gate_qdata, up_gate_scale, up_gate_global_scale
                n_out, n_in, sc, ss = up_gate_out, hidden_size, up_scale_cols, up_scale_stride
            else:
                q, s, gs = down_qdata, down_scale, down_global_scale
                n_out, n_in, sc, ss = (
                    hidden_size,
                    intermediate_size,
                    down_scale_cols,
                    down_scale_stride,
                )
            is_batched_sched = isinstance(sched, _schedule.Nvfp4BatchedSchedule)
            _nvfp4_triton.launch_nvfp4_proj(
                x,
                q,
                s,
                gs,
                routing,
                output,
                out_features=n_out,
                in_features=n_in,
                scale_cols=sc,
                scale_stride=ss,
                num_slots=num_slots,
                split_k=split_k,
                block_n=sched.block_n,
                block_k=sched.block_k,
                num_warps=sched.num_warps,
                high_first=high_first,
                has_global_scale=gs is not None,
                per_expert_input=per_expert_input,
                is_splitk=per_expert_input or split_k > 1,
                use_dot=sched.use_dot if is_batched_sched else False,
                cast_fp32_to_bf16=sched.cast_fp32_to_bf16 if is_batched_sched else False,
                use_sm120_e2m1x2=sched.use_sm120_e2m1x2,
                single_dot=sched.single_dot if is_batched_sched else False,
                is_amd_hip=resolved_target.is_amd_hip,
            )

        _run_a16_chunk(
            active_chunk,
            ids_chunk,
            weights_chunk,
            out_chunk,
            num_slots=num_slots,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            up_gate_out=up_gate_out,
            expert_to_slot=expert_to_slot,
            target=resolved_target,
            batched=batched,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            split_k_up=split_k_up,
            split_k_down=split_k_down,
            up_row_scale=None,
            down_row_scale=None,
            up_global_scale=up_gate_global_scale,
            down_global_scale=down_global_scale,
            post_scale_only=post_scale_only,
            launch_proj=_launch_nvfp4_step,
        )


def _run_int8_chunk(
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
        tile_slots, tile_pair_ids, _ = _launch_moe_bucket(topk_ids_flat, expert_to_slot, num_slots)
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
        _int8_triton.launch_int8_proj(
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
            is_amd_hip=target.is_amd_hip,
        )
        if split_k_up > 1:
            reduce_grid = _routing_triton.moe_2d_tile_grid(
                triton.cdiv(up_gate_out, _schedule.INT8_REDUCE_BLOCK_N),
                num_pairs,
                num_warps=4,
                is_amd_hip=target.is_amd_hip,
            )
            moe_splitk_reduce_kernel[reduce_grid](
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
        _int8_triton.launch_int8_proj(
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
            is_amd_hip=target.is_amd_hip,
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


def _run_prefill_int8_proj(
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
        _int8_triton.launch_int8_grouped_gemm(
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
            _int8_triton.launch_int8_grouped_gemm(
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


def _run_int8_prefill_chunk(
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
    ) = _launch_moe_prefill_bucket(topk_ids_chunk.reshape(-1), expert_to_slot, num_slots, block_m)
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
    _launch_moe_prefill_gather(
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
    _run_prefill_int8_proj(
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
    _run_prefill_int8_proj(
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
    _launch_moe_prefill_weighted_reduce(
        down_out,
        pair_to_compact,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        hidden_size,
        top_k,
        accumulate=accumulate,
    )


def _prepare_int8_input_gpu(
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


def _dispatch_int8_prefill_chunks(
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
        inp_q_chunk, inp_s_chunk = _prepare_int8_input_gpu(
            input_2d[start:end],
            None if prep_q_full is None else prep_q_full[start:end],
            None if prep_s_full is None else prep_s_full[start:end],
            group_size,
            target=resolved_target,
            int8_backend=int8_backend,
            up_prep_plan=up_prep_plan,
        )
        _run_int8_prefill_chunk(
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


def _dispatch_int8_moe(
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
        _dispatch_int8_prefill_chunks(
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

    inp_q_2d, inp_s_1d = _prepare_int8_input_gpu(
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
        _run_int8_chunk(
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


def _dispatch_trellis_moe(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_block_scales: torch.Tensor,
    up_gate_scale: torch.Tensor,
    up_gate_row_widths: torch.Tensor | None,
    up_gate_slot_offsets: torch.Tensor | None,
    up_gate_block_size: int,
    down_qdata: torch.Tensor,
    down_block_scales: torch.Tensor,
    down_scale: torch.Tensor,
    down_row_widths: torch.Tensor | None,
    down_slot_offsets: torch.Tensor | None,
    down_block_size: int,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
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
    num_sms = device_sm_count(device)
    rows, hidden_size = input_2d.shape
    num_slots, _, down_blocks = down_block_scales.shape
    intermediate_size = down_blocks * down_block_size
    up_gate_out, up_blocks = up_gate_block_scales.shape[1], up_gate_block_scales.shape[2]
    top_k = ids_2d.shape[1]

    int8_backend = (
        int8_backend_mod.select_linear_backend_for_target(resolved_target)
        if _routing_triton.supports_trellis_int8_prefill(resolved_target)
        else None
    )
    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=int8_backend is not None,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=out_2d.element_size(),
        group_size=group_size,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill" and int8_backend is not None:
        _dispatch_int8_prefill_chunks(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            prepared_data,
            None,
            out_2d,
            resolved_target=resolved_target,
            int8_backend=int8_backend,
            chunk_rows=chunk_rows,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
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
        return

    if regime == "prefill":
        # Non-INT8 accelerators (e.g. SM75) fall back to chunked batched A16 GEMV.
        a16_chunk_rows = min(
            rows, _schedule.BATCH_MAX_ROWS, max(1, _schedule.BUCKET_MAX_PAIRS // max(1, top_k))
        )
        batched = a16_chunk_rows > 1
    else:
        a16_chunk_rows = chunk_rows
        batched = regime == "batched"

    up_slot_bytes = up_gate_qdata.numel() // max(1, num_slots)
    down_slot_bytes = down_qdata.numel() // max(1, num_slots)
    expected_slots = min(num_slots, active_slots_hint) if active_slots_hint > 0 else num_slots
    split_k_up = split_k_down = 1
    schedule = _routing_triton.select_trellis_schedule(
        resolved_target,
        batched=batched,
        num_pairs=top_k,
        expected_slots=expected_slots,
    )
    if not batched:
        active_slots = _schedule.expected_bucket_tiles(
            top_k, num_slots, active_slots_hint, batched=False
        )
        split_k_up = _schedule.select_trellis_split_k(
            num_sms, active_slots, up_gate_out, up_blocks, max(1, up_slot_bytes * active_slots)
        )
        split_k_down = _schedule.select_trellis_split_k(
            num_sms, active_slots, hidden_size, down_blocks, max(1, down_slot_bytes * active_slots)
        )

    def _launch_trellis_step(
        per_expert_input: bool,
        x: torch.Tensor,
        routing: A16ChunkRouting,
        output: torch.Tensor,
        split_k: int,
    ) -> None:
        if not per_expert_input:
            q, bs, rs, rw, so, bsz, n_out, n_in = (
                up_gate_qdata,
                up_gate_block_scales,
                up_gate_scale,
                up_gate_row_widths,
                up_gate_slot_offsets,
                up_gate_block_size,
                up_gate_out,
                hidden_size,
            )
        else:
            q, bs, rs, rw, so, bsz, n_out, n_in = (
                down_qdata,
                down_block_scales,
                down_scale,
                down_row_widths,
                down_slot_offsets,
                down_block_size,
                hidden_size,
                intermediate_size,
            )
        _trellis_triton.launch_trellis_proj(
            x,
            q,
            bs,
            rs,
            rw,
            so,
            bsz,
            routing,
            expert_to_slot,
            output,
            out_features=n_out,
            in_features=n_in,
            num_slots=num_slots,
            split_k=split_k,
            batched=batched,
            per_expert_input=per_expert_input,
            is_splitk=per_expert_input or split_k > 1,
            schedule=schedule,
            is_amd_hip=resolved_target.is_amd_hip,
        )

    for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
        input_2d, ids_2d, weights_2d, out_2d, prepared_data, group_size, a16_chunk_rows
    ):
        if batched:
            num_pairs = active_chunk.shape[0] * top_k
            schedule = _routing_triton.select_trellis_schedule(
                resolved_target,
                batched=True,
                num_pairs=num_pairs,
                expected_slots=expected_slots,
            )
            weight_buckets = min(
                _schedule.max_bucket_tiles(num_pairs, num_slots, _schedule.BATCH_BLOCK_M),
                _schedule.expected_bucket_tiles(
                    num_pairs, num_slots, active_slots_hint, batched=True
                ),
            )
            active_proj_tiles = (
                min(
                    _schedule.max_bucket_tiles(num_pairs, num_slots, schedule.block_m),
                    _schedule.expected_bucket_tiles(
                        num_pairs,
                        num_slots,
                        active_slots_hint,
                        block_m=schedule.block_m,
                        batched=True,
                    ),
                )
                if schedule.use_subtile_splitk
                else weight_buckets
            )
            split_k_up = _schedule.select_trellis_split_k(
                num_sms,
                active_proj_tiles,
                triton.cdiv(up_gate_out, schedule.block_n),
                up_blocks,
                max(1, up_slot_bytes * weight_buckets),
            )
            split_k_down = _schedule.select_trellis_split_k(
                num_sms,
                active_proj_tiles,
                triton.cdiv(hidden_size, schedule.block_n),
                down_blocks,
                max(1, down_slot_bytes * weight_buckets),
            )
        _run_a16_chunk(
            active_chunk,
            ids_chunk,
            weights_chunk,
            out_chunk,
            num_slots=num_slots,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            up_gate_out=up_gate_out,
            expert_to_slot=expert_to_slot,
            target=resolved_target,
            batched=batched,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            split_k_up=split_k_up,
            split_k_down=split_k_down,
            up_row_scale=up_gate_scale,
            down_row_scale=down_scale,
            up_global_scale=None,
            down_global_scale=None,
            post_scale_only=True,
            launch_proj=_launch_trellis_step,
        )


def moe_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None = None,
    down_global_scale: torch.Tensor | None = None,
    expert_to_slot: torch.Tensor | None = None,
    prepared_data: torch.Tensor | None = None,
    prepared_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    target: AcceleratorTarget | None = None,
    accumulate: bool = False,
    group_size: int = 0,
    activation_fn: str = "swiglu",
    gate_first: bool = True,
    high_first: bool = False,
    active_slots_hint: int = 0,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_size: int = 0,
) -> torch.Tensor:
    """Execute single-token, small-batch, or prefill quantized MoE kernels on GPU."""
    device = input.device
    rows = math.prod(input.shape[:-1])
    hidden_size = input.shape[-1]
    top_k = topk_ids.shape[-1]
    result = torch.empty_like(input, memory_format=torch.contiguous_format) if out is None else out
    out_2d = result.view(rows, hidden_size)
    input_2d = input.reshape(rows, hidden_size)
    ids_2d = topk_ids.reshape(rows, top_k).contiguous()
    weights_2d = topk_weights.reshape(rows, top_k).contiguous()
    resolved_target = AcceleratorTarget.from_device(device) if target is None else target

    if up_gate_block_scales is not None:
        assert down_block_scales is not None
        _dispatch_trellis_moe(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_block_scales,
            up_gate_scale,
            up_gate_row_widths,
            up_gate_slot_offsets,
            up_gate_block_size,
            down_qdata,
            down_block_scales,
            down_scale,
            down_row_widths,
            down_slot_offsets,
            down_block_size,
            expert_to_slot,
            prepared_data,
            out_2d,
            resolved_target=resolved_target,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            active_slots_hint=active_slots_hint,
        )
    elif up_gate_qdata.dtype is torch.int8:
        _dispatch_int8_moe(
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
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            active_slots_hint=active_slots_hint,
        )
    else:
        _dispatch_nvfp4_moe(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            up_gate_global_scale,
            down_global_scale,
            expert_to_slot,
            prepared_data,
            out_2d,
            resolved_target=resolved_target,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            active_slots_hint=active_slots_hint,
        )
    return result


def compile_moe_kernels(
    target: GPUTarget,
    *,
    weight_format: str = "nvfp4_w4a16",
    logical_dtype: torch.dtype = torch.bfloat16,
    rows: int = 1,
    top_k: int = 1,
    num_slots: int = 64,
    active_slots_hint: int = 0,
    hidden_size: int = 256,
    intermediate_size: int = 256,
    group_size: int = 0,
    activation_fn: str = "swiglu",
    gate_first: bool = True,
    high_first: bool = False,
    has_slot_map: bool = True,
    has_global_scale: bool = True,
    split_k: int = 1,
    accumulate: bool = False,
    block_size: int = 256,
    variable_rows: bool = False,
    has_slot_offsets: bool = False,
    bytes_per_block: int | None = None,
    compiler: TritonCompileFn | None = None,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile quantized MoE Triton kernels matching runtime regime and chunking selection."""
    from . import _compile  # noqa: PLC0415

    return _compile.compile_moe_kernels(
        target,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        rows=rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        has_slot_map=has_slot_map,
        has_global_scale=has_global_scale,
        split_k=split_k,
        accumulate=accumulate,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
        bytes_per_block=bytes_per_block,
        compiler=compiler,
    )


__all__ = [
    "compile_moe_kernels",
    "int8_moe_batched_matmul_kernel",
    "int8_moe_grouped_gemm_kernel",
    "int8_moe_m1_matmul_kernel",
    "moe_bucket_tokens_kernel",
    "moe_ffn",
    "moe_m1_activate_rotate_kernel",
    "moe_m1_weighted_reduce_kernel",
    "moe_prefill_bucket_count_kernel",
    "moe_prefill_bucket_scatter_kernel",
    "moe_prefill_gather_kernel",
    "moe_splitk_reduce_kernel",
    "nvfp4_moe_batched_gemv_kernel",
    "nvfp4_moe_grouped_gemm_kernel",
    "nvfp4_moe_m1_gemv_kernel",
    "trellis_moe_gemv_kernel",
]
