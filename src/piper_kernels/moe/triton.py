"""Triton GPU dispatcher and chunk orchestration for quantized MoE decode and prefill."""

# This host orchestrator launches Triton routing kernels and mirrors their
# runtime tensor argument lists.
# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget, rdna4_int8_accumulator_seed
from piper_kernels.linear._splitk import device_sm_count
from piper_kernels.linear.convrot.int8.triton import splitk_accumulator_seed
from piper_kernels.trellis.triton import compute_row_offsets_2d, decode_slots_int8_out
from piper_kernels.weights.convrot.int8._quantization import dynamic_quantize_rows
from piper_kernels.weights.convrot.triton import rotate_input
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

from . import _int8_triton, _nvfp4_triton, _ops, _policy, _schedule, _trellis_triton
from ._compile import compile_moe_kernels
from ._nvidia import policy as _nvidia_policy
from ._routing_triton import (
    ChunkRouting,
    PrefillBuckets,
    moe_2d_tile_grid,
    moe_bucket_tokens_kernel,
    moe_m1_activate_rotate_kernel,
    moe_m1_weighted_reduce_kernel,
    moe_prefill_bucket_count_kernel,
    moe_prefill_bucket_scatter_kernel,
    moe_prefill_gather_kernel,
    moe_splitk_reduce_kernel,
)
from .reference import _ActivationConfig

if TYPE_CHECKING:
    from piper_kernels.linear.convrot.int8._backend import LinearBackend, LinearExecutionPlan


@dataclass(frozen=True, slots=True)
class _MoERunContext:
    weights: _ops._MoEWeightStorage
    expert_to_slot: torch.Tensor | None
    target: AcceleratorTarget
    accumulate: bool
    active_slots_hint: int
    num_slots: int
    hidden_size: int
    intermediate_size: int
    up_gate_out: int
    activation: _ActivationConfig


@dataclass(frozen=True, slots=True)
class _Nvfp4ScaleMeta:
    up_cols: int
    up_stride: int
    down_cols: int
    down_stride: int


@dataclass(frozen=True, slots=True)
class _Int8ChunkPlan:
    batched: bool
    launch_schedule: _schedule.Int8LaunchSchedule
    compiler_options: dict[str, object]
    split_k_up: int
    split_k_down: int
    up_seed: int
    down_seed: int
    up_aligned_nk: bool
    down_aligned_nk: bool
    up_tiles: int
    down_tiles: int


def _launch_moe_prefill_bucket(
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
    block_m: int,
) -> PrefillBuckets:
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
    return PrefillBuckets(
        tile_slots=tile_slots,
        tile_m_start=tile_m_start,
        tile_m_len=tile_m_len,
        compact_to_pair=compact_to_pair,
        pair_to_compact=pair_to_compact,
        max_tiles=max_tiles,
        slot_counts=slot_counts,
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


def _build_chunk_routing(
    topk_ids_chunk: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
    *,
    batched: bool,
) -> ChunkRouting:
    chunk_rows, top_k = topk_ids_chunk.shape
    topk_ids_flat = topk_ids_chunk.reshape(-1)
    tile_slots, tile_pair_ids = (
        _launch_moe_bucket(topk_ids_flat, expert_to_slot, num_slots)[:2]
        if batched
        else (None, None)
    )
    return ChunkRouting(
        topk_ids_flat=topk_ids_flat,
        expert_to_slot=expert_to_slot,
        tile_slots=tile_slots,
        tile_pair_ids=tile_pair_ids,
        num_experts=expert_to_slot.shape[0] if expert_to_slot is not None else num_slots,
        num_pairs=chunk_rows * top_k,
        top_k=top_k,
        has_slot_map=expert_to_slot is not None,
    )


def _launch_activate_rotate(
    ctx: _MoERunContext,
    mid: torch.Tensor,
    act: torch.Tensor,
    logical_ref: torch.Tensor,
    routing_ids: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    *,
    num_experts: int,
    num_slots: int,
    num_pairs: int,
    has_slot_map: bool,
) -> None:
    activate_grid = moe_2d_tile_grid(
        triton.cdiv(ctx.intermediate_size, _schedule.ACTIVATE_BLOCK_SIZE),
        num_pairs,
        num_warps=4,
        is_amd_hip=ctx.target.is_amd_hip,
    )
    with device_context(mid.device):
        moe_m1_activate_rotate_kernel[activate_grid](
            mid,
            act,
            logical_ref,
            routing_ids,
            expert_to_slot,
            ctx.intermediate_size,
            num_experts,
            num_slots,
            num_pairs,
            ctx.activation.gate_bound_value,
            ctx.activation.up_bound_value,
            block_size=_schedule.ACTIVATE_BLOCK_SIZE,
            group_size=ctx.weights.group_size,
            is_gated=ctx.activation.is_gated,
            gate_first=ctx.activation.gate_first,
            has_gate_bound=ctx.activation.gate_bound is not None,
            has_up_bound=ctx.activation.up_bound is not None,
            accelerator_backend=ctx.target.backend,
            has_slot_map=has_slot_map,
            num_warps=4,
        )


def _launch_splitk_reduce(
    ctx: _MoERunContext,
    partial: torch.Tensor,
    input_scale: torch.Tensor | None,
    weight_scale: torch.Tensor | None,
    weight_global_scale: torch.Tensor | None,
    routing: ChunkRouting,
    output: torch.Tensor,
    *,
    split_k: int,
    is_int8: bool,
    post_scale_only: bool = True,
    has_row_scale: bool = False,
) -> None:
    block_n = _schedule.INT8_REDUCE_BLOCK_N if is_int8 else _schedule.NVFP4_REDUCE_BLOCK_N
    reduce_grid = moe_2d_tile_grid(
        triton.cdiv(ctx.up_gate_out, block_n),
        routing.num_pairs,
        num_warps=4,
        is_amd_hip=ctx.target.is_amd_hip,
    )
    with device_context(partial.device):
        moe_splitk_reduce_kernel[reduce_grid](
            partial,
            input_scale,
            weight_scale,
            weight_global_scale,
            routing.topk_ids_flat,
            routing.expert_to_slot,
            output,
            ctx.up_gate_out,
            routing.num_experts,
            ctx.num_slots,
            routing.num_pairs,
            routing.top_k,
            split_k,
            block_n=block_n,
            is_int8=is_int8,
            has_slot_map=routing.has_slot_map,
            has_global_scale=weight_global_scale is not None,
            post_scale_only=post_scale_only,
            has_row_scale=has_row_scale,
            num_warps=4,
        )


def _launch_weighted_reduce(
    ctx: _MoERunContext,
    down_partial: torch.Tensor,
    input_scale: torch.Tensor | None,
    weight_scale: torch.Tensor | None,
    weight_global_scale: torch.Tensor | None,
    routing_ids: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    topk_weights_flat: torch.Tensor,
    out_chunk: torch.Tensor,
    *,
    num_experts: int,
    num_slots: int,
    num_pairs: int,
    top_k: int,
    split_k: int,
    is_int8: bool,
    has_slot_map: bool,
    post_scale_only: bool = True,
    use_compact_map: bool = False,
    has_row_scale: bool = False,
) -> None:
    with device_context(down_partial.device):
        moe_m1_weighted_reduce_kernel[
            (triton.cdiv(ctx.hidden_size, _schedule.WEIGHTED_REDUCE_BLOCK_N), out_chunk.shape[0])
        ](
            down_partial,
            input_scale,
            weight_scale,
            weight_global_scale,
            routing_ids,
            topk_weights_flat,
            expert_to_slot,
            out_chunk,
            ctx.hidden_size,
            out_chunk.stride(0),
            num_experts,
            num_slots,
            num_pairs,
            top_k,
            split_k,
            block_n=_schedule.WEIGHTED_REDUCE_BLOCK_N,
            is_int8=is_int8,
            has_slot_map=has_slot_map,
            accumulate=ctx.accumulate,
            has_global_scale=weight_global_scale is not None,
            post_scale_only=post_scale_only,
            use_compact_map=use_compact_map,
            has_row_scale=has_row_scale,
            num_warps=4,
        )


def _run_prefill_pipeline[TIn, TAct](
    ctx: _MoERunContext,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    out_chunk: torch.Tensor,
    *,
    block_m: int,
    mid_dtype: torch.dtype,
    gather_up_input: Callable[[PrefillBuckets], TIn],
    run_up: Callable[[TIn, PrefillBuckets, torch.Tensor], None],
    activate: Callable[[torch.Tensor, PrefillBuckets], TAct],
    run_down: Callable[[TAct, PrefillBuckets, torch.Tensor], None],
) -> None:
    """Shared bucket -> gather -> up -> activate -> down -> weighted_reduce prefill driver."""
    device = out_chunk.device
    top_k = topk_ids_chunk.shape[1]
    num_pairs = out_chunk.shape[0] * top_k
    buckets = _launch_moe_prefill_bucket(
        topk_ids_chunk.reshape(-1), ctx.expert_to_slot, ctx.num_slots, block_m
    )
    up_input = gather_up_input(buckets)
    mid = torch.empty((num_pairs, ctx.up_gate_out), device=device, dtype=mid_dtype)
    run_up(up_input, buckets, mid)
    del up_input
    act = activate(mid, buckets)
    del mid
    down_out = torch.empty((num_pairs, ctx.hidden_size), device=device, dtype=torch.float32)
    run_down(act, buckets, down_out)
    del act
    _launch_weighted_reduce(
        ctx,
        down_out,
        None,
        None,
        None,
        buckets.pair_to_compact,
        None,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        num_experts=1,
        num_slots=1,
        num_pairs=num_pairs,
        top_k=top_k,
        split_k=1,
        is_int8=False,
        has_slot_map=False,
        use_compact_map=True,
    )


def _run_a16_chunk(
    ctx: _MoERunContext,
    active_x_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    out_chunk: torch.Tensor,
    *,
    batched: bool,
    split_k_up: int,
    split_k_down: int,
    post_scale_only: bool,
    launch_proj: Callable[[bool, torch.Tensor, ChunkRouting, torch.Tensor, int], None],
) -> None:
    device = active_x_chunk.device
    w = ctx.weights
    routing = _build_chunk_routing(
        topk_ids_chunk, ctx.expert_to_slot, ctx.num_slots, batched=batched
    )
    has_row_scale = w.up_gate_block_scales is not None
    up_row_scale = w.up_gate_scale if has_row_scale else None
    down_row_scale = w.down_scale if has_row_scale else None
    up_global_scale = None if has_row_scale else w.up_gate_global_scale
    down_global_scale = None if has_row_scale else w.down_global_scale

    mid = torch.empty((routing.num_pairs, ctx.up_gate_out), device=device, dtype=torch.float32)
    up_target = (
        torch.empty(
            (split_k_up, routing.num_pairs, ctx.up_gate_out), device=device, dtype=torch.float32
        )
        if split_k_up > 1
        else mid
    )
    act_dtype = torch.float32 if w.group_size > 0 else out_chunk.dtype
    act = torch.empty((routing.num_pairs, ctx.intermediate_size), device=device, dtype=act_dtype)
    down_partial = torch.empty(
        (split_k_down, routing.num_pairs, ctx.hidden_size), device=device, dtype=torch.float32
    )

    launch_proj(False, active_x_chunk, routing, up_target, split_k_up)
    if split_k_up > 1:
        _launch_splitk_reduce(
            ctx,
            up_target,
            None,
            up_row_scale,
            up_global_scale,
            routing,
            mid,
            split_k=split_k_up,
            is_int8=False,
            post_scale_only=post_scale_only,
            has_row_scale=has_row_scale,
        )
    _launch_activate_rotate(
        ctx,
        mid,
        act,
        out_chunk,
        routing.topk_ids_flat,
        routing.expert_to_slot,
        num_experts=routing.num_experts,
        num_slots=ctx.num_slots,
        num_pairs=routing.num_pairs,
        has_slot_map=routing.has_slot_map,
    )
    launch_proj(True, act, routing, down_partial, split_k_down)
    _launch_weighted_reduce(
        ctx,
        down_partial,
        None,
        down_row_scale,
        down_global_scale,
        routing.topk_ids_flat,
        routing.expert_to_slot,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        num_experts=routing.num_experts,
        num_slots=ctx.num_slots,
        num_pairs=routing.num_pairs,
        top_k=routing.top_k,
        split_k=split_k_down,
        is_int8=False,
        has_slot_map=routing.has_slot_map,
        post_scale_only=post_scale_only,
        has_row_scale=has_row_scale,
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
    ctx: _MoERunContext,
    active_x_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    out_chunk: torch.Tensor,
    scales: _Nvfp4ScaleMeta,
) -> None:
    device = active_x_chunk.device
    chunk_rows = active_x_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    w = ctx.weights
    schedule = _nvidia_policy.select_nvfp4_prefill_schedule(
        ctx.target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=ctx.num_slots,
        active_slots_hint=ctx.active_slots_hint,
        active_dtype=active_x_chunk.dtype,
        logical_dtype=out_chunk.dtype,
    )

    def _gather(buckets: PrefillBuckets) -> torch.Tensor:
        x_compact = torch.empty(
            (num_pairs, ctx.hidden_size), device=device, dtype=active_x_chunk.dtype
        )
        _launch_moe_prefill_gather(
            active_x_chunk,
            None,
            buckets.compact_to_pair,
            x_compact,
            None,
            ctx.hidden_size,
            top_k,
            has_scale=False,
        )
        return x_compact

    def _proj(
        x: torch.Tensor, buckets: PrefillBuckets, out: torch.Tensor, *, is_down: bool
    ) -> None:
        _nvfp4_triton.launch_nvfp4_grouped_gemm(
            x,
            w.down_qdata if is_down else w.up_gate_qdata,
            w.down_scale if is_down else w.up_gate_scale,
            w.down_global_scale if is_down else w.up_gate_global_scale,
            buckets,
            out,
            out_features=ctx.hidden_size if is_down else ctx.up_gate_out,
            in_features=ctx.intermediate_size if is_down else ctx.hidden_size,
            scale_cols=scales.down_cols if is_down else scales.up_cols,
            scale_stride=scales.down_stride if is_down else scales.up_stride,
            schedule=schedule,
            high_first=w.high_first,
        )

    def _activate(mid: torch.Tensor, buckets: PrefillBuckets) -> torch.Tensor:
        act_dtype = torch.float32 if w.group_size > 0 else out_chunk.dtype
        act = torch.empty((num_pairs, ctx.intermediate_size), device=device, dtype=act_dtype)
        _launch_activate_rotate(
            ctx,
            mid,
            act,
            out_chunk,
            buckets.compact_to_pair,
            None,
            num_experts=num_pairs,
            num_slots=num_pairs,
            num_pairs=num_pairs,
            has_slot_map=False,
        )
        return act

    _run_prefill_pipeline(
        ctx,
        topk_ids_chunk,
        topk_weights_chunk,
        out_chunk,
        block_m=schedule.block_m,
        mid_dtype=torch.float32,
        gather_up_input=_gather,
        run_up=lambda x, b, o: _proj(x, b, o, is_down=False),
        activate=_activate,
        run_down=lambda x, b, o: _proj(x, b, o, is_down=True),
    )


def _run_nvfp4_w4a4_prefill_chunk(
    ctx: _MoERunContext,
    src_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    out_chunk: torch.Tensor,
    scales: _Nvfp4ScaleMeta,
    *,
    input_is_rotated: bool,
) -> None:
    device = src_chunk.device
    chunk_rows = src_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    packed_hidden = ctx.hidden_size // 2
    w = ctx.weights
    schedule = _nvidia_policy.select_nvfp4_w4a4_prefill_schedule(
        ctx.target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=ctx.num_slots,
        active_slots_hint=ctx.active_slots_hint,
    )

    def _alloc_fp4_rows(
        row_count: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((row_count, width // 2), device=device, dtype=torch.uint8),
            torch.empty((row_count, width // 16), device=device, dtype=torch.float8_e4m3fn),
            torch.empty((row_count,), device=device, dtype=torch.float32),
        )

    def _gather(buckets: PrefillBuckets) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_q, x_sc, x_rs = _alloc_fp4_rows(chunk_rows, ctx.hidden_size)
        _nvfp4_triton.launch_nvfp4_quant(
            src_chunk,
            x_q,
            x_sc,
            x_rs,
            width=ctx.hidden_size,
            group_size=0 if input_is_rotated else w.group_size,
            high_first=w.high_first,
        )
        x_q_c, x_sc_c, x_rs_c = _alloc_fp4_rows(num_pairs, ctx.hidden_size)
        _nvfp4_triton.launch_nvfp4_gather(
            x_q,
            x_sc,
            x_rs,
            buckets.compact_to_pair,
            x_q_c,
            x_sc_c,
            x_rs_c,
            packed_k=packed_hidden,
            top_k=top_k,
        )
        return x_q_c, x_sc_c, x_rs_c

    def _proj(
        inp: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        buckets: PrefillBuckets,
        out: torch.Tensor,
        *,
        is_down: bool,
    ) -> None:
        _nvfp4_triton.launch_nvfp4_w4a4_grouped_gemm(
            *inp,
            w.down_qdata if is_down else w.up_gate_qdata,
            w.down_scale if is_down else w.up_gate_scale,
            w.down_global_scale if is_down else w.up_gate_global_scale,
            buckets,
            out,
            out_features=ctx.hidden_size if is_down else ctx.up_gate_out,
            in_features=ctx.intermediate_size if is_down else ctx.hidden_size,
            scale_cols=scales.down_cols if is_down else scales.up_cols,
            scale_stride=scales.down_stride if is_down else scales.up_stride,
            schedule=schedule,
        )

    def _activate(
        mid: torch.Tensor, buckets: PrefillBuckets
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        act_q, act_sc, act_rs = _alloc_fp4_rows(num_pairs, ctx.intermediate_size)
        _nvfp4_triton.launch_nvfp4_quant(
            mid,
            act_q,
            act_sc,
            act_rs,
            width=ctx.intermediate_size,
            group_size=w.group_size,
            high_first=w.high_first,
            compact_to_pair=buckets.compact_to_pair,
            activation=ctx.activation,
            accelerator_backend=ctx.target.backend,
        )
        return act_q, act_sc, act_rs

    _run_prefill_pipeline(
        ctx,
        topk_ids_chunk,
        topk_weights_chunk,
        out_chunk,
        block_m=schedule.block_m,
        mid_dtype=out_chunk.dtype,
        gather_up_input=_gather,
        run_up=lambda x, b, o: _proj(x, b, o, is_down=False),
        activate=_activate,
        run_down=lambda x, b, o: _proj(x, b, o, is_down=True),
    )


def _dispatch_nvfp4_moe(
    ctx: _MoERunContext,
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    out_2d: torch.Tensor,
) -> None:
    device = input_2d.device
    rows = input_2d.shape[0]
    top_k = ids_2d.shape[1]
    w = ctx.weights
    scales = _Nvfp4ScaleMeta(
        up_cols=int(triton.cdiv(ctx.hidden_size, nvfp4_layout.SCALE_COLUMN_TILE)),
        up_stride=w.up_gate_scale.shape[1] * w.up_gate_scale.shape[2],
        down_cols=int(triton.cdiv(ctx.intermediate_size, nvfp4_layout.SCALE_COLUMN_TILE)),
        down_stride=w.down_scale.shape[1] * w.down_scale.shape[2],
    )
    is_w4a4 = w.act_quant == "dynamic"

    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=False,
        hidden_size=ctx.hidden_size,
        intermediate_size=ctx.intermediate_size,
        up_gate_out=ctx.up_gate_out,
        logical_elem_bytes=input_2d.element_size(),
        group_size=w.group_size,
        is_nvfp4_w4a4=is_w4a4,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill":
        if is_w4a4:
            prep_full = (
                None if prepared_data is None else prepared_data.reshape(rows, ctx.hidden_size)
            )
            input_is_rotated = prep_full is not None and w.group_size > 0
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                src_chunk = (
                    input_2d[start:end] if prep_full is None else prep_full[start:end]
                ).contiguous()
                _run_nvfp4_w4a4_prefill_chunk(
                    ctx,
                    src_chunk,
                    ids_2d[start:end],
                    weights_2d[start:end],
                    out_2d[start:end],
                    scales,
                    input_is_rotated=input_is_rotated,
                )
            return
        for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
            input_2d, ids_2d, weights_2d, out_2d, prepared_data, w.group_size, chunk_rows
        ):
            _run_nvfp4_prefill_chunk(ctx, active_chunk, ids_chunk, weights_chunk, out_chunk, scales)
        return

    batched = regime == "batched"
    split_k_up = split_k_down = 1
    m1_schedule = _nvidia_policy.select_nvfp4_m1_schedule(ctx.target)
    active_schedule: _schedule.Nvfp4M1Schedule | _schedule.Nvfp4BatchedSchedule = m1_schedule
    post_scale_only = True
    num_sms = device_sm_count(device)
    if not batched:
        active_slots = _schedule.expected_bucket_tiles(
            top_k, ctx.num_slots, ctx.active_slots_hint, batched=False
        )
        split_k_up = _schedule.select_nvfp4_split_k(
            num_sms,
            active_slots,
            ctx.up_gate_out,
            ctx.hidden_size,
            block_n=m1_schedule.block_n,
            block_k=m1_schedule.block_k,
            cta_multiplier=m1_schedule.splitk_cta_multiplier,
        )
        split_k_down = _schedule.select_nvfp4_split_k(
            num_sms,
            active_slots,
            ctx.hidden_size,
            ctx.intermediate_size,
            block_n=m1_schedule.block_n,
            block_k=m1_schedule.block_k,
            cta_multiplier=m1_schedule.splitk_cta_multiplier,
        )

    for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
        input_2d, ids_2d, weights_2d, out_2d, prepared_data, w.group_size, chunk_rows
    ):
        if batched:
            chunk_n = active_chunk.shape[0]
            batched_schedule = _nvidia_policy.select_nvfp4_batched_schedule(
                ctx.target,
                rows=chunk_n,
                active_dtype=active_chunk.dtype,
                logical_dtype=out_chunk.dtype,
            )
            active_schedule = batched_schedule
            post_scale_only = batched_schedule.post_scale_only
            tiles = _schedule.expected_bucket_tiles(
                chunk_n * top_k, ctx.num_slots, ctx.active_slots_hint, batched=True
            )
            split_k_up = _schedule.select_nvfp4_split_k(
                num_sms,
                tiles,
                ctx.up_gate_out,
                ctx.hidden_size,
                block_n=batched_schedule.block_n,
                block_k=batched_schedule.block_k,
            )
            split_k_down = _schedule.select_nvfp4_split_k(
                num_sms,
                tiles,
                ctx.hidden_size,
                ctx.intermediate_size,
                block_n=batched_schedule.block_n,
                block_k=batched_schedule.block_k,
            )

        def _launch_nvfp4_step(
            per_expert_input: bool,
            x: torch.Tensor,
            routing: ChunkRouting,
            output: torch.Tensor,
            split_k: int,
            *,
            sched: _schedule.Nvfp4M1Schedule | _schedule.Nvfp4BatchedSchedule = active_schedule,
        ) -> None:
            if not per_expert_input:
                q, s, gs = w.up_gate_qdata, w.up_gate_scale, w.up_gate_global_scale
                n_out, n_in, sc, ss = (
                    ctx.up_gate_out,
                    ctx.hidden_size,
                    scales.up_cols,
                    scales.up_stride,
                )
            else:
                q, s, gs = w.down_qdata, w.down_scale, w.down_global_scale
                n_out, n_in, sc, ss = (
                    ctx.hidden_size,
                    ctx.intermediate_size,
                    scales.down_cols,
                    scales.down_stride,
                )
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
                num_slots=ctx.num_slots,
                split_k=split_k,
                schedule=sched,
                high_first=w.high_first,
                per_expert_input=per_expert_input,
                is_splitk=per_expert_input or split_k > 1,
            )

        _run_a16_chunk(
            ctx,
            active_chunk,
            ids_chunk,
            weights_chunk,
            out_chunk,
            batched=batched,
            split_k_up=split_k_up,
            split_k_down=split_k_down,
            post_scale_only=post_scale_only,
            launch_proj=_launch_nvfp4_step,
        )


def _run_int8_chunk(
    ctx: _MoERunContext,
    input_q_chunk: torch.Tensor,
    input_s_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    out_chunk: torch.Tensor,
    plan: _Int8ChunkPlan,
) -> None:
    device = input_q_chunk.device
    w = ctx.weights
    routing = _build_chunk_routing(
        topk_ids_chunk, ctx.expert_to_slot, ctx.num_slots, batched=plan.batched
    )
    block_m = _schedule.BATCH_BLOCK_M if plan.batched else _schedule.INT8_GEMM_BLOCK_M

    mid = torch.empty((routing.num_pairs, ctx.up_gate_out), device=device, dtype=out_chunk.dtype)
    up_target = (
        torch.empty(
            (plan.split_k_up, routing.num_pairs, ctx.up_gate_out), device=device, dtype=torch.int32
        )
        if plan.split_k_up > 1
        else mid
    )
    _int8_triton.launch_int8_proj(
        input_q_chunk,
        w.up_gate_qdata,
        input_s_chunk,
        w.up_gate_scale,
        routing,
        up_target,
        out_tiles=plan.up_tiles,
        out_features=ctx.up_gate_out,
        in_features=ctx.hidden_size,
        num_slots=ctx.num_slots,
        split_k=plan.split_k_up,
        block_m=block_m,
        schedule=plan.launch_schedule,
        compiler_options=plan.compiler_options,
        per_expert_input=False,
        is_splitk=plan.split_k_up > 1,
        aligned_nk=plan.up_aligned_nk,
        accumulator_seed=plan.up_seed,
        is_amd_hip=ctx.target.is_amd_hip,
    )
    if plan.split_k_up > 1:
        _launch_splitk_reduce(
            ctx,
            up_target,
            input_s_chunk,
            w.up_gate_scale,
            None,
            routing,
            mid,
            split_k=plan.split_k_up,
            is_int8=True,
        )

    act_q = torch.empty((routing.num_pairs, ctx.intermediate_size), device=device, dtype=torch.int8)
    act_s = torch.empty((routing.num_pairs,), device=device, dtype=torch.float32)
    _int8_triton.launch_int8_activate_quant(
        mid,
        routing.topk_ids_flat,
        routing.expert_to_slot,
        act_q,
        act_s,
        intermediate_size=ctx.intermediate_size,
        group_size=w.group_size,
        num_experts=routing.num_experts,
        num_slots=ctx.num_slots,
        activation=ctx.activation,
        has_slot_map=routing.has_slot_map,
        use_compact_map=False,
        accelerator_backend=ctx.target.backend,
    )
    del mid
    down_partial = torch.empty(
        (plan.split_k_down, routing.num_pairs, ctx.hidden_size), device=device, dtype=torch.int32
    )
    _int8_triton.launch_int8_proj(
        act_q,
        w.down_qdata,
        act_s,
        w.down_scale,
        routing,
        down_partial,
        out_tiles=plan.down_tiles,
        out_features=ctx.hidden_size,
        in_features=ctx.intermediate_size,
        num_slots=ctx.num_slots,
        split_k=plan.split_k_down,
        block_m=block_m,
        schedule=plan.launch_schedule,
        compiler_options=plan.compiler_options,
        per_expert_input=True,
        is_splitk=True,
        aligned_nk=plan.down_aligned_nk,
        accumulator_seed=plan.down_seed,
        is_amd_hip=ctx.target.is_amd_hip,
    )
    _launch_weighted_reduce(
        ctx,
        down_partial,
        act_s,
        w.down_scale,
        None,
        routing.topk_ids_flat,
        routing.expert_to_slot,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        num_experts=routing.num_experts,
        num_slots=ctx.num_slots,
        num_pairs=routing.num_pairs,
        top_k=routing.top_k,
        split_k=plan.split_k_down,
        is_int8=True,
        has_slot_map=routing.has_slot_map,
    )


def _run_prefill_int8_proj(
    ctx: _MoERunContext,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    buckets: PrefillBuckets,
    output: torch.Tensor,
    *,
    is_down: bool,
    schedule: _schedule.Int8PrefillSchedule,
    compiler_options: dict[str, object],
    accumulator_seed: int,
) -> None:
    w = ctx.weights
    qdata = w.down_qdata if is_down else w.up_gate_qdata
    scale = w.down_scale if is_down else w.up_gate_scale
    trellis = w.trellis_proj(is_down=is_down)
    out_features = ctx.hidden_size if is_down else ctx.up_gate_out
    in_features = ctx.intermediate_size if is_down else ctx.hidden_size

    if trellis is None:
        _int8_triton.launch_int8_grouped_gemm(
            x_q,
            qdata,
            buckets,
            x_scale,
            scale,
            output,
            out_features=out_features,
            in_features=in_features,
            schedule=schedule,
            compiler_options=compiler_options,
            aligned_nk=(
                out_features % schedule.block_n == 0 and in_features % schedule.block_k == 0
            ),
            accumulator_seed=accumulator_seed,
        )
        return

    slot_decoded_bytes = out_features * in_features
    if slot_decoded_bytes <= _schedule.TRELLIS_SCRATCH_BYTES:
        max_chunk_slots = min(
            ctx.num_slots, _schedule.TRELLIS_SCRATCH_BYTES // max(1, slot_decoded_bytes)
        )
        n_chunk = out_features
    else:
        max_chunk_slots = 1
        raw_chunk_rows = max(1, _schedule.TRELLIS_SCRATCH_BYTES // max(1, in_features))
        n_chunk = (
            (raw_chunk_rows // schedule.block_n) * schedule.block_n
            if raw_chunk_rows >= schedule.block_n
            else raw_chunk_rows
        )

    scratch = torch.empty(
        (max_chunk_slots, n_chunk, in_features), device=x_q.device, dtype=torch.int8
    )
    block_scales_u8 = trellis.block_scales.view(torch.uint8)
    for slot_start in range(0, ctx.num_slots, max_chunk_slots):
        slot_end = min(slot_start + max_chunk_slots, ctx.num_slots)
        window_slots = slot_end - slot_start
        window_counts = buckets.slot_counts[slot_start:slot_end]
        window_slot_offsets = (
            None if trellis.slot_offsets is None else trellis.slot_offsets[slot_start:slot_end]
        )
        window_widths = (
            None if trellis.row_widths is None else trellis.row_widths[slot_start:slot_end]
        )
        window_offsets = (
            None
            if window_widths is None
            else compute_row_offsets_2d(window_widths, dtype=torch.int32)
        )
        for n_start in range(0, out_features, n_chunk):
            n_end = min(n_start + n_chunk, out_features)
            n_rows = n_end - n_start
            q_view = scratch[:window_slots, :n_rows]
            if trellis.row_widths is None:
                chunk_streams = (
                    qdata[slot_start:slot_end, n_start:n_end].contiguous()
                    if n_rows < out_features
                    else qdata[slot_start:slot_end]
                )
            elif trellis.slot_offsets is None:
                chunk_streams = qdata[slot_start:slot_end]
            else:
                chunk_streams = qdata
            decode_slots_int8_out(
                chunk_streams,
                block_scales_u8[slot_start:slot_end, n_start:n_end],
                q_view,
                block_size=trellis.block_size,
                row_widths=None if window_widths is None else window_widths[:, n_start:n_end],
                slot_offsets=window_slot_offsets,
                active_counts=window_counts,
                row_offsets=None if window_offsets is None else window_offsets[:, n_start:n_end],
            )
            _int8_triton.launch_int8_grouped_gemm(
                x_q,
                q_view,
                buckets,
                x_scale,
                scale,
                output,
                out_features=out_features,
                in_features=in_features,
                schedule=schedule,
                compiler_options=compiler_options,
                aligned_nk=(n_rows % schedule.block_n == 0 and in_features % schedule.block_k == 0),
                accumulator_seed=accumulator_seed,
                slot_start=slot_start,
                slot_end=slot_end,
                n_start=n_start,
                n_rows=n_rows,
                slot_window=True,
            )


def _run_int8_prefill_chunk(
    ctx: _MoERunContext,
    input_q_chunk: torch.Tensor,
    input_s_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    out_chunk: torch.Tensor,
) -> None:
    device = input_q_chunk.device
    chunk_rows = input_q_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    schedule = _policy.select_int8_prefill_schedule(
        ctx.target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=ctx.num_slots,
        active_slots_hint=ctx.active_slots_hint,
    )
    compiler_options = amd_compiler_workarounds.lds_only_fence_options(schedule.lds_only_fences)
    expected_tiles = _schedule.expected_bucket_tiles(
        num_pairs, ctx.num_slots, ctx.active_slots_hint, block_m=schedule.block_m, batched=True
    )
    up_seed = rdna4_int8_accumulator_seed(
        ctx.target, ctx.hidden_size, expected_tiles * triton.cdiv(ctx.up_gate_out, schedule.block_n)
    )
    down_seed = rdna4_int8_accumulator_seed(
        ctx.target,
        ctx.intermediate_size,
        expected_tiles * triton.cdiv(ctx.hidden_size, schedule.block_n),
    )

    def _gather(buckets: PrefillBuckets) -> tuple[torch.Tensor, torch.Tensor]:
        x_compact = torch.empty((num_pairs, ctx.hidden_size), device=device, dtype=torch.int8)
        x_scale_compact = torch.empty((num_pairs,), device=device, dtype=torch.float32)
        _launch_moe_prefill_gather(
            input_q_chunk,
            input_s_chunk,
            buckets.compact_to_pair,
            x_compact,
            x_scale_compact,
            ctx.hidden_size,
            top_k,
            has_scale=True,
        )
        return x_compact, x_scale_compact

    def _up(
        x_compact: tuple[torch.Tensor, torch.Tensor], buckets: PrefillBuckets, mid: torch.Tensor
    ) -> None:
        _run_prefill_int8_proj(
            ctx,
            x_compact[0],
            x_compact[1],
            buckets,
            mid,
            is_down=False,
            schedule=schedule,
            compiler_options=compiler_options,
            accumulator_seed=up_seed,
        )

    def _activate(mid: torch.Tensor, buckets: PrefillBuckets) -> tuple[torch.Tensor, torch.Tensor]:
        act_q = torch.empty((num_pairs, ctx.intermediate_size), device=device, dtype=torch.int8)
        act_s = torch.empty((num_pairs,), device=device, dtype=torch.float32)
        _int8_triton.launch_int8_activate_quant(
            mid,
            buckets.compact_to_pair,
            None,
            act_q,
            act_s,
            intermediate_size=ctx.intermediate_size,
            group_size=ctx.weights.group_size,
            num_experts=num_pairs,
            num_slots=num_pairs,
            activation=ctx.activation,
            has_slot_map=False,
            use_compact_map=True,
            accelerator_backend=ctx.target.backend,
        )
        return act_q, act_s

    def _down(
        act: tuple[torch.Tensor, torch.Tensor], buckets: PrefillBuckets, down_out: torch.Tensor
    ) -> None:
        _run_prefill_int8_proj(
            ctx,
            act[0],
            act[1],
            buckets,
            down_out,
            is_down=True,
            schedule=schedule,
            compiler_options=compiler_options,
            accumulator_seed=down_seed,
        )

    _run_prefill_pipeline(
        ctx,
        topk_ids_chunk,
        topk_weights_chunk,
        out_chunk,
        block_m=schedule.block_m,
        mid_dtype=out_chunk.dtype,
        gather_up_input=_gather,
        run_up=_up,
        activate=_activate,
        run_down=_down,
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
    ctx: _MoERunContext,
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    int8_backend: LinearBackend,
    chunk_rows: int,
) -> None:
    rows = input_2d.shape[0]
    prep_q_full = None if prepared_data is None else prepared_data.reshape(rows, ctx.hidden_size)
    prep_s_full = None if prepared_scale is None else prepared_scale.reshape(rows)
    up_prep_plan = (
        None
        if prepared_data is not None
        else int8_backend.policy.select_execution_plan(ctx.target, in_features=ctx.hidden_size)
    )
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        inp_q_chunk, inp_s_chunk = _prepare_int8_input_gpu(
            input_2d[start:end],
            None if prep_q_full is None else prep_q_full[start:end],
            None if prep_s_full is None else prep_s_full[start:end],
            ctx.weights.group_size,
            target=ctx.target,
            int8_backend=int8_backend,
            up_prep_plan=up_prep_plan,
        )
        _run_int8_prefill_chunk(
            ctx,
            inp_q_chunk,
            inp_s_chunk,
            ids_2d[start:end],
            weights_2d[start:end],
            out_2d[start:end],
        )


def _dispatch_int8_moe(
    ctx: _MoERunContext,
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out_2d: torch.Tensor,
) -> None:
    from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

    device = input_2d.device
    rows = input_2d.shape[0]
    top_k = ids_2d.shape[1]
    int8_backend = int8_backend_mod.select_linear_backend_for_target(ctx.target)
    assert int8_backend is not None
    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=True,
        hidden_size=ctx.hidden_size,
        intermediate_size=ctx.intermediate_size,
        up_gate_out=ctx.up_gate_out,
        logical_elem_bytes=out_2d.element_size(),
        group_size=ctx.weights.group_size,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill":
        _dispatch_int8_prefill_chunks(
            ctx,
            input_2d,
            ids_2d,
            weights_2d,
            prepared_data,
            prepared_scale,
            out_2d,
            int8_backend=int8_backend,
            chunk_rows=chunk_rows,
        )
        return

    up_prep_plan = (
        None
        if prepared_data is not None
        else int8_backend.policy.select_execution_plan(ctx.target, in_features=ctx.hidden_size)
    )
    batched = regime == "batched"
    expected_tiles = _schedule.expected_bucket_tiles(
        chunk_rows * top_k, ctx.num_slots, ctx.active_slots_hint, batched=batched
    )
    launch_schedule = _policy.select_int8_launch_schedule(ctx.target)
    block_k = launch_schedule.block_k
    num_sms = device_sm_count(device)
    split_k_up = _schedule.select_int8_split_k(
        num_sms, expected_tiles, ctx.up_gate_out, ctx.hidden_size, block_k
    )
    split_k_down = _schedule.select_int8_split_k(
        num_sms, expected_tiles, ctx.hidden_size, ctx.intermediate_size, block_k
    )
    seed_rows = expected_tiles * _schedule.INT8_GEMM_BLOCK_M
    plan = _Int8ChunkPlan(
        batched=batched,
        launch_schedule=launch_schedule,
        compiler_options=amd_compiler_workarounds.lds_only_fence_options(
            launch_schedule.lds_only_fences
        ),
        split_k_up=split_k_up,
        split_k_down=split_k_down,
        up_seed=splitk_accumulator_seed(
            ctx.target,
            rows=seed_rows,
            out_features=ctx.up_gate_out,
            in_features=ctx.hidden_size,
            block_m=_schedule.INT8_GEMM_BLOCK_M,
            block_n=_schedule.INT8_GEMM_BLOCK_N,
            block_k=block_k,
            paired=False,
            split_k=split_k_up,
        ),
        down_seed=splitk_accumulator_seed(
            ctx.target,
            rows=seed_rows,
            out_features=ctx.hidden_size,
            in_features=ctx.intermediate_size,
            block_m=_schedule.INT8_GEMM_BLOCK_M,
            block_n=_schedule.INT8_GEMM_BLOCK_N,
            block_k=block_k,
            paired=False,
            split_k=split_k_down,
        ),
        up_aligned_nk=(
            ctx.up_gate_out % _schedule.INT8_GEMM_BLOCK_N == 0 and ctx.hidden_size % block_k == 0
        ),
        down_aligned_nk=(
            ctx.hidden_size % _schedule.INT8_GEMM_BLOCK_N == 0
            and ctx.intermediate_size % block_k == 0
        ),
        up_tiles=triton.cdiv(ctx.up_gate_out, _schedule.INT8_GEMM_BLOCK_N),
        down_tiles=triton.cdiv(ctx.hidden_size, _schedule.INT8_GEMM_BLOCK_N),
    )

    inp_q_2d, inp_s_1d = _prepare_int8_input_gpu(
        input_2d,
        prepared_data,
        prepared_scale,
        ctx.weights.group_size,
        target=ctx.target,
        int8_backend=int8_backend,
        up_prep_plan=up_prep_plan,
    )
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        _run_int8_chunk(
            ctx,
            inp_q_2d[start:end],
            inp_s_1d[start:end],
            ids_2d[start:end],
            weights_2d[start:end],
            out_2d[start:end],
            plan,
        )


def _dispatch_trellis_moe(
    ctx: _MoERunContext,
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    out_2d: torch.Tensor,
) -> None:
    from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

    w = ctx.weights
    assert w.up_gate_block_scales is not None
    assert w.down_block_scales is not None
    device = input_2d.device
    num_sms = device_sm_count(device)
    rows = input_2d.shape[0]
    top_k = ids_2d.shape[1]
    up_blocks = w.up_gate_block_scales.shape[2]
    down_blocks = w.down_block_scales.shape[2]

    int8_backend = (
        int8_backend_mod.select_linear_backend_for_target(ctx.target)
        if _policy.supports_trellis_int8_prefill(ctx.target)
        else None
    )
    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=int8_backend is not None,
        hidden_size=ctx.hidden_size,
        intermediate_size=ctx.intermediate_size,
        up_gate_out=ctx.up_gate_out,
        logical_elem_bytes=out_2d.element_size(),
        group_size=w.group_size,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill" and int8_backend is not None:
        _dispatch_int8_prefill_chunks(
            ctx,
            input_2d,
            ids_2d,
            weights_2d,
            prepared_data,
            None,
            out_2d,
            int8_backend=int8_backend,
            chunk_rows=chunk_rows,
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

    up_slot_bytes = w.up_gate_qdata.numel() // max(1, ctx.num_slots)
    down_slot_bytes = w.down_qdata.numel() // max(1, ctx.num_slots)
    expected_slots = (
        min(ctx.num_slots, ctx.active_slots_hint) if ctx.active_slots_hint > 0 else ctx.num_slots
    )
    split_k_up = split_k_down = 1
    schedule = _policy.select_trellis_schedule(
        ctx.target,
        batched=batched,
        num_pairs=top_k,
        expected_slots=expected_slots,
    )
    if not batched:
        active_slots = _schedule.expected_bucket_tiles(
            top_k, ctx.num_slots, ctx.active_slots_hint, batched=False
        )
        split_k_up = _schedule.select_trellis_split_k(
            num_sms, active_slots, ctx.up_gate_out, up_blocks, max(1, up_slot_bytes * active_slots)
        )
        split_k_down = _schedule.select_trellis_split_k(
            num_sms,
            active_slots,
            ctx.hidden_size,
            down_blocks,
            max(1, down_slot_bytes * active_slots),
        )

    for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
        input_2d, ids_2d, weights_2d, out_2d, prepared_data, w.group_size, a16_chunk_rows
    ):
        if batched:
            num_pairs = active_chunk.shape[0] * top_k
            schedule = _policy.select_trellis_schedule(
                ctx.target,
                batched=True,
                num_pairs=num_pairs,
                expected_slots=expected_slots,
            )
            weight_buckets = min(
                _schedule.max_bucket_tiles(num_pairs, ctx.num_slots, _schedule.BATCH_BLOCK_M),
                _schedule.expected_bucket_tiles(
                    num_pairs, ctx.num_slots, ctx.active_slots_hint, batched=True
                ),
            )
            active_proj_tiles = (
                min(
                    _schedule.max_bucket_tiles(num_pairs, ctx.num_slots, schedule.block_m),
                    _schedule.expected_bucket_tiles(
                        num_pairs,
                        ctx.num_slots,
                        ctx.active_slots_hint,
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
                triton.cdiv(ctx.up_gate_out, schedule.block_n),
                up_blocks,
                max(1, up_slot_bytes * weight_buckets),
            )
            split_k_down = _schedule.select_trellis_split_k(
                num_sms,
                active_proj_tiles,
                triton.cdiv(ctx.hidden_size, schedule.block_n),
                down_blocks,
                max(1, down_slot_bytes * weight_buckets),
            )

        def _launch_trellis_step(
            per_expert_input: bool,
            x: torch.Tensor,
            routing: ChunkRouting,
            output: torch.Tensor,
            split_k: int,
            *,
            sched: _schedule.TrellisMoeSchedule = schedule,
        ) -> None:
            proj = w.trellis_proj(is_down=per_expert_input)
            assert proj is not None
            q = w.down_qdata if per_expert_input else w.up_gate_qdata
            rs = w.down_scale if per_expert_input else w.up_gate_scale
            n_out = ctx.hidden_size if per_expert_input else ctx.up_gate_out
            n_in = ctx.intermediate_size if per_expert_input else ctx.hidden_size
            _trellis_triton.launch_trellis_proj(
                x,
                q,
                proj.block_scales,
                rs,
                proj.row_widths,
                proj.slot_offsets,
                proj.block_size,
                routing,
                output,
                out_features=n_out,
                in_features=n_in,
                num_slots=ctx.num_slots,
                split_k=split_k,
                batched=batched,
                per_expert_input=per_expert_input,
                is_splitk=per_expert_input or split_k > 1,
                schedule=sched,
                is_amd_hip=ctx.target.is_amd_hip,
            )

        _run_a16_chunk(
            ctx,
            active_chunk,
            ids_chunk,
            weights_chunk,
            out_chunk,
            batched=batched,
            split_k_up=split_k_up,
            split_k_down=split_k_down,
            post_scale_only=True,
            launch_proj=_launch_trellis_step,
        )


def run_moe_ffn(
    ops: _ops._MoEOperands,
    *,
    target: AcceleratorTarget | None = None,
) -> torch.Tensor:
    """Execute single-token, small-batch, or prefill quantized MoE kernels on GPU."""
    inp = ops.input
    w = ops.weights
    rows = math.prod(inp.shape[:-1])
    hidden_size = inp.shape[-1]
    top_k = ops.topk_ids.shape[-1]
    result = (
        torch.empty_like(inp, memory_format=torch.contiguous_format) if ops.out is None else ops.out
    )
    out_2d = result.view(rows, hidden_size)
    input_2d = inp.reshape(rows, hidden_size)
    ids_2d = ops.topk_ids.reshape(rows, top_k).contiguous()
    weights_2d = ops.topk_weights.reshape(rows, top_k).contiguous()
    resolved_target = AcceleratorTarget.from_device(inp.device) if target is None else target

    if w.up_gate_block_scales is not None:
        assert w.down_block_scales is not None
        num_slots = w.down_block_scales.shape[0]
        intermediate_size = w.down_block_scales.shape[2] * w.down_block_size
        up_gate_out = w.up_gate_block_scales.shape[1]
    elif w.up_gate_qdata.dtype is torch.int8:
        num_slots, _, intermediate_size = w.down_qdata.shape
        up_gate_out = w.up_gate_qdata.shape[1]
    else:
        num_slots, _, packed_inter = w.down_qdata.shape
        intermediate_size = 2 * packed_inter
        up_gate_out = w.up_gate_qdata.shape[1]

    ctx = _MoERunContext(
        weights=w,
        expert_to_slot=ops.expert_to_slot,
        target=resolved_target,
        accumulate=ops.accumulate,
        active_slots_hint=ops.active_slots_hint,
        num_slots=num_slots,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        activation=w.activation,
    )
    if w.up_gate_block_scales is not None:
        _dispatch_trellis_moe(ctx, input_2d, ids_2d, weights_2d, ops.prepared_data, out_2d)
    elif w.up_gate_qdata.dtype is torch.int8:
        _dispatch_int8_moe(
            ctx, input_2d, ids_2d, weights_2d, ops.prepared_data, ops.prepared_scale, out_2d
        )
    else:
        _dispatch_nvfp4_moe(ctx, input_2d, ids_2d, weights_2d, ops.prepared_data, out_2d)
    return result


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
    act_quant: str | None = None,
    gate_bound: float | None = None,
    up_bound: float | None = None,
) -> torch.Tensor:
    """Thin flat-argument wrapper around ``run_moe_ffn``."""
    return run_moe_ffn(
        _ops._MoEOperands(
            input,
            topk_ids,
            topk_weights,
            _ops._MoEWeightStorage(
                up_gate_qdata,
                up_gate_scale,
                down_qdata,
                down_scale,
                up_gate_global_scale,
                down_global_scale,
                group_size,
                activation_fn,
                gate_first,
                high_first,
                up_gate_block_scales,
                down_block_scales,
                up_gate_row_widths,
                down_row_widths,
                up_gate_slot_offsets,
                down_slot_offsets,
                up_gate_block_size,
                down_block_size,
                act_quant,
                gate_bound,
                up_bound,
            ),
            expert_to_slot,
            prepared_data,
            prepared_scale,
            out,
            accumulate,
            active_slots_hint,
        ),
        target=target,
    )


__all__ = [
    "compile_moe_kernels",
    "moe_ffn",
]
