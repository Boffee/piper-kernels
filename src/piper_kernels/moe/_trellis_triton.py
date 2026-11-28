"""ConvRot Trellis W4A16 MoE Triton kernels and host dispatcher."""

# Triton's JIT launcher, constexpr arguments, and wide kernel signatures are not
# fully represented by Python typing rules.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

from typing import Literal

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._splitk import device_sm_count
from piper_kernels.linear.convrot.int8.trellis_triton import _gemv_accumulate_k_blocks
from piper_kernels.trellis._layout import WRAP_GUARD_BYTES, TrellisLayout
from piper_kernels.trellis.triton import _le_step_bits_4, compute_row_offsets_2d

from . import _routing_triton, _schedule
from ._routing_triton import _resolve_slot

_WRAP_GUARD_BYTES = tl.constexpr(WRAP_GUARD_BYTES)
_BATCH_BLOCK_M_TL = tl.constexpr(_schedule.BATCH_BLOCK_M)


@triton.jit
def _trellis_slot_row_params(
    row_widths_ptr,
    row_offsets_ptr,
    slot_offsets_ptr,
    slot,
    width_slot_idx,
    pid_n,
    out_features,
    blocks_per_row,
    bitstreams_bytes,
    denom,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    slot_bytes,
    variable_rows: tl.constexpr,
    has_slot_offsets: tl.constexpr,
):
    """Resolve slot/row bitstream offsets, byte bounds, and LE step widths for MoE GEMV."""
    if variable_rows:
        width_idx = width_slot_idx * out_features + pid_n
        width = tl.load(row_widths_ptr + width_idx).to(tl.int32)
        row_offset = tl.load(row_offsets_ptr + width_idx).to(tl.int64)
        width64 = width.to(tl.int64)
        eff_code_bits = tl.maximum(width - _WRAP_GUARD_BYTES, 0) * 8
        eff_step_bits = eff_code_bits // denom
        eff_wide_steps = eff_code_bits - eff_step_bits * denom
        if has_slot_offsets:
            raw_slot_base = tl.load(slot_offsets_ptr + slot).to(tl.int64)
            valid_base = raw_slot_base >= 0
            slot_base = tl.where(valid_base, raw_slot_base, 0)
            row_base_bytes = slot_base + row_offset * blocks_per_row
            limit_bytes = tl.where(valid_base, bitstreams_bytes.to(tl.int64), 0)
            non_empty = valid_base & (width > _WRAP_GUARD_BYTES)
        else:
            slot_base = slot * slot_bytes.to(tl.int64)
            row_base_bytes = slot_base + row_offset * blocks_per_row
            limit_bytes = tl.minimum(
                slot_base + slot_bytes.to(tl.int64), bitstreams_bytes.to(tl.int64)
            )
            non_empty = width > _WRAP_GUARD_BYTES
    else:
        width64 = bytes_per_block.to(tl.int64)
        row_base_bytes = (slot * out_features.to(tl.int64) + pid_n) * blocks_per_row * width64
        limit_bytes = bitstreams_bytes.to(tl.int64)
        eff_step_bits = step_bits
        eff_wide_steps = num_wide_steps
        non_empty = True
    return row_base_bytes, width64, limit_bytes, eff_step_bits, eff_wide_steps, non_empty


@triton.jit
def _trellis_slot_row_step(
    row_widths_ptr,
    row_offsets_ptr,
    slot_offsets_ptr,
    slot,
    width_slot_idx,
    pid_n,
    out_features,
    blocks_per_row,
    bitstreams_bytes,
    denom,
    block_size,
    first_step,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    slot_bytes,
    variable_rows: tl.constexpr,
    has_slot_offsets: tl.constexpr,
    exact_pow2: tl.constexpr,
):
    """Resolve one MoE trellis output row's storage offsets and 4-step bit indices."""
    row_base, width64, limit_bytes, esb, ews, non_empty = _trellis_slot_row_params(
        row_widths_ptr, row_offsets_ptr, slot_offsets_ptr, slot, width_slot_idx, pid_n,
        out_features, blocks_per_row, bitstreams_bytes, denom, step_bits, num_wide_steps,
        bytes_per_block, slot_bytes, variable_rows=variable_rows, has_slot_offsets=has_slot_offsets,
    )  # fmt: skip
    sb0, sb1, sb2, sb3 = _le_step_bits_4(first_step, esb, ews, denom, block_size, exact_pow2)
    row_blk = (slot * out_features.to(tl.int64) + pid_n) * blocks_per_row
    return row_blk, row_base, width64, limit_bytes, sb0, sb1, sb2, sb3, sb0 >> 3, non_empty


@triton.jit
def _trellis_slot_second_row(
    row_widths_ptr,
    row_offsets_ptr,
    slot_offsets_ptr,
    slot,
    width_slot_idx,
    n1,
    n1_in_bounds,
    out_features,
    blocks_per_row,
    bitstreams_bytes,
    denom,
    block_size,
    first_step,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    slot_bytes,
    width64_0,
    sb0_0,
    sb0_1,
    sb0_2,
    sb0_3,
    fb0,
    variable_rows: tl.constexpr,
    has_slot_offsets: tl.constexpr,
    exact_pow2: tl.constexpr,
):
    """Resolve the paired second output row for ``block_n == 2`` MoE Trellis GEMV."""
    if variable_rows:
        row_blk1, row_base1, width64_1, _, sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1_raw = (
            _trellis_slot_row_step(
                row_widths_ptr, row_offsets_ptr, slot_offsets_ptr, slot, width_slot_idx, n1,
                out_features, blocks_per_row, bitstreams_bytes, denom, block_size, first_step,
                step_bits, num_wide_steps, bytes_per_block, slot_bytes, variable_rows=True,
                has_slot_offsets=has_slot_offsets, exact_pow2=exact_pow2,
            )
        )  # fmt: skip
        ne1 = ne1_raw & n1_in_bounds
    else:
        row_blk1 = (slot * out_features.to(tl.int64) + n1) * blocks_per_row
        width64_1 = width64_0
        row_base1 = row_blk1 * width64_1
        sb1_0, sb1_1, sb1_2, sb1_3, fb1 = sb0_0, sb0_1, sb0_2, sb0_3, fb0
        ne1 = n1_in_bounds
    return row_blk1, row_base1, width64_1, sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1


@triton.jit
def _store_moe_trellis_row(
    acc,
    row_scale_ptr,
    output_ptr,
    slot,
    safe_pairs,
    m_mask,
    pid_k,
    pid_n,
    n_in_bounds,
    num_pairs,
    out_features,
    is_splitk: tl.constexpr,
):
    """Reduce K-group accumulators for one MoE output row and store partials or outputs."""
    sums = tl.sum(acc, axis=1)
    n_valid = m_mask & n_in_bounds
    if is_splitk:
        partial_offsets = (
            pid_k.to(tl.int64) * (num_pairs.to(tl.int64) * out_features)
            + safe_pairs * out_features
            + pid_n
        )
        tl.store(output_ptr + partial_offsets, sums, mask=n_valid)
    else:
        row_scale = tl.load(
            row_scale_ptr + slot * out_features + pid_n,
            mask=n_in_bounds,
            other=0.0,
        ).to(tl.float32)
        tl.store(output_ptr + safe_pairs * out_features + pid_n, sums * row_scale, mask=n_valid)


@triton.jit(
    do_not_specialize=[
        "out_features",
        "in_features",
        "blocks_per_row",
        "bitstreams_bytes",
        "block_size",
        "step_bits",
        "num_wide_steps",
        "bytes_per_block",
        "slot_bytes",
        "num_experts",
        "num_slots",
        "num_pairs",
        "top_k",
        "split_k",
    ]
)
def trellis_moe_gemv_kernel(
    x_rot_ptr,
    bitstreams_ptr,
    block_scales_ptr,
    row_scale_ptr,
    row_widths_ptr,
    row_offsets_ptr,
    slot_offsets_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    tile_slots_ptr,
    tile_pair_ids_ptr,
    output_ptr,
    out_features,
    in_features,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    slot_bytes,
    num_experts,
    num_slots,
    num_pairs,
    top_k,
    split_k,
    tile_steps: tl.constexpr,
    block_m: tl.constexpr,
    batched: tl.constexpr,
    has_slot_map: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    variable_rows: tl.constexpr,
    has_slot_offsets: tl.constexpr,
    exact_pow2: tl.constexpr,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    x_mul4: tl.constexpr,
    unroll_k: tl.constexpr = 1,
    block_n: tl.constexpr = 1,
    use_sad_u8: tl.constexpr = False,
    split_x_loads: tl.constexpr = False,
):
    """Fused in-register LE Trellis -> A16 Split-K GEMV for ``m1`` and bucketed ``batched`` MoE."""
    tl.static_assert((block_n == 1) | (block_n == 2))
    n_tiles = tl.cdiv(out_features, block_n)
    pid_0 = tl.program_id(0)
    pid_n_tile = pid_0 % n_tiles
    pid_tile = pid_0 // n_tiles
    pid_k = tl.program_id(1)
    n0 = pid_n_tile * block_n

    if batched:
        sub_tiles = _BATCH_BLOCK_M_TL // block_m
        bucket_idx = pid_tile // sub_tiles
        sub_idx = pid_tile - bucket_idx * sub_tiles
        slot = tl.load(tile_slots_ptr + bucket_idx).to(tl.int64)
        if slot < 0:
            return
        pair_offs = (
            bucket_idx.to(tl.int64) * _BATCH_BLOCK_M_TL + sub_idx * block_m + tl.arange(0, block_m)
        )
        pair_ids = tl.load(tile_pair_ids_ptr + pair_offs)
        m_mask = pair_ids >= 0
        if tl.sum(m_mask.to(tl.int32)) == 0:
            return
        safe_pairs = tl.where(m_mask, pair_ids, 0).to(tl.int64)
        ms = safe_pairs if per_expert_input else (safe_pairs // top_k)
        m0 = tl.full((), 0, tl.int64)
        width_slot_idx = bucket_idx.to(tl.int64)
    else:
        slot = _resolve_slot(
            topk_ids_ptr,
            expert_to_slot_ptr,
            pid_tile,
            num_experts,
            num_slots,
            has_slot_map,
        )
        if slot < 0:
            return
        m_mask = tl.arange(0, 1) >= 0
        safe_pairs = tl.full((1,), pid_tile, tl.int64)
        m0 = pid_tile.to(tl.int64) if per_expert_input else tl.full((), 0, tl.int64)
        ms = tl.full((1,), pid_tile if per_expert_input else 0, tl.int64)
        width_slot_idx = pid_tile.to(tl.int64)

    denom = tile_steps if exact_pow2 else block_size
    first_step = tl.arange(0, tile_steps // 4) * 4

    row_blk0, row_base0, width64_0, limit_bytes, sb0_0, sb0_1, sb0_2, sb0_3, fb0, ne0 = (
        _trellis_slot_row_step(
            row_widths_ptr, row_offsets_ptr, slot_offsets_ptr, slot, width_slot_idx, n0,
            out_features, blocks_per_row, bitstreams_bytes, denom, block_size, first_step,
            step_bits, num_wide_steps, bytes_per_block, slot_bytes, variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets, exact_pow2=exact_pow2,
        )
    )  # fmt: skip

    if block_n == 2:
        n1_in_bounds = (n0 + 1) < out_features
        n1 = tl.minimum(n0 + 1, out_features - 1)
        row_blk1, row_base1, width64_1, sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1 = (
            _trellis_slot_second_row(
                row_widths_ptr, row_offsets_ptr, slot_offsets_ptr, slot, width_slot_idx, n1,
                n1_in_bounds, out_features, blocks_per_row, bitstreams_bytes, denom, block_size,
                first_step, step_bits, num_wide_steps, bytes_per_block, slot_bytes, width64_0,
                sb0_0, sb0_1, sb0_2, sb0_3, fb0, variable_rows=variable_rows,
                has_slot_offsets=has_slot_offsets, exact_pow2=exact_pow2,
            )
        )  # fmt: skip
    else:
        n1_in_bounds = False
        n1 = n0
        row_blk1, row_base1, width64_1, sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1 = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    acc0, acc1 = _gemv_accumulate_k_blocks(
        x_rot_ptr, bitstreams_ptr, block_scales_ptr, row_blk0, row_base0, width64_0,
        sb0_0, sb0_1, sb0_2, sb0_3, fb0, m0, ms, m_mask, pid_k, blocks_per_row,
        limit_bytes, block_size, in_features, split_k, ne0, row_blk1, row_base1, width64_1,
        sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1, tile_steps=tile_steps, block_m=block_m,
        exact_pow2=exact_pow2, needs_w2=needs_w2, use_dp4a=use_dp4a,
        has_tail_bytes=has_tail_bytes, ptr_aligned=ptr_aligned, x_mul4=x_mul4,
        unroll_k=unroll_k, block_n=block_n, use_sad_u8=use_sad_u8,
        split_x_loads=split_x_loads,
    )  # fmt: skip
    _store_moe_trellis_row(
        acc0, row_scale_ptr, output_ptr, slot, safe_pairs, m_mask, pid_k, n0,
        n0 < out_features, num_pairs, out_features, is_splitk=is_splitk,
    )  # fmt: skip
    if block_n == 2:
        _store_moe_trellis_row(
            acc1, row_scale_ptr, output_ptr, slot, safe_pairs, m_mask, pid_k, n1,
            n1_in_bounds, num_pairs, out_features, is_splitk=is_splitk,
        )  # fmt: skip


def _resolve_m1_active_slots(
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
) -> torch.Tensor:
    if expert_to_slot is None:
        return topk_ids_flat.to(torch.int64).clamp(0, num_slots - 1)
    num_experts = expert_to_slot.shape[0]
    in_range = (topk_ids_flat >= 0) & (topk_ids_flat < num_experts)
    safe_experts = torch.where(in_range, topk_ids_flat.to(torch.int64), 0)
    slots = expert_to_slot.index_select(0, safe_experts).to(torch.int64)
    valid = in_range & (slots >= 0) & (slots < num_slots)
    return torch.where(valid, slots, 0)


def launch_trellis_proj(
    x_rot: torch.Tensor,
    qdata: torch.Tensor,
    block_scales: torch.Tensor,
    row_scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    slot_offsets: torch.Tensor | None,
    block_size: int,
    routing: _routing_triton.A16ChunkRouting,
    expert_to_slot: torch.Tensor | None,
    output: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    num_slots: int,
    split_k: int,
    batched: bool,
    per_expert_input: bool,
    is_splitk: bool,
    schedule: _schedule.TrellisMoeSchedule,
) -> None:
    blocks_per_row = block_scales.shape[2]
    tile_steps = triton.next_power_of_2(block_size)
    exact_pow2 = block_size == tile_steps
    variable_rows = row_widths is not None
    has_slot_offsets = slot_offsets is not None

    if batched:
        assert routing.tile_slots is not None
        assert routing.tile_pair_ids is not None
        sub_tiles = _schedule.BATCH_BLOCK_M // schedule.block_m
        grid_tiles = routing.tile_slots.shape[0] * sub_tiles
        active_slots = (
            routing.tile_slots.clamp(0, num_slots - 1).to(torch.int64) if variable_rows else None
        )
    else:
        grid_tiles = routing.top_k
        active_slots = (
            _resolve_m1_active_slots(routing.topk_ids_flat, expert_to_slot, num_slots)
            if variable_rows
            else None
        )

    per_slot_2d = variable_rows and not has_slot_offsets
    if variable_rows:
        assert row_widths is not None
        assert active_slots is not None
        active_widths = row_widths.index_select(0, active_slots).contiguous()
        active_offsets = compute_row_offsets_2d(active_widths, dtype=torch.int32)
        slot_bytes = 0 if has_slot_offsets else qdata.shape[1]
        step_bits = num_wide_steps = bytes_per_block = 0
        needs_w2 = True
    else:
        layout = TrellisLayout.from_storage(block_size, qdata.shape[3])
        active_widths = active_offsets = None
        slot_bytes = 0
        step_bits = layout.step_bits
        num_wide_steps = layout.num_wide_steps
        bytes_per_block = layout.bytes_per_block
        needs_w2 = layout.block_bits > 5 * layout.block_size

    flat_qdata = qdata.view(-1)
    ptr_aligned = qdata.is_contiguous() and qdata.data_ptr() % 4 == 0
    has_tail_bytes = per_slot_2d or (qdata.numel() % 4 != 0)
    x_mul4 = block_size % 4 == 0 and in_features % 4 == 0

    grid = (triton.cdiv(out_features, schedule.block_n) * grid_tiles, split_k)
    with device_context(x_rot.device):
        trellis_moe_gemv_kernel[grid](
            x_rot,
            flat_qdata,
            block_scales.view(torch.uint8),
            row_scale,
            active_widths,
            active_offsets,
            slot_offsets,
            None if batched else routing.topk_ids_flat,
            None if batched else routing.expert_to_slot,
            routing.tile_slots,
            routing.tile_pair_ids,
            output,
            out_features,
            in_features,
            blocks_per_row,
            qdata.numel(),
            block_size,
            step_bits,
            num_wide_steps,
            bytes_per_block,
            slot_bytes,
            routing.num_experts,
            num_slots,
            routing.num_pairs,
            routing.top_k,
            split_k,
            tile_steps=tile_steps,
            block_m=schedule.block_m,
            batched=batched,
            has_slot_map=False if batched else routing.has_slot_map,
            per_expert_input=per_expert_input,
            is_splitk=is_splitk,
            variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets,
            exact_pow2=exact_pow2,
            needs_w2=needs_w2,
            use_dp4a=schedule.use_dp4a,
            has_tail_bytes=has_tail_bytes,
            ptr_aligned=ptr_aligned,
            x_mul4=x_mul4,
            unroll_k=schedule.unroll_k,
            block_n=schedule.block_n,
            use_sad_u8=schedule.use_sad_u8,
            split_x_loads=schedule.split_x_loads,
            num_warps=schedule.num_warps,
        )


def dispatch_trellis_a16_moe(
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
    regime: Literal["m1", "batched", "prefill"],
    chunk_rows: int,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    active_slots_hint: int,
) -> None:
    device = input_2d.device
    num_sms = device_sm_count(device)
    rows, hidden_size = input_2d.shape
    num_slots, _, down_blocks = down_block_scales.shape
    intermediate_size = down_blocks * down_block_size
    up_gate_out, up_blocks = up_gate_block_scales.shape[1], up_gate_block_scales.shape[2]
    top_k = ids_2d.shape[1]

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
        routing: _routing_triton.A16ChunkRouting,
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
        launch_trellis_proj(
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
        )

    for active_chunk, ids_chunk, weights_chunk, out_chunk in _routing_triton.iter_a16_chunks(
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
        _routing_triton.run_a16_chunk(
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
