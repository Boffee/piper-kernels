"""Portable PyTorch reference for offload-friendly quantized MoE FFNs."""

# Custom-op aligned signature mirrors the flat tensor arguments passed from _ops.py.
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable

import torch

from piper_kernels._input_activations import apply_input_activation, parse_input_activation
from piper_kernels.trellis._layout import TrellisLayout
from piper_kernels.trellis._reference import decode_reference, decode_variable_rows_reference
from piper_kernels.weights.convrot._rotation import rotate_groups


def _autocast_disabled(device: torch.device) -> contextlib.AbstractContextManager[object]:
    if device.type in ("cpu", "cuda"):
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def _activate_intermediate(
    values: torch.Tensor,
    activation_fn: str,
    *,
    gate_first: bool,
) -> torch.Tensor:
    activated_fn = parse_input_activation(activation_fn, name="MoE")
    if activated_fn == "swiglu":
        first, second = values.float().chunk(2, dim=-1)
        if gate_first:
            return torch.nn.functional.silu(first) * second
        return first * torch.nn.functional.silu(second)
    return apply_input_activation(values, activated_fn)


def _resolve_slot_tensor(
    expert_id: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_i64 = expert_id.to(torch.int64)
    if expert_to_slot is None:
        slot = expert_i64
        in_expert_range = torch.ones((), device=expert_i64.device, dtype=torch.bool)
    else:
        num_experts = expert_to_slot.shape[0]
        in_expert_range = (expert_i64 >= 0) & (expert_i64 < num_experts)
        safe_expert = torch.where(in_expert_range, expert_i64, 0).unsqueeze(0)
        slot = torch.index_select(expert_to_slot, 0, safe_expert).squeeze(0).to(torch.int64)
    is_hit = in_expert_range & (slot >= 0) & (slot < num_slots)
    safe_slot_1d = torch.where(is_hit, slot, 0).unsqueeze(0)
    return safe_slot_1d, is_hit


def _prepare_nvfp4_reference_rows(
    input_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    if prepared_data is not None:
        return prepared_data.reshape(input_2d.shape)
    if group_size > 0:
        return rotate_groups(input_2d.float(), group_size)
    return input_2d


def _decode_trellis_slot(
    qdata: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor | None,
    slot_offsets: torch.Tensor | None,
    safe_slot_1d: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    slot_block_scales = torch.index_select(block_scales, 0, safe_slot_1d).squeeze(0)
    if row_widths is None:
        layout = TrellisLayout.from_storage(block_size, qdata.shape[3])
        slot_qdata = torch.index_select(qdata, 0, safe_slot_1d).squeeze(0)
        return decode_reference(slot_qdata, slot_block_scales, layout, round_int8=False).reshape(
            slot_block_scales.shape[0], -1
        )
    slot_widths = torch.index_select(row_widths, 0, safe_slot_1d).squeeze(0)
    if slot_offsets is None:
        stream = torch.index_select(qdata, 0, safe_slot_1d).squeeze(0)
        base_offset = None
    else:
        stream = qdata
        base_offset = torch.index_select(slot_offsets, 0, safe_slot_1d).squeeze(0)
    return decode_variable_rows_reference(
        stream,
        slot_block_scales,
        slot_widths,
        block_size,
        base_offset=base_offset,
        round_int8=False,
    )


def _make_slot_evaluator(
    input_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    *,
    logical_dtype: torch.dtype,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    up_gate_block_scales: torch.Tensor | None,
    down_block_scales: torch.Tensor | None,
    up_gate_row_widths: torch.Tensor | None,
    down_row_widths: torch.Tensor | None,
    up_gate_slot_offsets: torch.Tensor | None,
    down_slot_offsets: torch.Tensor | None,
    up_gate_block_size: int,
    down_block_size: int,
) -> Callable[[int, torch.Tensor], torch.Tensor]:
    rows, hidden_size = input_2d.shape
    if up_gate_block_scales is not None:
        assert down_block_scales is not None
        active_2d = _prepare_nvfp4_reference_rows(input_2d, prepared_data, group_size)

        def eval_trellis(row_index: int, safe_slot_1d: torch.Tensor) -> torch.Tensor:
            up_q = _decode_trellis_slot(
                up_gate_qdata,
                up_gate_block_scales,
                up_gate_row_widths,
                up_gate_slot_offsets,
                safe_slot_1d,
                up_gate_block_size,
            )
            up_s = torch.index_select(up_gate_scale, 0, safe_slot_1d).squeeze(0)
            up_w = up_q.float() * up_s.unsqueeze(1)
            mid = active_2d[row_index : row_index + 1].float() @ up_w.t()
            activated = _activate_intermediate(mid, activation_fn, gate_first=gate_first).to(
                logical_dtype
            )
            act_rot = rotate_groups(activated.float(), group_size)
            down_q = _decode_trellis_slot(
                down_qdata,
                down_block_scales,
                down_row_widths,
                down_slot_offsets,
                safe_slot_1d,
                down_block_size,
            )
            down_s = torch.index_select(down_scale, 0, safe_slot_1d).squeeze(0)
            down_w = down_q.float() * down_s.unsqueeze(1)
            return act_rot @ down_w.t()

        return eval_trellis

    if up_gate_qdata.dtype is torch.int8:
        from piper_kernels.linear.convrot.int8 import reference as int8_ref  # noqa: PLC0415

        if prepared_data is None:
            input_q_2d, input_s_1d = int8_ref.prepare_input(input_2d, group_size)
        else:
            if prepared_scale is None:
                raise ValueError("ConvRot INT8 prepared_data requires prepared_scale")
            input_q_2d = prepared_data.reshape(rows, hidden_size)
            input_s_1d = prepared_scale.reshape(rows)

        def eval_int8(row_index: int, safe_slot_1d: torch.Tensor) -> torch.Tensor:
            up_q = torch.index_select(up_gate_qdata, 0, safe_slot_1d).squeeze(0)
            up_s = torch.index_select(up_gate_scale, 0, safe_slot_1d).squeeze(0)
            down_q_w = torch.index_select(down_qdata, 0, safe_slot_1d).squeeze(0)
            down_s_w = torch.index_select(down_scale, 0, safe_slot_1d).squeeze(0)
            mid = int8_ref.linear_prepared(
                input_q_2d[row_index : row_index + 1],
                input_s_1d[row_index : row_index + 1],
                up_q,
                up_s,
                logical_dtype,
            )
            activated = _activate_intermediate(mid, activation_fn, gate_first=gate_first)
            down_q, down_s = int8_ref.prepare_input(activated, group_size)
            return int8_ref.linear_prepared(
                down_q,
                down_s,
                down_q_w,
                down_s_w,
                torch.float32,
            )

        return eval_int8

    from piper_kernels.linear.nvfp4 import reference as nvfp4_ref  # noqa: PLC0415

    active_2d = _prepare_nvfp4_reference_rows(input_2d, prepared_data, group_size)

    def eval_nvfp4(row_index: int, safe_slot_1d: torch.Tensor) -> torch.Tensor:
        up_q = torch.index_select(up_gate_qdata, 0, safe_slot_1d).squeeze(0)
        up_s = torch.index_select(up_gate_scale, 0, safe_slot_1d).squeeze(0)
        down_q = torch.index_select(down_qdata, 0, safe_slot_1d).squeeze(0)
        down_s = torch.index_select(down_scale, 0, safe_slot_1d).squeeze(0)
        up_gs = (
            None
            if up_gate_global_scale is None
            else torch.index_select(up_gate_global_scale, 0, safe_slot_1d).squeeze(0)
        )
        down_gs = (
            None
            if down_global_scale is None
            else torch.index_select(down_global_scale, 0, safe_slot_1d).squeeze(0)
        )
        mid = nvfp4_ref.linear_w4a16(
            active_2d[row_index : row_index + 1].float(),
            up_q,
            up_s,
            up_gs,
            None,
            high_first=high_first,
            group_size=0,
        )
        activated = _activate_intermediate(mid, activation_fn, gate_first=gate_first).to(
            logical_dtype
        )
        return nvfp4_ref.linear_w4a16(
            activated.float(),
            down_q,
            down_s,
            down_gs,
            None,
            high_first=high_first,
            group_size=group_size,
        )

    return eval_nvfp4


def moe_ffn_reference(
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
    accumulate: bool = False,
    group_size: int = 0,
    activation_fn: str = "swiglu",
    gate_first: bool = True,
    high_first: bool = False,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_size: int = 0,
) -> torch.Tensor:
    """Run an independent PyTorch reference across routed expert slots."""
    parse_input_activation(activation_fn, name="MoE")
    rows = math.prod(input.shape[:-1])
    hidden_size = input.shape[-1]
    top_k = topk_ids.shape[-1]
    num_slots = down_block_scales.shape[0] if down_block_scales is not None else down_qdata.shape[0]
    result = torch.zeros_like(input, memory_format=torch.contiguous_format) if out is None else out
    out_2d = result.view(rows, hidden_size) if rows else result.reshape(0, hidden_size)

    num_experts = num_slots if expert_to_slot is None else expert_to_slot.shape[0]
    if rows == 0 or top_k == 0 or num_slots == 0 or num_experts == 0:
        if out is not None and not accumulate:
            out.zero_()
        return result

    with _autocast_disabled(input.device):
        eval_slot = _make_slot_evaluator(
            input.reshape(rows, hidden_size),
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            logical_dtype=input.dtype,
            up_gate_global_scale=up_gate_global_scale,
            down_global_scale=down_global_scale,
            prepared_data=prepared_data,
            prepared_scale=prepared_scale,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            up_gate_block_scales=up_gate_block_scales,
            down_block_scales=down_block_scales,
            up_gate_row_widths=up_gate_row_widths,
            down_row_widths=down_row_widths,
            up_gate_slot_offsets=up_gate_slot_offsets,
            down_slot_offsets=down_slot_offsets,
            up_gate_block_size=up_gate_block_size,
            down_block_size=down_block_size,
        )
        topk_ids_2d = topk_ids.reshape(rows, top_k)
        topk_weights_2d = topk_weights.reshape(rows, top_k)

        for row_index in range(rows):
            accumulated = torch.zeros((hidden_size,), device=input.device, dtype=torch.float32)
            had_hit = torch.zeros((), device=input.device, dtype=torch.bool)
            for k_index in range(top_k):
                safe_slot_1d, is_hit = _resolve_slot_tensor(
                    topk_ids_2d[row_index, k_index], expert_to_slot, num_slots
                )
                if input.device.type == "cpu" and not bool(is_hit):
                    continue
                expert_out = eval_slot(row_index, safe_slot_1d)
                router_weight = topk_weights_2d[row_index, k_index].float()
                accumulated = accumulated + torch.where(
                    is_hit,
                    expert_out[0].float() * router_weight,
                    torch.zeros_like(accumulated),
                )
                had_hit = had_hit | is_hit

            if accumulate:
                updated = (out_2d[row_index].float() + accumulated).to(input.dtype)
                out_2d[row_index].copy_(torch.where(had_hit, updated, out_2d[row_index]))
            else:
                out_2d[row_index].copy_(accumulated.to(input.dtype))
    return result


__all__ = ["moe_ffn_reference"]
