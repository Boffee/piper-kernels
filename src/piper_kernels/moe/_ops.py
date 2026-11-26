"""Stable MoE FFN custom-op schemas and runtime accelerator dispatch."""

# Explicit custom-op schemas take flat tensor arguments required by torch.library.
# ruff: noqa: PLR0912, PLR0913, PLR0915, PLR0917

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor

from piper_kernels._input_activations import parse_input_activation
from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.trellis._layout import (
    MAX_BITS_PER_WEIGHT,
    MIN_BITS_PER_WEIGHT,
    TrellisLayout,
    validate_block_size,
)
from piper_kernels.weights.convrot._rotation import validate_group_size

from . import reference

type _MoEWeightKind = Literal["nvfp4", "convrot_int8", "convrot_trellis"]

_SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Largest K avoiding signed int32 wrap on 128 x 128 worst-case int8 products.
_INT8_MAX_REDUCTION_WIDTH = (2**31 - 1) // (128 * 128)
_PREVALIDATED_STATE: ContextVar[
    tuple[tuple[object, ...], tuple[_MoEWeightKind, int, int]] | None
] = ContextVar("piper_moe_prevalidated_state", default=None)


@dataclass(frozen=True, slots=True)
class _MoEOperands:
    input: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    up_gate_qdata: torch.Tensor
    up_gate_scale: torch.Tensor
    down_qdata: torch.Tensor
    down_scale: torch.Tensor
    up_gate_global_scale: torch.Tensor | None = None
    down_global_scale: torch.Tensor | None = None
    expert_to_slot: torch.Tensor | None = None
    prepared_data: torch.Tensor | None = None
    prepared_scale: torch.Tensor | None = None
    out: torch.Tensor | None = None
    accumulate: bool = False
    group_size: int = 0
    activation_fn: str = "swiglu"
    gate_first: bool = True
    high_first: bool = False
    active_slots_hint: int = 0
    up_gate_block_scales: torch.Tensor | None = None
    down_block_scales: torch.Tensor | None = None
    up_gate_row_widths: torch.Tensor | None = None
    down_row_widths: torch.Tensor | None = None
    up_gate_slot_offsets: torch.Tensor | None = None
    down_slot_offsets: torch.Tensor | None = None
    up_gate_block_size: int = 0
    down_block_size: int = 0

    def weight_tensors(self) -> tuple[torch.Tensor | None, ...]:
        return (
            self.up_gate_qdata,
            self.up_gate_scale,
            self.down_qdata,
            self.down_scale,
            self.up_gate_global_scale,
            self.down_global_scale,
            self.up_gate_block_scales,
            self.down_block_scales,
            self.up_gate_row_widths,
            self.down_row_widths,
            self.up_gate_slot_offsets,
            self.down_slot_offsets,
        )

    def op_args(self, *, inplace: bool) -> tuple[Any, ...]:
        head: tuple[Any, ...] = (
            self.input,
            self.topk_ids,
            self.topk_weights,
            self.up_gate_qdata,
            self.up_gate_scale,
            self.down_qdata,
            self.down_scale,
        )
        mid: tuple[Any, ...] = (
            self.up_gate_global_scale,
            self.down_global_scale,
            self.expert_to_slot,
            self.prepared_data,
            self.prepared_scale,
        )
        tail: tuple[Any, ...] = (
            self.group_size,
            self.activation_fn,
            self.gate_first,
            self.high_first,
            self.active_slots_hint,
            self.up_gate_block_scales,
            self.down_block_scales,
            self.up_gate_row_widths,
            self.down_row_widths,
            self.up_gate_slot_offsets,
            self.down_slot_offsets,
            self.up_gate_block_size,
            self.down_block_size,
        )
        if inplace:
            assert self.out is not None
            return (*head, self.out, *mid, self.accumulate, *tail)
        return (*head, *mid, *tail)

    def validation_key(self) -> tuple[object, ...]:
        return (
            *(
                None if tensor is None else id(tensor)
                for tensor in (
                    self.input,
                    self.topk_ids,
                    self.topk_weights,
                    self.expert_to_slot,
                    self.prepared_data,
                    self.prepared_scale,
                    self.out,
                    *self.weight_tensors(),
                )
            ),
            self.accumulate,
            self.group_size,
            self.activation_fn,
            self.gate_first,
            self.high_first,
            self.active_slots_hint,
            self.up_gate_block_size,
            self.down_block_size,
        )


def _require_strided_contiguous(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.layout is not torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous per expert slot")
    if tensor.is_neg() or tensor.is_conj():
        raise ValueError(f"{name} must not use negated or conjugated views")


def _validate_global_scale(
    scale: torch.Tensor | None,
    *,
    name: str,
    num_slots: int,
    device: torch.device,
) -> None:
    if scale is None:
        return
    if scale.dtype is not torch.float32 or tuple(scale.shape) != (num_slots,):
        raise ValueError(
            f"{name} must be float32 with shape {(num_slots,)}, "
            f"got {scale.dtype} {tuple(scale.shape)}"
        )
    _require_strided_contiguous(scale, name=name, device=device)


def _validate_trellis_projection(
    *,
    name: str,
    qdata: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    slot_offsets: torch.Tensor | None,
    block_size: int | torch.SymInt,
    num_slots: int,
    out_features: int,
    in_features: int,
    device: torch.device,
) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, (int, torch.SymInt)):
        raise TypeError(f"MoE {name}_block_size must be an integer, got {block_size!r}")
    resolved_block_size = int(block_size) if isinstance(block_size, torch.SymInt) else block_size
    validate_block_size(resolved_block_size)
    if in_features % resolved_block_size:
        raise ValueError(
            f"ConvRot Trellis MoE {name} width {in_features} must be divisible by "
            f"block_size {resolved_block_size}"
        )
    expected_blocks = in_features // resolved_block_size

    _require_strided_contiguous(qdata, name=f"{name}_qdata", device=device)
    _require_strided_contiguous(block_scales, name=f"{name}_block_scales", device=device)
    _require_strided_contiguous(scale, name=f"{name}_scale", device=device)

    if qdata.dtype is not torch.uint8:
        raise ValueError(f"ConvRot Trellis MoE {name}_qdata must be uint8, got {qdata.dtype}")
    if block_scales.dtype is not torch.float8_e4m3fn or tuple(block_scales.shape) != (
        num_slots,
        out_features,
        expected_blocks,
    ):
        raise ValueError(
            f"ConvRot Trellis MoE {name}_block_scales must be float8_e4m3fn with shape "
            f"{(num_slots, out_features, expected_blocks)}, "
            f"got {block_scales.dtype} {tuple(block_scales.shape)}"
        )
    if scale.dtype is not torch.float32 or tuple(scale.shape) != (num_slots, out_features):
        raise ValueError(
            f"ConvRot Trellis MoE {name}_scale must be float32 with shape "
            f"{(num_slots, out_features)}, got {scale.dtype} {tuple(scale.shape)}"
        )

    if row_widths is None:
        if slot_offsets is not None:
            raise ValueError(f"ConvRot Trellis MoE {name}_slot_offsets requires {name}_row_widths")
        if qdata.ndim != 4 or tuple(qdata.shape[:3]) != (num_slots, out_features, expected_blocks):
            raise ValueError(
                f"ConvRot Trellis MoE uniform {name}_qdata must have shape "
                f"{(num_slots, out_features, expected_blocks, 'bytes')}, got {tuple(qdata.shape)}"
            )
        bytes_per_block = qdata.shape[3]
        if type(bytes_per_block) is int:
            TrellisLayout.from_storage(resolved_block_size, bytes_per_block)
        else:
            lowest = MIN_BITS_PER_WEIGHT * resolved_block_size
            highest = MAX_BITS_PER_WEIGHT * resolved_block_size
            if 8 * bytes_per_block < lowest or 8 * bytes_per_block > highest:
                raise ValueError(
                    f"ConvRot Trellis blocks of {resolved_block_size} weights take "
                    f"{(lowest + 7) // 8} to {highest // 8} bytes, got {8 * bytes_per_block} bits"
                )
        return resolved_block_size

    _require_strided_contiguous(row_widths, name=f"{name}_row_widths", device=device)
    if row_widths.dtype is not torch.uint8 or tuple(row_widths.shape) != (num_slots, out_features):
        raise ValueError(
            f"ConvRot Trellis MoE {name}_row_widths must be uint8 with shape "
            f"{(num_slots, out_features)}, got {row_widths.dtype} {tuple(row_widths.shape)}"
        )
    if slot_offsets is None:
        if qdata.ndim != 2 or qdata.shape[0] != num_slots:
            raise ValueError(
                f"ConvRot Trellis MoE 2-D variable-row {name}_qdata must have shape "
                f"{(num_slots, 'slot_bytes')}, got {tuple(qdata.shape)}"
            )
        return resolved_block_size

    _require_strided_contiguous(slot_offsets, name=f"{name}_slot_offsets", device=device)
    if slot_offsets.dtype is not torch.int64 or tuple(slot_offsets.shape) != (num_slots,):
        raise ValueError(
            f"ConvRot Trellis MoE {name}_slot_offsets must be int64 with shape {(num_slots,)}, "
            f"got {slot_offsets.dtype} {tuple(slot_offsets.shape)}"
        )
    if qdata.ndim != 1:
        raise ValueError(
            f"ConvRot Trellis MoE 1-D variable-row {name}_qdata must be 1-D [total_bytes], "
            f"got {tuple(qdata.shape)}"
        )
    return resolved_block_size


class _WeightStorage(Protocol):
    @property
    def up_gate_qdata(self) -> torch.Tensor: ...
    @property
    def up_gate_scale(self) -> torch.Tensor: ...
    @property
    def down_qdata(self) -> torch.Tensor: ...
    @property
    def down_scale(self) -> torch.Tensor: ...
    @property
    def up_gate_global_scale(self) -> torch.Tensor | None: ...
    @property
    def down_global_scale(self) -> torch.Tensor | None: ...
    @property
    def group_size(self) -> int: ...
    @property
    def activation_fn(self) -> str: ...
    @property
    def gate_first(self) -> bool: ...
    @property
    def high_first(self) -> bool: ...
    @property
    def up_gate_block_scales(self) -> torch.Tensor | None: ...
    @property
    def down_block_scales(self) -> torch.Tensor | None: ...
    @property
    def up_gate_row_widths(self) -> torch.Tensor | None: ...
    @property
    def down_row_widths(self) -> torch.Tensor | None: ...
    @property
    def up_gate_slot_offsets(self) -> torch.Tensor | None: ...
    @property
    def down_slot_offsets(self) -> torch.Tensor | None: ...
    @property
    def up_gate_block_size(self) -> int: ...
    @property
    def down_block_size(self) -> int: ...


def _validate_weight_storage(
    ops: _WeightStorage,
) -> tuple[_MoEWeightKind, int, int, int, torch.device]:
    parse_input_activation(ops.activation_fn, name="MoE")
    if not isinstance(ops.gate_first, bool) or not isinstance(ops.high_first, bool):
        raise TypeError("MoE gate_first and high_first must be booleans")
    group_size = ops.group_size
    if isinstance(group_size, bool) or not isinstance(group_size, (int, torch.SymInt)):
        raise TypeError(f"MoE group_size must be an integer, got {group_size!r}")
    if isinstance(group_size, torch.SymInt):
        group_size = int(group_size)

    device = ops.up_gate_qdata.device
    has_trellis_arg = (
        any(
            tensor is not None
            for tensor in (
                ops.up_gate_block_scales,
                ops.down_block_scales,
                ops.up_gate_row_widths,
                ops.down_row_widths,
                ops.up_gate_slot_offsets,
                ops.down_slot_offsets,
            )
        )
        or (ops.up_gate_block_size != 0)
        or (ops.down_block_size != 0)
    )
    if has_trellis_arg:
        if ops.high_first:
            raise ValueError("ConvRot Trellis MoE weights do not support high_first nibble packing")
        validate_group_size(group_size)
        if ops.up_gate_global_scale is not None or ops.down_global_scale is not None:
            raise ValueError("ConvRot Trellis MoE weights do not use per-expert global scales")
        if (
            ops.up_gate_block_scales is None
            or ops.down_block_scales is None
            or ops.up_gate_block_scales.ndim != 3
            or ops.down_block_scales.ndim != 3
        ):
            raise ValueError(
                "ConvRot Trellis MoE weights require 3-D up_gate_block_scales and down_block_scales"
            )
        if ops.up_gate_block_scales.shape[0] != ops.down_block_scales.shape[0]:
            raise ValueError("MoE up_gate and down tensors must have matching slot counts")
        if isinstance(ops.down_block_size, bool) or not isinstance(
            ops.down_block_size, (int, torch.SymInt)
        ):
            raise TypeError(f"MoE down_block_size must be an integer, got {ops.down_block_size!r}")
        resolved_down_bs = (
            int(ops.down_block_size)
            if isinstance(ops.down_block_size, torch.SymInt)
            else ops.down_block_size
        )
        validate_block_size(resolved_down_bs)
        num_slots, hidden_size, down_blocks = ops.down_block_scales.shape
        intermediate_size = down_blocks * resolved_down_bs
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("MoE hidden_size and intermediate_size must be positive")
        if hidden_size % group_size or intermediate_size % group_size:
            raise ValueError(
                f"ConvRot Trellis MoE dimensions must be divisible by group_size {group_size}"
            )
        if hidden_size > _INT8_MAX_REDUCTION_WIDTH or intermediate_size > _INT8_MAX_REDUCTION_WIDTH:
            raise ValueError(
                f"ConvRot Trellis MoE widths must not exceed {_INT8_MAX_REDUCTION_WIDTH} "
                f"for INT32 accumulation, got hidden_size={hidden_size}, "
                f"intermediate_size={intermediate_size}"
            )
        up_gate_out = intermediate_size * (2 if ops.activation_fn == "swiglu" else 1)
        _validate_trellis_projection(
            name="up_gate",
            qdata=ops.up_gate_qdata,
            block_scales=ops.up_gate_block_scales,
            scale=ops.up_gate_scale,
            row_widths=ops.up_gate_row_widths,
            slot_offsets=ops.up_gate_slot_offsets,
            block_size=ops.up_gate_block_size,
            num_slots=num_slots,
            out_features=up_gate_out,
            in_features=hidden_size,
            device=device,
        )
        _validate_trellis_projection(
            name="down",
            qdata=ops.down_qdata,
            block_scales=ops.down_block_scales,
            scale=ops.down_scale,
            row_widths=ops.down_row_widths,
            slot_offsets=ops.down_slot_offsets,
            block_size=resolved_down_bs,
            num_slots=num_slots,
            out_features=hidden_size,
            in_features=intermediate_size,
            device=device,
        )
        return "convrot_trellis", num_slots, hidden_size, intermediate_size, device

    if ops.up_gate_qdata.ndim != 3 or ops.down_qdata.ndim != 3:
        raise ValueError("MoE qdata tensors must be 3-D [num_slots, out, in]")
    if ops.up_gate_qdata.shape[0] != ops.down_qdata.shape[0]:
        raise ValueError("MoE up_gate and down tensors must have matching slot counts")

    for tensor, name in (
        (ops.up_gate_qdata, "up_gate_qdata"),
        (ops.up_gate_scale, "up_gate_scale"),
        (ops.down_qdata, "down_qdata"),
        (ops.down_scale, "down_scale"),
    ):
        _require_strided_contiguous(tensor, name=name, device=device)

    if ops.up_gate_qdata.dtype is torch.uint8:
        from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout  # noqa: PLC0415

        num_slots, hidden_size, packed_intermediate = ops.down_qdata.shape
        intermediate_size = 2 * packed_intermediate
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("MoE hidden_size and intermediate_size must be positive")
        if hidden_size % nvfp4_layout.BLOCK_SIZE or intermediate_size % nvfp4_layout.BLOCK_SIZE:
            raise ValueError("NVFP4 MoE hidden_size and intermediate_size must be multiples of 16")
        if group_size != 0:
            validate_group_size(group_size)
            if hidden_size % group_size or intermediate_size % group_size:
                raise ValueError(
                    f"ConvRot NVFP4 MoE dimensions must be divisible by group_size {group_size}"
                )
        up_gate_out = intermediate_size * (2 if ops.activation_fn == "swiglu" else 1)
        expected_up_scale = (num_slots, *nvfp4_layout.scale_shape(up_gate_out, hidden_size))
        expected_down_scale = (
            num_slots,
            *nvfp4_layout.scale_shape(hidden_size, intermediate_size),
        )
        if (
            ops.down_qdata.dtype is not torch.uint8
            or ops.up_gate_scale.dtype is not torch.float8_e4m3fn
            or ops.down_scale.dtype is not torch.float8_e4m3fn
            or tuple(ops.up_gate_qdata.shape) != (num_slots, up_gate_out, hidden_size // 2)
            or tuple(ops.up_gate_scale.shape) != expected_up_scale
            or tuple(ops.down_scale.shape) != expected_down_scale
        ):
            raise ValueError("NVFP4 MoE weight storage shapes or dtypes are invalid")
        _validate_global_scale(
            ops.up_gate_global_scale,
            name="up_gate_global_scale",
            num_slots=num_slots,
            device=device,
        )
        _validate_global_scale(
            ops.down_global_scale,
            name="down_global_scale",
            num_slots=num_slots,
            device=device,
        )
        return "nvfp4", num_slots, hidden_size, intermediate_size, device

    if ops.up_gate_qdata.dtype is torch.int8:
        if ops.high_first:
            raise ValueError("ConvRot INT8 MoE weights do not support high_first nibble packing")
        validate_group_size(group_size)
        if ops.up_gate_global_scale is not None or ops.down_global_scale is not None:
            raise ValueError("ConvRot INT8 MoE weights do not use per-expert global scales")
        num_slots, hidden_size, intermediate_size = ops.down_qdata.shape
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("MoE hidden_size and intermediate_size must be positive")
        if hidden_size % group_size or intermediate_size % group_size:
            raise ValueError(
                f"ConvRot INT8 MoE dimensions must be divisible by group_size {group_size}"
            )
        if hidden_size > _INT8_MAX_REDUCTION_WIDTH or intermediate_size > _INT8_MAX_REDUCTION_WIDTH:
            raise ValueError(
                f"ConvRot INT8 MoE widths must not exceed {_INT8_MAX_REDUCTION_WIDTH} "
                f"for INT32 accumulation, got hidden_size={hidden_size}, "
                f"intermediate_size={intermediate_size}"
            )
        up_gate_out = intermediate_size * (2 if ops.activation_fn == "swiglu" else 1)
        if (
            ops.down_qdata.dtype is not torch.int8
            or ops.up_gate_scale.dtype is not torch.float32
            or ops.down_scale.dtype is not torch.float32
            or tuple(ops.up_gate_qdata.shape) != (num_slots, up_gate_out, hidden_size)
            or tuple(ops.up_gate_scale.shape) != (num_slots, up_gate_out)
            or tuple(ops.down_scale.shape) != (num_slots, hidden_size)
        ):
            raise ValueError("ConvRot INT8 MoE weight storage shapes or dtypes are invalid")
        return "convrot_int8", num_slots, hidden_size, intermediate_size, device

    raise ValueError(
        f"MoE up_gate_qdata must be uint8 (NVFP4/Trellis) or int8 (ConvRot INT8), "
        f"got {ops.up_gate_qdata.dtype}"
    )


def _validate_op_operands(ops: _MoEOperands) -> tuple[_MoEWeightKind, int, int]:
    if not isinstance(ops.accumulate, bool):
        raise TypeError("MoE accumulate must be a bool")
    if isinstance(ops.active_slots_hint, bool) or not isinstance(
        ops.active_slots_hint, (int, torch.SymInt)
    ):
        raise TypeError("MoE active_slots_hint must be an integer")
    if ops.active_slots_hint < 0:
        raise ValueError("MoE active_slots_hint must be non-negative")

    weight_format, num_slots, hidden_size, _inter, weight_device = _validate_weight_storage(ops)
    inp = ops.input
    if inp.ndim < 2 or inp.shape[-1] != hidden_size:
        raise ValueError(f"MoE input must have shape [..., {hidden_size}], got {tuple(inp.shape)}")
    if inp.dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(f"MoE input must be fp16, bf16, or fp32, got {inp.dtype}")
    if inp.device != weight_device:
        raise ValueError(f"MoE input device {inp.device} must match weights on {weight_device}")
    if inp.layout is not torch.strided:
        raise ValueError("MoE input must use strided layout")
    rows = math.prod(inp.shape[:-1])

    if ops.topk_ids.shape[:-1] != inp.shape[:-1] or ops.topk_weights.shape != ops.topk_ids.shape:
        raise ValueError(
            "MoE topk_ids and topk_weights must share leading shape "
            f"{tuple(inp.shape[:-1])} and matching top_k"
        )
    if ops.topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"MoE topk_ids must be int32 or int64, got {ops.topk_ids.dtype}")
    if ops.topk_weights.dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            f"MoE topk_weights must be fp16, bf16, or fp32, got {ops.topk_weights.dtype}"
        )
    if ops.topk_ids.device != weight_device or ops.topk_weights.device != weight_device:
        raise ValueError("MoE routing tensors must share the weight device")
    if ops.topk_ids.layout is not torch.strided or ops.topk_weights.layout is not torch.strided:
        raise ValueError("MoE routing tensors must use strided layout")

    if ops.expert_to_slot is not None:
        if ops.expert_to_slot.ndim != 1 or ops.expert_to_slot.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("MoE expert_to_slot must be 1-D int32 or int64")
        if (
            ops.expert_to_slot.device != weight_device
            or ops.expert_to_slot.layout is not torch.strided
        ):
            raise ValueError("MoE expert_to_slot must be strided on the weight device")

    if ops.prepared_data is None:
        if ops.prepared_scale is not None:
            raise ValueError("MoE prepared_input scale must be None when prepared_data is None")
    else:
        _require_strided_contiguous(
            ops.prepared_data, name="MoE prepared_input data", device=weight_device
        )
        if tuple(ops.prepared_data.shape) != (rows, hidden_size):
            raise ValueError("MoE prepared_input does not match input shape")
        if weight_format == "convrot_int8":
            if ops.prepared_data.dtype is not torch.int8:
                raise ValueError("ConvRot INT8 prepared_input data must be int8")
            if ops.prepared_scale is None or ops.prepared_scale.dtype is not torch.float32:
                raise ValueError(
                    "ConvRot INT8 prepared_input requires contiguous [rows] float32 scale"
                )
            _require_strided_contiguous(
                ops.prepared_scale, name="MoE prepared_input scale", device=weight_device
            )
            if tuple(ops.prepared_scale.shape) != (rows,):
                raise ValueError(
                    "ConvRot INT8 prepared_input requires contiguous [rows] float32 scale"
                )
        else:
            if ops.prepared_scale is not None:
                raise ValueError("MoE prepared_input scale must be None for rotated A16 weights")
            expected_prepared_dtype = torch.float32 if ops.group_size > 0 else inp.dtype
            if ops.prepared_data.dtype is not expected_prepared_dtype:
                raise ValueError(
                    f"MoE prepared_input data must be {expected_prepared_dtype}, "
                    f"got {ops.prepared_data.dtype}"
                )

    out = ops.out
    if out is not None:
        if (
            out.shape != inp.shape
            or out.dtype is not inp.dtype
            or out.device != weight_device
            or out.layout is not torch.strided
            or out.stride(-1) != 1
        ):
            raise ValueError(
                "MoE out tensor must match input shape/dtype/device with stride(-1)==1"
            )
        if out.is_neg() or out.is_conj():
            raise ValueError("MoE out tensor must not use negated or conjugated views")
        if rows > 0:
            try:
                out_2d = out.view(rows, hidden_size)
            except RuntimeError as error:
                raise ValueError(
                    "MoE out tensor leading dimensions must be viewable as 2-D [rows, H]"
                ) from error
            if rows > 1 and out_2d.stride(0) < hidden_size:
                raise ValueError("MoE out tensor rows must not overlap in storage")
        if out.is_inference() and not torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Inplace update to inference tensor outside InferenceMode is not allowed."
            )

    return weight_format, num_slots, rows


def _reject_autograd(ops: _MoEOperands) -> None:
    if not torch.is_grad_enabled():
        return
    if ops.input.requires_grad:
        raise RuntimeError("MoE FFN is inference-only and does not support gradients on input")
    if ops.topk_weights.requires_grad:
        raise RuntimeError("MoE FFN does not support gradients on topk_weights")
    if any(
        tensor is not None and tensor.requires_grad
        for tensor in (
            ops.up_gate_scale,
            ops.down_scale,
            ops.up_gate_global_scale,
            ops.down_global_scale,
            ops.up_gate_block_scales,
            ops.down_block_scales,
        )
    ):
        raise RuntimeError("MoE weights do not support autograd")
    if (ops.prepared_data is not None and ops.prepared_data.requires_grad) or (
        ops.prepared_scale is not None and ops.prepared_scale.requires_grad
    ):
        raise RuntimeError("MoE prepared_input does not support autograd")
    if ops.out is not None and ops.out.requires_grad:
        raise RuntimeError("MoE out tensor must not require gradients")


def _concrete_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _tensor_span_elements(tensor: torch.Tensor) -> int | None:
    span = 0
    for size_raw, stride_raw in zip(tensor.shape, tensor.stride(), strict=True):
        size = _concrete_int(size_raw)
        stride = _concrete_int(stride_raw)
        if size is None or stride is None:
            return None
        span += (size - 1) * stride
    return span


def _storage_byte_range(tensor: torch.Tensor) -> tuple[int, int] | None:
    numel = _concrete_int(tensor.numel())
    storage_offset = _concrete_int(tensor.storage_offset())
    span = _tensor_span_elements(tensor)
    if numel in (None, 0) or storage_offset is None or span is None:
        return None
    elem_size = tensor.element_size()
    start = storage_offset * elem_size
    end = start + (span + 1) * elem_size
    if isinstance(tensor, FakeTensor):
        return start, end
    if tensor.device.type == "meta":
        return None
    storage_ptr = tensor.untyped_storage().data_ptr()
    return None if storage_ptr == 0 else (storage_ptr + start, storage_ptr + end)


def _tensors_overlap(first: torch.Tensor, second: torch.Tensor) -> bool:
    if first.device != second.device:
        return False
    is_fake = isinstance(first, FakeTensor) or isinstance(second, FakeTensor)
    if is_fake and first.untyped_storage() is not second.untyped_storage():
        return False
    first_range = _storage_byte_range(first)
    second_range = _storage_byte_range(second)
    if first_range is None or second_range is None:
        if is_fake and first.untyped_storage() is second.untyped_storage():
            return _concrete_int(first.numel()) != 0 and _concrete_int(second.numel()) != 0
        return False
    return first_range[0] < second_range[1] and second_range[0] < first_range[1]


def _reject_weight_aliasing(ops: _MoEOperands) -> None:
    out = ops.out
    assert out is not None
    for weight_tensor in ops.weight_tensors():
        if weight_tensor is not None and _tensors_overlap(out, weight_tensor):
            raise ValueError("MoE out tensor must not alias weight storage")


def _validate_before_mutating(ops: _MoEOperands) -> tuple[_MoEWeightKind, int, int]:
    validated = _validate_op_operands(ops)
    if ops.input.device.type != "meta":
        _reject_weight_aliasing(ops)
    return validated


@contextmanager
def _prevalidated_inplace(ops: _MoEOperands) -> Iterator[None]:
    validated = _validate_before_mutating(ops)
    token = _PREVALIDATED_STATE.set((ops.validation_key(), validated))
    try:
        yield
    finally:
        _PREVALIDATED_STATE.reset(token)


def _consume_prevalidated(ops: _MoEOperands) -> tuple[_MoEWeightKind, int, int] | None:
    entry = _PREVALIDATED_STATE.get()
    if entry is None:
        return None
    key, validated = entry
    if key != ops.validation_key():
        return None
    _PREVALIDATED_STATE.set(None)
    return validated


def _resolve_gpu_target(
    device: torch.device,
    weight_format: _MoEWeightKind,
) -> AcceleratorTarget | None:
    if device.type != "cuda" or not runtime.supports_device(device):
        return None
    target = AcceleratorTarget.from_device(device)
    if weight_format == "convrot_int8":
        if target.is_nvidia_cuda and not target.cuda_capability_at_least(8):
            return None
        from piper_kernels.linear.convrot.int8 import _backend as int8_backend  # noqa: PLC0415

        return target if int8_backend.select_linear_backend_for_target(target) is not None else None
    return target if target.supports_nvfp4_w4a16 else None


def _is_eager_execution(ops: _MoEOperands) -> bool:
    tensors = (ops.input,) if ops.out is None else (ops.input, ops.out)
    return not torch.compiler.is_compiling() and detect_fake_mode(tensors) is None


def _run_moe_ffn(ops: _MoEOperands) -> torch.Tensor:
    is_eager = _is_eager_execution(ops)
    is_concrete = is_eager and ops.input.device.type != "meta"
    prevalidated = _consume_prevalidated(ops) if (ops.out is not None and is_eager) else None
    if prevalidated is not None:
        weight_format, num_slots, rows = prevalidated
    else:
        weight_format, num_slots, rows = _validate_op_operands(ops)
        if ops.out is not None and is_concrete:
            _reject_weight_aliasing(ops)

    out = ops.out
    top_k = ops.topk_ids.shape[-1]
    num_experts = num_slots if ops.expert_to_slot is None else ops.expert_to_slot.shape[0]
    if rows == 0 or top_k == 0 or num_slots == 0 or num_experts == 0:
        if out is None:
            return torch.zeros_like(ops.input, memory_format=torch.contiguous_format)
        if not ops.accumulate:
            out.zero_()
        return out

    resolved_input = ops.input.resolve_neg().resolve_conj()
    topk_weights = ops.topk_weights.resolve_neg().resolve_conj().contiguous()
    topk_ids = ops.topk_ids.contiguous()
    expert_to_slot = None if ops.expert_to_slot is None else ops.expert_to_slot.contiguous()
    prepared_data = ops.prepared_data
    prepared_scale = ops.prepared_scale

    if out is not None and is_concrete:
        if _tensors_overlap(out, topk_ids):
            topk_ids = topk_ids.clone()
        if _tensors_overlap(out, topk_weights):
            topk_weights = topk_weights.clone()
        if expert_to_slot is not None and _tensors_overlap(out, expert_to_slot):
            expert_to_slot = expert_to_slot.clone()
        if rows > 1:
            if _tensors_overlap(out, resolved_input):
                resolved_input = resolved_input.clone()
            if prepared_data is not None and _tensors_overlap(out, prepared_data):
                prepared_data = prepared_data.clone()
            if prepared_scale is not None and _tensors_overlap(out, prepared_scale):
                prepared_scale = prepared_scale.clone()

    target = _resolve_gpu_target(resolved_input.device, weight_format)
    if target is not None:
        from . import triton as moe_triton  # noqa: PLC0415

        return moe_triton.moe_ffn(
            resolved_input,
            topk_ids,
            topk_weights,
            ops.up_gate_qdata,
            ops.up_gate_scale,
            ops.down_qdata,
            ops.down_scale,
            ops.up_gate_global_scale,
            ops.down_global_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            out,
            target=target,
            accumulate=ops.accumulate,
            group_size=ops.group_size,
            activation_fn=ops.activation_fn,
            gate_first=ops.gate_first,
            high_first=ops.high_first,
            active_slots_hint=ops.active_slots_hint,
            up_gate_block_scales=ops.up_gate_block_scales,
            down_block_scales=ops.down_block_scales,
            up_gate_row_widths=ops.up_gate_row_widths,
            down_row_widths=ops.down_row_widths,
            up_gate_slot_offsets=ops.up_gate_slot_offsets,
            down_slot_offsets=ops.down_slot_offsets,
            up_gate_block_size=ops.up_gate_block_size,
            down_block_size=ops.down_block_size,
        )

    return reference.moe_ffn_reference(
        resolved_input,
        topk_ids,
        topk_weights,
        ops.up_gate_qdata,
        ops.up_gate_scale,
        ops.down_qdata,
        ops.down_scale,
        ops.up_gate_global_scale,
        ops.down_global_scale,
        expert_to_slot,
        prepared_data,
        prepared_scale,
        out,
        accumulate=ops.accumulate,
        group_size=ops.group_size,
        activation_fn=ops.activation_fn,
        gate_first=ops.gate_first,
        high_first=ops.high_first,
        up_gate_block_scales=ops.up_gate_block_scales,
        down_block_scales=ops.down_block_scales,
        up_gate_row_widths=ops.up_gate_row_widths,
        down_row_widths=ops.down_row_widths,
        up_gate_slot_offsets=ops.up_gate_slot_offsets,
        down_slot_offsets=ops.down_slot_offsets,
        up_gate_block_size=ops.up_gate_block_size,
        down_block_size=ops.down_block_size,
    )


@torch.library.custom_op("piper_kernels::moe_ffn", mutates_args=())
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
    """Execute a slot-indirect quantized MoE FFN and return a fresh output tensor."""
    return _run_moe_ffn(
        _MoEOperands(
            input,
            topk_ids,
            topk_weights,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            up_gate_global_scale,
            down_global_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            None,
            False,
            group_size,
            activation_fn,
            gate_first,
            high_first,
            active_slots_hint,
            up_gate_block_scales,
            down_block_scales,
            up_gate_row_widths,
            down_row_widths,
            up_gate_slot_offsets,
            down_slot_offsets,
            up_gate_block_size,
            down_block_size,
        )
    )


@moe_ffn.register_fake
def _moe_ffn_fake(
    input: torch.Tensor,  # noqa: A002
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
    ops = _MoEOperands(
        input,
        topk_ids,
        topk_weights,
        up_gate_qdata,
        up_gate_scale,
        down_qdata,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        expert_to_slot,
        prepared_data,
        prepared_scale,
        None,
        False,
        group_size,
        activation_fn,
        gate_first,
        high_first,
        active_slots_hint,
        up_gate_block_scales,
        down_block_scales,
        up_gate_row_widths,
        down_row_widths,
        up_gate_slot_offsets,
        down_slot_offsets,
        up_gate_block_size,
        down_block_size,
    )
    _reject_autograd(ops)
    _validate_op_operands(ops)
    return torch.empty_like(input, memory_format=torch.contiguous_format)


@torch.library.custom_op("piper_kernels::moe_ffn_", mutates_args=("out",))
def moe_ffn_(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    out: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None = None,
    down_global_scale: torch.Tensor | None = None,
    expert_to_slot: torch.Tensor | None = None,
    prepared_data: torch.Tensor | None = None,
    prepared_scale: torch.Tensor | None = None,
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
) -> None:
    """Execute a slot-indirect quantized MoE FFN into caller-owned ``out``."""
    _run_moe_ffn(
        _MoEOperands(
            input,
            topk_ids,
            topk_weights,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            up_gate_global_scale,
            down_global_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            out,
            accumulate,
            group_size,
            activation_fn,
            gate_first,
            high_first,
            active_slots_hint,
            up_gate_block_scales,
            down_block_scales,
            up_gate_row_widths,
            down_row_widths,
            up_gate_slot_offsets,
            down_slot_offsets,
            up_gate_block_size,
            down_block_size,
        )
    )


@moe_ffn_.register_fake
def _moe_ffn_inplace_fake(
    input: torch.Tensor,  # noqa: A002
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    out: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None = None,
    down_global_scale: torch.Tensor | None = None,
    expert_to_slot: torch.Tensor | None = None,
    prepared_data: torch.Tensor | None = None,
    prepared_scale: torch.Tensor | None = None,
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
) -> None:
    ops = _MoEOperands(
        input,
        topk_ids,
        topk_weights,
        up_gate_qdata,
        up_gate_scale,
        down_qdata,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        expert_to_slot,
        prepared_data,
        prepared_scale,
        out,
        accumulate,
        group_size,
        activation_fn,
        gate_first,
        high_first,
        active_slots_hint,
        up_gate_block_scales,
        down_block_scales,
        up_gate_row_widths,
        down_row_widths,
        up_gate_slot_offsets,
        down_slot_offsets,
        up_gate_block_size,
        down_block_size,
    )
    _reject_autograd(ops)
    _validate_op_operands(ops)
    _reject_weight_aliasing(ops)


@torch.library.impl("piper_kernels::moe_ffn", "Autograd")
def _moe_ffn_autograd(
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
    ops = _MoEOperands(
        input,
        topk_ids,
        topk_weights,
        up_gate_qdata,
        up_gate_scale,
        down_qdata,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        expert_to_slot,
        prepared_data,
        prepared_scale,
        None,
        False,
        group_size,
        activation_fn,
        gate_first,
        high_first,
        active_slots_hint,
        up_gate_block_scales,
        down_block_scales,
        up_gate_row_widths,
        down_row_widths,
        up_gate_slot_offsets,
        down_slot_offsets,
        up_gate_block_size,
        down_block_size,
    )
    _reject_autograd(ops)
    with torch._C._AutoDispatchBelowAutograd():
        return cast(
            torch.Tensor, torch.ops.piper_kernels.moe_ffn.default(*ops.op_args(inplace=False))
        )


@torch.library.impl("piper_kernels::moe_ffn_", "Autograd")
def _moe_ffn_inplace_autograd(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    out: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None = None,
    down_global_scale: torch.Tensor | None = None,
    expert_to_slot: torch.Tensor | None = None,
    prepared_data: torch.Tensor | None = None,
    prepared_scale: torch.Tensor | None = None,
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
) -> None:
    ops = _MoEOperands(
        input,
        topk_ids,
        topk_weights,
        up_gate_qdata,
        up_gate_scale,
        down_qdata,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        expert_to_slot,
        prepared_data,
        prepared_scale,
        out,
        accumulate,
        group_size,
        activation_fn,
        gate_first,
        high_first,
        active_slots_hint,
        up_gate_block_scales,
        down_block_scales,
        up_gate_row_widths,
        down_row_widths,
        up_gate_slot_offsets,
        down_slot_offsets,
        up_gate_block_size,
        down_block_size,
    )
    _reject_autograd(ops)
    # Validate eagerly above ADInplaceOrView so rejected calls leave out._version intact.
    if _is_eager_execution(ops):
        with (
            _prevalidated_inplace(ops),
            torch._C._AutoDispatchBelowAutograd(),
        ):
            torch.ops.piper_kernels.moe_ffn_.default(*ops.op_args(inplace=True))
        return
    with torch._C._AutoDispatchBelowAutograd():
        torch.ops.piper_kernels.moe_ffn_.default(*ops.op_args(inplace=True))


def dispatch_moe_ffn(ops: _MoEOperands) -> torch.Tensor:
    """Dispatch public ``moe_ffn`` while avoiding duplicate concrete validation on ``out``."""
    out = ops.out
    if out is None:
        return moe_ffn(*ops.op_args(inplace=False))
    # InferenceMode skips the Autograd key; validate eager out calls once before ADInplaceOrView.
    if _is_eager_execution(ops) and torch.is_inference_mode_enabled():
        with _prevalidated_inplace(ops):
            moe_ffn_(*ops.op_args(inplace=True))
    else:
        moe_ffn_(*ops.op_args(inplace=True))
    return out


__all__ = ["moe_ffn", "moe_ffn_"]
