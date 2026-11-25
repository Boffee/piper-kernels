"""Stable MoE FFN custom-op schemas and runtime accelerator dispatch."""

# Explicit custom-op schemas take flat tensor arguments required by torch.library.
# ruff: noqa: PLR0912, PLR0913, PLR0915, PLR0917

from __future__ import annotations

import math
from typing import Literal

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


def _validate_weight_storage(
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    *,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_size: int = 0,
) -> tuple[_MoEWeightKind, int, int, int, torch.device]:
    parse_input_activation(activation_fn, name="MoE")
    if not isinstance(gate_first, bool) or not isinstance(high_first, bool):
        raise TypeError("MoE gate_first and high_first must be booleans")
    if isinstance(group_size, bool) or not isinstance(group_size, (int, torch.SymInt)):
        raise TypeError(f"MoE group_size must be an integer, got {group_size!r}")
    if isinstance(group_size, torch.SymInt):
        group_size = int(group_size)

    device = up_gate_qdata.device
    has_trellis_arg = (
        any(
            tensor is not None
            for tensor in (
                up_gate_block_scales,
                down_block_scales,
                up_gate_row_widths,
                down_row_widths,
                up_gate_slot_offsets,
                down_slot_offsets,
            )
        )
        or (up_gate_block_size != 0)
        or (down_block_size != 0)
    )
    if has_trellis_arg:
        if high_first:
            raise ValueError("ConvRot Trellis MoE weights do not support high_first nibble packing")
        validate_group_size(group_size)
        if up_gate_global_scale is not None or down_global_scale is not None:
            raise ValueError("ConvRot Trellis MoE weights do not use per-expert global scales")
        if (
            up_gate_block_scales is None
            or down_block_scales is None
            or up_gate_block_scales.ndim != 3
            or down_block_scales.ndim != 3
        ):
            raise ValueError(
                "ConvRot Trellis MoE weights require 3-D up_gate_block_scales and down_block_scales"
            )
        if up_gate_block_scales.shape[0] != down_block_scales.shape[0]:
            raise ValueError("MoE up_gate and down tensors must have matching slot counts")
        if isinstance(down_block_size, bool) or not isinstance(
            down_block_size, (int, torch.SymInt)
        ):
            raise TypeError(f"MoE down_block_size must be an integer, got {down_block_size!r}")
        resolved_down_bs = (
            int(down_block_size) if isinstance(down_block_size, torch.SymInt) else down_block_size
        )
        validate_block_size(resolved_down_bs)
        num_slots, hidden_size, down_blocks = down_block_scales.shape
        intermediate_size = int(down_blocks) * resolved_down_bs
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
        up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
        _validate_trellis_projection(
            name="up_gate",
            qdata=up_gate_qdata,
            block_scales=up_gate_block_scales,
            scale=up_gate_scale,
            row_widths=up_gate_row_widths,
            slot_offsets=up_gate_slot_offsets,
            block_size=up_gate_block_size,
            num_slots=int(num_slots),
            out_features=up_gate_out,
            in_features=int(hidden_size),
            device=device,
        )
        _validate_trellis_projection(
            name="down",
            qdata=down_qdata,
            block_scales=down_block_scales,
            scale=down_scale,
            row_widths=down_row_widths,
            slot_offsets=down_slot_offsets,
            block_size=resolved_down_bs,
            num_slots=int(num_slots),
            out_features=int(hidden_size),
            in_features=intermediate_size,
            device=device,
        )
        return "convrot_trellis", int(num_slots), int(hidden_size), intermediate_size, device

    if up_gate_qdata.ndim != 3 or down_qdata.ndim != 3:
        raise ValueError("MoE qdata tensors must be 3-D [num_slots, out, in]")
    if up_gate_qdata.shape[0] != down_qdata.shape[0]:
        raise ValueError("MoE up_gate and down tensors must have matching slot counts")

    for tensor, name in (
        (up_gate_qdata, "up_gate_qdata"),
        (up_gate_scale, "up_gate_scale"),
        (down_qdata, "down_qdata"),
        (down_scale, "down_scale"),
    ):
        _require_strided_contiguous(tensor, name=name, device=device)

    if up_gate_qdata.dtype is torch.uint8:
        from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout  # noqa: PLC0415

        num_slots, hidden_size, packed_intermediate = down_qdata.shape
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
        up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
        expected_up_scale = (num_slots, *nvfp4_layout.scale_shape(up_gate_out, hidden_size))
        expected_down_scale = (
            num_slots,
            *nvfp4_layout.scale_shape(hidden_size, intermediate_size),
        )
        if (
            down_qdata.dtype is not torch.uint8
            or up_gate_scale.dtype is not torch.float8_e4m3fn
            or down_scale.dtype is not torch.float8_e4m3fn
            or tuple(up_gate_qdata.shape) != (num_slots, up_gate_out, hidden_size // 2)
            or tuple(up_gate_scale.shape) != expected_up_scale
            or tuple(down_scale.shape) != expected_down_scale
        ):
            raise ValueError("NVFP4 MoE weight storage shapes or dtypes are invalid")
        _validate_global_scale(
            up_gate_global_scale,
            name="up_gate_global_scale",
            num_slots=num_slots,
            device=device,
        )
        _validate_global_scale(
            down_global_scale,
            name="down_global_scale",
            num_slots=num_slots,
            device=device,
        )
        return "nvfp4", int(num_slots), int(hidden_size), int(intermediate_size), device

    if up_gate_qdata.dtype is torch.int8:
        if high_first:
            raise ValueError("ConvRot INT8 MoE weights do not support high_first nibble packing")
        validate_group_size(group_size)
        if up_gate_global_scale is not None or down_global_scale is not None:
            raise ValueError("ConvRot INT8 MoE weights do not use per-expert global scales")
        num_slots, hidden_size, intermediate_size = down_qdata.shape
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
        up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
        if (
            down_qdata.dtype is not torch.int8
            or up_gate_scale.dtype is not torch.float32
            or down_scale.dtype is not torch.float32
            or tuple(up_gate_qdata.shape) != (num_slots, up_gate_out, hidden_size)
            or tuple(up_gate_scale.shape) != (num_slots, up_gate_out)
            or tuple(down_scale.shape) != (num_slots, hidden_size)
        ):
            raise ValueError("ConvRot INT8 MoE weight storage shapes or dtypes are invalid")
        return "convrot_int8", int(num_slots), int(hidden_size), int(intermediate_size), device

    raise ValueError(
        f"MoE up_gate_qdata must be uint8 (NVFP4/Trellis) or int8 (ConvRot INT8), "
        f"got {up_gate_qdata.dtype}"
    )


def _validate_op_operands(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out: torch.Tensor | None,
    *,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_size: int = 0,
) -> tuple[_MoEWeightKind, int, int]:
    if not isinstance(accumulate, bool):
        raise TypeError("MoE accumulate must be a bool")
    if isinstance(active_slots_hint, bool) or not isinstance(
        active_slots_hint, (int, torch.SymInt)
    ):
        raise TypeError("MoE active_slots_hint must be an integer")
    if isinstance(active_slots_hint, torch.SymInt):
        active_slots_hint = int(active_slots_hint)
    if active_slots_hint < 0:
        raise ValueError("MoE active_slots_hint must be non-negative")

    weight_format, num_slots, hidden_size, _inter, weight_device = _validate_weight_storage(
        up_gate_qdata,
        up_gate_scale,
        down_qdata,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
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

    if input.ndim < 2 or input.shape[-1] != hidden_size:
        raise ValueError(
            f"MoE input must have shape [..., {hidden_size}], got {tuple(input.shape)}"
        )
    if input.dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(f"MoE input must be fp16, bf16, or fp32, got {input.dtype}")
    if input.device != weight_device:
        raise ValueError(f"MoE input device {input.device} must match weights on {weight_device}")
    if input.layout is not torch.strided:
        raise ValueError("MoE input must use strided layout")
    rows = math.prod(input.shape[:-1])

    if topk_ids.shape[:-1] != input.shape[:-1] or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MoE topk_ids and topk_weights must share leading shape "
            f"{tuple(input.shape[:-1])} and matching top_k"
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"MoE topk_ids must be int32 or int64, got {topk_ids.dtype}")
    if topk_weights.dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(f"MoE topk_weights must be fp16, bf16, or fp32, got {topk_weights.dtype}")
    if topk_ids.device != weight_device or topk_weights.device != weight_device:
        raise ValueError("MoE routing tensors must share the weight device")
    if topk_ids.layout is not torch.strided or topk_weights.layout is not torch.strided:
        raise ValueError("MoE routing tensors must use strided layout")

    if expert_to_slot is not None:
        if expert_to_slot.ndim != 1 or expert_to_slot.dtype not in (torch.int32, torch.int64):
            raise ValueError("MoE expert_to_slot must be 1-D int32 or int64")
        if expert_to_slot.device != weight_device or expert_to_slot.layout is not torch.strided:
            raise ValueError("MoE expert_to_slot must be strided on the weight device")

    if prepared_data is None:
        if prepared_scale is not None:
            raise ValueError("MoE prepared_input scale must be None when prepared_data is None")
    else:
        _require_strided_contiguous(
            prepared_data, name="MoE prepared_input data", device=weight_device
        )
        if tuple(prepared_data.shape) != (rows, hidden_size):
            raise ValueError("MoE prepared_input does not match input shape")
        if weight_format == "convrot_int8":
            if prepared_data.dtype is not torch.int8:
                raise ValueError("ConvRot INT8 prepared_input data must be int8")
            if prepared_scale is None or prepared_scale.dtype is not torch.float32:
                raise ValueError(
                    "ConvRot INT8 prepared_input requires contiguous [rows] float32 scale"
                )
            _require_strided_contiguous(
                prepared_scale, name="MoE prepared_input scale", device=weight_device
            )
            if tuple(prepared_scale.shape) != (rows,):
                raise ValueError(
                    "ConvRot INT8 prepared_input requires contiguous [rows] float32 scale"
                )
        else:
            if prepared_scale is not None:
                raise ValueError("MoE prepared_input scale must be None for rotated A16 weights")
            expected_prepared_dtype = torch.float32 if group_size > 0 else input.dtype
            if prepared_data.dtype is not expected_prepared_dtype:
                raise ValueError(
                    f"MoE prepared_input data must be {expected_prepared_dtype}, "
                    f"got {prepared_data.dtype}"
                )

    if out is not None:
        if (
            out.shape != input.shape
            or out.dtype is not input.dtype
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


def _reject_autograd(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_weights: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out: torch.Tensor | None,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
) -> None:
    if not torch.is_grad_enabled():
        return
    if input.requires_grad:
        raise RuntimeError("MoE FFN is inference-only and does not support gradients on input")
    if topk_weights.requires_grad:
        raise RuntimeError("MoE FFN does not support gradients on topk_weights")
    if any(
        tensor is not None and tensor.requires_grad
        for tensor in (
            up_gate_scale,
            down_scale,
            up_gate_global_scale,
            down_global_scale,
            up_gate_block_scales,
            down_block_scales,
        )
    ):
        raise RuntimeError("MoE weights do not support autograd")
    if (prepared_data is not None and prepared_data.requires_grad) or (
        prepared_scale is not None and prepared_scale.requires_grad
    ):
        raise RuntimeError("MoE prepared_input does not support autograd")
    if out is not None and out.requires_grad:
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


def _reject_weight_aliasing(
    out: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
) -> None:
    for weight_tensor in (
        up_gate_qdata,
        up_gate_scale,
        down_qdata,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        up_gate_block_scales,
        down_block_scales,
        up_gate_row_widths,
        down_row_widths,
        up_gate_slot_offsets,
        down_slot_offsets,
    ):
        if weight_tensor is not None and _tensors_overlap(out, weight_tensor):
            raise ValueError("MoE out tensor must not alias weight storage")


def _validate_before_mutating(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out: torch.Tensor,
    *,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_size: int = 0,
) -> None:
    _validate_op_operands(
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
        accumulate=accumulate,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        active_slots_hint=active_slots_hint,
        up_gate_block_scales=up_gate_block_scales,
        down_block_scales=down_block_scales,
        up_gate_row_widths=up_gate_row_widths,
        down_row_widths=down_row_widths,
        up_gate_slot_offsets=up_gate_slot_offsets,
        down_slot_offsets=down_slot_offsets,
        up_gate_block_size=up_gate_block_size,
        down_block_size=down_block_size,
    )
    if input.device.type != "meta":
        _reject_weight_aliasing(
            out,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            up_gate_global_scale,
            down_global_scale,
            up_gate_block_scales,
            down_block_scales,
            up_gate_row_widths,
            down_row_widths,
            up_gate_slot_offsets,
            down_slot_offsets,
        )


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
    if weight_format == "convrot_trellis":
        return target
    return target if target.supports_nvfp4_w4a16 else None


def _run_moe_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out: torch.Tensor | None,
    *,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_size: int = 0,
) -> torch.Tensor:
    weight_format, num_slots, rows = _validate_op_operands(
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
        accumulate=accumulate,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        active_slots_hint=active_slots_hint,
        up_gate_block_scales=up_gate_block_scales,
        down_block_scales=down_block_scales,
        up_gate_row_widths=up_gate_row_widths,
        down_row_widths=down_row_widths,
        up_gate_slot_offsets=up_gate_slot_offsets,
        down_slot_offsets=down_slot_offsets,
        up_gate_block_size=up_gate_block_size,
        down_block_size=down_block_size,
    )

    is_concrete = (
        input.device.type != "meta"
        and not torch.compiler.is_compiling()
        and detect_fake_mode() is None
    )
    if out is not None and is_concrete:
        _reject_weight_aliasing(
            out,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            up_gate_global_scale,
            down_global_scale,
            up_gate_block_scales,
            down_block_scales,
            up_gate_row_widths,
            down_row_widths,
            up_gate_slot_offsets,
            down_slot_offsets,
        )

    top_k = topk_ids.shape[-1]
    num_experts = num_slots if expert_to_slot is None else expert_to_slot.shape[0]
    if rows == 0 or top_k == 0 or num_slots == 0 or num_experts == 0:
        if out is None:
            return torch.zeros_like(input, memory_format=torch.contiguous_format)
        if not accumulate:
            out.zero_()
        return out

    resolved_input = input.resolve_neg().resolve_conj()
    topk_weights = topk_weights.resolve_neg().resolve_conj()
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    expert_to_slot = None if expert_to_slot is None else expert_to_slot.contiguous()

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
            target=target,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            active_slots_hint=active_slots_hint,
            up_gate_block_scales=up_gate_block_scales,
            down_block_scales=down_block_scales,
            up_gate_row_widths=up_gate_row_widths,
            down_row_widths=down_row_widths,
            up_gate_slot_offsets=up_gate_slot_offsets,
            down_slot_offsets=down_slot_offsets,
            up_gate_block_size=up_gate_block_size,
            down_block_size=down_block_size,
        )

    return reference.moe_ffn_reference(
        resolved_input,
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
        accumulate=accumulate,
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
        accumulate=False,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        active_slots_hint=active_slots_hint,
        up_gate_block_scales=up_gate_block_scales,
        down_block_scales=down_block_scales,
        up_gate_row_widths=up_gate_row_widths,
        down_row_widths=down_row_widths,
        up_gate_slot_offsets=up_gate_slot_offsets,
        down_slot_offsets=down_slot_offsets,
        up_gate_block_size=up_gate_block_size,
        down_block_size=down_block_size,
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
    _reject_autograd(
        input,
        topk_weights,
        up_gate_scale,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        prepared_data,
        prepared_scale,
        None,
        up_gate_block_scales,
        down_block_scales,
    )
    _validate_op_operands(
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
        accumulate=False,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        active_slots_hint=active_slots_hint,
        up_gate_block_scales=up_gate_block_scales,
        down_block_scales=down_block_scales,
        up_gate_row_widths=up_gate_row_widths,
        down_row_widths=down_row_widths,
        up_gate_slot_offsets=up_gate_slot_offsets,
        down_slot_offsets=down_slot_offsets,
        up_gate_block_size=up_gate_block_size,
        down_block_size=down_block_size,
    )
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
        accumulate=accumulate,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        active_slots_hint=active_slots_hint,
        up_gate_block_scales=up_gate_block_scales,
        down_block_scales=down_block_scales,
        up_gate_row_widths=up_gate_row_widths,
        down_row_widths=down_row_widths,
        up_gate_slot_offsets=up_gate_slot_offsets,
        down_slot_offsets=down_slot_offsets,
        up_gate_block_size=up_gate_block_size,
        down_block_size=down_block_size,
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
    _reject_autograd(
        input,
        topk_weights,
        up_gate_scale,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        prepared_data,
        prepared_scale,
        out,
        up_gate_block_scales,
        down_block_scales,
    )
    _validate_op_operands(
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
        accumulate=accumulate,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        active_slots_hint=active_slots_hint,
        up_gate_block_scales=up_gate_block_scales,
        down_block_scales=down_block_scales,
        up_gate_row_widths=up_gate_row_widths,
        down_row_widths=down_row_widths,
        up_gate_slot_offsets=up_gate_slot_offsets,
        down_slot_offsets=down_slot_offsets,
        up_gate_block_size=up_gate_block_size,
        down_block_size=down_block_size,
    )
    _reject_weight_aliasing(
        out,
        up_gate_qdata,
        up_gate_scale,
        down_qdata,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        up_gate_block_scales,
        down_block_scales,
        up_gate_row_widths,
        down_row_widths,
        up_gate_slot_offsets,
        down_slot_offsets,
    )


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
    _reject_autograd(
        input,
        topk_weights,
        up_gate_scale,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        prepared_data,
        prepared_scale,
        None,
        up_gate_block_scales,
        down_block_scales,
    )
    with torch._C._AutoDispatchBelowAutograd():
        return torch.ops.piper_kernels.moe_ffn.default(
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
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            active_slots_hint=active_slots_hint,
            up_gate_block_scales=up_gate_block_scales,
            down_block_scales=down_block_scales,
            up_gate_row_widths=up_gate_row_widths,
            down_row_widths=down_row_widths,
            up_gate_slot_offsets=up_gate_slot_offsets,
            down_slot_offsets=down_slot_offsets,
            up_gate_block_size=up_gate_block_size,
            down_block_size=down_block_size,
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
    _reject_autograd(
        input,
        topk_weights,
        up_gate_scale,
        down_scale,
        up_gate_global_scale,
        down_global_scale,
        prepared_data,
        prepared_scale,
        out,
        up_gate_block_scales,
        down_block_scales,
    )
    # Validate eagerly above ADInplaceOrView so rejected calls leave out._version intact.
    if not torch.compiler.is_compiling() and detect_fake_mode() is None:
        _validate_before_mutating(
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
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            active_slots_hint=active_slots_hint,
            up_gate_block_scales=up_gate_block_scales,
            down_block_scales=down_block_scales,
            up_gate_row_widths=up_gate_row_widths,
            down_row_widths=down_row_widths,
            up_gate_slot_offsets=up_gate_slot_offsets,
            down_slot_offsets=down_slot_offsets,
            up_gate_block_size=up_gate_block_size,
            down_block_size=down_block_size,
        )
    with torch._C._AutoDispatchBelowAutograd():
        torch.ops.piper_kernels.moe_ffn_.default(
            input,
            topk_ids,
            topk_weights,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            out,
            up_gate_global_scale,
            down_global_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            active_slots_hint=active_slots_hint,
            up_gate_block_scales=up_gate_block_scales,
            down_block_scales=down_block_scales,
            up_gate_row_widths=up_gate_row_widths,
            down_row_widths=down_row_widths,
            up_gate_slot_offsets=up_gate_slot_offsets,
            down_slot_offsets=down_slot_offsets,
            up_gate_block_size=up_gate_block_size,
            down_block_size=down_block_size,
        )


__all__ = ["moe_ffn", "moe_ffn_"]
