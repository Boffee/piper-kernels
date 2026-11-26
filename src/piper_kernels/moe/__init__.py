"""Offload-first quantized MoE FFN operators and expert-major weight storage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch._guards import detect_fake_mode

from piper_kernels._input_activations import InputActivation, parse_input_activation
from piper_kernels._triton import runtime
from piper_kernels.trellis._layout import (
    TrellisLayout,
    resolve_block_size,
    resolve_layout,
)
from piper_kernels.weights.convrot._rotation import rotate_groups, validate_group_size

from . import _ops
from .reference import _autocast_disabled

type MoEWeightFormat = Literal[
    "nvfp4_w4a16",
    "convrot_nvfp4_w4a16",
    "convrot_int8",
    "convrot_trellis",
]

_SUPPORTED_LOGICAL_DTYPES = _ops._SUPPORTED_LOGICAL_DTYPES
_SUPPORTED_WEIGHT_FORMATS = (
    "nvfp4_w4a16",
    "convrot_nvfp4_w4a16",
    "convrot_int8",
    "convrot_trellis",
)


def _normalize_nvfp4_global_scale(
    scale: torch.Tensor | None,
    *,
    name: str,
) -> torch.Tensor | None:
    if scale is None or scale.ndim == 1:
        return scale
    if scale.ndim == 2 and scale.shape[1] == 1:
        return scale.squeeze(1)
    if scale.ndim == 3 and scale.shape[1:] == (1, 1):
        return scale.squeeze(2).squeeze(1)
    raise ValueError(f"{name} must have shape [num_slots]")


@dataclass(frozen=True, slots=True)
class MoEWeights:
    """Zero-copy view of expert-major quantized MoE weights stored as ``[num_slots, ...]``.

    For ``"convrot_trellis"``, wrapping and dispatch perform no host readbacks: callers must
    ensure E4M3 ``block_scales`` are non-NaN (not ``0x7F`` or ``0xFF``), variable-row
    ``row_widths`` lie in ``[ceil(2 * block_size / 8), floor(6 * block_size / 8)]`` with each
    slot fitting in ``slot_bytes`` (2-D) or ``qdata.numel() - slot_offsets[s]`` (1-D), and
    ``slot_offsets`` are non-negative and within ``qdata.numel()``.
    """

    up_gate_qdata: torch.Tensor
    up_gate_scale: torch.Tensor
    down_qdata: torch.Tensor
    down_scale: torch.Tensor
    logical_dtype: torch.dtype
    up_gate_global_scale: torch.Tensor | None = None
    down_global_scale: torch.Tensor | None = None
    group_size: int = 0
    activation_fn: InputActivation = "swiglu"
    gate_first: bool = True
    high_first: bool = False
    up_gate_block_scales: torch.Tensor | None = None
    down_block_scales: torch.Tensor | None = None
    up_gate_row_widths: torch.Tensor | None = None
    down_row_widths: torch.Tensor | None = None
    up_gate_slot_offsets: torch.Tensor | None = None
    down_slot_offsets: torch.Tensor | None = None
    up_gate_block_size: int = 0
    down_block_size: int = 0

    def __post_init__(self) -> None:
        if self.logical_dtype not in _SUPPORTED_LOGICAL_DTYPES:
            raise ValueError(
                f"MoE logical_dtype must be float16, bfloat16, or float32, got {self.logical_dtype}"
            )
        _ops._validate_weight_storage(self)

    @property
    def weight_format(self) -> MoEWeightFormat:
        """Return the storage format identifier."""
        if self.up_gate_block_scales is not None or self.down_block_scales is not None:
            return "convrot_trellis"
        if self.up_gate_qdata.dtype is torch.int8:
            return "convrot_int8"
        return "convrot_nvfp4_w4a16" if self.group_size > 0 else "nvfp4_w4a16"

    @property
    def num_slots(self) -> int:
        """Return the number of resident expert slots wrapped by this storage."""
        if self.down_block_scales is not None:
            return self.down_block_scales.shape[0]
        return self.down_qdata.shape[0]

    @property
    def hidden_size(self) -> int:
        """Return the logical model hidden width ``H``."""
        if self.down_block_scales is not None:
            return self.down_block_scales.shape[1]
        return self.down_qdata.shape[1]

    @property
    def intermediate_size(self) -> int:
        """Return the per-expert FFN intermediate width ``I``."""
        if self.down_block_scales is not None:
            return self.down_block_scales.shape[2] * self.down_block_size
        factor = 2 if self.up_gate_qdata.dtype is torch.uint8 else 1
        return self.down_qdata.shape[2] * factor

    @property
    def device(self) -> torch.device:
        """Return the storage device shared by all tensors."""
        return self.down_qdata.device

    @classmethod
    def from_quantized(  # noqa: PLR0913 - flat storage constructor mirrors fields
        cls,
        up_gate_qdata: torch.Tensor,
        up_gate_scale: torch.Tensor,
        down_qdata: torch.Tensor,
        down_scale: torch.Tensor,
        *,
        logical_dtype: torch.dtype,
        up_gate_global_scale: torch.Tensor | None = None,
        down_global_scale: torch.Tensor | None = None,
        group_size: int = 0,
        activation_fn: InputActivation = "swiglu",
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
    ) -> MoEWeights:
        """Wrap existing expert-major quantized storage without copying."""
        has_trellis = (
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
            or up_gate_block_size != 0
            or down_block_size != 0
        )
        if has_trellis or up_gate_qdata.dtype is torch.int8:
            if up_gate_scale.ndim == 3 and up_gate_scale.shape[2] == 1:
                up_gate_scale = up_gate_scale.squeeze(2)
            if down_scale.ndim == 3 and down_scale.shape[2] == 1:
                down_scale = down_scale.squeeze(2)
        else:
            up_gate_global_scale = _normalize_nvfp4_global_scale(
                up_gate_global_scale, name="up_gate_global_scale"
            )
            down_global_scale = _normalize_nvfp4_global_scale(
                down_global_scale, name="down_global_scale"
            )
        return cls(
            up_gate_qdata=up_gate_qdata,
            up_gate_scale=up_gate_scale,
            down_qdata=down_qdata,
            down_scale=down_scale,
            logical_dtype=logical_dtype,
            up_gate_global_scale=up_gate_global_scale,
            down_global_scale=down_global_scale,
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

    @classmethod
    def from_hp(
        cls,
        up_gate_hp: torch.Tensor,
        down_hp: torch.Tensor,
        *,
        weight_format: MoEWeightFormat = "nvfp4_w4a16",
        group_size: int = 0,
        activation_fn: InputActivation = "swiglu",
        gate_first: bool = True,
        high_first: bool = False,
        compute_per_tensor_scale: bool = True,
        bits_per_weight: float | None = None,
        bytes_per_block: int | None = None,
    ) -> MoEWeights:
        """Quantize ``[S, up_gate_out, H]`` and ``[S, H, I]`` weights into slot storage."""
        _validate_from_hp_inputs(
            up_gate_hp,
            down_hp,
            weight_format=weight_format,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            compute_per_tensor_scale=compute_per_tensor_scale,
            bits_per_weight=bits_per_weight,
            bytes_per_block=bytes_per_block,
        )
        if weight_format == "convrot_trellis":
            return _quantize_hp_trellis(
                up_gate_hp,
                down_hp,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                high_first=high_first,
                bits_per_weight=bits_per_weight,
                bytes_per_block=bytes_per_block,
            )
        if weight_format == "convrot_int8":
            return _quantize_hp_int8(
                up_gate_hp,
                down_hp,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                high_first=high_first,
            )
        return _quantize_hp_nvfp4(
            up_gate_hp,
            down_hp,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            compute_per_tensor_scale=compute_per_tensor_scale,
        )


def _validate_from_hp_inputs(
    up_gate_hp: torch.Tensor,
    down_hp: torch.Tensor,
    *,
    weight_format: MoEWeightFormat,
    group_size: int,
    activation_fn: InputActivation,
    gate_first: bool,
    high_first: bool,
    compute_per_tensor_scale: bool,
    bits_per_weight: float | None,
    bytes_per_block: int | None,
) -> None:
    parse_input_activation(activation_fn, name="MoE")
    if weight_format not in _SUPPORTED_WEIGHT_FORMATS:
        raise ValueError(f"Unsupported MoE weight_format {weight_format!r}")
    if (
        not isinstance(gate_first, bool)
        or not isinstance(high_first, bool)
        or not isinstance(compute_per_tensor_scale, bool)
    ):
        raise TypeError("MoE gate_first, high_first, and compute_per_tensor_scale must be booleans")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError(f"MoE group_size must be an integer, got {group_size!r}")
    if weight_format != "convrot_trellis" and (
        bits_per_weight is not None or bytes_per_block is not None
    ):
        raise ValueError("bits_per_weight and bytes_per_block apply only to convrot_trellis")
    if weight_format == "nvfp4_w4a16" and group_size != 0:
        raise ValueError("nvfp4_w4a16 requires group_size=0; use convrot_nvfp4_w4a16")
    if weight_format == "convrot_nvfp4_w4a16" and group_size == 0:
        raise ValueError("convrot_nvfp4_w4a16 requires group_size in (16, 64, 256)")
    if up_gate_hp.layout is not torch.strided or down_hp.layout is not torch.strided:
        raise ValueError("MoE from_hp weights must use strided layout")
    if up_gate_hp.ndim != 3 or down_hp.ndim != 3:
        raise ValueError("MoE from_hp requires 3-D [num_slots, out, in] tensors")
    logical_dtype = up_gate_hp.dtype
    if logical_dtype not in _SUPPORTED_LOGICAL_DTYPES or down_hp.dtype is not logical_dtype:
        raise ValueError("MoE from_hp weights must share fp16, bf16, or fp32 dtype")
    if up_gate_hp.device != down_hp.device or up_gate_hp.device.type == "meta":
        raise ValueError("MoE from_hp weights must share a non-meta device")
    num_slots, hidden_size, intermediate_size = down_hp.shape
    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    if (
        num_slots != up_gate_hp.shape[0]
        or hidden_size <= 0
        or intermediate_size <= 0
        or tuple(up_gate_hp.shape) != (num_slots, up_gate_out, hidden_size)
    ):
        raise ValueError("MoE from_hp up_gate_hp and down_hp shapes are incompatible")


def _quantize_hp_trellis(
    up_gate_hp: torch.Tensor,
    down_hp: torch.Tensor,
    *,
    group_size: int,
    activation_fn: InputActivation,
    gate_first: bool,
    high_first: bool,
    bits_per_weight: float | None,
    bytes_per_block: int | None,
) -> MoEWeights:
    if high_first:
        raise ValueError("ConvRot Trellis MoE weights do not support high_first")
    validate_group_size(group_size)
    num_slots, hidden_size, intermediate_size = down_hp.shape
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError(f"ConvRot Trellis MoE widths must be divisible by group_size {group_size}")
    if (
        hidden_size > _ops._INT8_MAX_REDUCTION_WIDTH
        or intermediate_size > _ops._INT8_MAX_REDUCTION_WIDTH
    ):
        raise ValueError(
            f"ConvRot Trellis MoE widths must not exceed {_ops._INT8_MAX_REDUCTION_WIDTH} "
            f"for INT32 accumulation, got hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}"
        )
    if (bits_per_weight is None) == (bytes_per_block is None):
        raise ValueError("specify exactly one of bits_per_weight or bytes_per_block")
    if bits_per_weight is not None:
        up_layout = resolve_layout(hidden_size, bits_per_weight)
        down_layout = resolve_layout(intermediate_size, bits_per_weight)
    else:
        assert bytes_per_block is not None
        up_layout = TrellisLayout.from_storage(resolve_block_size(hidden_size), bytes_per_block)
        down_layout = TrellisLayout.from_storage(
            resolve_block_size(intermediate_size), bytes_per_block
        )

    up_src = up_gate_hp.detach()
    down_src = down_hp.detach()
    if not (bool(torch.isfinite(up_src).all()) and bool(torch.isfinite(down_src).all())):
        raise ValueError("cannot quantize non-finite MoE weights")
    up_out = up_src.shape[1]
    up_blocks = hidden_size // up_layout.block_size
    down_blocks = intermediate_size // down_layout.block_size
    if num_slots == 0:
        return MoEWeights(
            up_gate_qdata=up_src.new_empty(
                (0, up_out, up_blocks, up_layout.bytes_per_block), dtype=torch.uint8
            ),
            up_gate_scale=up_src.new_empty((0, up_out), dtype=torch.float32),
            down_qdata=down_src.new_empty(
                (0, hidden_size, down_blocks, down_layout.bytes_per_block), dtype=torch.uint8
            ),
            down_scale=down_src.new_empty((0, hidden_size), dtype=torch.float32),
            logical_dtype=up_src.dtype,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            up_gate_block_scales=up_src.new_empty(
                (0, up_out, up_blocks), dtype=torch.float8_e4m3fn
            ),
            down_block_scales=down_src.new_empty(
                (0, hidden_size, down_blocks), dtype=torch.float8_e4m3fn
            ),
            up_gate_block_size=up_layout.block_size,
            down_block_size=down_layout.block_size,
        )
    from piper_kernels.trellis import TrellisStorage  # noqa: PLC0415

    with _autocast_disabled(up_src.device):
        up_items = [
            TrellisStorage.from_hp(
                up_src[s],
                group_size=group_size,
                bits_per_weight=bits_per_weight,
                bytes_per_block=bytes_per_block,
            )
            for s in range(num_slots)
        ]
        down_items = [
            TrellisStorage.from_hp(
                down_src[s],
                group_size=group_size,
                bits_per_weight=bits_per_weight,
                bytes_per_block=bytes_per_block,
            )
            for s in range(num_slots)
        ]
    return MoEWeights(
        up_gate_qdata=torch.stack([item.bitstreams for item in up_items]).contiguous(),
        up_gate_scale=torch.stack([item.scale for item in up_items]).squeeze(-1).contiguous(),
        down_qdata=torch.stack([item.bitstreams for item in down_items]).contiguous(),
        down_scale=torch.stack([item.scale for item in down_items]).squeeze(-1).contiguous(),
        logical_dtype=up_src.dtype,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        up_gate_block_scales=torch.stack([item.block_scales for item in up_items]).contiguous(),
        down_block_scales=torch.stack([item.block_scales for item in down_items]).contiguous(),
        up_gate_block_size=up_layout.block_size,
        down_block_size=down_layout.block_size,
    )


def _quantize_hp_int8(
    up_gate_hp: torch.Tensor,
    down_hp: torch.Tensor,
    *,
    group_size: int,
    activation_fn: InputActivation,
    gate_first: bool,
    high_first: bool,
) -> MoEWeights:
    if high_first:
        raise ValueError("ConvRot INT8 MoE weights do not support high_first")
    validate_group_size(group_size)
    num_slots, hidden_size, intermediate_size = down_hp.shape
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError(f"ConvRot INT8 MoE widths must be divisible by group_size {group_size}")
    if (
        hidden_size > _ops._INT8_MAX_REDUCTION_WIDTH
        or intermediate_size > _ops._INT8_MAX_REDUCTION_WIDTH
    ):
        raise ValueError(
            f"ConvRot INT8 MoE widths must not exceed {_ops._INT8_MAX_REDUCTION_WIDTH} "
            f"for INT32 accumulation, got hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}"
        )
    up_src = up_gate_hp.detach()
    down_src = down_hp.detach()
    if not (bool(torch.isfinite(up_src).all()) and bool(torch.isfinite(down_src).all())):
        raise ValueError("cannot quantize non-finite MoE weights")
    if num_slots == 0:
        up_out = up_src.shape[1]
        return MoEWeights(
            up_gate_qdata=up_src.new_empty((0, up_out, hidden_size), dtype=torch.int8),
            up_gate_scale=up_src.new_empty((0, up_out), dtype=torch.float32),
            down_qdata=down_src.new_empty((0, hidden_size, intermediate_size), dtype=torch.int8),
            down_scale=down_src.new_empty((0, hidden_size), dtype=torch.float32),
            logical_dtype=up_src.dtype,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
        )
    from piper_kernels.weights.convrot.int8._quantization import quantize_weight  # noqa: PLC0415

    with _autocast_disabled(up_src.device):
        up_pairs = [quantize_weight(up_src[s], group_size) for s in range(num_slots)]
        down_pairs = [quantize_weight(down_src[s], group_size) for s in range(num_slots)]
        up_scale = torch.stack([pair[1] for pair in up_pairs]).squeeze(-1).contiguous()
        down_scale = torch.stack([pair[1] for pair in down_pairs]).squeeze(-1).contiguous()
    if not (bool(torch.isfinite(up_scale).all()) and bool(torch.isfinite(down_scale).all())):
        raise ValueError("ConvRot INT8 quantization produced non-finite scales")
    return MoEWeights(
        up_gate_qdata=torch.stack([pair[0] for pair in up_pairs]).contiguous(),
        up_gate_scale=up_scale,
        down_qdata=torch.stack([pair[0] for pair in down_pairs]).contiguous(),
        down_scale=down_scale,
        logical_dtype=up_src.dtype,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
    )


def _quantize_nvfp4_batch(
    source: torch.Tensor,
    *,
    group_size: int,
    high_first: bool,
    compute_per_tensor_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor  # noqa: PLC0415
    from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor  # noqa: PLC0415
    from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout  # noqa: PLC0415

    num_slots, out_features, in_features = source.shape
    if num_slots == 0:
        gs = source.new_empty((0,), dtype=torch.float32) if compute_per_tensor_scale else None
        return (
            source.new_empty((0, out_features, in_features // 2), dtype=torch.uint8),
            source.new_empty(
                (0, *nvfp4_layout.scale_shape(out_features, in_features)),
                dtype=torch.float8_e4m3fn,
            ),
            gs,
        )
    q_list: list[torch.Tensor] = []
    s_list: list[torch.Tensor] = []
    gs_list: list[torch.Tensor] = []
    with _autocast_disabled(source.device):
        for slot in range(num_slots):
            slot_source = source[slot]
            if group_size > 0:
                quantized_weight = ConvRotNVFP4Tensor.from_hp(
                    slot_source,
                    group_size=group_size,
                    compute_per_tensor_scale=compute_per_tensor_scale,
                    is_swizzled_scales=True,
                    act_quant=None,
                )
            else:
                quantized_weight = PiperNVFP4Tensor.from_hp(
                    slot_source,
                    compute_per_tensor_scale=compute_per_tensor_scale,
                    is_swizzled_scales=True,
                    act_quant=None,
                )
            qdata = quantized_weight.qdata
            if high_first:
                qdata = nvfp4_layout.swap_packed_pairs(qdata)
            q_list.append(qdata)
            s_list.append(quantized_weight.scale)
            if compute_per_tensor_scale:
                assert quantized_weight.per_tensor_scale is not None
                gs_list.append(quantized_weight.per_tensor_scale)
    stacked_scales = torch.stack(s_list).contiguous()
    global_scales = torch.stack(gs_list).contiguous() if compute_per_tensor_scale else None
    if not bool(torch.isfinite(stacked_scales.float()).all()) or (
        global_scales is not None and not bool(torch.isfinite(global_scales).all())
    ):
        raise ValueError("NVFP4 quantization produced non-finite scales")
    return torch.stack(q_list).contiguous(), stacked_scales, global_scales


def _quantize_hp_nvfp4(
    up_gate_hp: torch.Tensor,
    down_hp: torch.Tensor,
    *,
    group_size: int,
    activation_fn: InputActivation,
    gate_first: bool,
    high_first: bool,
    compute_per_tensor_scale: bool,
) -> MoEWeights:
    from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout  # noqa: PLC0415

    hidden_size, intermediate_size = down_hp.shape[1], down_hp.shape[2]
    if hidden_size % nvfp4_layout.BLOCK_SIZE or intermediate_size % nvfp4_layout.BLOCK_SIZE:
        raise ValueError("NVFP4 MoE widths must be multiples of 16")
    if group_size != 0:
        validate_group_size(group_size)
        if hidden_size % group_size or intermediate_size % group_size:
            raise ValueError(f"ConvRot NVFP4 MoE widths must be divisible by {group_size}")
    up_src = up_gate_hp.detach()
    down_src = down_hp.detach()
    if not (bool(torch.isfinite(up_src).all()) and bool(torch.isfinite(down_src).all())):
        raise ValueError("cannot quantize non-finite MoE weights")
    up_q, up_s, up_gs = _quantize_nvfp4_batch(
        up_src,
        group_size=group_size,
        high_first=high_first,
        compute_per_tensor_scale=compute_per_tensor_scale,
    )
    down_q, down_s, down_gs = _quantize_nvfp4_batch(
        down_src,
        group_size=group_size,
        high_first=high_first,
        compute_per_tensor_scale=compute_per_tensor_scale,
    )
    return MoEWeights(
        up_gate_qdata=up_q,
        up_gate_scale=up_s,
        down_qdata=down_q,
        down_scale=down_s,
        logical_dtype=up_src.dtype,
        up_gate_global_scale=up_gs,
        down_global_scale=down_gs,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
    )


@dataclass(frozen=True, slots=True)
class PreparedMoEInput:
    """Reusable prepared activation shared across resident/streamed MoE waves."""

    data: torch.Tensor
    scale: torch.Tensor | None
    logical_dtype: torch.dtype
    group_size: int
    is_int8: bool


def _validate_input_tensor(input: torch.Tensor, weights: MoEWeights) -> int:  # noqa: A002
    if input.ndim < 2 or input.shape[-1] != weights.hidden_size:
        raise ValueError(
            f"MoE input must have shape [..., {weights.hidden_size}], got {tuple(input.shape)}"
        )
    if input.dtype is not weights.logical_dtype:
        raise ValueError(
            f"MoE input dtype {input.dtype} must match weight logical_dtype {weights.logical_dtype}"
        )
    if input.device != weights.device:
        raise ValueError(f"MoE input device {input.device} must match weights on {weights.device}")
    if input.layout is not torch.strided:
        raise ValueError("MoE input must use strided layout")
    if torch.is_grad_enabled() and input.requires_grad:
        raise RuntimeError("MoE FFN is inference-only and does not support gradients on input")
    return math.prod(input.shape[:-1])


def prepare_moe_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weights: MoEWeights,
) -> PreparedMoEInput:
    """Prepare/rotate/quantize ``input`` once for reuse across MoE waves or splits."""
    rows = _validate_input_tensor(input, weights)
    hidden_size = weights.hidden_size
    is_int8 = weights.weight_format == "convrot_int8"
    input_2d = input.resolve_neg().resolve_conj().reshape(rows, hidden_size).contiguous()
    if is_int8:
        if rows == 0:
            return PreparedMoEInput(
                data=input.new_empty((0, hidden_size), dtype=torch.int8),
                scale=input.new_empty((0,), dtype=torch.float32),
                logical_dtype=weights.logical_dtype,
                group_size=weights.group_size,
                is_int8=True,
            )
        from piper_kernels.linear.convrot.int8 import _ops as int8_ops  # noqa: PLC0415

        with _autocast_disabled(input.device):
            data, scale = int8_ops.prepare_input(input_2d, weights.group_size)
        return PreparedMoEInput(
            data=data,
            scale=scale,
            logical_dtype=weights.logical_dtype,
            group_size=weights.group_size,
            is_int8=True,
        )

    if weights.group_size > 0:
        if rows == 0:
            return PreparedMoEInput(
                data=input.new_empty((0, hidden_size), dtype=torch.float32),
                scale=None,
                logical_dtype=weights.logical_dtype,
                group_size=weights.group_size,
                is_int8=False,
            )
        if (
            input.device.type == "cuda"
            and not torch.compiler.is_compiling()
            and detect_fake_mode() is None
            and runtime.supports_device(input.device)
        ):
            from piper_kernels.weights.convrot.triton import rotate_input  # noqa: PLC0415

            rotated = torch.empty((rows, hidden_size), device=input.device, dtype=torch.float32)
            rotate_input(input_2d, rotated, weights.group_size, num_warps=4)
        else:
            with _autocast_disabled(input.device):
                rotated = rotate_groups(input_2d.float(), weights.group_size)
        return PreparedMoEInput(
            data=rotated,
            scale=None,
            logical_dtype=weights.logical_dtype,
            group_size=weights.group_size,
            is_int8=False,
        )

    return PreparedMoEInput(
        data=input_2d.clone(),
        scale=None,
        logical_dtype=weights.logical_dtype,
        group_size=0,
        is_int8=False,
    )


def _unwrap_prepared_input(
    prepared_input: PreparedMoEInput | None,
    weights: MoEWeights,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if prepared_input is None:
        return None, None
    is_int8 = weights.weight_format == "convrot_int8"
    if (
        prepared_input.is_int8 is not is_int8
        or prepared_input.group_size != weights.group_size
        or prepared_input.logical_dtype is not weights.logical_dtype
    ):
        raise ValueError("MoE prepared_input does not match weights")
    if not is_int8 and prepared_input.scale is not None:
        raise ValueError("MoE prepared_input scale must be None")
    return prepared_input.data, prepared_input.scale


def moe_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weights: MoEWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    expert_to_slot: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    prepared_input: PreparedMoEInput | None = None,
    active_slots_hint: int | None = None,
) -> torch.Tensor:
    """Execute a routed MoE FFN with optional slot indirection and in-place accumulation."""
    if not isinstance(accumulate, bool):
        raise TypeError("MoE accumulate must be a bool")
    if accumulate and out is None:
        raise ValueError("accumulate=True requires a caller-owned out tensor")
    if input.dtype is not weights.logical_dtype:
        raise ValueError(
            f"MoE input dtype {input.dtype} must match weight logical_dtype {weights.logical_dtype}"
        )

    prepared_data, prepared_scale = _unwrap_prepared_input(prepared_input, weights)

    if active_slots_hint is None:
        hint = 0
    else:
        if isinstance(active_slots_hint, bool) or not isinstance(
            active_slots_hint, (int, torch.SymInt)
        ):
            raise TypeError("MoE active_slots_hint must be an integer or None")
        if active_slots_hint <= 0:
            raise ValueError("MoE active_slots_hint must be positive")
        hint = active_slots_hint

    return _ops.dispatch_moe_ffn(
        _ops._MoEOperands(
            input,
            topk_ids,
            topk_weights,
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata,
            weights.down_scale,
            weights.up_gate_global_scale,
            weights.down_global_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            out,
            accumulate,
            weights.group_size,
            weights.activation_fn,
            weights.gate_first,
            weights.high_first,
            hint,
            weights.up_gate_block_scales,
            weights.down_block_scales,
            weights.up_gate_row_widths,
            weights.down_row_widths,
            weights.up_gate_slot_offsets,
            weights.down_slot_offsets,
            weights.up_gate_block_size,
            weights.down_block_size,
        )
    )


__all__ = [
    "MoEWeightFormat",
    "MoEWeights",
    "PreparedMoEInput",
    "moe_ffn",
    "prepare_moe_input",
]
