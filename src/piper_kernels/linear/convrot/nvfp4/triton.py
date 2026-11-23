"""Exact static and dynamic ConvRot NVFP4 activation preparation."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._input_activations import input_activation_width
from piper_kernels._input_activations.triton import gelu_tanh, swiglu
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.nvfp4._storage import prepare_activation_storage
from piper_kernels.weights.convrot import triton as rotation
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout
from piper_kernels.weights.nvfp4.triton import dynamic_scale as nvfp4_dynamic_scale
from piper_kernels.weights.nvfp4.triton import encode_nvfp4_blocks, swizzled_scale_offsets

_NVFP4_BLOCK_SIZE = nvfp4_layout.BLOCK_SIZE
_NVFP4_BLOCK_SIZE_TL = tl.constexpr(_NVFP4_BLOCK_SIZE)
_AMAX_BLOCK_SIZE = 1_024
_AMAX_TILES_PER_PROGRAM = 2
_AMAX_NUM_WARPS = 2
_QUANTIZE_BLOCK_SIZE = 1_024
_QUANTIZE_NUM_WARPS = 2


type _ValidatedInput = tuple[
    torch.Tensor,
    int,
    int,
    AcceleratorTarget,
]


@triton.jit
def _load_activated_rotated_tile(
    input_ptr,
    offsets,
    valid,
    features: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    block_size: tl.constexpr,
):
    """Load, optionally activate, and ConvRot-rotate one group-aligned 1-D tile in FP32."""
    if activation_fn == "swiglu":
        input_offsets = (offsets // features) * (2 * features) + (offsets % features)
        values = tl.load(input_ptr + input_offsets, mask=valid, other=0.0).to(tl.float32)
        gate = tl.load(
            input_ptr + input_offsets + features,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        values = swiglu(values, gate)
    else:
        values = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        if activation_fn == "gelu_tanh":
            values = gelu_tanh(values, accelerator_backend)
    return rotation.rotate_hadamard_groups(values, block_size, group_size) * inverse_sqrt_group


@triton.jit
def _rotated_amax_kernel(
    input_ptr,
    partial_ptr,
    elements,
    features: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    block_size: tl.constexpr,
    tiles_per_program: tl.constexpr,
):
    """Reduce post-rotation absolute maxima across block-tiled 1-D slices."""
    program_id = tl.program_id(0).to(tl.int64)
    base = program_id * (block_size * tiles_per_program)
    lane = tl.arange(0, block_size)
    tile_amax = tl.full((), 0.0, tl.float32)
    for tile in tl.static_range(0, tiles_per_program):  # pyright: ignore[reportGeneralTypeIssues]
        offsets = base + tile * block_size + lane
        valid = offsets < elements
        rotated = _load_activated_rotated_tile(
            input_ptr,
            offsets,
            valid,
            features,
            group_size,
            inverse_sqrt_group,
            activation_fn,
            accelerator_backend,
            block_size,
        )
        rotated = tl.where(valid, rotated, 0.0)
        tile_amax = tl.maximum(tile_amax, tl.max(tl.abs(rotated), axis=0))
    tl.store(partial_ptr + program_id, tile_amax)


@triton.jit
def _rotate_quantize_kernel(
    input_ptr,
    per_tensor_scale_ptr,
    qdata_ptr,
    scale_ptr,
    elements,
    features: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    block_size: tl.constexpr,
    high_first: tl.constexpr,
):
    """Apply optional activation and ConvRot rotation, then write hardware-ready NVFP4 storage."""
    start = tl.program_id(0).to(tl.int64) * block_size
    offsets = start + tl.arange(0, block_size)
    valid = offsets < elements
    rotated = _load_activated_rotated_tile(
        input_ptr,
        offsets,
        valid,
        features,
        group_size,
        inverse_sqrt_group,
        activation_fn,
        accelerator_backend,
        block_size,
    )
    block_count: tl.constexpr = block_size // _NVFP4_BLOCK_SIZE_TL
    packed, encoded_scale = encode_nvfp4_blocks(  # pyright: ignore[reportGeneralTypeIssues]
        tl.reshape(rotated, (block_count, _NVFP4_BLOCK_SIZE_TL)),
        tl.load(per_tensor_scale_ptr).to(tl.float32),
        block_count,
        high_first,
    )
    qdata_offsets = start // 2 + tl.arange(0, block_size // 2)
    tl.store(
        qdata_ptr + qdata_offsets,
        tl.reshape(packed, (block_size // 2,)),
        mask=qdata_offsets < elements // 2,
    )
    blocks = start // _NVFP4_BLOCK_SIZE_TL + tl.arange(0, block_count)
    feature_blocks: tl.constexpr = features // _NVFP4_BLOCK_SIZE_TL
    scale_offsets = swizzled_scale_offsets(
        blocks // feature_blocks,
        blocks % feature_blocks,
        scale_column_blocks,
    )
    tl.store(
        scale_ptr + scale_offsets, encoded_scale, mask=blocks < elements // _NVFP4_BLOCK_SIZE_TL
    )


def _validate_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
) -> _ValidatedInput:
    validate_group_size(group_size)
    if input.ndim == 0 or input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("ConvRot NVFP4 input must be a non-scalar FP16, BF16, or FP32 tensor")
    source_features = int(input.shape[-1])
    if source_features < 1:
        raise ValueError("ConvRot NVFP4 requires a nonempty feature dimension")
    activation_width = input_activation_width(activation_fn)
    if source_features % activation_width:
        raise ValueError("ConvRot NVFP4 input activation requires equal feature partitions")
    input_features = source_features // activation_width
    rows = int(input.numel() // source_features)
    if rows < 1 or input_features % group_size or input_features % _NVFP4_BLOCK_SIZE:
        raise ValueError(
            "ConvRot NVFP4 requires nonempty rows divisible by the rotation and FP4 blocks"
        )
    if input.device.type != "cuda":
        raise ValueError("ConvRot NVFP4 currently requires CUDA")
    target = AcceleratorTarget.from_device(input.device)
    if not target.is_cuda_capability(12, 0):
        raise ValueError("ConvRot NVFP4 requires exact NVIDIA SM120")
    return input, rows, input_features, target


def _validate_per_tensor_scale(per_tensor_scale: torch.Tensor, device: torch.device) -> None:
    if (
        per_tensor_scale.shape != ()
        or per_tensor_scale.dtype is not torch.float32
        or per_tensor_scale.device != device
        or not per_tensor_scale.is_contiguous()
    ):
        raise ValueError("ConvRot NVFP4 scale must be a contiguous FP32 scalar on the input device")


def _launch_amax(
    contiguous_input: torch.Tensor,
    rows: int,
    input_features: int,
    group_size: int,
    target: AcceleratorTarget,
    activation_fn: str | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    elements = rows * input_features
    partial_count = int(triton.cdiv(elements, _AMAX_BLOCK_SIZE * _AMAX_TILES_PER_PROGRAM))
    partial = torch.empty(partial_count, device=contiguous_input.device, dtype=torch.float32)
    with device_context(contiguous_input.device):
        _rotated_amax_kernel[(partial_count,)](
            contiguous_input,
            partial,
            elements,
            features=input_features,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=activation_fn,
            accelerator_backend=target.backend,
            block_size=_AMAX_BLOCK_SIZE,
            tiles_per_program=_AMAX_TILES_PER_PROGRAM,
            num_warps=_AMAX_NUM_WARPS,
        )
        return nvfp4_dynamic_scale(partial, out=out)


def _launch_quantize(
    contiguous_input: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    rows: int,
    input_features: int,
    group_size: int,
    target: AcceleratorTarget,
    activation_fn: str | None,
    high_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    elements = rows * input_features
    with device_context(contiguous_input.device):
        _rotate_quantize_kernel[(triton.cdiv(elements, _QUANTIZE_BLOCK_SIZE),)](
            contiguous_input,
            per_tensor_scale,
            qdata,
            scale,
            elements,
            features=input_features,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            scale_column_blocks=(input_features + nvfp4_layout.SCALE_COLUMN_TILE - 1)
            // nvfp4_layout.SCALE_COLUMN_TILE,
            activation_fn=activation_fn,
            accelerator_backend=target.backend,
            block_size=_QUANTIZE_BLOCK_SIZE,
            high_first=high_first,
            num_warps=_QUANTIZE_NUM_WARPS,
        )
        return qdata, scale


def dynamic_scale(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    *,
    activation_fn: str | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate the exact dynamic scale after optional activation and grouped rotation."""
    validated_input, rows, input_features, target = _validate_input(
        input, group_size, activation_fn
    )
    if out is not None and (
        out.shape != ()
        or out.dtype is not torch.float32
        or out.device != validated_input.device
        or not out.is_contiguous()
    ):
        raise ValueError("dynamic NVFP4 scale output must be a contiguous FP32 scalar")
    return _launch_amax(
        validated_input.contiguous(),
        rows,
        input_features,
        group_size,
        target,
        activation_fn,
        out=out,
    )


def _prepare_static_checked(
    validated_input: _ValidatedInput,
    per_tensor_scale: torch.Tensor,
    group_size: int,
    out: tuple[torch.Tensor, torch.Tensor] | None,
    activation_fn: str | None,
    high_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_input, rows, input_features, target = validated_input
    _validate_per_tensor_scale(per_tensor_scale, raw_input.device)
    qdata, scale = prepare_activation_storage(raw_input, rows, input_features, out)
    return _launch_quantize(
        raw_input.contiguous(),
        per_tensor_scale,
        qdata,
        scale,
        rows,
        input_features,
        group_size,
        target,
        activation_fn,
        high_first,
    )


def prepare_static(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    group_size: int,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then rotate and pack using a supplied scale."""
    validated_input = _validate_input(input, group_size, activation_fn)
    qdata, scale = _prepare_static_checked(
        validated_input,
        per_tensor_scale,
        group_size,
        None,
        activation_fn,
        high_first,
    )
    return qdata, scale, per_tensor_scale.clone()


def prepare_static_out(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    per_tensor_scale: torch.Tensor,
    group_size: int,
    out: tuple[torch.Tensor, torch.Tensor],
    high_first: bool = False,
    *,
    activation_fn: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally activate, rotate, and pack into reusable caller-owned NVFP4 storage."""
    validated_input = _validate_input(input, group_size, activation_fn)
    return _prepare_static_checked(
        validated_input,
        per_tensor_scale,
        group_size,
        out,
        activation_fn,
        high_first,
    )


def prepare_dynamic(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then prepare exact dynamic ConvRot NVFP4 storage."""
    raw_input, rows, input_features, target = _validate_input(input, group_size, activation_fn)
    qdata, scale = prepare_activation_storage(
        raw_input,
        rows,
        input_features,
        out,
        initialize_padding=False,
    )
    contiguous_input = raw_input.contiguous()
    per_tensor_scale = _launch_amax(
        contiguous_input,
        rows,
        input_features,
        group_size,
        target,
        activation_fn,
    )
    if nvfp4_layout.has_scale_padding(rows, input_features):
        scale.zero_()
    _launch_quantize(
        contiguous_input,
        per_tensor_scale,
        qdata,
        scale,
        rows,
        input_features,
        group_size,
        target,
        activation_fn,
        high_first,
    )
    return qdata, scale, per_tensor_scale


__all__ = [
    "dynamic_scale",
    "prepare_dynamic",
    "prepare_static",
    "prepare_static_out",
]
