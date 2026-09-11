"""Owning partitions of already quantized Piper weights for checkpoint loading."""

from __future__ import annotations

from typing import cast

import torch
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked

from piper_kernels.weights._dispatch import QuantizedWeight
from piper_kernels.weights._views import require_untransposed
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor, _layout


def _validate_partition(weight: QuantizedWeight, dim: int, start: int, length: int) -> int:
    if not isinstance(weight, (ConvRotInt8Tensor, PiperNVFP4Tensor)):
        raise TypeError("shard_quantized_weight requires a Piper quantized weight")
    if weight.ndim != 2:
        raise NotImplementedError("quantized weight sharding requires a 2-D weight")
    require_untransposed(weight, "sharding")
    if any(type(value) is not int for value in (dim, start, length)):
        raise TypeError("shard dim, start, and length must be concrete integers")
    if dim not in (-2, -1, 0, 1):
        raise IndexError("shard dimension out of range for a 2-D weight")
    dim %= 2
    if start < 0 or length <= 0 or start + length > weight.shape[dim]:
        raise ValueError("shard must be a nonempty partition within the weight")
    if not weight.qdata.is_contiguous() or not weight.scale.is_contiguous():
        raise NotImplementedError("quantized weight sharding requires contiguous storage")
    if dim == 1:
        group_size = getattr(weight, "group_size", 1)
        if start % group_size or length % group_size:
            raise ValueError(f"input-channel shards must align to rotation group size {group_size}")
    return dim


def _nvfp4_scales(weight: PiperNVFP4Tensor, dim: int, start: int, length: int) -> torch.Tensor:
    rows, features = weight.shape
    if weight.block_size != _layout.BLOCK_SIZE or features % _layout.BLOCK_SIZE:
        raise NotImplementedError("NVFP4 sharding requires complete 16-value quantization blocks")
    if weight.qdata.dtype is not torch.uint8 or weight.scale.dtype is not torch.float8_e4m3fn:
        raise NotImplementedError("NVFP4 sharding requires packed UINT8 data and FP8 E4M3 scales")
    if dim == 1 and (start % weight.block_size or length % weight.block_size):
        raise ValueError("NVFP4 input-channel shards must align to 16-value quantization blocks")
    for scale in (weight.per_tensor_scale, weight.act_per_tensor_scale):
        if scale is not None and (
            scale.ndim != 0 or scale.dtype is not torch.float32 or scale.device != weight.device
        ):
            raise NotImplementedError(
                "NVFP4 sharding requires scalar FP32 global scales on the weight device"
            )
    scale_shape = (
        _layout.scale_shape(rows, features)
        if weight.is_swizzled_scales
        else (rows, features // weight.block_size)
    )
    if (
        tuple(weight.scale.shape) not in (scale_shape, (scale_shape[0] * scale_shape[1],))
        or weight.scale.device != weight.device
    ):
        raise NotImplementedError("NVFP4 sharding requires canonical or flat block-scale storage")

    # Rearrange bytes so this also works on CPUs with limited FP8 operations.
    # Only block scales are unpacked; the FP4 weight payload stays quantized.
    scale_bytes = weight.scale.view(torch.uint8)
    plain = (
        from_blocked(scale_bytes, rows, features // weight.block_size)
        if weight.is_swizzled_scales
        else scale_bytes.view(rows, features // weight.block_size)
    )
    divisor = weight.block_size if dim == 1 else 1
    selected = plain.narrow(dim, start // divisor, length // divisor)
    if weight.is_swizzled_scales:
        shard_rows = length if dim == 0 else rows
        shard_features = length if dim == 1 else features
        selected = to_blocked(selected).view(_layout.scale_shape(shard_rows, shard_features))
    return selected.view(weight.scale.dtype)


def shard_quantized_weight[Weight: QuantizedWeight](
    weight: Weight, *, dim: int, start: int, length: int
) -> Weight:
    """Copy one aligned row or input-channel partition without requantization.

    Accepts contiguous, untransposed 2-D ``ConvRotInt8Tensor``,
    ``PiperNVFP4Tensor``, and ``ConvRotNVFP4Tensor`` weights on CPU or CUDA.
    ``dim`` is 0 (output rows) or 1 (input channels), with negative dimensions
    also accepted. ``start`` must be nonnegative and ``length`` positive.
    Input partitions must align to NVFP4's 16-value blocks and ConvRot groups.
    NVFP4 accepts ordinary or swizzled FP8 scales, stored flat or in canonical
    2-D form, and optional scalar FP32 global scales.

    The result owns its tensor storage and retains the concrete wrapper,
    scales, nibble order, rotation, and activation-quantization configuration.
    NVFP4 block scales are repacked with fresh padding. No high-precision
    weight is materialized. This is a copy operation; ordinary slice/narrow
    views remain unsupported.

    For DTensor integration, install the result with ``DTensor.from_local``
    before applying a matching ``ColwiseParallel`` or ``RowwiseParallel``
    plan. See the README example for device movement and global shape metadata.
    """
    dim = _validate_partition(weight, dim, start, length)
    if isinstance(weight, ConvRotInt8Tensor):
        scale = weight.scale.narrow(0, start, length) if dim == 0 else weight.scale
        divisor = 1
    else:
        scale = _nvfp4_scales(weight, dim, start, length)
        divisor = 2 if dim == 1 else 1
    qdata = weight.qdata.narrow(dim, start // divisor, length // divisor)
    names, metadata = weight.__tensor_flatten__()
    tensors = {name: getattr(weight, name) for name in names}
    tensors.update(qdata=qdata, scale=scale)
    # Copy every tensor, including global scales, at the ownership boundary.
    tensors = {
        name: value.clone(memory_format=torch.contiguous_format) for name, value in tensors.items()
    }
    return cast(Weight, type(weight).__tensor_unflatten__(tensors, metadata, None, None))


__all__ = ["shard_quantized_weight"]
