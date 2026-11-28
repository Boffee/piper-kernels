"""Owning partitions of already quantized Piper weights for checkpoint loading."""

from __future__ import annotations

from typing import NamedTuple, cast

import torch
from torch._subclasses.fake_tensor import FakeTensor

from piper_kernels.trellis._layout import (
    stored_bytes_bounds,
    validate_block_size,
)
from piper_kernels.weights._dispatch import QuantizedWeight, require_untransposed
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.trellis.tensor import (
    ConvRotTrellisTensor,
)
from piper_kernels.weights.convrot.trellis.tensor import (
    validate_storage as validate_trellis_storage,
)
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor, _layout

_GPU_GATHER_CHUNK_BYTES = 2 * 1024 * 1024
_INT32_MAX = (1 << 31) - 1


class ByteRange(NamedTuple):
    """Contiguous byte span ``[offset, offset + length)`` inside a 1-D bitstream."""

    offset: int
    length: int


def _validate_trellis_row_widths(
    row_widths: torch.Tensor,
    *,
    block_size: int | None,
    start: int | None = None,
    length: int | None = None,
) -> torch.Tensor:
    if not isinstance(row_widths, torch.Tensor):
        raise TypeError(
            f"ConvRot Trellis row_widths must be a torch.Tensor, got {type(row_widths).__name__}"
        )
    if row_widths.layout is not torch.strided:
        raise ValueError("ConvRot Trellis row_widths must use strided layout")
    if row_widths.ndim != 1 or row_widths.dtype is not torch.uint8:
        raise ValueError(
            "ConvRot Trellis row_widths must be a 1-D uint8 tensor, "
            f"got {row_widths.dtype} {tuple(row_widths.shape)}"
        )
    size = None if block_size is None else validate_block_size(block_size)
    if (
        start is not None
        and length is not None
        and (start < 0 or length <= 0 or start + length > int(row_widths.shape[0]))
    ):
        raise ValueError("row shard must be a nonempty partition within row_widths")
    if row_widths.device.type == "meta" or isinstance(row_widths, FakeTensor):
        raise NotImplementedError(
            "ConvRot Trellis variable-row byte ranges require materialized row_widths"
        )
    source_widths = (
        row_widths[: start + length]
        if size is None and start is not None and length is not None
        else row_widths
    )
    widths_cpu = source_widths.detach().to(device="cpu", dtype=torch.int64)
    if size is not None:
        min_bytes, max_bytes = stored_bytes_bounds(size)
        if widths_cpu.numel() > 0:
            low = int(widths_cpu.min().item())
            high = int(widths_cpu.max().item())
            if low < min_bytes or high > max_bytes:
                raise ValueError(
                    f"trellis variable-row widths for block size {size} must lie in "
                    f"[{min_bytes}, {max_bytes}] bytes, got [{low}, {high}]"
                )
    return widths_cpu


def _row_shard_byte_range_from_widths(
    widths_cpu: torch.Tensor,
    *,
    blocks_per_row: int,
    start: int,
    length: int,
) -> ByteRange:
    offset = int(widths_cpu[:start].sum().item()) * blocks_per_row
    byte_length = int(widths_cpu[start : start + length].sum().item()) * blocks_per_row
    return ByteRange(offset, byte_length)


def trellis_row_shard_byte_range(
    row_widths: torch.Tensor,
    *,
    blocks_per_row: int,
    start: int,
    length: int,
    block_size: int | None = None,
) -> ByteRange:
    """Compute the contiguous bitstream ``ByteRange(offset, length)`` for output rows.

    Intended for zero-materialization Direct I/O ``Shard(0)`` reads of variable-row
    2-6 bpw Trellis weights. Pass ``block_size`` to also enforce per-row legal byte
    width bounds.
    """
    if any(type(value) is not int for value in (blocks_per_row, start, length)):
        raise TypeError("blocks_per_row, start, and length must be concrete integers")
    if blocks_per_row <= 0:
        raise ValueError(f"blocks_per_row must be positive, got {blocks_per_row}")
    widths_cpu = _validate_trellis_row_widths(
        row_widths,
        block_size=block_size,
        start=start,
        length=length,
    )
    return _row_shard_byte_range_from_widths(
        widths_cpu,
        blocks_per_row=blocks_per_row,
        start=start,
        length=length,
    )


def _validate_partition(weight: QuantizedWeight, dim: int, start: int, length: int) -> int:
    if not isinstance(weight, (ConvRotInt8Tensor, PiperNVFP4Tensor, ConvRotTrellisTensor)):
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
    names, _ = weight.__tensor_flatten__()
    if any(not getattr(weight, name).is_contiguous() for name in names):
        raise NotImplementedError("quantized weight sharding requires contiguous storage")
    if dim == 1:
        group_size = getattr(weight, "group_size", 1)
        if start % group_size or length % group_size:
            raise ValueError(f"input-channel shards must align to rotation group size {group_size}")
    return dim


def _nvfp4_scales(weight: PiperNVFP4Tensor, dim: int, start: int, length: int) -> torch.Tensor:
    rows, features = weight.shape
    canonical_scale = _layout.validate_nvfp4_weight(
        weight,
        allowed_ndims=(2,),
        error_type=NotImplementedError,
        prefix="NVFP4 sharding",
    )
    if dim == 1 and (start % weight.block_size or length % weight.block_size):
        raise ValueError("NVFP4 input-channel shards must align to 16-value quantization blocks")

    # Rearrange bytes so this also works on CPUs with limited FP8 operations.
    # Only block scales are unpacked; the FP4 weight payload stays quantized.
    scale_bytes = canonical_scale.view(torch.uint8)
    plain = (
        _layout.from_blocked(scale_bytes, rows, features // weight.block_size)
        if weight.is_swizzled_scales
        else scale_bytes
    )
    divisor = weight.block_size if dim == 1 else 1
    selected = plain.narrow(dim, start // divisor, length // divisor)
    if weight.is_swizzled_scales:
        shard_rows = length if dim == 0 else rows
        shard_features = length if dim == 1 else features
        selected = _layout.to_blocked(selected).view(
            _layout.scale_shape(shard_rows, shard_features)
        )
    return selected.view(weight.scale.dtype)


def _gather_variable_row_blocks(
    bitstreams: torch.Tensor,
    widths_cpu: torch.Tensor,
    *,
    blocks_per_row: int,
    block_start: int,
    block_length: int,
) -> torch.Tensor:
    """Gather a contiguous block slice per variable-width row into an owned 1-D buffer."""
    rows = int(widths_cpu.shape[0])
    if rows == 0:
        return bitstreams.new_empty((0,))
    width_sum = int(widths_cpu.sum().item())
    total_required = width_sum * blocks_per_row
    total_out_bytes = width_sum * block_length
    if total_out_bytes == 0:
        return bitstreams.new_empty((0,))
    if block_start == 0 and block_length == blocks_per_row:
        return bitstreams.narrow(0, 0, total_required).clone(memory_format=torch.contiguous_format)
    first_width = int(widths_cpu[0].item())
    if bool((widths_cpu == first_width).all()):
        return (
            bitstreams.narrow(0, 0, total_required)
            .view(rows, blocks_per_row, first_width)
            .narrow(1, block_start, block_length)
            .clone(memory_format=torch.contiguous_format)
            .reshape(-1)
        )
    if bitstreams.device.type == "cpu":
        out = torch.empty((total_out_bytes,), dtype=torch.uint8, device="cpu")
        dst_pos = 0
        row_src_base = 0
        for width in widths_cpu.tolist():
            row_in_bytes = width * blocks_per_row
            row_out_bytes = width * block_length
            if row_out_bytes > 0:
                src_start = row_src_base + width * block_start
                out.narrow(0, dst_pos, row_out_bytes).copy_(
                    bitstreams.narrow(0, src_start, row_out_bytes)
                )
            row_src_base += row_in_bytes
            dst_pos += row_out_bytes
        return out

    device = bitstreams.device
    index_dtype = torch.int32 if total_required <= _INT32_MAX else torch.int64
    in_row_bytes_cpu = widths_cpu * blocks_per_row
    out_row_bytes_cpu = widths_cpu * block_length
    src_row_starts_cpu = in_row_bytes_cpu.cumsum(0) - in_row_bytes_cpu + widths_cpu * block_start
    dst_row_starts_cpu = out_row_bytes_cpu.cumsum(0) - out_row_bytes_cpu
    dst_row_ends_cpu = dst_row_starts_cpu + out_row_bytes_cpu
    delta_cpu = src_row_starts_cpu - dst_row_starts_cpu

    out = torch.empty((total_out_bytes,), dtype=torch.uint8, device=device)
    row_start = 0
    while row_start < rows:
        dst_base = int(dst_row_starts_cpu[row_start].item())
        target_end = min(total_out_bytes, dst_base + _GPU_GATHER_CHUNK_BYTES)
        row_end = int(torch.searchsorted(dst_row_ends_cpu, target_end, right=True).item())
        row_end = max(row_start + 1, min(row_end, rows))
        chunk_bytes = int(dst_row_ends_cpu[row_end - 1].item()) - dst_base
        if chunk_bytes > 0:
            chunk_row_lens = out_row_bytes_cpu[row_start:row_end].to(
                device=device, dtype=index_dtype
            )
            chunk_delta = (delta_cpu[row_start:row_end] + dst_base).to(
                device=device, dtype=index_dtype
            )
            src_idx = torch.repeat_interleave(
                chunk_delta, chunk_row_lens, output_size=chunk_bytes
            ) + torch.arange(chunk_bytes, device=device, dtype=index_dtype)
            out.narrow(0, dst_base, chunk_bytes).copy_(bitstreams.index_select(0, src_idx))
        row_start = row_end
    return out


def _trellis_shard(
    weight: ConvRotTrellisTensor,
    dim: int,
    start: int,
    length: int,
) -> ConvRotTrellisTensor:
    validate_trellis_storage(
        weight.bitstreams,
        weight.block_scales,
        weight.scale,
        weight.block_size,
        weight.group_size,
        weight.dtype,
        row_widths=weight.row_widths,
        require_contiguous=True,
    )
    if dim == 1 and (start % weight.block_size or length % weight.block_size):
        raise ValueError(
            f"Trellis input-channel shards must align to block size {weight.block_size}"
        )
    _, metadata = weight.__tensor_flatten__()
    block_start = start // weight.block_size
    block_length = length // weight.block_size

    if weight.row_widths is None:
        if dim == 0:
            bitstreams = weight.bitstreams.narrow(0, start, length)
            block_scales = weight.block_scales.narrow(0, start, length)
            scale = weight.scale.narrow(0, start, length)
        else:
            bitstreams = weight.bitstreams.narrow(1, block_start, block_length)
            block_scales = weight.block_scales.narrow(1, block_start, block_length)
            scale = weight.scale
        tensors = {
            name: value.clone(memory_format=torch.contiguous_format)
            for name, value in (
                ("bitstreams", bitstreams),
                ("block_scales", block_scales),
                ("scale", scale),
            )
        }
        return type(weight).__tensor_unflatten__(tensors, metadata, None, None)

    blocks_per_row = int(weight.block_scales.shape[1])
    if dim == 0:
        widths_cpu = _validate_trellis_row_widths(
            weight.row_widths,
            block_size=None,
            start=start,
            length=length,
        )
        byte_range = _row_shard_byte_range_from_widths(
            widths_cpu,
            blocks_per_row=blocks_per_row,
            start=start,
            length=length,
        )
        required_end = byte_range.offset + byte_range.length
        if required_end > weight.bitstreams.numel():
            raise ValueError(
                "ConvRot Trellis variable-row bitstreams is shorter than row_widths requires: "
                f"need {required_end} bytes, got {weight.bitstreams.numel()}"
            )
        bitstreams = weight.bitstreams.narrow(0, byte_range.offset, byte_range.length).clone(
            memory_format=torch.contiguous_format
        )
        block_scales = weight.block_scales.narrow(0, start, length)
        scale = weight.scale.narrow(0, start, length)
        row_widths = weight.row_widths.narrow(0, start, length)
    else:
        widths_cpu = _validate_trellis_row_widths(weight.row_widths, block_size=None)
        total_required = int(widths_cpu.sum().item()) * blocks_per_row
        if total_required > weight.bitstreams.numel():
            raise ValueError(
                "ConvRot Trellis variable-row bitstreams is shorter than row_widths requires: "
                f"need {total_required} bytes, got {weight.bitstreams.numel()}"
            )
        bitstreams = _gather_variable_row_blocks(
            weight.bitstreams,
            widths_cpu,
            blocks_per_row=blocks_per_row,
            block_start=block_start,
            block_length=block_length,
        )
        block_scales = weight.block_scales.narrow(1, block_start, block_length)
        scale = weight.scale
        row_widths = weight.row_widths
    tensors = {
        "bitstreams": bitstreams,
        "block_scales": block_scales.clone(memory_format=torch.contiguous_format),
        "scale": scale.clone(memory_format=torch.contiguous_format),
        "row_widths": row_widths.clone(memory_format=torch.contiguous_format),
    }
    return type(weight).__tensor_unflatten__(tensors, metadata, None, None)


def shard_quantized_weight[Weight: QuantizedWeight](
    weight: Weight, *, dim: int, start: int, length: int
) -> Weight:
    """Copy one aligned row or input-channel partition without requantization.

    Accepts contiguous, untransposed 2-D ``ConvRotInt8Tensor``,
    ``PiperNVFP4Tensor``, ``ConvRotNVFP4Tensor``, and ``ConvRotTrellisTensor``
    (uniform and variable-row ``row_widths``) weights on CPU or CUDA (variable-row
    Trellis requires materialized ``row_widths``, not ``meta``). ``dim`` is 0
    (output rows) or 1 (input channels), with negative dimensions also accepted.
    ``start`` must be nonnegative and ``length`` positive. Input partitions must
    align to NVFP4's 16-value blocks, Trellis's ``block_size``, and ConvRot groups.
    NVFP4 accepts ordinary or swizzled FP8 scales, stored flat or in canonical
    2-D form, and optional scalar FP32 global scales.

    The result owns its tensor storage and retains the concrete wrapper,
    scales, nibble order, rotation, rate layout, and activation-quantization
    configuration. NVFP4 block scales are repacked with fresh padding, and
    variable-row Trellis ``dim=1`` bitstreams are repacked from each row's
    selected blocks. No high-precision weight is materialized. This is a copy
    operation; ordinary slice/narrow views remain unsupported.

    For DTensor integration, install the result with ``DTensor.from_local``
    before applying a matching ``ColwiseParallel`` or ``RowwiseParallel``
    plan. See docs/weights.md#shard-a-quantized-weight for device movement
    and global shape metadata.
    """
    dim = _validate_partition(weight, dim, start, length)
    if isinstance(weight, ConvRotTrellisTensor):
        return cast(Weight, _trellis_shard(weight, dim, start, length))
    if isinstance(weight, ConvRotInt8Tensor):
        scale = weight.scale.narrow(0, start, length) if dim == 0 else weight.scale
        divisor = 1
    elif isinstance(weight, PiperNVFP4Tensor):
        scale = _nvfp4_scales(weight, dim, start, length)
        divisor = 2 if dim == 1 else 1
    else:
        raise TypeError("shard_quantized_weight requires a Piper quantized weight")
    qdata = weight.qdata.narrow(dim, start // divisor, length // divisor)
    names, metadata = weight.__tensor_flatten__()
    tensors = {name: getattr(weight, name) for name in names}
    tensors.update(qdata=qdata, scale=scale)
    # Copy every tensor, including global scales, at the ownership boundary.
    tensors = {
        name: value.clone(memory_format=torch.contiguous_format) for name, value in tensors.items()
    }
    return cast(Weight, type(weight).__tensor_unflatten__(tensors, metadata, None, None))


__all__ = ["ByteRange", "shard_quantized_weight", "trellis_row_shard_byte_range"]
