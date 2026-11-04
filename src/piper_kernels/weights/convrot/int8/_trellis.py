"""Decode trellis-coded storage directly into ConvRot INT8 storage."""

from __future__ import annotations

import torch

from piper_kernels._triton import runtime
from piper_kernels.trellis._layout import TrellisLayout


def _validate_scale(scale: torch.Tensor, rows: int, device: torch.device) -> torch.Tensor:
    """Return the row scale as ``[rows, 1]`` after checking its metadata."""
    if tuple(scale.shape) not in ((rows,), (rows, 1)):
        raise ValueError(
            f"ConvRot INT8 trellis scale must have shape ({rows},) or ({rows}, 1), "
            f"got {tuple(scale.shape)}"
        )
    if scale.dtype is not torch.float32 or scale.device != device:
        raise ValueError(
            f"ConvRot INT8 trellis scale must be float32 on {device}, "
            f"got {scale.dtype} on {scale.device}"
        )
    return scale.view(rows, 1)


def convert(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one trellis-coded matrix into ``(qdata, scale)``.

    The rate comes from the bitstreams' bytes per block. Without ``out``,
    ``qdata`` is new storage and ``scale`` is the supplied row scale as ``[rows, 1]``,
    reused when contiguous. With ``out``, both are written in place, and
    ``block_size`` defaults to the output width over the blocks per row.
    """
    if bitstreams.dtype is not torch.uint8 or bitstreams.ndim != 3:
        raise ValueError(
            "ConvRot INT8 trellis bitstreams must be a 3-D uint8 tensor "
            f"[out, blocks, bytes], got {bitstreams.dtype} {tuple(bitstreams.shape)}"
        )
    rows, blocks_per_row, num_bytes = bitstreams.shape
    if block_size is None:
        if out is None or blocks_per_row == 0 or out[0].shape[-1] % blocks_per_row:
            raise ValueError("ConvRot INT8 trellis output storage is incompatible")
        block_size = out[0].shape[-1] // blocks_per_row
    layout = TrellisLayout.from_storage(block_size, num_bytes)
    device = bitstreams.device
    if (
        block_scales.dtype is not torch.float8_e4m3fn
        or tuple(block_scales.shape) != (rows, blocks_per_row)
        or block_scales.device != device
    ):
        raise ValueError(
            "ConvRot INT8 trellis block_scales must be float8_e4m3fn with shape "
            f"{(rows, blocks_per_row)} on {device}, got {block_scales.dtype} "
            f"{tuple(block_scales.shape)} on {block_scales.device}"
        )
    row_scale = _validate_scale(scale, rows, device)
    if not runtime.supports_device(device):
        raise ValueError(
            "ConvRot INT8 trellis decoding requires a supported Triton accelerator; "
            f"decoding is unavailable on {device}"
        )

    expected_qdata = (rows, blocks_per_row * block_size)
    if out is None:
        qdata = torch.empty(expected_qdata, device=device, dtype=torch.int8)
        output_scale = row_scale.contiguous()
    else:
        qdata, output_scale = out
        if (
            tuple(qdata.shape) != expected_qdata
            or qdata.dtype is not torch.int8
            or qdata.device != device
            or not qdata.is_contiguous()
        ):
            raise ValueError("ConvRot INT8 trellis output storage is incompatible")
        output_scale.copy_(row_scale)

    from piper_kernels.trellis.triton import decode_int8_out  # noqa: PLC0415

    decode_int8_out(
        bitstreams.contiguous().view(rows * blocks_per_row, num_bytes),
        block_scales.contiguous().view(torch.uint8).view(-1),
        layout,
        qdata,
    )
    return qdata, output_scale


__all__ = ["convert"]
