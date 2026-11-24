"""Decode trellis-coded storage directly into ConvRot INT8 storage."""

from __future__ import annotations

import torch

from piper_kernels._triton import runtime
from piper_kernels.trellis._layout import TrellisLayout, validate_block_size


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


def _resolve_block_size(
    block_size: int | None,
    blocks_per_row: int,
    out: tuple[torch.Tensor, torch.Tensor] | None,
) -> int:
    if block_size is not None:
        return block_size
    if out is None or blocks_per_row == 0 or out[0].shape[-1] % blocks_per_row:
        raise ValueError("ConvRot INT8 trellis output storage is incompatible")
    return out[0].shape[-1] // blocks_per_row


def _validate_inputs(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int | None,
    row_widths: torch.Tensor | None,
    out: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[int, int, int, torch.Tensor, TrellisLayout | None]:
    device = bitstreams.device
    layout: TrellisLayout | None = None
    if row_widths is None:
        if bitstreams.dtype is not torch.uint8 or bitstreams.ndim != 3:
            raise ValueError(
                "ConvRot INT8 trellis bitstreams must be a 3-D uint8 tensor "
                f"[out, blocks, bytes], got {bitstreams.dtype} {tuple(bitstreams.shape)}"
            )
        rows, blocks_per_row, num_bytes = bitstreams.shape
        resolved_block_size = _resolve_block_size(block_size, blocks_per_row, out)
        layout = TrellisLayout.from_storage(resolved_block_size, num_bytes)
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
    else:
        if bitstreams.dtype is not torch.uint8 or bitstreams.ndim != 1:
            raise ValueError(
                "ConvRot INT8 variable-row trellis bitstreams must be a 1-D uint8 tensor "
                f"[total_bytes], got {bitstreams.dtype} {tuple(bitstreams.shape)}"
            )
        if (
            block_scales.dtype is not torch.float8_e4m3fn
            or block_scales.ndim != 2
            or block_scales.device != device
        ):
            raise ValueError(
                "ConvRot INT8 trellis block_scales must be a 2-D float8_e4m3fn tensor "
                f"[out, blocks] on {device}, got {block_scales.dtype} "
                f"{tuple(block_scales.shape)} on {block_scales.device}"
            )
        rows, blocks_per_row = block_scales.shape
        if (
            row_widths.dtype is not torch.uint8
            or tuple(row_widths.shape) != (rows,)
            or row_widths.device != device
        ):
            raise ValueError(
                "ConvRot INT8 trellis row_widths must be uint8 with shape "
                f"{(rows,)} on {device}, got {row_widths.dtype} "
                f"{tuple(row_widths.shape)} on {row_widths.device}"
            )
        resolved_block_size = validate_block_size(
            _resolve_block_size(block_size, blocks_per_row, out)
        )

    row_scale = _validate_scale(scale, rows, device)
    if not runtime.supports_device(device):
        raise ValueError(
            "ConvRot INT8 trellis decoding requires a supported Triton accelerator; "
            f"decoding is unavailable on {device}"
        )
    return rows, blocks_per_row, resolved_block_size, row_scale, layout


def convert(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int | None = None,
    row_widths: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one trellis-coded matrix into ``(qdata, scale)``.

    With ``row_widths=None``, ``bitstreams`` is uint8 ``[out, blocks, bytes]`` and
    the rate comes from its trailing dimension. With ``row_widths`` (uint8 ``[out]``),
    ``bitstreams`` is 1-D uint8 holding concatenated blocks-per-row streams per row
    and ``block_scales`` is 2-D float8_e4m3fn ``[out, blocks]``. Without ``out``,
    ``qdata`` is new storage and ``scale`` is the supplied row scale as ``[rows, 1]``,
    reused when contiguous. With ``out``, both are written in place, and
    ``block_size`` defaults to the output width over the blocks per row.
    """
    rows, blocks_per_row, resolved_block_size, row_scale, layout = _validate_inputs(
        bitstreams,
        block_scales,
        scale,
        block_size=block_size,
        row_widths=row_widths,
        out=out,
    )
    device = bitstreams.device
    expected_qdata = (rows, blocks_per_row * resolved_block_size)
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
            or tuple(output_scale.shape) != (rows, 1)
            or output_scale.dtype is not torch.float32
            or output_scale.device != device
        ):
            raise ValueError("ConvRot INT8 trellis output storage is incompatible")

    if layout is None:
        assert row_widths is not None
        from piper_kernels.trellis.triton import decode_variable_rows_int8_out  # noqa: PLC0415

        decode_variable_rows_int8_out(
            bitstreams.contiguous(),
            block_scales.contiguous().view(torch.uint8),
            row_widths.contiguous(),
            resolved_block_size,
            qdata,
        )
    else:
        from piper_kernels.trellis.triton import decode_int8_out  # noqa: PLC0415

        decode_int8_out(
            bitstreams.contiguous().view(rows * blocks_per_row, layout.bytes_per_block),
            block_scales.contiguous().view(torch.uint8).view(-1),
            layout,
            qdata,
        )
    if out is not None:
        output_scale.copy_(row_scale)
    return qdata, output_scale


__all__ = ["convert"]
