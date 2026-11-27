"""Decode trellis-coded storage directly into ConvRot INT8 storage."""

from __future__ import annotations

import torch

from piper_kernels._triton import runtime
from piper_kernels.trellis._layout import TrellisLayout
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.convrot.int8._quantization import _SUPPORTED_LOGICAL_DTYPES
from piper_kernels.weights.convrot.trellis.tensor import validate_trellis_tensors


def _validate_inputs(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int | None,
    group_size: int | None,
    logical_dtype: torch.dtype | None,
    row_widths: torch.Tensor | None,
    out: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[int, int, int, torch.Tensor, TrellisLayout | None]:
    if group_size is not None:
        validate_group_size(group_size)
    if logical_dtype is not None and logical_dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            f"ConvRot logical dtype must be float16, bfloat16, or float32, got {logical_dtype}"
        )
    device = bitstreams.device
    rows, blocks_per_row, resolved_block_size, layout, row_scale = validate_trellis_tensors(
        bitstreams,
        block_scales,
        scale,
        block_size,
        row_widths=row_widths,
        inferred_width=None if out is None else out[0].shape[-1],
        require_contiguous=False,
        allow_1d_scale=True,
        prefix="ConvRot INT8 trellis",
    )
    in_features = blocks_per_row * resolved_block_size
    if group_size is not None and in_features % group_size:
        raise ValueError(
            f"ConvRot in_features {in_features} is not divisible by group size {group_size}"
        )
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
    group_size: int | None = None,
    logical_dtype: torch.dtype | None = None,
    row_widths: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one trellis-coded matrix into ``(qdata, scale)``.

    With ``row_widths=None``, ``bitstreams`` is uint8 ``[out, blocks, bytes]`` and
    the rate comes from its trailing dimension (including 2 wrap guard bytes). With
    ``row_widths`` (uint8 ``[out]``), ``bitstreams`` is 1-D uint8 holding concatenated
    blocks-per-row streams per row (each block storing ``row_widths[r]`` bytes including
    2 wrap guard bytes) and ``block_scales`` is 2-D float8_e4m3fn ``[out, blocks]``.
    Without ``out``, ``qdata`` is new storage and ``scale`` is the supplied row scale
    reused when contiguous. With ``out``, both are written in place, and
    ``block_size`` defaults to the output width over the blocks per row.
    """
    rows, blocks_per_row, resolved_block_size, row_scale, layout = _validate_inputs(
        bitstreams,
        block_scales,
        scale,
        block_size=block_size,
        group_size=group_size,
        logical_dtype=logical_dtype,
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
