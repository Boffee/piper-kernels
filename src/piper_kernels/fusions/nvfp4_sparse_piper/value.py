"""Chunked NVFP4 value projection and sparse-Piper preparation."""

from __future__ import annotations

import torch

from piper_kernels.fusions.attention import _sparse_output as output_common
from piper_kernels.linear.nvfp4._chunking import (
    DEFAULT_CHUNK_ROWS,
    PreparedProjection,
    run_chunked_projection,
)

from . import _epilogue
from ._validation import validate_block_lengths, validate_projection


@torch.library.custom_op("piper_kernels::nvfp4_sparse_piper_project_value", mutates_args=())
def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length, heads = validate_projection(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        chunk_rows,
        "V projection",
        head_dim,
    )
    validate_block_lengths(
        block_lengths,
        sequence_length,
        input_qdata.device,
        "V projection",
    )
    value, value_scale, value_mean = output_common.allocate_value_storage(
        input_qdata,
        1,
        heads,
        sequence_length,
        head_dim,
    )
    projection = PreparedProjection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
    )

    def consume(chunk: torch.Tensor, start: int) -> None:
        _epilogue.launch_value(
            chunk,
            input_per_tensor_scale,
            weight_per_tensor_scale,
            bias,
            value,
            value_scale,
            value_mean,
            block_lengths,
            start,
            sequence_length,
        )

    consumer_tensors = [input_per_tensor_scale, value, value_scale, value_mean]
    consumer_tensors.extend(
        operand for operand in (weight_per_tensor_scale, bias) if operand is not None
    )
    if block_lengths is not None:
        consumer_tensors.append(block_lengths)
    run_chunked_projection(
        projection,
        chunk_rows,
        consume,
        consumer_tensors,
    )
    return value, value_scale, value_mean


@project_value.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _project_value_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _chunk_rows: int = DEFAULT_CHUNK_ROWS,
    _block_lengths: torch.Tensor | None = None,
    *,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = input_qdata.shape[0]
    heads = weight_qdata.shape[0] // head_dim
    return output_common.allocate_value_storage(
        input_qdata,
        1,
        heads,
        sequence_length,
        head_dim,
    )


__all__ = ["project_value"]
