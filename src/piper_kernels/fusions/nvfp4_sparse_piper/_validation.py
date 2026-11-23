"""Validation shared by NVFP4 sparse-Piper projection operators."""

from __future__ import annotations

from typing import cast

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS, TILE_ROWS
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation


def validate_projection(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    chunk_rows: int,
    name: str,
    head_dim: int = 128,
) -> tuple[int, int]:
    """Validate one batch-one prepared NVFP4 projection and return length/heads."""
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"{name} requires head_dim=64 or 128")
    shape = nvfp4_validation.validate_prepared_linear(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        torch.bfloat16,
        name,
    )
    sequence_length = shape.rows
    output_features = shape.output_features
    if isinstance(sequence_length, int) and sequence_length < TILE_ROWS:
        raise ValueError(f"{name} input must contain at least K64 rows")
    if not isinstance(output_features, int) or output_features % head_dim:
        raise ValueError(f"{name} weight must map to complete packed D64/D128 heads")
    if chunk_rows < 128 or chunk_rows % 128:
        raise ValueError(f"{name} chunk rows must be a positive multiple of 128")
    return cast(int, sequence_length), output_features // head_dim


__all__ = ["validate_projection"]
