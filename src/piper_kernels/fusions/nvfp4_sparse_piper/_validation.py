"""Validation shared by NVFP4 sparse-Piper projection operators."""

from __future__ import annotations

import math
from typing import cast

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import HEAD_DIM, TILE_ROWS
from piper_kernels.attention.sparse_piper_attention._block_layout import (
    validate_block_lengths as validate_k64_block_lengths,
)
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
) -> tuple[int, int]:
    """Validate one batch-one prepared NVFP4 projection and return length/heads."""
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
    if not isinstance(output_features, int) or output_features % HEAD_DIM:
        raise ValueError(f"{name} weight must map to complete packed D128 heads")
    if chunk_rows < 128 or chunk_rows % 128:
        raise ValueError(f"{name} chunk rows must be a positive multiple of 128")
    return cast(int, sequence_length), output_features // HEAD_DIM


def validate_qk_epilogue(
    input_qdata: torch.Tensor,
    sequence_length: int,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    name: str,
) -> None:
    """Validate the RMSNorm/RoPE inputs shared by Q and K epilogues."""
    rotary_dim = cos.shape[1] if cos.ndim == 2 else 0
    operands = (norm_weight, cos, sin)
    if (
        norm_weight.shape != (HEAD_DIM,)
        or norm_weight.dtype is not torch.bfloat16
        or cos.ndim != 2
        or sin.shape != cos.shape
        or cos.shape[0] != sequence_length
        or cos.dtype is not torch.float32
        or sin.dtype is not torch.float32
        or not 2 <= rotary_dim <= HEAD_DIM
        or rotary_dim % 2
        or any(operand.device != input_qdata.device for operand in operands)
        or any(not operand.is_contiguous() for operand in operands)
    ):
        raise ValueError(f"{name} requires contiguous BF16 norm and FP32 split-half RoPE")
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError(f"{name} norm epsilon must be finite and positive")


def validate_block_lengths(
    block_lengths: torch.Tensor | None,
    sequence_length: int,
    device: torch.device,
    name: str,
) -> None:
    """Validate optional trusted valid-prefix metadata for one NVFP4 projection."""
    if block_lengths is not None:
        validate_k64_block_lengths(
            block_lengths,
            sequence_length=sequence_length,
            device=device,
            context=name,
            require_contiguous=True,
            check_values=False,
        )


__all__ = ["validate_block_lengths", "validate_projection", "validate_qk_epilogue"]
