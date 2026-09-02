"""Policy-independent coarse-attention residuals for Sparse Piper."""

from __future__ import annotations

import torch
from torch.nn import functional

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS


def mean_pool_block_values(
    value: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return padding-aware FP32 K64 value means as ``[B,H,blocks,D]``.

    Without ``block_lengths``, value uses compact logical sequence storage and
    only its final block may be partial. Supplying lengths selects internally
    padded storage: every physical block contains its valid rows first and its
    corresponding INT32 length lies in ``[1, 64]``.
    """
    if value.ndim != 4:
        raise ValueError("coarse attention V must use [batch,sequence,heads,features]")
    if not value.is_floating_point():
        raise TypeError("coarse attention V must be floating-point")
    _batch, sequence_length, _heads, _features = value.shape
    if sequence_length < 1:
        raise ValueError("coarse attention V must contain at least one sequence row")

    if block_lengths is None:
        block_count = (sequence_length + TILE_ROWS - 1) // TILE_ROWS
        storage_length = block_count * TILE_ROWS
        padded = functional.pad(value, (0, 0, 0, 0, 0, storage_length - sequence_length))
        lengths = torch.minimum(
            torch.full(
                (block_count,),
                TILE_ROWS,
                dtype=torch.int32,
                device=value.device,
            ),
            sequence_length
            - torch.arange(block_count, dtype=torch.int32, device=value.device) * TILE_ROWS,
        )
        blocks = padded.reshape(
            value.shape[0],
            block_count,
            TILE_ROWS,
            value.shape[2],
            value.shape[3],
        )
    else:
        if (
            block_lengths.ndim != 1
            or block_lengths.numel() < 1
            or block_lengths.dtype is not torch.int32
        ):
            raise ValueError("coarse attention block lengths must be a nonempty INT32 vector")
        if block_lengths.device != value.device:
            raise ValueError("coarse attention V and block lengths must share a device")
        block_count = block_lengths.numel()
        if sequence_length != block_count * TILE_ROWS:
            raise ValueError("coarse attention padded V must contain one physical K64 per length")
        lengths = block_lengths
        blocks = value.reshape(
            value.shape[0],
            block_count,
            TILE_ROWS,
            value.shape[2],
            value.shape[3],
        )
        valid_rows = torch.arange(TILE_ROWS, device=value.device)[None, :] < lengths[:, None]
        blocks = torch.where(valid_rows[None, :, :, None, None], blocks, 0)

    pooled = blocks.sum(dim=2, dtype=torch.float32)
    pooled /= lengths.reshape(1, block_count, 1, 1)
    return pooled.permute(0, 2, 1, 3).contiguous()


def coarse_attention(
    block_scores: torch.Tensor,
    pooled_value: torch.Tensor,
) -> torch.Tensor:
    """Apply dense FP32 attention to caller-scaled block logits and pooled V."""
    if block_scores.ndim != 4 or pooled_value.ndim != 4:
        raise ValueError("coarse scores and pooled V must use rank-four block tensors")
    if block_scores.dtype is not torch.float32 or pooled_value.dtype is not torch.float32:
        raise ValueError("coarse scores and pooled V must use FP32")
    if block_scores.device != pooled_value.device:
        raise ValueError("coarse scores and pooled V must share a device")
    if block_scores.shape[:2] != pooled_value.shape[:2]:
        raise ValueError("coarse scores and pooled V batch/head dimensions must match")
    if block_scores.shape[-1] != pooled_value.shape[2]:
        raise ValueError("coarse score key blocks must match pooled V blocks")
    if block_scores.shape[-2] < 1 or block_scores.shape[-1] < 1:
        raise ValueError("coarse attention requires query and key blocks")
    return torch.matmul(torch.softmax(block_scores, dim=-1), pooled_value)


def apply_coarse_attention_residual(
    fine_output: torch.Tensor,
    coarse_output: torch.Tensor,
    compression_gate: torch.Tensor,
) -> torch.Tensor:
    """Expand block outputs, apply the token gate, and add to fine attention."""
    if fine_output.ndim != 4 or coarse_output.ndim != 4:
        raise ValueError("fine and coarse attention outputs must use rank-four tensors")
    if compression_gate.shape != fine_output.shape:
        raise ValueError("coarse attention compression gate must match fine output")
    if not fine_output.is_floating_point() or not compression_gate.is_floating_point():
        raise TypeError("fine output and compression gate must be floating-point")
    if compression_gate.dtype is not fine_output.dtype:
        raise ValueError("fine output and compression gate must share a dtype")
    if coarse_output.dtype is not torch.float32:
        raise ValueError("coarse attention output must use FP32")
    if any(tensor.device != fine_output.device for tensor in (coarse_output, compression_gate)):
        raise ValueError("fine output, coarse output, and compression gate must share a device")
    batch, sequence_length, heads, features = fine_output.shape
    query_blocks = (sequence_length + TILE_ROWS - 1) // TILE_ROWS
    if coarse_output.shape != (batch, heads, query_blocks, features):
        raise ValueError("coarse output must contain one K64 result per fine query block")

    expanded = coarse_output.permute(0, 2, 1, 3).repeat_interleave(TILE_ROWS, dim=1)
    expanded = expanded[:, :sequence_length]
    output = fine_output.float() + compression_gate.float() * expanded
    return output.to(fine_output.dtype)


def coarse_attention_residual(
    fine_output: torch.Tensor,
    block_scores: torch.Tensor,
    pooled_value: torch.Tensor,
    compression_gate: torch.Tensor,
) -> torch.Tensor:
    """Add a policy-independent coarse-attention branch to fine attention."""
    return apply_coarse_attention_residual(
        fine_output,
        coarse_attention(block_scores, pooled_value),
        compression_gate,
    )


__all__ = [
    "apply_coarse_attention_residual",
    "coarse_attention",
    "coarse_attention_residual",
    "mean_pool_block_values",
]
