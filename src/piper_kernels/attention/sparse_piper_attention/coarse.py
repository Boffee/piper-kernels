"""Policy-independent coarse-attention residuals and compiler boundary for Sparse Piper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch

from piper_kernels._validation import validate_positive_finite_float
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from ._contracts import (
    _ROUTING_NAME_BY_MODE,
    routing_mode_from_name,
    valid_block_rows,
    validate_block_lengths,
    validate_routing_mode,
)


def _preserve_coarse_residual_in_graph(*tensors: torch.Tensor) -> bool:
    """Use the semantic boundary only when no compiled backward is required."""
    return torch.compiler.is_compiling() and not (
        torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)
    )


def validate_coarse_scale(coarse_scale: float) -> float:
    """Reject scales that do not preserve fine-route score ordering."""
    return validate_positive_finite_float(
        coarse_scale,
        "coarse attention scale must be finite and positive",
    )


def _validate_coarse_key_blocks(
    sparse_key_blocks: int,
    coarse_key_blocks: int,
    *,
    available_coarse_key_blocks: int,
) -> None:
    """Check that a fused coarse scope covers the sparse prefix routing scores."""
    if not sparse_key_blocks <= coarse_key_blocks <= available_coarse_key_blocks:
        raise ValueError(
            "coarse_key_blocks must include the sparse prefix and fit the available K64 blocks"
        )


def validate_coarse_residual_inputs(  # noqa: PLR0912
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coarse_gate: torch.Tensor,
    coarse_key_blocks: int | None,
    coarse_scale: float,
    block_lengths: torch.Tensor | None,
    *,
    routing_label: str,
) -> int:
    """Validate the shared contract and resolve the coarse K64 prefix length."""
    tensors = (query, key, value, coarse_gate)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError(
            f"{routing_label} coarse attention tensors must use [batch,sequence,heads,features]"
        )
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(f"{routing_label} coarse attention requires equal Q/K/V shapes")
    if query.shape[1] < 1 or query.shape[2] < 1:
        raise ValueError(
            f"{routing_label} coarse attention requires nonempty sequence and head dimensions"
        )
    if coarse_gate.shape != query.shape:
        raise ValueError(f"{routing_label} coarse attention gate must match Q/K/V")
    if any(tensor.device != query.device for tensor in (key, value, coarse_gate)):
        raise ValueError(f"{routing_label} coarse attention tensors must share a device")
    if any(not tensor.is_floating_point() for tensor in (query, key, value)):
        raise TypeError(f"{routing_label} coarse attention tensors must be floating-point")
    if not coarse_gate.is_floating_point() or coarse_gate.dtype is not query.dtype:
        raise ValueError(f"{routing_label} coarse attention gate must share the Q/K/V dtype")
    if (
        query.layout is not torch.strided
        or key.layout is not torch.strided
        or query.stride(-1) != 1
        or key.stride(-1) != 1
    ):
        raise ValueError(f"{routing_label} coarse attention Q/K features must be contiguous")

    validate_coarse_scale(coarse_scale)
    if block_lengths is None:
        available_coarse_key_blocks = (query.shape[1] + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    else:
        available_coarse_key_blocks = validate_block_lengths(
            block_lengths,
            sequence_length=query.shape[1],
            device=query.device,
            context=f"{routing_label} coarse attention",
        )
    resolved_coarse_key_blocks = (
        available_coarse_key_blocks if coarse_key_blocks is None else coarse_key_blocks
    )
    if isinstance(resolved_coarse_key_blocks, bool):
        raise TypeError("coarse_key_blocks must be an integer")
    if torch.compiler.is_compiling():
        torch._check(
            resolved_coarse_key_blocks >= 1,
            lambda: "coarse_key_blocks must be positive",
        )
        torch._check(
            resolved_coarse_key_blocks <= available_coarse_key_blocks,
            lambda: "coarse_key_blocks cannot exceed the available K64 blocks",
        )
    elif not isinstance(resolved_coarse_key_blocks, int):
        raise TypeError("coarse_key_blocks must be an integer")
    elif not 1 <= resolved_coarse_key_blocks <= available_coarse_key_blocks:
        raise ValueError("coarse_key_blocks must fit the available K64 blocks")
    return resolved_coarse_key_blocks


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

    if block_lengths is not None:
        validate_block_lengths(
            block_lengths,
            sequence_length=sequence_length,
            device=value.device,
            context="coarse attention",
            require_contiguous=False,
        )
    return _mean_pool_token_blocks(value, block_lengths)


def _mean_pool_token_blocks(
    sequence: torch.Tensor,
    block_lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-pool validated token-major Q, K, or V storage into FP32 blocks."""
    pooled = _mean_pool_sequence_blocks(sequence, sequence_dim=1, block_lengths=block_lengths)
    return pooled.permute(0, 2, 1, 3).contiguous()


def _mean_pool_head_major_blocks(
    sequence: torch.Tensor,
    block_lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-pool validated head-major Q or K storage into FP32 blocks."""
    return _mean_pool_sequence_blocks(
        sequence,
        sequence_dim=2,
        block_lengths=block_lengths,
    ).contiguous()


def _mean_pool_sequence_blocks(
    sequence: torch.Tensor,
    *,
    sequence_dim: int,
    block_lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Pool K64 storage along one sequence dimension without materializing padding."""
    sequence_length = sequence.shape[sequence_dim]
    if block_lengths is not None:
        block_count = block_lengths.numel()
        blocks = sequence.unflatten(sequence_dim, (block_count, _BLOCK_ROWS))
        mask_shape = [1] * blocks.ndim
        mask_shape[sequence_dim : sequence_dim + 2] = (block_count, _BLOCK_ROWS)
        blocks = torch.where(valid_block_rows(block_lengths).reshape(mask_shape), blocks, 0)
        pooled = blocks.sum(dim=sequence_dim + 1, dtype=torch.float32)
        length_shape = [1] * sequence.ndim
        length_shape[sequence_dim] = block_count
        return pooled / block_lengths.reshape(length_shape)

    full_rows = sequence_length // _BLOCK_ROWS * _BLOCK_ROWS
    summaries = []
    if full_rows:
        blocks = sequence.narrow(sequence_dim, 0, full_rows).unflatten(
            sequence_dim,
            (full_rows // _BLOCK_ROWS, _BLOCK_ROWS),
        )
        summaries.append(blocks.sum(dim=sequence_dim + 1, dtype=torch.float32) / _BLOCK_ROWS)
    if full_rows != sequence_length:
        tail = sequence.narrow(sequence_dim, full_rows, sequence_length - full_rows)
        summaries.append(tail.mean(dim=sequence_dim, keepdim=True, dtype=torch.float32))
    return summaries[0] if len(summaries) == 1 else torch.cat(summaries, dim=sequence_dim)


def _coarse_attention_unchecked(
    block_scores: torch.Tensor,
    pooled_value: torch.Tensor,
) -> torch.Tensor:
    """Apply dense FP32 attention to validated block logits and pooled V."""
    return torch.matmul(torch.softmax(block_scores, dim=-1), pooled_value)


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
    return _coarse_attention_unchecked(block_scores, pooled_value)


def _apply_coarse_attention_residual_unchecked(
    coarse_output: torch.Tensor,
    coarse_gate: torch.Tensor,
) -> torch.Tensor:
    sequence_length = coarse_gate.shape[1]
    expanded = coarse_output.permute(0, 2, 1, 3).repeat_interleave(_BLOCK_ROWS, dim=1)
    expanded = expanded[:, :sequence_length]
    return (coarse_gate.float() * expanded).to(coarse_gate.dtype).contiguous()


def apply_coarse_attention_residual(
    coarse_output: torch.Tensor,
    coarse_gate: torch.Tensor,
) -> torch.Tensor:
    """Expand block outputs and apply the token-resolution gate."""
    if coarse_gate.ndim != 4 or coarse_output.ndim != 4:
        raise ValueError("coarse attention output and gate must use rank-four tensors")
    if not coarse_gate.is_floating_point():
        raise TypeError("coarse attention gate must be floating-point")
    if coarse_output.dtype is not torch.float32:
        raise ValueError("coarse attention output must use FP32")
    if coarse_output.device != coarse_gate.device:
        raise ValueError("coarse attention output and coarse gate must share a device")
    batch, sequence_length, heads, features = coarse_gate.shape
    query_blocks = (sequence_length + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    if coarse_output.shape != (batch, heads, query_blocks, features):
        raise ValueError("coarse output must contain one Q64 result per fine query block")
    return _apply_coarse_attention_residual_unchecked(coarse_output, coarse_gate)


def _apply_chunked_coarse_residual(
    pooled_value: torch.Tensor,
    coarse_gate: torch.Tensor,
    score_chunks: Iterable[tuple[int, torch.Tensor]],
) -> torch.Tensor:
    """Apply and concatenate validated coarse-score chunks before the gated residual."""
    coarse_chunks = [
        _coarse_attention_unchecked(scores, pooled_value) for _query_offset, scores in score_chunks
    ]
    coarse_output = coarse_chunks[0] if len(coarse_chunks) == 1 else torch.cat(coarse_chunks, dim=2)
    return _apply_coarse_attention_residual_unchecked(coarse_output, coarse_gate)


def coarse_attention_residual(
    block_scores: torch.Tensor,
    pooled_value: torch.Tensor,
    coarse_gate: torch.Tensor,
) -> torch.Tensor:
    """Return a gated, token-resolution coarse-attention residual."""
    return apply_coarse_attention_residual(
        coarse_attention(block_scores, pooled_value),
        coarse_gate,
    )


def _coarse_residual_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coarse_gate: torch.Tensor,
    *,
    coarse_key_blocks: int | None,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the common Q/K/V-derived coarse residual implementation."""
    from ._routing import score_chunks  # noqa: PLC0415
    from ._summaries import _sequence_block_summaries_validated  # noqa: PLC0415

    validate_routing_mode(routing_mode)
    resolved_blocks = validate_coarse_residual_inputs(
        query,
        key,
        value,
        coarse_gate,
        coarse_key_blocks,
        coarse_scale,
        block_lengths,
        routing_label=_ROUTING_NAME_BY_MODE[routing_mode],
    )
    query_head_major = query.transpose(1, 2)
    coarse_key = key.transpose(1, 2)[:, :, : resolved_blocks * _BLOCK_ROWS]
    query_summary, key_primary, key_aux = _sequence_block_summaries_validated(
        query_head_major,
        coarse_key,
        routing_mode,
        block_lengths,
    )
    pooled_value = _mean_pool_token_blocks(value, block_lengths)[:, :, :resolved_blocks]
    return _apply_chunked_coarse_residual(
        pooled_value,
        coarse_gate,
        score_chunks(
            query_summary,
            key_primary,
            key_aux,
            routing_mode,
            score_scale=coarse_scale,
        ),
    )


@torch.library.custom_op(
    "piper_kernels::sparse_piper_coarse_residual",
    mutates_args=(),
)
def _sparse_piper_coarse_residual_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coarse_gate: torch.Tensor,
    coarse_key_blocks: int,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve a Q/K/V-derived coarse residual as one semantic operation."""
    return _coarse_residual_impl(
        query,
        key,
        value,
        coarse_gate,
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        routing_mode=routing_mode,
        block_lengths=block_lengths,
    )


@_sparse_piper_coarse_residual_op.register_fake
def _sparse_piper_coarse_residual_op_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coarse_gate: torch.Tensor,
    coarse_key_blocks: int,
    coarse_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_routing_mode(routing_mode)
    validate_coarse_residual_inputs(
        query,
        key,
        value,
        coarse_gate,
        coarse_key_blocks,
        coarse_scale,
        block_lengths,
        routing_label=_ROUTING_NAME_BY_MODE[routing_mode],
    )
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def sparse_piper_coarse_residual(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coarse_gate: torch.Tensor,
    *,
    routing: Literal["mean", "minmax"],
    coarse_key_blocks: int | None = None,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a gated coarse branch using the selected block-summary policy.

    All token tensors use ``[batch,sequence,heads,features]``. Compact storage
    may end in one partial block. With ``block_lengths``, storage instead holds
    complete physical K64 blocks whose valid rows occupy each block's prefix.
    ``coarse_key_blocks`` defaults to every available K64 block.
    """
    routing_mode = routing_mode_from_name(routing)
    if _preserve_coarse_residual_in_graph(query, key, value, coarse_gate):
        resolved_coarse_key_blocks = (
            (
                block_lengths.numel()
                if block_lengths is not None
                else (query.shape[1] + _BLOCK_ROWS - 1) // _BLOCK_ROWS
            )
            if coarse_key_blocks is None
            else coarse_key_blocks
        )
        return torch.ops.piper_kernels.sparse_piper_coarse_residual.default(
            query,
            key,
            value,
            coarse_gate,
            resolved_coarse_key_blocks,
            coarse_scale,
            routing_mode,
            block_lengths,
        )
    return _coarse_residual_impl(
        query,
        key,
        value,
        coarse_gate,
        coarse_key_blocks=coarse_key_blocks,
        coarse_scale=coarse_scale,
        routing_mode=routing_mode,
        block_lengths=block_lengths,
    )


__all__ = [
    "apply_coarse_attention_residual",
    "coarse_attention",
    "coarse_attention_residual",
    "mean_pool_block_values",
    "sparse_piper_coarse_residual",
]
