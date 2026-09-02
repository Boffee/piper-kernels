"""Policy-independent coarse-attention residuals for Sparse Piper."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.nn import functional

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS


def _preserve_coarse_residual_in_graph(*tensors: torch.Tensor) -> bool:
    """Use the semantic boundary only when no compiled backward is required."""
    return torch.compiler.is_compiling() and not (
        torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)
    )


def validate_coarse_scale(coarse_scale: float) -> None:
    """Reject scales that do not preserve fine-route score ordering."""
    if not math.isfinite(coarse_scale) or coarse_scale <= 0:
        raise ValueError("coarse attention scale must be finite and positive")


def _validate_block_lengths(
    block_lengths: torch.Tensor,
    *,
    sequence_length: int,
    device: torch.device,
    context: str,
    require_contiguous: bool = False,
) -> int:
    """Validate public valid-prefix block metadata without synchronizing CUDA."""
    if (
        block_lengths.ndim != 1
        or block_lengths.numel() < 1
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != device
        or sequence_length != block_lengths.numel() * _BLOCK_ROWS
        or (require_contiguous and not block_lengths.is_contiguous())
    ):
        contiguity = " contiguous" if require_contiguous else ""
        raise ValueError(
            f"{context} block lengths must be one{contiguity} device INT32 value per K64"
        )
    torch._assert_async(
        torch.all((block_lengths >= 1) & (block_lengths <= _BLOCK_ROWS)),
        f"{context} block lengths must lie in [1, {_BLOCK_ROWS}]",
    )
    return block_lengths.numel()


def _validate_key_block_scopes(
    sparse_key_blocks: int,
    coarse_key_blocks: int | None,
    *,
    available_sparse_key_blocks: int,
    available_coarse_key_blocks: int,
) -> int:
    """Validate the sparse prefix and resolve its enclosing coarse prefix."""
    if isinstance(sparse_key_blocks, bool):
        raise TypeError("sparse_key_blocks must be an integer")
    if torch.compiler.is_compiling():
        torch._check(sparse_key_blocks >= 1, lambda: "sparse_key_blocks must be positive")
        torch._check(
            sparse_key_blocks <= available_sparse_key_blocks,
            lambda: "sparse_key_blocks cannot exceed the available K64 blocks",
        )
    else:
        if not isinstance(sparse_key_blocks, int):
            raise TypeError("sparse_key_blocks must be an integer")
        if not 1 <= sparse_key_blocks <= available_sparse_key_blocks:
            raise ValueError("sparse_key_blocks must fit the available K64 blocks")
    return _resolve_coarse_key_blocks(
        sparse_key_blocks,
        coarse_key_blocks,
        available_coarse_key_blocks=available_coarse_key_blocks,
    )


def _resolve_coarse_key_blocks(
    sparse_key_blocks: int,
    coarse_key_blocks: int | None,
    *,
    available_coarse_key_blocks: int,
) -> int:
    """Resolve a coarse prefix that contains the sparse-routing prefix."""
    resolved = sparse_key_blocks if coarse_key_blocks is None else coarse_key_blocks
    if isinstance(resolved, bool):
        raise TypeError("coarse_key_blocks must be an integer")
    if torch.compiler.is_compiling():
        torch._check(
            resolved >= sparse_key_blocks,
            lambda: "coarse_key_blocks cannot exclude sparse key blocks",
        )
        torch._check(
            resolved <= available_coarse_key_blocks,
            lambda: "coarse_key_blocks cannot exceed the available K64 blocks",
        )
    elif not isinstance(resolved, int):
        raise TypeError("coarse_key_blocks must be an integer")
    elif not sparse_key_blocks <= resolved <= available_coarse_key_blocks:
        raise ValueError(
            "coarse_key_blocks must include the sparse prefix and fit the available K64 blocks"
        )
    return resolved


def validate_coarse_residual_inputs(
    fine_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compression_gate: torch.Tensor,
    sparse_key_blocks: int,
    coarse_key_blocks: int | None,
    coarse_scale: float,
    block_lengths: torch.Tensor | None,
    *,
    routing_label: str,
) -> int:
    """Validate the shared contract and resolve the coarse K64 prefix length."""
    tensors = (fine_output, query, key, value, compression_gate)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError(
            f"{routing_label} coarse attention tensors must use [batch,sequence,heads,features]"
        )
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(f"{routing_label} coarse attention requires equal Q/K/V shapes")
    if fine_output.shape != query.shape:
        raise ValueError(f"{routing_label} coarse attention must match the fine output shape")
    if compression_gate.shape != fine_output.shape:
        raise ValueError(f"{routing_label} coarse attention gate must match the fine output shape")
    if any(tensor.device != query.device for tensor in (fine_output, key, value, compression_gate)):
        raise ValueError(f"{routing_label} coarse attention tensors must share a device")
    if any(not tensor.is_floating_point() for tensor in (fine_output, query, key, value)):
        raise TypeError(f"{routing_label} coarse attention tensors must be floating-point")
    if not compression_gate.is_floating_point() or compression_gate.dtype is not fine_output.dtype:
        raise ValueError(f"{routing_label} coarse attention gate must share the fine output dtype")

    validate_coarse_scale(coarse_scale)
    if block_lengths is None:
        available_sparse_key_blocks = query.shape[1] // _BLOCK_ROWS
        available_coarse_key_blocks = (query.shape[1] + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    else:
        available_sparse_key_blocks = available_coarse_key_blocks = _validate_block_lengths(
            block_lengths,
            sequence_length=query.shape[1],
            device=query.device,
            context=f"{routing_label} coarse attention",
        )
    return _validate_key_block_scopes(
        sparse_key_blocks,
        coarse_key_blocks,
        available_sparse_key_blocks=available_sparse_key_blocks,
        available_coarse_key_blocks=available_coarse_key_blocks,
    )


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
        _validate_block_lengths(
            block_lengths,
            sequence_length=sequence_length,
            device=value.device,
            context="coarse attention",
        )
    return _mean_pool_token_blocks(value, block_lengths)


def _mean_pool_token_blocks(
    sequence: torch.Tensor,
    block_lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-pool validated token-major Q, K, or V storage into FP32 blocks."""
    sequence_length = sequence.shape[1]

    if block_lengths is None:
        block_count = (sequence_length + _BLOCK_ROWS - 1) // _BLOCK_ROWS
        storage_length = block_count * _BLOCK_ROWS
        padded = functional.pad(sequence, (0, 0, 0, 0, 0, storage_length - sequence_length))
        lengths = torch.minimum(
            torch.full(
                (block_count,),
                _BLOCK_ROWS,
                dtype=torch.int32,
                device=sequence.device,
            ),
            sequence_length
            - torch.arange(block_count, dtype=torch.int32, device=sequence.device) * _BLOCK_ROWS,
        )
        blocks = padded.reshape(
            sequence.shape[0],
            block_count,
            _BLOCK_ROWS,
            sequence.shape[2],
            sequence.shape[3],
        )
    else:
        block_count = block_lengths.numel()
        lengths = block_lengths
        blocks = sequence.reshape(
            sequence.shape[0],
            block_count,
            _BLOCK_ROWS,
            sequence.shape[2],
            sequence.shape[3],
        )
        valid_rows = torch.arange(_BLOCK_ROWS, device=sequence.device)[None, :] < lengths[:, None]
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
    query_blocks = (sequence_length + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    if coarse_output.shape != (batch, heads, query_blocks, features):
        raise ValueError("coarse output must contain one Q64 result per fine query block")

    expanded = coarse_output.permute(0, 2, 1, 3).repeat_interleave(_BLOCK_ROWS, dim=1)
    expanded = expanded[:, :sequence_length]
    output = fine_output.float() + compression_gate.float() * expanded
    return output.to(fine_output.dtype)


def _apply_chunked_coarse_residual(
    fine_output: torch.Tensor,
    pooled_value: torch.Tensor,
    compression_gate: torch.Tensor,
    score_chunks: Iterable[tuple[int, torch.Tensor]],
) -> torch.Tensor:
    """Apply and concatenate coarse-score chunks before the gated residual."""
    coarse_chunks = [
        coarse_attention(scores, pooled_value) for _query_offset, scores in score_chunks
    ]
    coarse_output = coarse_chunks[0] if len(coarse_chunks) == 1 else torch.cat(coarse_chunks, dim=2)
    return apply_coarse_attention_residual(
        fine_output,
        coarse_output,
        compression_gate,
    )


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
