"""Shared block-summary orchestration for sparse Piper routing."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from ._block_layout import valid_block_rows, validate_block_lengths
from ._routes import _MEAN_ROUTING, validate_routing_mode
from .coarse import _mean_pool_head_major_blocks

try:
    from ._summaries_triton import sequence_block_summaries as _sm120_sequence_block_summaries
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("triton"):
        raise
    _sm120_sequence_block_summaries = None


def sequence_block_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy-specific Q/K summaries through one fixed tensor contract."""
    validate_routing_mode(routing_mode)
    _validate_sequences(query, key, block_lengths)
    if _supports_sm120_sequence_summaries(query, key):
        assert _sm120_sequence_block_summaries is not None
        return _sm120_sequence_block_summaries(query, key, routing_mode, block_lengths)

    key_lengths = None if block_lengths is None else block_lengths[: key.shape[2] // _BLOCK_ROWS]
    if routing_mode == _MEAN_ROUTING:
        query_summary = _mean_pool_head_major_blocks(query, block_lengths)
        key_primary = _mean_pool_head_major_blocks(key, key_lengths)
        return query_summary, key_primary, key_primary[:, :, :0]

    if block_lengths is None:
        query_max, query_min = _compact_block_extrema(query)
        key_primary, key_aux = _compact_block_extrema(key)
    else:
        assert key_lengths is not None
        query_max, query_min = _padded_block_extrema(query, block_lengths)
        key_primary, key_aux = _padded_block_extrema(key, key_lengths)
    return query_max + query_min, key_primary, key_aux


def _compact_block_extrema(sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 extrema for compact blocks, including a ragged tail."""
    extrema = []
    for start in range(0, sequence.shape[2], _BLOCK_ROWS):
        block = sequence[:, :, start : start + _BLOCK_ROWS].float()
        extrema.append((block.amax(dim=2), block.amin(dim=2)))
    maxima, minima = zip(*extrema, strict=True)
    return torch.stack(maxima, dim=2), torch.stack(minima, dim=2)


def _padded_block_extrema(
    sequence: torch.Tensor,
    block_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = sequence.unflatten(2, (block_lengths.numel(), _BLOCK_ROWS)).float()
    valid_rows = valid_block_rows(block_lengths)[None, None, :, :, None]
    maximum = torch.where(valid_rows, blocks, -float("inf")).amax(dim=3)
    minimum = torch.where(valid_rows, blocks, float("inf")).amin(dim=3)
    return maximum, minimum


def _supports_sm120_sequence_summaries(query: torch.Tensor, key: torch.Tensor) -> bool:
    target = AcceleratorTarget.from_device(query.device)
    return (
        _sm120_sequence_block_summaries is not None
        and target.is_cuda_capability(12, 0)
        and query.shape[-1] == 128
        and key.shape[-1] == 128
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype
    )


def _validate_sequences(
    query: torch.Tensor,
    key: torch.Tensor,
    block_lengths: torch.Tensor | None,
) -> None:
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("routing query and key sequences must be rank-four tensors")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("routing query and key batch/head/feature dimensions must match")
    if query.shape[2] < 1 or key.shape[2] < 1:
        raise ValueError("routing summaries require nonempty Q/K sequences")
    if query.device != key.device:
        raise ValueError("routing query and key sequences must share a device")
    if not query.is_floating_point() or not key.is_floating_point():
        raise TypeError("routing query and key sequences must be floating-point tensors")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("routing query and key features must be contiguous")
    if block_lengths is not None:
        block_count = validate_block_lengths(
            block_lengths,
            sequence_length=query.shape[2],
            device=query.device,
            context="padded sparse routing",
            require_contiguous=True,
        )
        if key.shape[2] % _BLOCK_ROWS:
            raise ValueError("padded routing keys must use complete physical K64 blocks")
        if key.shape[2] // _BLOCK_ROWS > block_count:
            raise ValueError("padded routing keys cannot exceed the block-length layout")


__all__ = ["sequence_block_summaries"]
