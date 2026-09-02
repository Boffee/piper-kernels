"""Portable references for Sparse Piper Attention."""

from __future__ import annotations

import torch

from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    quantize_query_key,
)
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from ._block_layout import valid_block_rows
from ._routes import PackedRoutes

_RECURRENCE_ROWS = 128
_P_UINT8_RANGE = 255.0
_V_INT8_RANGE = 127.0
_SCALE_EPSILON = 1e-7


def _active_indices(
    routes: PackedRoutes,
    *,
    batch: int,
    head: int,
    query_block: int,
    sparse_key_blocks: int,
    sequence_length: int,
    route_head_offsets: list[int],
    block_lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    route_start, route_stop = route_head_offsets[head : head + 2]
    selected_blocks = routes.indices[
        batch,
        query_block,
        route_start:route_stop,
    ].long()
    row_offsets = torch.arange(_BLOCK_ROWS, device=routes.indices.device)
    sparse_rows = (selected_blocks[:, None] * _BLOCK_ROWS + row_offsets).flatten()
    sparse_key_rows = sparse_key_blocks * _BLOCK_ROWS
    suffix_rows = torch.arange(
        sparse_key_rows,
        sequence_length,
        device=routes.indices.device,
    )
    key_indices = torch.cat((sparse_rows, suffix_rows))

    suffix_tiles = torch.arange(
        sparse_key_blocks,
        (sequence_length + _BLOCK_ROWS - 1) // _BLOCK_ROWS,
        device=routes.indices.device,
    )
    tile_indices = torch.cat((selected_blocks, suffix_tiles))
    valid_keys = (
        None
        if block_lengths is None
        else valid_block_rows(block_lengths.index_select(0, tile_indices)).flatten()
    )
    return key_indices, tile_indices, valid_keys


def _quantize_value_per_tile(
    value_centered: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize centered V with one scalar scale per logical K64 tile."""
    batch, heads, key_length, head_dim = value_centered.shape
    tile_count = (key_length + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    storage_length = tile_count * _BLOCK_ROWS
    padded = value_centered.new_zeros((batch, heads, storage_length, head_dim))
    padded[:, :, :key_length] = value_centered
    grouped = padded.reshape(batch, heads, tile_count, _BLOCK_ROWS, head_dim)
    if block_lengths is not None:
        valid_rows = valid_block_rows(block_lengths)
        grouped = torch.where(valid_rows[None, None, :, :, None], grouped, 0)
    scale = grouped.abs().amax(dim=(3, 4)) / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = (
        (grouped / scale[..., None, None])
        .round()
        .clamp(-_V_INT8_RANGE, _V_INT8_RANGE)
        .to(torch.int8)
    )
    return quantized.flatten(2, 3)[:, :, :key_length], scale


def _quantize_query_key(
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match SM120 per-Q32 and per-K64 grouped quantization."""
    return quantize_query_key(
        query,
        key,
        granularity="per_warp",
    )


def reference_sparse_piper_attention(  # noqa: PLR0915
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    routes: PackedRoutes,
    *,
    sparse_key_blocks: int,
    scale: float,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the selected quantized Sparse Piper algorithm in PyTorch.

    This matches the SM120 algorithmic contract: grouped INT8 Q/K, centered
    tile-scaled INT8 V, paired K128 recurrence, UINT8 probabilities, and a
    pre-rounding FP32 denominator. It is a correctness reference, not a fast
    fallback.
    """
    batch, sequence, heads, head_dim = query.shape
    query_blocks = (sequence + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)

    if block_lengths is not None:
        valid_rows = valid_block_rows(block_lengths).reshape(-1)
        valid_rows = valid_rows[None, None, :, None]
        query_head_major = torch.where(valid_rows, query_head_major, 0)
        key_head_major = torch.where(valid_rows, key_head_major, 0)
        value_head_major = torch.where(valid_rows, value_head_major, 0)

    value_float = value_head_major.float()
    value_mean = value_float.mean(dim=2, keepdim=True)
    value_centered = value_float - value_mean
    query_int8, key_int8, query_scale, key_scale = _quantize_query_key(
        query_head_major,
        key_head_major,
    )
    value_int8, value_scale = _quantize_value_per_tile(value_centered, block_lengths)
    output = torch.empty_like(query_head_major)
    route_head_offsets = routes.route_head_offsets.detach().cpu().tolist()

    for batch_index in range(batch):
        for head in range(heads):
            for query_block in range(query_blocks):
                key_indices, tile_indices, valid_keys = _active_indices(
                    routes,
                    batch=batch_index,
                    head=head,
                    query_block=query_block,
                    sparse_key_blocks=sparse_key_blocks,
                    sequence_length=sequence,
                    route_head_offsets=route_head_offsets,
                    block_lengths=block_lengths,
                )
                selected_key = key_int8[batch_index, head].index_select(0, key_indices)
                selected_value = value_int8[batch_index, head].index_select(0, key_indices)
                selected_key_scale = key_scale[batch_index, head].index_select(0, key_indices)
                selected_value_scale = value_scale[batch_index, head].index_select(
                    0,
                    tile_indices,
                )
                query_start = query_block * _BLOCK_ROWS
                query_stop = min(query_start + _BLOCK_ROWS, sequence)
                block_query = query_int8[batch_index, head, query_start:query_stop]
                block_query_scale = query_scale[batch_index, head, query_start:query_stop]
                numerator = torch.zeros(
                    (query_stop - query_start, head_dim),
                    device=query.device,
                    dtype=torch.float32,
                )
                denominator = torch.zeros(
                    (query_stop - query_start,),
                    device=query.device,
                    dtype=torch.float32,
                )
                running_max = torch.full_like(denominator, -float("inf"))

                for pair_start in range(0, key_indices.numel(), _RECURRENCE_ROWS):
                    pair_stop = min(pair_start + _RECURRENCE_ROWS, key_indices.numel())
                    integer_scores = (
                        block_query.float() @ selected_key[pair_start:pair_stop].float().mT
                    )
                    scores = (
                        integer_scores
                        * block_query_scale[:, None]
                        * selected_key_scale[None, pair_start:pair_stop]
                        * scale
                    )
                    if valid_keys is not None:
                        scores = scores.masked_fill(
                            ~valid_keys[pair_start:pair_stop][None, :],
                            -float("inf"),
                        )
                    pair_tile_start = pair_start // _BLOCK_ROWS
                    pair_tile_stop = (pair_stop + _BLOCK_ROWS - 1) // _BLOCK_ROWS
                    pair_value_scales = selected_value_scale[pair_tile_start:pair_tile_stop]
                    row_value_scales = pair_value_scales.repeat_interleave(
                        _BLOCK_ROWS,
                    )[: pair_stop - pair_start]
                    shifted_scores = scores + torch.log(row_value_scales[None, :])
                    block_max = shifted_scores.amax(dim=-1)
                    next_max = torch.maximum(running_max, block_max)
                    old_weight = torch.exp(running_max - next_max)
                    current_weight = torch.exp(block_max - next_max)
                    probabilities = torch.exp(scores - block_max[:, None])

                    numerator *= old_weight[:, None]
                    denominator = (
                        denominator * old_weight + probabilities.sum(dim=-1) * current_weight
                    )
                    for tile_start in range(pair_start, pair_stop, _BLOCK_ROWS):
                        tile_stop = min(tile_start + _BLOCK_ROWS, pair_stop)
                        probability_start = tile_start - pair_start
                        probability_stop = tile_stop - pair_start
                        tile_ordinal = tile_start // _BLOCK_ROWS
                        probability_codes = (
                            (
                                probabilities[:, probability_start:probability_stop]
                                * selected_value_scale[tile_ordinal]
                                * _P_UINT8_RANGE
                            )
                            .round()
                            .clamp(0, _P_UINT8_RANGE)
                        )
                        partial = probability_codes @ selected_value[tile_start:tile_stop].float()
                        numerator += partial * current_weight[:, None]
                    running_max = next_max

                block_output = numerator / (denominator.clamp_min(1e-30)[:, None] * _P_UINT8_RANGE)
                block_output += value_mean[batch_index, head]
                output[batch_index, head, query_start:query_stop] = block_output.to(query.dtype)

    return output.transpose(1, 2).contiguous()


def reference_exact_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    routes: PackedRoutes,
    *,
    sparse_key_blocks: int,
    scale: float,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the same routes with exact BF16 inputs and FP32 attention math."""
    batch, sequence, heads, _head_dim = query.shape
    query_blocks = (sequence + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    output = torch.empty_like(query)
    route_head_offsets = routes.route_head_offsets.detach().cpu().tolist()

    for batch_index in range(batch):
        for head in range(heads):
            for query_block in range(query_blocks):
                key_indices, _tile_indices, valid_keys = _active_indices(
                    routes,
                    batch=batch_index,
                    head=head,
                    query_block=query_block,
                    sparse_key_blocks=sparse_key_blocks,
                    sequence_length=sequence,
                    route_head_offsets=route_head_offsets,
                    block_lengths=block_lengths,
                )
                query_start = query_block * _BLOCK_ROWS
                query_stop = min(query_start + _BLOCK_ROWS, sequence)
                block_query = query[batch_index, query_start:query_stop, head].float()
                selected_key = key[batch_index, key_indices, head].float()
                selected_value = value[batch_index, key_indices, head].float()
                scores = block_query @ selected_key.mT * scale
                if valid_keys is not None:
                    scores = scores.masked_fill(~valid_keys[None, :], -float("inf"))
                probability = torch.softmax(scores, dim=-1)
                output[batch_index, query_start:query_stop, head] = (
                    probability @ selected_value
                ).to(value.dtype)
    return output
