"""Bounded-memory reference checks for sparse Piper attention."""

from __future__ import annotations

from typing import Protocol

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat


class _PreparedQuery(Protocol):
    @property
    def data(self) -> torch.Tensor: ...
    @property
    def scale(self) -> torch.Tensor: ...
    @property
    def routes(self) -> torch.Tensor: ...
    @property
    def global_block_offset(self) -> int: ...


class _PreparedContext(Protocol):
    @property
    def key(self) -> torch.Tensor: ...
    @property
    def value(self) -> torch.Tensor: ...
    @property
    def value_format(self) -> ValueFormat: ...
    @property
    def key_scale(self) -> torch.Tensor: ...
    @property
    def key_center(self) -> torch.Tensor: ...
    @property
    def key_center_scale(self) -> torch.Tensor: ...
    @property
    def value_scale_multiplier(self) -> torch.Tensor: ...
    @property
    def value_mean(self) -> torch.Tensor: ...
    @property
    def route_head_offsets(self) -> torch.Tensor: ...
    @property
    def routes_per_query(self) -> int: ...
    @property
    def block_lengths(self) -> torch.Tensor | None: ...
    @property
    def sparse_key_blocks(self) -> int: ...
    @property
    def sparse_query_blocks(self) -> int | None: ...
    @property
    def logical_sequence_length(self) -> int: ...


class _PreparedSparseAttention(Protocol):
    @property
    def query(self) -> _PreparedQuery: ...
    @property
    def context(self) -> _PreparedContext: ...


def reference_value_storage(value: torch.Tensor, value_format: ValueFormat) -> torch.Tensor:
    """Encode canonical [B,H,D,S] codes for reference fixtures, outside measured paths."""
    if value_format is ValueFormat.TRANSPOSED:
        return value
    batch, heads, head_dim, sequence = value.shape
    return (
        value.reshape(batch, heads, head_dim, sequence // 64, 2, 2, 2, 8)
        .transpose(-3, -2)
        .reshape(batch, heads, head_dim, sequence // 64, 64)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


def _selected_tiles_and_validity(
    prepared: _PreparedSparseAttention,
    batch: int,
    head: int,
    query_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Resolve one query block's paired block indices and valid-token mask."""
    query, context = prepared.query, prepared.context
    device = query.data.device
    global_block = query.global_block_offset + query_block
    stored_blocks = context.key.shape[2] // 64
    use_routes = context.sparse_query_blocks is None or global_block < context.sparse_query_blocks
    if use_routes and context.routes_per_query != 0:
        start, stop = context.route_head_offsets[head : head + 2].tolist()
        tiles = query.routes[batch, query_block, start:stop].long()
    else:
        tiles = torch.arange(context.sparse_key_blocks, device=device)
    tiles = torch.cat(
        (tiles, torch.arange(context.sparse_key_blocks, stored_blocks, device=device))
    )
    tile_count = tiles.numel()
    if tile_count % 2:
        tiles = torch.cat((tiles, tiles[-1:]))
    offsets = torch.arange(64, device=device)
    indices = (tiles[:, None] * 64 + offsets).flatten()
    if context.block_lengths is None:
        valid = indices < context.logical_sequence_length
        rows = min(64, context.logical_sequence_length - global_block * 64)
    else:
        valid = (offsets < context.block_lengths[tiles, None]).flatten()
        rows = 64
    valid &= torch.arange(indices.numel(), device=device) < tile_count * 64
    return tiles, indices, valid, rows


def reference_prepared_query(
    prepared: _PreparedSparseAttention,
    batch: int,
    head: int,
    query_block: int,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Independent FP64 paired attention for one query block and one head.

    Quantized operands are shared with the kernel, but matrix products and
    probability quantization use PyTorch FP64. Pair results are combined with
    a single global normalization, not the kernel's online FP32 recurrence.
    Memory scales with one selected sequence, never the full attention matrix.
    """
    query, context = prepared.query, prepared.context
    kv_head = head // (query.data.shape[1] // context.key.shape[1])
    tiles, indices, valid, rows = _selected_tiles_and_validity(prepared, batch, head, query_block)
    pairs = tiles.numel() // 2
    q = query.data[batch, head, query_block * 64 : query_block * 64 + rows].double()
    q_scale = (
        query.scale[batch, head, query_block * 2 : query_block * 2 + 2]
        .repeat_interleave(32)
        .double()
    )
    k = context.key[batch, kv_head].index_select(0, indices).double()
    k_scale = context.key_scale[batch, kv_head, tiles].repeat_interleave(64).double()
    key_center = context.key_center[batch, kv_head, tiles]
    key_center_scale = context.key_center_scale[batch, kv_head, tiles]
    rotated_center = (
        key_center[:, 0].double() * key_center_scale[:, 0, None].double()
        + key_center[:, 1].double() * key_center_scale[:, 1, None].double()
    )
    k_bias = ((q @ rotated_center.T) * q_scale[:rows, None]).repeat_interleave(64, dim=-1)
    multiplier = (
        context.value_scale_multiplier[batch, kv_head, tiles, 0].repeat_interleave(64).double()
    )
    scores = (q @ k.T) * q_scale[:rows, None] * k_scale[None, :] + k_bias
    scores = (
        scores.masked_fill(~valid[None, :], -torch.inf).reshape(rows, pairs, 128).transpose(0, 1)
    )
    multipliers = multiplier.reshape(pairs, 1, 128)
    pair_max = (scores + torch.log2(multipliers / 255)).amax(dim=-1)
    probabilities = torch.exp2(scores - pair_max[:, :, None])
    codes = (probabilities * multipliers + 0.5).floor().clamp(0, 255)
    if context.value_format is ValueFormat.AMD_WMMA:
        # Decode with a reshape/transpose of token bit axes, independently of
        # the producer's address arithmetic. Gather first to bound memory.
        selected = context.value[batch, kv_head].index_select(0, tiles)
        values = selected.reshape(-1, query.data.shape[-1], 2, 2, 2, 8)
        values = values.transpose(-3, -2).reshape(-1, query.data.shape[-1], 64)
        values = values.transpose(1, 2)
    else:
        values = context.value[batch, kv_head].index_select(1, indices).T
    values = values.reshape(pairs, 128, query.data.shape[-1]).double()
    products = torch.bmm(codes, values)
    weights = torch.exp2(pair_max - pair_max.amax(dim=0))
    tile_masses = (
        probabilities.reshape(pairs, rows, 2, 64).sum(dim=-1) * weights[:, :, None] * 255.0
    )
    tile_means = context.value_mean[batch, kv_head, tiles].reshape(pairs, 2, -1).double()
    numerator = (products * weights[:, :, None]).sum(dim=0) + (
        tile_masses.transpose(0, 1).reshape(rows, pairs * 2) @ tile_means.reshape(pairs * 2, -1)
    )
    denominator = (probabilities.sum(dim=-1) * weights).sum(dim=0)
    output = numerator / (denominator.clamp_min(1e-30)[:, None] * 255)
    return output.to(output_dtype)
