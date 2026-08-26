"""Portable exact reference for sparse Piper's routing contract."""

from __future__ import annotations

import torch

from .dsa import PackedDsaRoutes


def reference_sparse_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    routes: PackedDsaRoutes,
    *,
    suffix_start: int,
    valid_sequence_length: int,
    scale: float,
) -> torch.Tensor:
    """Apply packed prefix routes plus the complete suffix in one softmax."""
    batch, sequence, heads, _head_dim = query.shape
    query_blocks = sequence // 64
    output = torch.empty_like(query)
    suffix_indices = torch.arange(
        suffix_start,
        valid_sequence_length,
        device=query.device,
    )
    row_offsets = torch.arange(64, device=query.device)
    offsets = routes.head_offsets.detach().cpu().tolist()

    for batch_index in range(batch):
        for head in range(heads):
            route_start, route_stop = offsets[head : head + 2]
            for query_block in range(query_blocks):
                selected_blocks = routes.indices[
                    batch_index,
                    query_block,
                    route_start:route_stop,
                ].long()
                prefix_indices = (selected_blocks[:, None] * 64 + row_offsets).flatten()
                key_indices = torch.cat((prefix_indices, suffix_indices))
                query_start = query_block * 64
                query_stop = query_start + 64
                block_query = query[batch_index, query_start:query_stop, head].float()
                selected_key = key[batch_index, key_indices, head].float()
                selected_value = value[batch_index, key_indices, head].float()
                probability = torch.softmax(block_query @ selected_key.mT * scale, dim=-1)
                output[batch_index, query_start:query_stop, head] = (
                    probability @ selected_value
                ).to(value.dtype)
    return output
