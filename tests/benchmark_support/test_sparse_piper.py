from unittest.mock import Mock

import pytest
import torch
from lib.sparse_piper import reference_prepared_query, reference_value_storage

from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _PreparedSparsePiperAttention,
    _PreparedSparsePiperContext,
    _PreparedSparsePiperQuery,
)


@pytest.mark.parametrize("value_format", list(ValueFormat))
@pytest.mark.parametrize("internal_padding", [False, True])
@pytest.mark.parametrize("global_offset", [0, 1])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_bounded_reference_selects_routes_dense_queries_and_valid_rows(
    internal_padding, global_offset, head_dim, kv_heads, value_format
):
    generator = torch.Generator().manual_seed(713)
    lengths = [3, 64, 7, 41] if internal_padding else [64, 64, 64, 1]
    value = torch.randint(
        -127, 128, (2, kv_heads, head_dim, 256), dtype=torch.int8, generator=generator
    )
    mean = torch.randn((2, kv_heads, 4, head_dim), dtype=torch.bfloat16, generator=generator)
    context = _PreparedSparsePiperContext(
        key=torch.zeros((2, kv_heads, 256, head_dim), dtype=torch.int8),
        value=reference_value_storage(value, value_format),
        value_format=value_format,
        key_scale=torch.ones((2, kv_heads, 4)),
        key_center=torch.zeros((2, kv_heads, 4, 2, head_dim), dtype=torch.int8),
        key_center_scale=torch.ones((2, kv_heads, 4, 2)),
        value_scale_multiplier=torch.full((2, kv_heads, 4, 1), 255.0),
        value_mean=mean,
        route_head_offsets=torch.tensor([0, 1, 3], dtype=torch.int32),
        head_keep_blocks=torch.tensor([1, 2], dtype=torch.int32),
        routes_per_query=3,
        block_lengths=torch.tensor(lengths, dtype=torch.int32) if internal_padding else None,
        sparse_key_blocks=2,
        sparse_query_blocks=2,
        logical_sequence_length=256 if internal_padding else 193,
    )
    query = _PreparedSparsePiperQuery(
        data=torch.zeros((2, 2, (4 - global_offset) * 64, head_dim), dtype=torch.int8),
        scale=torch.ones((2, 2, (4 - global_offset) * 2)),
        routes=torch.tensor([1, 1, 0], dtype=torch.uint16).repeat(2, 4 - global_offset, 1),
        global_block_offset=global_offset,
    )
    prepared = _PreparedSparsePiperAttention(Mock(context=context), query)
    for batch in range(2):
        for head in range(2):
            kv_head = head // (2 // kv_heads)
            for block in range(4 - global_offset):
                global_block = block + global_offset
                prefix = [1] if head == 0 and global_block < 2 else [0, 1]
                indices = [
                    tile * 64 + row for tile in [*prefix, 2, 3] for row in range(lengths[tile])
                ]
                tile_ids = torch.tensor([*prefix, 2, 3], dtype=torch.long)
                expected = value[batch, kv_head, :, indices].double().mean(dim=1) + (
                    mean[batch, kv_head, tile_ids].double()
                    * torch.tensor(
                        [lengths[tile] for tile in tile_ids.tolist()], dtype=torch.float64
                    )[:, None]
                ).sum(dim=0) / sum(lengths[tile] for tile in tile_ids.tolist())
                rows = 64 if internal_padding else min(64, 193 - global_block * 64)
                expected = expected.to(torch.bfloat16).expand(rows, head_dim)
                torch.testing.assert_close(
                    reference_prepared_query(prepared, batch, head, block), expected, rtol=0, atol=0
                )
