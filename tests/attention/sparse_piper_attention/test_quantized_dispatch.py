"""Ragged already-quantized sparse Piper orchestration tests."""

import pytest
import torch

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
)
from piper_kernels.attention.sparse_piper_attention.dsa import (
    _sequence_block_summaries,
    packed_dsa_routes_from_sequences,
)
from piper_kernels.attention.sparse_piper_attention.triton import (
    _prepare_sparse_piper_attention,
)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("sequence_length", [65, 193])
def test_ragged_quantized_path_matches_materialized_dispatch(sequence_length: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(66 + sequence_length)
    shape = (1, sequence_length, 2, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    ratios = (0.5, 1.0)
    ratio_units = _normalize_head_keep_ratios(ratios)
    sparse_key_blocks = sequence_length // 64
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)
    sparse_key = key_head_major[:, :, : sparse_key_blocks * 64]
    layout = _resolve_route_layout(ratio_units, sparse_key_blocks, query.device)
    routes = packed_dsa_routes_from_sequences(query_head_major, sparse_key, layout)
    query_summary, key_max, key_min = _sequence_block_summaries(
        query_head_major,
        sparse_key,
    )
    prepared = _prepare_sparse_piper_attention(
        query_head_major,
        routes.indices,
        routes.keep_blocks,
        128**-0.5,
        sparse_key_blocks=sparse_key_blocks,
        route_head_offsets=routes.head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
    )

    quantized_arguments = (
        prepared.query,
        prepared.query_scale,
        query_summary,
        prepared.key,
        prepared.key_scale,
        key_max,
        key_min,
        prepared.value,
        prepared.value_scale_multiplier,
        prepared.value_mean,
        list(ratio_units),
        sparse_key_blocks,
        sequence_length,
    )
    with torch.no_grad():
        expected = SparsePiperAttention(ratios)(
            query,
            key,
            value,
            sparse_key_blocks=sparse_key_blocks,
        )
        actual = _sparse_piper_attention_from_quantized_op(*quantized_arguments)
        opcheck = torch.library.opcheck(
            _sparse_piper_attention_from_quantized_op,
            quantized_arguments,
        )

    assert actual.shape == query.shape
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert set(opcheck.values()) == {"SUCCESS"}
