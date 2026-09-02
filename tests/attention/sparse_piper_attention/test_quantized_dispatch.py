"""Already-quantized sparse Piper orchestration tests."""

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
from piper_kernels.attention.sparse_piper_attention.gluon import (
    _launch_sparse_piper_attention,
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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("sequence_length", [65, 193, 256])
def test_query_block_ranges_match_full_launch_and_preserve_guards(
    sequence_length: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(91 + sequence_length)
    shape = (1, sequence_length, 2, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)
    sparse_key_blocks = sequence_length // 64
    layout = _resolve_route_layout(
        _normalize_head_keep_ratios((0.5, 1.0)),
        sparse_key_blocks,
        query.device,
    )
    routes = packed_dsa_routes_from_sequences(
        query_head_major,
        key_head_major[:, :, : sparse_key_blocks * 64],
        layout,
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

    full_output = torch.empty_like(query)
    chunks: list[torch.Tensor] = []
    query_block_count = (sequence_length + 63) // 64
    query_block_offset = 0
    with torch.no_grad():
        _launch_sparse_piper_attention(prepared, full_output.transpose(1, 2))
        while query_block_offset < query_block_count:
            remaining_blocks = query_block_count - query_block_offset
            range_block_count = 1 if query_block_offset == 0 else min(2, remaining_blocks)
            range_rows = min(
                range_block_count * 64,
                sequence_length - query_block_offset * 64,
            )
            guarded = torch.full(
                (shape[0], range_rows + 2, shape[2], shape[3]),
                123.0,
                dtype=torch.bfloat16,
                device=query.device,
            )
            output = guarded[:, 1 : range_rows + 1]
            _launch_sparse_piper_attention(
                prepared,
                output.transpose(1, 2),
                query_block_offset=query_block_offset,
                query_block_count=range_block_count,
            )
            assert bool(torch.all(guarded[:, 0] == 123.0))
            assert bool(torch.all(guarded[:, -1] == 123.0))
            chunks.append(output.clone())
            query_block_offset += range_block_count

    ranged_output = torch.cat(chunks, dim=1)
    assert torch.equal(ranged_output, full_output)


def _block_length_case():
    generator = torch.Generator(device="cuda").manual_seed(417)
    storage_sequence_length = 3 * 64
    shape = (1, storage_sequence_length, 2, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)
    sparse_key_blocks = 2
    ratio_units = _normalize_head_keep_ratios((1.0, 1.0))
    layout = _resolve_route_layout(ratio_units, sparse_key_blocks, query.device)
    routes = packed_dsa_routes_from_sequences(
        query_head_major,
        key_head_major[:, :, : sparse_key_blocks * 64],
        layout,
    )
    query_summary, key_max, key_min = _sequence_block_summaries(
        query_head_major,
        key_head_major,
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
    arguments = (
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
        storage_sequence_length,
    )
    block_lengths = torch.tensor([64, 17, 51], dtype=torch.int32, device=query.device)
    return shape, query, prepared, arguments, block_lengths


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_block_lengths_mask_internal_key_padding() -> None:
    shape, query, prepared, arguments, block_lengths = _block_length_case()
    storage_sequence_length = shape[1]
    full_block_lengths = torch.full(
        (storage_sequence_length // 64,),
        64,
        dtype=torch.int32,
        device=query.device,
    )
    valid_rows = torch.arange(
        storage_sequence_length, device=query.device
    ) % 64 < block_lengths.repeat_interleave(64)
    corrupted_key = prepared.key.clone()
    corrupted_value = prepared.value.clone()
    corrupted_key[:, :, ~valid_rows] = 127
    corrupted_value[..., ~valid_rows] = -127
    corrupted_arguments = list(arguments)
    corrupted_arguments[3] = corrupted_key
    corrupted_arguments[7] = corrupted_value
    partial_arguments = list(arguments)
    partial_arguments[-1] = 64 + 17 + 51
    corrupted_arguments[-1] = partial_arguments[-1]

    with torch.no_grad():
        legacy = _sparse_piper_attention_from_quantized_op(*arguments)
        full_blocks = _sparse_piper_attention_from_quantized_op(
            *arguments,
            full_block_lengths,
        )
        expected = _sparse_piper_attention_from_quantized_op(
            *partial_arguments,
            block_lengths,
        )
        actual = _sparse_piper_attention_from_quantized_op(
            *corrupted_arguments,
            block_lengths,
        )
        opcheck = torch.library.opcheck(
            _sparse_piper_attention_from_quantized_op,
            (*partial_arguments, block_lengths),
        )

        captured_graphs = []

        def capture_backend(graph, _example_inputs):
            captured_graphs.append(graph)
            return graph.forward

        def run(candidate_block_lengths):
            return _sparse_piper_attention_from_quantized_op(
                *partial_arguments,
                candidate_block_lengths,
            )

        compiled = torch.compile(run, backend=capture_backend, fullgraph=True)
        compiled(block_lengths)
        compiled(torch.tensor([64, 18, 50], dtype=torch.int32, device=query.device))

    torch.testing.assert_close(full_blocks, legacy, atol=0.001953125, rtol=0)
    assert torch.equal(actual, expected)
    assert actual.shape == shape
    assert bool(torch.isfinite(actual).all())
    assert set(opcheck.values()) == {"SUCCESS"}
    assert len(captured_graphs) == 1

    with pytest.raises(ValueError, match="one INT32 value per K64"):
        _sparse_piper_attention_from_quantized_op(
            *partial_arguments,
            block_lengths[:-1],
        )
    with pytest.raises(ValueError, match="one INT32 value per K64"):
        _sparse_piper_attention_from_quantized_op(
            *partial_arguments,
            block_lengths.to(torch.int64),
        )
