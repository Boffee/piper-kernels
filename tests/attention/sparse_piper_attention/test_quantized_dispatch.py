"""Already-quantized sparse Piper orchestration tests."""

import pytest
import torch

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS, TILE_ROWS
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
    _sparse_piper_attention_with_coarse_residual_from_quantized_op,
)
from piper_kernels.attention.sparse_piper_attention._routes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention._routing import (
    packed_routes_from_sequences,
    routing_scores,
)
from piper_kernels.attention.sparse_piper_attention._summaries import (
    sequence_block_summaries,
)
from piper_kernels.attention.sparse_piper_attention.coarse import (
    _mean_pool_head_major_blocks,
    apply_coarse_attention_residual,
    coarse_attention,
    mean_pool_block_values,
)
from piper_kernels.attention.sparse_piper_attention.gluon import (
    _launch_sparse_piper_attention,
)
from piper_kernels.attention.sparse_piper_attention.triton import (
    _prepare_sparse_piper_attention,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
)


def _sequence_block_means(sequence, block_lengths=None):
    return _mean_pool_head_major_blocks(sequence, block_lengths)


def _minmax_pool_block_summaries(query, key, block_lengths=None):
    return sequence_block_summaries(query, key, _MINMAX_ROUTING, block_lengths)


def _minmax_pool_scores(query_summary, key_max, key_min):
    return routing_scores(query_summary, key_max, key_min, _MINMAX_ROUTING)


def packed_minmax_pool_routes_from_sequences(query, key, layout, block_lengths=None):
    return packed_routes_from_sequences(
        query,
        key,
        layout,
        _MINMAX_ROUTING,
        block_lengths,
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
    routes = packed_minmax_pool_routes_from_sequences(query_head_major, sparse_key, layout)
    query_summary, key_max, key_min = _minmax_pool_block_summaries(
        query_head_major,
        sparse_key,
    )
    prepared = _prepare_sparse_piper_attention(
        query_head_major,
        routes.indices,
        routes.head_keep_blocks,
        128**-0.5,
        sparse_key_blocks=sparse_key_blocks,
        route_head_offsets=routes.route_head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
    )

    quantized_arguments = (
        prepared.query.data,
        prepared.query.scale,
        query_summary,
        prepared.context.key,
        prepared.context.key_scale,
        key_max,
        key_min,
        prepared.context.value,
        prepared.context.value_scale_multiplier,
        prepared.context.value_mean,
        list(ratio_units),
        sparse_key_blocks,
        sequence_length,
        _MINMAX_ROUTING,
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


def _local_query_range(
    prepared: _PreparedSparsePiperAttention,
    global_block_offset: int,
    block_count: int,
) -> _PreparedSparsePiperAttention:
    storage_start = global_block_offset * TILE_ROWS
    storage_stop = storage_start + block_count * TILE_ROWS
    scale_start = storage_start // QUERY_SCALE_ROWS
    scale_stop = storage_stop // QUERY_SCALE_ROWS
    return _PreparedSparsePiperAttention(
        context=prepared.context,
        query=_prepare_sparse_piper_query_from_quantized(
            prepared.query.data[:, :, storage_start:storage_stop].contiguous(),
            prepared.query.scale[:, :, scale_start:scale_stop].contiguous(),
            prepared.query.routes[
                :,
                global_block_offset : global_block_offset + block_count,
            ].contiguous(),
            prepared.context,
            global_block_offset=global_block_offset,
        ),
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
def test_quantized_coarse_residual_matches_explicit_composition(
    routing_mode: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1221 + routing_mode)
    shape = (1, 193, 2, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    coarse_gate = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)
    sparse_key_blocks = shape[1] // 64
    coarse_key_blocks = (shape[1] + 63) // 64
    sparse_key = key_head_major[:, :, : sparse_key_blocks * 64]
    ratio_units = _normalize_head_keep_ratios((0.5, 1.0))
    layout = _resolve_route_layout(ratio_units, sparse_key_blocks, query.device)
    placeholder_routes = packed_minmax_pool_routes_from_sequences(
        query_head_major,
        sparse_key,
        layout,
    )
    prepared = _prepare_sparse_piper_attention(
        query_head_major,
        placeholder_routes.indices,
        placeholder_routes.head_keep_blocks,
        128**-0.5,
        sparse_key_blocks=sparse_key_blocks,
        route_head_offsets=placeholder_routes.route_head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
    )
    if routing_mode == _MEAN_ROUTING:
        query_summary = _sequence_block_means(query_head_major)
        key_summary = _sequence_block_means(key_head_major)
        key_aux = key_summary[:, :, :0]
        scores = query_summary @ key_summary.mT
    else:
        query_summary, key_summary, key_aux = _minmax_pool_block_summaries(
            query_head_major,
            key_head_major,
        )
        scores = _minmax_pool_scores(query_summary, key_summary, key_aux)
    fine_arguments = (
        prepared.query.data,
        prepared.query.scale,
        query_summary,
        prepared.context.key,
        prepared.context.key_scale,
        key_summary,
        key_aux,
        prepared.context.value,
        prepared.context.value_scale_multiplier,
        prepared.context.value_mean,
        list(ratio_units),
        sparse_key_blocks,
        shape[1],
        routing_mode,
    )
    block_mean = mean_pool_block_values(value)
    coarse_scale = 128**-0.5
    coarse_arguments = (
        *fine_arguments[:10],
        block_mean,
        coarse_gate,
        *fine_arguments[10:],
        coarse_scale,
        None,
        coarse_key_blocks,
    )

    with torch.no_grad():
        fine_output = _sparse_piper_attention_from_quantized_op(*fine_arguments)
        expected_coarse = coarse_attention(
            scores * coarse_scale,
            block_mean[:, :, :coarse_key_blocks],
        )
        expected = fine_output + apply_coarse_attention_residual(
            expected_coarse,
            coarse_gate,
        )
        actual = _sparse_piper_attention_with_coarse_residual_from_quantized_op(
            *coarse_arguments,
        )
        zero_gate_arguments = list(coarse_arguments)
        zero_gate_arguments[11] = torch.zeros_like(coarse_gate)
        zero_gate_output = _sparse_piper_attention_with_coarse_residual_from_quantized_op(
            *zero_gate_arguments,
        )
        opcheck = torch.library.opcheck(
            _sparse_piper_attention_with_coarse_residual_from_quantized_op,
            coarse_arguments,
        )

    assert torch.equal(actual, expected)
    assert torch.equal(zero_gate_output, fine_output)
    assert set(opcheck.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("sequence_length", [65, 193, 256])
@pytest.mark.parametrize("mixed_query_scope", [False, True])
def test_query_block_ranges_match_full_launch_and_preserve_guards(
    sequence_length: int,
    mixed_query_scope: bool,
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
    routes = packed_minmax_pool_routes_from_sequences(
        query_head_major,
        key_head_major[:, :, : sparse_key_blocks * 64],
        layout,
    )
    query_block_count = (sequence_length + 63) // 64
    sparse_query_blocks = query_block_count - 1 if mixed_query_scope else None
    prepared = _prepare_sparse_piper_attention(
        query_head_major,
        routes.indices,
        routes.head_keep_blocks,
        128**-0.5,
        sparse_key_blocks=sparse_key_blocks,
        route_head_offsets=routes.route_head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
        sparse_query_blocks=sparse_query_blocks,
    )

    coarse_output = torch.randn(
        (shape[0], shape[2], query_block_count, shape[3]),
        dtype=torch.float32,
        device=query.device,
        generator=generator,
    )
    coarse_gate = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device=query.device,
        generator=generator,
    )
    fine_output = torch.empty_like(query)
    full_output = torch.empty_like(query)
    chunks: list[torch.Tensor] = []
    query_block_offset = 0
    with torch.no_grad():
        _launch_sparse_piper_attention(prepared, fine_output.transpose(1, 2))
        expected = fine_output + apply_coarse_attention_residual(
            coarse_output,
            coarse_gate,
        )
        _launch_sparse_piper_attention(
            prepared,
            full_output.transpose(1, 2),
            coarse_output=coarse_output,
            coarse_gate=coarse_gate,
        )
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
                coarse_output=coarse_output,
                coarse_gate=coarse_gate[
                    :,
                    query_block_offset * 64 : query_block_offset * 64 + range_rows,
                ],
            )
            local_prepared = _local_query_range(
                prepared,
                query_block_offset,
                range_block_count,
            )
            local_output = torch.empty_like(output)
            _launch_sparse_piper_attention(
                local_prepared,
                local_output.transpose(1, 2),
                coarse_output=coarse_output[
                    :,
                    :,
                    query_block_offset : query_block_offset + range_block_count,
                ].contiguous(),
                coarse_gate=coarse_gate[
                    :,
                    query_block_offset * 64 : query_block_offset * 64 + range_rows,
                ],
            )
            assert bool(torch.all(guarded[:, 0] == 123.0))
            assert bool(torch.all(guarded[:, -1] == 123.0))
            assert torch.equal(local_output, output)
            chunks.append(output.clone())
            query_block_offset += range_block_count

    ranged_output = torch.cat(chunks, dim=1)
    torch.testing.assert_close(full_output, expected, atol=0.0078125, rtol=0.0078125)
    assert torch.equal(ranged_output, full_output)


def _block_length_case(routing_mode: int):
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
    routes = packed_minmax_pool_routes_from_sequences(
        query_head_major,
        key_head_major[:, :, : sparse_key_blocks * 64],
        layout,
    )
    if routing_mode == _MEAN_ROUTING:
        query_summary = _sequence_block_means(query_head_major)
        key_summary = _sequence_block_means(key_head_major)
        key_aux = key_summary[:, :, :0]
    else:
        query_summary, key_summary, key_aux = _minmax_pool_block_summaries(
            query_head_major,
            key_head_major,
        )
    prepared = _prepare_sparse_piper_attention(
        query_head_major,
        routes.indices,
        routes.head_keep_blocks,
        128**-0.5,
        sparse_key_blocks=sparse_key_blocks,
        route_head_offsets=routes.route_head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
    )
    arguments = (
        prepared.query.data,
        prepared.query.scale,
        query_summary,
        prepared.context.key,
        prepared.context.key_scale,
        key_summary,
        key_aux,
        prepared.context.value,
        prepared.context.value_scale_multiplier,
        prepared.context.value_mean,
        list(ratio_units),
        sparse_key_blocks,
        storage_sequence_length,
        routing_mode,
    )
    block_lengths = torch.tensor([64, 17, 51], dtype=torch.int32, device=query.device)
    return shape, query, value, prepared, arguments, block_lengths


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
def test_block_lengths_mask_internal_key_padding(routing_mode: int) -> None:
    shape, query, _value, prepared, arguments, block_lengths = _block_length_case(
        routing_mode,
    )
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
    corrupted_key = prepared.context.key.clone()
    corrupted_value = prepared.context.value.clone()
    corrupted_key[:, :, ~valid_rows] = 127
    corrupted_value[..., ~valid_rows] = -127
    corrupted_arguments = list(arguments)
    corrupted_arguments[3] = corrupted_key
    corrupted_arguments[7] = corrupted_value
    partial_arguments = list(arguments)
    partial_arguments[-2] = 64 + 17 + 51
    corrupted_arguments[-2] = partial_arguments[-2]

    with torch.no_grad():
        legacy = _sparse_piper_attention_from_quantized_op(*arguments)
        full_blocks = _sparse_piper_attention_from_quantized_op(
            *arguments,
            full_block_lengths,
        )
        expected = _sparse_piper_attention_from_quantized_op(
            *partial_arguments,
            block_lengths,
            2,
        )
        actual = _sparse_piper_attention_from_quantized_op(
            *corrupted_arguments,
            block_lengths,
            2,
        )
        opcheck = torch.library.opcheck(
            _sparse_piper_attention_from_quantized_op,
            (*partial_arguments, block_lengths, 2),
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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
def test_quantized_coarse_residual_supports_internal_block_padding(
    routing_mode: int,
) -> None:
    _shape, _query, value, _prepared, arguments, block_lengths = _block_length_case(
        routing_mode,
    )
    logical_sequence_length = int(block_lengths.sum().item())
    fine_arguments = list(arguments)
    fine_arguments[-2] = logical_sequence_length
    generator = torch.Generator(device="cuda").manual_seed(1223 + routing_mode)
    coarse_gate = torch.randn(
        value.shape,
        dtype=torch.bfloat16,
        device=value.device,
        generator=generator,
    )
    block_mean = mean_pool_block_values(value, block_lengths)
    coarse_key_blocks = block_lengths.numel()
    routing_query_summary = fine_arguments[2]
    routing_key_summary = fine_arguments[5][:, :, :coarse_key_blocks]
    if routing_mode == _MEAN_ROUTING:
        scores = routing_query_summary @ routing_key_summary.mT
    else:
        scores = _minmax_pool_scores(
            routing_query_summary,
            routing_key_summary,
            fine_arguments[6][:, :, :coarse_key_blocks],
        )
    coarse_scale = 128**-0.5
    coarse_arguments = (
        *fine_arguments[:10],
        block_mean,
        coarse_gate,
        *fine_arguments[10:],
        coarse_scale,
        block_lengths,
        coarse_key_blocks,
        2,
    )

    with torch.no_grad():
        fine_output = _sparse_piper_attention_from_quantized_op(
            *fine_arguments,
            block_lengths,
            2,
        )
        expected = fine_output + apply_coarse_attention_residual(
            coarse_attention(
                scores * coarse_scale,
                block_mean[:, :, :coarse_key_blocks],
            ),
            coarse_gate,
        )
        actual = _sparse_piper_attention_with_coarse_residual_from_quantized_op(
            *coarse_arguments,
        )
        opcheck = torch.library.opcheck(
            _sparse_piper_attention_with_coarse_residual_from_quantized_op,
            coarse_arguments,
        )

    assert actual.shape == value.shape
    torch.testing.assert_close(actual, expected, atol=0.00390625, rtol=0.01)
    assert set(opcheck.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_mean_pool_summaries_feed_the_common_quantized_attention() -> None:
    generator = torch.Generator(device="cuda").manual_seed(418)
    shape = (1, 3 * 64, 2, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)
    sparse_key_blocks = 2
    ratio_units = _normalize_head_keep_ratios((0.5, 1.0))
    layout = _resolve_route_layout(ratio_units, sparse_key_blocks, query.device)
    placeholder_routes = packed_minmax_pool_routes_from_sequences(
        query_head_major,
        key_head_major[:, :, : sparse_key_blocks * 64],
        layout,
    )
    prepared = _prepare_sparse_piper_attention(
        query_head_major,
        placeholder_routes.indices,
        placeholder_routes.head_keep_blocks,
        128**-0.5,
        sparse_key_blocks=sparse_key_blocks,
        route_head_offsets=placeholder_routes.route_head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
    )
    query_mean = _sequence_block_means(query_head_major)
    key_mean = _sequence_block_means(key_head_major)
    arguments = (
        prepared.query.data,
        prepared.query.scale,
        query_mean,
        prepared.context.key,
        prepared.context.key_scale,
        key_mean,
        key_mean[:, :, :0],
        prepared.context.value,
        prepared.context.value_scale_multiplier,
        prepared.context.value_mean,
        list(ratio_units),
        sparse_key_blocks,
        shape[1],
        _MEAN_ROUTING,
    )

    with torch.no_grad():
        expected = SparsePiperAttention((0.5, 1.0), routing="mean")(
            query,
            key,
            value,
            sparse_key_blocks=sparse_key_blocks,
        )
        actual = _sparse_piper_attention_from_quantized_op(*arguments)
        opcheck = torch.library.opcheck(
            _sparse_piper_attention_from_quantized_op,
            arguments,
        )

    assert torch.equal(actual, expected)
    assert set(opcheck.values()) == {"SUCCESS"}
