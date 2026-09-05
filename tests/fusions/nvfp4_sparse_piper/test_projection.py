"""Tests for chunked NVFP4 sparse-Piper projection epilogues."""

from __future__ import annotations

import pytest
import torch

from piper_kernels.attention.sparse_piper_attention._routes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.nvfp4_sparse_piper import key, query, value
from piper_kernels.linear.nvfp4.triton import linear_mean

from ._helpers import (
    exact_sm120_available,
    key_reference,
    make_operands,
    materialize_projection,
    materialize_qk,
    query_reference,
    value_reference,
)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_chunked_qkv_epilogues_match_materialized_fp32_contract() -> None:
    operands = make_operands()
    q_projection = operands.projection(0)
    k_projection = operands.projection(1)
    v_projection = operands.projection(2)
    biases = tuple(torch.randn(256, device="cuda", dtype=torch.float32) for _ in range(3))
    expected_query = query_reference(
        materialize_qk(
            q_projection,
            operands.query_norm,
            operands.cos,
            operands.sin,
            bias=biases[0],
            norm_epsilon=1e-5,
        ),
        128**-0.5,
    )
    expected_key = key_reference(
        materialize_qk(
            k_projection,
            operands.key_norm,
            operands.cos,
            operands.sin,
            bias=biases[1],
            norm_epsilon=1e-5,
        )
    )
    value_mean = linear_mean(*v_projection.as_tuple(), biases[2], 1, 193).view(1, 2, 128)
    materialized_value = materialize_projection(v_projection, biases[2]).view(193, 2, 128)
    expected_value = value_reference(materialized_value, value_mean)

    actual_query = query.project_query(
        *q_projection.as_tuple(),
        biases[0],
        operands.query_norm,
        operands.cos,
        operands.sin,
        1e-5,
        128**-0.5,
        128,
    )
    actual_key = key.project_key(
        *k_projection.as_tuple(),
        biases[1],
        operands.key_norm,
        operands.cos,
        operands.sin,
        1e-5,
        128,
    )
    actual_value = value.project_value_with_block_means(
        *v_projection.as_tuple(),
        biases[2],
        value_mean,
        128,
    )

    assert int((actual_query[0].to(torch.int16) - expected_query[0]).abs().max()) <= 1
    torch.testing.assert_close(actual_query[1], expected_query[1], atol=1e-5, rtol=2e-3)
    torch.testing.assert_close(actual_query[2], expected_query[2], atol=0.125, rtol=0.01)
    assert int((actual_key[0].to(torch.int16) - expected_key[0]).abs().max()) <= 1
    torch.testing.assert_close(actual_key[1], expected_key[1], atol=1e-4, rtol=3e-3)
    torch.testing.assert_close(actual_key[2], expected_key[2], atol=0.125, rtol=0.01)
    torch.testing.assert_close(actual_key[3], expected_key[3], atol=0.125, rtol=0.01)
    assert int((actual_value[0].to(torch.int16) - expected_value[0]).abs().max()) <= 1
    torch.testing.assert_close(actual_value[1], expected_value[1], atol=2e-5, rtol=2e-3)
    expected_block_mean = torch.stack(
        [
            materialized_value[start : start + 64].float().mean(dim=0)
            for start in range(0, materialized_value.shape[0], 64)
        ],
        dim=1,
    )[None]
    torch.testing.assert_close(actual_value[2], expected_block_mean, atol=0.015625, rtol=0.01)

    exact_mean = materialized_value.float().mean(dim=0)[None]
    smallest_tile_step = (expected_value[1] / 255.0).amin(dim=2)
    assert bool(((value_mean - exact_mean).abs() <= smallest_tile_step * 0.05).all())


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_mean_pool_summaries_match_materialized_fp32_contract() -> None:
    operands = make_operands()
    q_projection = operands.projection(0)
    k_projection = operands.projection(1)
    transformed_query = materialize_qk(
        q_projection,
        operands.query_norm,
        operands.cos,
        operands.sin,
        norm_epsilon=1e-5,
    )
    transformed_key = materialize_qk(
        k_projection,
        operands.key_norm,
        operands.cos,
        operands.sin,
        norm_epsilon=1e-5,
    )

    actual_query = query.project_query(
        *q_projection.as_tuple(),
        None,
        operands.query_norm,
        operands.cos,
        operands.sin,
        1e-5,
        128**-0.5,
        128,
        _MEAN_ROUTING,
    )
    actual_key = key.project_key(
        *k_projection.as_tuple(),
        None,
        operands.key_norm,
        operands.cos,
        operands.sin,
        1e-5,
        128,
        _MEAN_ROUTING,
    )

    def block_means(sequence: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                sequence[:, :, start : start + 64].mean(dim=2)
                for start in range(0, sequence.shape[2], 64)
            ],
            dim=2,
        )

    torch.testing.assert_close(
        actual_query[2],
        block_means(transformed_query),
        atol=0.125,
        rtol=0.01,
    )
    torch.testing.assert_close(
        actual_key[2],
        block_means(transformed_key),
        atol=0.125,
        rtol=0.01,
    )
    assert actual_key[3].numel() == 0


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("routing_mode", [_MEAN_ROUTING, _MINMAX_ROUTING])
def test_query_projection_range_uses_compact_storage_and_global_rows(
    routing_mode: int,
) -> None:
    operands = make_operands(sequence_length=193)
    projection = operands.projection(0)
    start, rows = 128, 65
    actual = query._launch_query_range(
        *projection.as_tuple(),
        None,
        operands.query_norm,
        operands.cos,
        operands.sin,
        1e-5,
        128**-0.5,
        128,
        routing_mode,
        None,
        chunk_start=start,
        chunk_rows=rows,
    )
    transformed = materialize_qk(
        projection,
        operands.query_norm,
        operands.cos,
        operands.sin,
        norm_epsilon=1e-5,
    )[:, :, start : start + rows]
    expected = query_reference(transformed, 128**-0.5)

    assert int((actual[0].to(torch.int16) - expected[0]).abs().max()) <= 1
    torch.testing.assert_close(actual[1], expected[1], atol=1e-5, rtol=2e-3)
    expected_summary = (
        torch.stack((transformed[:, :, :64].mean(dim=2), transformed[:, :, 64:].mean(dim=2)), 2)
        if routing_mode == _MEAN_ROUTING
        else expected[2]
    )
    torch.testing.assert_close(actual[2], expected_summary, atol=0.125, rtol=0.01)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_value_block_means_respect_internal_block_lengths() -> None:
    operands = make_operands(sequence_length=128)
    projection = operands.projection(2)
    block_lengths = torch.tensor([64, 17], device="cuda", dtype=torch.int32)
    value_mean = torch.zeros((1, 2, 128), device="cuda", dtype=torch.float32)

    actual = value.project_value_with_block_means(
        *projection.as_tuple(),
        None,
        value_mean,
        128,
        block_lengths,
    )
    materialized = materialize_projection(projection).view(128, 2, 128)
    expected = torch.stack(
        (materialized[:64].mean(dim=0), materialized[64:81].mean(dim=0)),
        dim=1,
    )[None]

    torch.testing.assert_close(actual[2], expected, atol=0.015625, rtol=0.01)
    assert not bool(actual[0][..., 81:].any())


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
def test_qkv_projection_respects_internal_block_lengths(routing_mode: int) -> None:
    operands = make_operands(sequence_length=128)
    q_projection = operands.projection(0)
    k_projection = operands.projection(1)
    v_projection = operands.projection(2)
    block_lengths = torch.tensor([64, 17], device="cuda", dtype=torch.int32)
    valid_rows = (torch.arange(64, device="cuda")[None, :] < block_lengths[:, None]).flatten()
    transformed_query = materialize_qk(
        q_projection,
        operands.query_norm,
        operands.cos,
        operands.sin,
        norm_epsilon=1e-5,
    )
    transformed_key = materialize_qk(
        k_projection,
        operands.key_norm,
        operands.cos,
        operands.sin,
        norm_epsilon=1e-5,
    )
    value_mean = linear_mean(
        *v_projection.as_tuple(),
        None,
        1,
        128,
        block_lengths,
    ).view(1, 2, 128)

    actual_query = query.project_query(
        *q_projection.as_tuple(),
        None,
        operands.query_norm,
        operands.cos,
        operands.sin,
        1e-5,
        128**-0.5,
        128,
        routing_mode,
        block_lengths,
    )
    actual_key = key.project_key(
        *k_projection.as_tuple(),
        None,
        operands.key_norm,
        operands.cos,
        operands.sin,
        1e-5,
        128,
        routing_mode,
        block_lengths,
    )
    actual_value = value.project_value(
        *v_projection.as_tuple(),
        None,
        value_mean,
        128,
        block_lengths,
    )

    assert not bool(actual_query[0][:, :, ~valid_rows].any())
    assert not bool(actual_key[0][:, :, ~valid_rows].any())
    assert not bool(actual_value[0][..., ~valid_rows].any())

    valid = valid_rows.unflatten(0, (2, 64))
    mask = valid[None, None, :, :, None]
    query_blocks = transformed_query.unflatten(2, (2, 64))
    key_blocks = transformed_key.unflatten(2, (2, 64))
    if routing_mode == _MEAN_ROUTING:
        denominator = block_lengths[None, None, :, None]
        expected_query_summary = query_blocks.masked_fill(~mask, 0).sum(dim=3) / denominator
        expected_key_summary = key_blocks.masked_fill(~mask, 0).sum(dim=3) / denominator
        assert actual_key[3].numel() == 0
    else:
        expected_query_summary = query_blocks.masked_fill(~mask, -torch.inf).amax(
            dim=3
        ) + query_blocks.masked_fill(~mask, torch.inf).amin(dim=3)
        expected_key_summary = key_blocks.masked_fill(~mask, -torch.inf).amax(dim=3)
        expected_key_aux = key_blocks.masked_fill(~mask, torch.inf).amin(dim=3)
        torch.testing.assert_close(actual_key[3], expected_key_aux, atol=0.125, rtol=0.01)
    torch.testing.assert_close(actual_query[2], expected_query_summary, atol=0.125, rtol=0.01)
    torch.testing.assert_close(actual_key[2], expected_key_summary, atol=0.125, rtol=0.01)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_nvfp4_projection_custom_ops_pass_opcheck() -> None:
    operands = make_operands(sequence_length=128)
    q_projection = operands.projection(0)
    k_projection = operands.projection(1)
    v_projection = operands.projection(2)
    block_lengths = torch.tensor([64, 31], device="cuda", dtype=torch.int32)
    value_mean = linear_mean(
        *v_projection.as_tuple(),
        None,
        1,
        128,
        block_lengths,
    ).view(1, 2, 128)
    cases = (
        (
            query.project_query,
            (
                *q_projection.as_tuple(),
                None,
                operands.query_norm,
                operands.cos,
                operands.sin,
                1e-5,
                128**-0.5,
                128,
                _MINMAX_ROUTING,
                block_lengths,
            ),
        ),
        (
            key.project_key,
            (
                *k_projection.as_tuple(),
                None,
                operands.key_norm,
                operands.cos,
                operands.sin,
                1e-5,
                128,
                _MINMAX_ROUTING,
                block_lengths,
            ),
        ),
        (
            value.project_value,
            (*v_projection.as_tuple(), None, value_mean, 128, block_lengths),
        ),
        (
            value.project_value_with_block_means,
            (
                *v_projection.as_tuple(),
                None,
                value_mean,
                128,
                block_lengths,
            ),
        ),
    )

    for operation, arguments in cases:
        # SchemaCheckMode compares every operand with allclose, whose CUDA
        # implementation does not support FP8 scale tensors yet.
        result = torch.library.opcheck(
            operation,
            arguments,
            test_utils=(
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ),
        )
        assert set(result.values()) == {"SUCCESS"}


def test_projection_fake_kernels_propagate_padded_shapes() -> None:
    input_qdata = torch.empty((193, 128), device="meta", dtype=torch.uint8)
    input_scale = torch.empty((64, 64), device="meta", dtype=torch.float8_e4m3fn)
    input_per_tensor_scale = torch.empty((), device="meta", dtype=torch.float32)
    weight_qdata = torch.empty((3 * 128, 128), device="meta", dtype=torch.uint8)
    weight_scale = torch.empty((96, 64), device="meta", dtype=torch.float8_e4m3fn)
    weight_per_tensor_scale = torch.empty((), device="meta", dtype=torch.float32)
    norm = torch.empty(128, device="meta", dtype=torch.bfloat16)
    cos = torch.empty((193, 96), device="meta", dtype=torch.float32)
    sin = torch.empty_like(cos)
    prepared = (
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        None,
    )

    projected_query = query.project_query(*prepared, norm, cos, sin, 1e-5, 128**-0.5, 128)
    projected_key = key.project_key(*prepared, norm, cos, sin, 1e-5, 128)
    projected_value = value.project_value(
        *prepared,
        torch.empty((1, 3, 128), device="meta", dtype=torch.float32),
        128,
    )
    summarized_value = value.project_value_with_block_means(
        *prepared,
        torch.empty((1, 3, 128), device="meta", dtype=torch.float32),
        128,
    )

    assert [output.shape for output in projected_query] == [
        (1, 3, 256, 128),
        (1, 3, 8),
        (1, 3, 4, 128),
    ]
    assert [output.shape for output in projected_key] == [
        (1, 3, 256, 128),
        (1, 3, 4),
        (1, 3, 4, 128),
        (1, 3, 4, 128),
    ]
    assert [output.shape for output in projected_value] == [
        (1, 3, 128, 256),
        (1, 3, 4, 1),
    ]
    assert [output.shape for output in summarized_value] == [
        (1, 3, 128, 256),
        (1, 3, 4, 1),
        (1, 3, 4, 128),
    ]
