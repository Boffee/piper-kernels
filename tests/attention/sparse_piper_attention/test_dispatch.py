"""Public sparse Piper Attention contract tests."""

import pytest
import torch

import piper_kernels
from piper_kernels import SparsePiperAttention


def _inputs(
    device: str = "cpu",
    sequence_length: int = 3 * 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(52)
    shape = (1, sequence_length, 2, 128)
    query = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    key = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    return query, key, value


def _attention(
    ratios: tuple[float, ...],
) -> SparsePiperAttention:
    return SparsePiperAttention(ratios)


def test_public_api_exports_sparse_attention_backend() -> None:
    assert piper_kernels.SparsePiperAttention is SparsePiperAttention


def test_mean_pool_backend_runs_through_the_common_attention_path() -> None:
    query, key, value = _inputs()
    attention = SparsePiperAttention((0.5, 1.0), routing="mean_pool")

    with torch.no_grad():
        output = attention(query, key, value, sparse_key_blocks=2)

    assert output.shape == query.shape
    assert output.dtype is torch.bfloat16
    assert torch.isfinite(output).all()


def test_every_query_uses_sparse_prefix_plus_dense_suffix_on_cpu() -> None:
    query, key, value = _inputs()
    attention = _attention((0.5, 1.0))

    with torch.no_grad():
        output = attention(
            query,
            key,
            value,
            sparse_key_blocks=2,
        )

    assert output.shape == query.shape
    assert output.dtype is torch.bfloat16
    assert torch.isfinite(output).all()


def test_backend_owns_only_an_immutable_semantic_ratio_profile() -> None:
    attention = SparsePiperAttention((0.75, 0.25, 1.0, 0.5))

    assert attention.head_keep_ratios == (0.75, 0.25, 1.0, 0.5)
    assert attention._head_keep_ratio_units == (750_000, 250_000, 1_000_000, 500_000)
    assert attention.routing == "dsa"


def test_routing_policy_is_validated_at_construction() -> None:
    with pytest.raises(ValueError, match="'dsa' or 'mean_pool'"):
        SparsePiperAttention((1.0,), routing="unknown")


def test_dense_suffix_is_included_for_prefix_and_suffix_queries() -> None:
    shape = (1, 3 * 64, 2, 128)
    query = torch.zeros(shape, dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.empty_like(query)
    value[:, :64] = 1
    value[:, 64:128] = 2
    value[:, 128:] = 10
    attention = _attention((0.5, 1.0))

    with torch.no_grad():
        output = attention(query, key, value, sparse_key_blocks=2)

    expected_by_head = torch.tensor([5.5, 13 / 3], dtype=torch.bfloat16)
    torch.testing.assert_close(output[0, 0, :, 0], expected_by_head)
    torch.testing.assert_close(output[0, -1, :, 0], expected_by_head)


def test_backend_accepts_sparse_prefix_length_changes_without_derived_state() -> None:
    query, key, value = _inputs()
    attention = _attention((0.5, 0.5))

    with torch.no_grad():
        one_block = attention(query, key, value, sparse_key_blocks=1)
        two_blocks = attention(query, key, value, sparse_key_blocks=2)

    assert one_block.shape == two_blocks.shape == query.shape


@pytest.mark.parametrize("sequence_length", [64, 65, 127, 128, 129, 181, 191, 192, 193])
def test_public_contract_accepts_ragged_logical_lengths(sequence_length: int) -> None:
    query, key, value = _inputs(sequence_length=sequence_length)
    attention = _attention((0.5, 1.0))

    with torch.no_grad():
        output = attention(
            query,
            key,
            value,
            sparse_key_blocks=sequence_length // 64,
        )

    assert output.shape == query.shape
    assert output.is_contiguous()
    assert torch.isfinite(output).all()


def test_partial_dense_suffix_attends_only_valid_rows() -> None:
    sequence_length = 65
    shape = (1, sequence_length, 2, 128)
    query = torch.zeros(shape, dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.ones_like(query)
    value[:, -1] = 10
    attention = _attention((1.0, 1.0))

    with torch.no_grad():
        output = attention(query, key, value, sparse_key_blocks=1)

    expected = torch.full_like(output, (64 + 10) / sequence_length)
    torch.testing.assert_close(output, expected, atol=0.015625, rtol=0)


@pytest.mark.parametrize("routing", ["mean_pool", "dsa"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_internal_block_lengths_make_padded_values_unobservable(
    routing: str,
    device: str,
) -> None:
    if device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)
    ):
        pytest.skip("requires exact NVIDIA SM120")
    query, key, value = _inputs(device=device)
    block_lengths = torch.tensor([64, 17, 51], device=device, dtype=torch.int32)
    valid_rows = torch.arange(query.shape[1], device=device) % 64
    valid_rows = valid_rows < block_lengths.repeat_interleave(64)
    corrupted = [tensor.clone() for tensor in (query, key, value)]
    for tensor in corrupted:
        tensor[:, ~valid_rows] = torch.randn_like(tensor[:, ~valid_rows]).mul_(100)
    attention = SparsePiperAttention((0.5, 1.0), routing=routing)

    with torch.no_grad():
        expected = attention(
            query,
            key,
            value,
            sparse_key_blocks=2,
            block_lengths=block_lengths,
        )
        actual = attention(
            *corrupted,
            sparse_key_blocks=2,
            block_lengths=block_lengths,
        )

    assert actual.shape == query.shape
    torch.testing.assert_close(actual[:, valid_rows], expected[:, valid_rows], atol=0, rtol=0)


@pytest.mark.parametrize(
    "block_lengths",
    [
        torch.tensor([64, 64], dtype=torch.int32),
        torch.tensor([64, 64, 64], dtype=torch.int64),
    ],
)
def test_internal_block_lengths_reject_invalid_metadata(block_lengths: torch.Tensor) -> None:
    query, key, value = _inputs()
    attention = SparsePiperAttention((0.5, 1.0))

    with pytest.raises(ValueError, match="block lengths"):
        attention(
            query,
            key,
            value,
            sparse_key_blocks=2,
            block_lengths=block_lengths,
        )


def test_contract_rejects_sparse_prefix_larger_than_the_sequence() -> None:
    query, key, value = _inputs()
    attention = _attention((0.5, 0.5))

    with pytest.raises(ValueError, match="sparse_key_blocks"):
        attention(query, key, value, sparse_key_blocks=4)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_path_runs_and_writes_engine_layout() -> None:
    query, key, value = _inputs("cuda")
    attention = _attention((0.5, 1.0))

    with torch.no_grad():
        output = attention(query, key, value, sparse_key_blocks=2)

    assert output.shape == query.shape
    assert output.is_contiguous()
    assert torch.isfinite(output).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_sm120_path_returns_contiguous_output_for_noncontiguous_inputs() -> None:
    query, key, value = (
        tensor.transpose(1, 2).contiguous().transpose(1, 2)
        for tensor in _inputs("cuda", sequence_length=193)
    )
    attention = _attention((0.5, 1.0))

    with torch.no_grad():
        output = attention(query, key, value, sparse_key_blocks=3)

    assert not query.is_contiguous()
    assert output.is_contiguous()
    assert torch.isfinite(output).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("sequence_length", [192, 193])
def test_sm120_custom_op_passes_opcheck(sequence_length: int) -> None:
    from piper_kernels.attention.sparse_piper_attention.dispatch import (  # noqa: PLC0415
        _sparse_piper_attention_op,
    )

    query, key, value = _inputs("cuda", sequence_length)
    attention = _attention((0.5, 1.0))
    result = torch.library.opcheck(
        _sparse_piper_attention_op,
        (
            query,
            key,
            value,
            list(attention._head_keep_ratio_units),
            sequence_length // 64,
            128**-0.5,
            attention._routing_mode,
        ),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize(
    ("sparse_key_blocks", "ratios"),
    [(1, (1.0, 1.0)), (2, (0.5, 1.0)), (3, (1 / 3, 2 / 3))],
)
def test_sm120_matches_the_portable_quantized_reference(
    sparse_key_blocks: int,
    ratios: tuple[float, float],
) -> None:
    query, key, value = _inputs()
    cpu_attention = _attention(ratios)
    cuda_attention = _attention(ratios)

    with torch.no_grad():
        reference = cpu_attention(
            query,
            key,
            value,
            sparse_key_blocks=sparse_key_blocks,
        )
        actual = cuda_attention(
            query.cuda(),
            key.cuda(),
            value.cuda(),
            sparse_key_blocks=sparse_key_blocks,
        ).cpu()

    relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2 < 0.015


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("sequence_length", [64, 65, 127, 128, 129, 181, 191, 192, 193])
def test_sm120_ragged_lengths_match_the_portable_reference(sequence_length: int) -> None:
    query, key, value = _inputs(sequence_length=sequence_length)
    attention = _attention((0.5, 1.0))
    sparse_key_blocks = sequence_length // 64

    with torch.no_grad():
        expected = attention(
            query,
            key,
            value,
            sparse_key_blocks=sparse_key_blocks,
        )
        actual = attention(
            query.cuda(),
            key.cuda(),
            value.cuda(),
            sparse_key_blocks=sparse_key_blocks,
        ).cpu()

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert actual.shape == query.shape
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert relative_l2 < 0.015


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_operator_is_opaque_to_a_full_compile_graph() -> None:
    query, key, value = _inputs("cuda", sequence_length=193)
    attention = _attention((0.5, 1.0))

    def run(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return attention(query, key, value, sparse_key_blocks=3)

    compiled = torch.compile(run, fullgraph=True)
    with torch.no_grad():
        expected = run(query, key, value)
        actual = compiled(query, key, value)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_mean_pool_operator_is_opaque_to_a_full_compile_graph() -> None:
    query, key, value = _inputs("cuda", sequence_length=193)
    attention = SparsePiperAttention((0.5, 1.0), routing="mean_pool")

    def run(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return attention(query, key, value, sparse_key_blocks=3)

    compiled = torch.compile(run, fullgraph=True)
    with torch.no_grad():
        expected = run(query, key, value)
        actual = compiled(query, key, value)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
