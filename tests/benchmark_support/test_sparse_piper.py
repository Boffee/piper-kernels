import pytest
import torch
from benchmark_sparse_piper import _parse_args
from lib.sparse_piper import (
    assert_equal_finite,
    reference_prepared_query,
    useful_integer_operations,
)

from piper_kernels.attention.sparse_piper_attention._prepared import (
    _PreparedSparsePiperAttention,
    _PreparedSparsePiperContext,
    _PreparedSparsePiperQuery,
)


@pytest.mark.parametrize("option", ["--sequence", "--batch", "--heads", "--rep-ms", "--samples"])
def test_nonpositive_counts_are_rejected(option):
    with pytest.raises(SystemExit):
        _parse_args([option, "0"])


@pytest.mark.parametrize("ratio", ["0", "-1", "1.1", "nan", "inf"])
def test_invalid_ratios_are_rejected(ratio):
    with pytest.raises(SystemExit):
        _parse_args(["--ratios", ratio])


def test_ragged_benchmark_counts_only_valid_selected_work():
    assert _parse_args(["--sequence", "100000"]).sequence == [100000]
    assert useful_integer_operations(65, [1, 1]) == 4 * 65 * 128 * 2 * 65
    assert useful_integer_operations(4096, [16] * 8) == 17179869184


def test_defaults_include_sparse_and_dense_multiple_sample_measurements():
    args = _parse_args([])
    assert args.ratios == [0.25, 1.0]
    assert args.sequence == [1024, 4096]
    assert args.samples == 3


@pytest.mark.parametrize("internal_padding", [False, True])
@pytest.mark.parametrize("global_offset", [0, 1])
def test_bounded_reference_selects_routes_dense_queries_and_valid_rows(
    internal_padding, global_offset
):
    generator = torch.Generator().manual_seed(713)
    lengths = [3, 64, 7, 41] if internal_padding else [64, 64, 64, 1]
    value = torch.randint(-127, 128, (2, 2, 128, 256), dtype=torch.int8, generator=generator)
    mean = torch.randn((2, 2, 128), generator=generator)
    context = _PreparedSparsePiperContext(
        key=torch.zeros((2, 2, 256, 128), dtype=torch.int8),
        value=value,
        key_scale=torch.ones((2, 2, 4)),
        value_scale_multiplier=torch.full((2, 2, 4, 1), 255.0),
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
        data=torch.zeros((2, 2, (4 - global_offset) * 64, 128), dtype=torch.int8),
        scale=torch.ones((2, 2, (4 - global_offset) * 2)),
        routes=torch.tensor([1, 1, 0], dtype=torch.uint16).repeat(2, 4 - global_offset, 1),
        global_block_offset=global_offset,
    )
    prepared = _PreparedSparsePiperAttention(context, query)
    for batch in range(2):
        for head in range(2):
            for block in range(4 - global_offset):
                global_block = block + global_offset
                prefix = [1] if head == 0 and global_block < 2 else [0, 1]
                indices = [
                    tile * 64 + row for tile in [*prefix, 2, 3] for row in range(lengths[tile])
                ]
                expected = (
                    value[batch, head, :, indices].double().mean(dim=1) + mean[batch, head].double()
                )
                rows = 64 if internal_padding else min(64, 193 - global_block * 64)
                expected = expected.to(torch.bfloat16).expand(rows, 128)
                torch.testing.assert_close(
                    reference_prepared_query(prepared, batch, head, block), expected, rtol=0, atol=0
                )


def test_complete_output_check_rejects_nonfinite_and_late_mismatches():
    expected = torch.zeros((1 << 20) + 1)
    actual = expected.clone()
    assert_equal_finite(actual, expected)
    actual[-1] = 1
    with pytest.raises(AssertionError):
        assert_equal_finite(actual, expected)
    actual[-1] = torch.nan
    with pytest.raises(AssertionError):
        assert_equal_finite(actual, actual)
