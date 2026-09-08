"""Host-only sparse Piper execution-policy checks."""

import pytest

from piper_kernels.attention.sparse_piper_attention._nvidia import policy


@pytest.mark.parametrize(
    ("width", "queries", "keys", "skip_dense_routing", "coarse", "expected"),
    [
        (128, 100000, 100000, False, False, (64, 4)),
        (64, 8191, 8192, False, False, (64, 4)),
        (64, 8192, 8192, False, False, (64, 2)),
        (64, 32767, 32768, True, False, (64, 4)),
        (64, 32768, 32768, True, False, (128, 4)),
        (64, 32768, 32768, True, True, (64, 4)),
        (64, 64, 100000, True, False, (64, 4)),
    ],
)
def test_schedule_respects_width_ranges_and_coarse(
    width, queries, keys, skip_dense_routing, coarse, expected
):
    assert (
        policy.select_attention_schedule(
            width,
            queries,
            keys,
            skip_dense_routing=skip_dense_routing,
            has_coarse_residual=coarse,
            selected_key_rows=2048,
        )
        == expected
    )


def test_very_sparse_long_sequences_keep_four_warps():
    assert policy.select_attention_schedule(
        64,
        8192,
        8192,
        skip_dense_routing=False,
        has_coarse_residual=False,
        selected_key_rows=128,
    ) == (64, 4)
