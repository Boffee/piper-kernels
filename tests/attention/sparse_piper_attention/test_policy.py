"""Host-only sparse Piper execution-policy checks."""

import pytest

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._amd import policy as amd_policy
from piper_kernels.attention.sparse_piper_attention._contracts import MinmaxScoreSchedule
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
        policy.select_sm120_attention_schedule(
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
    assert policy.select_sm120_attention_schedule(
        64,
        8192,
        8192,
        skip_dense_routing=False,
        has_coarse_residual=False,
        selected_key_rows=128,
    ) == (64, 4)


@pytest.mark.parametrize(
    ("target", "tma", "async_copy"),
    [
        (AcceleratorTarget("cuda", "sm120"), True, False),
        *[(AcceleratorTarget("cuda", f"sm{arch}"), False, True) for arch in (80, 86, 87, 89)],
        (AcceleratorTarget("cuda", "sm121"), False, False),
        (AcceleratorTarget("cuda", "sm90"), False, False),
        (AcceleratorTarget("cuda", "sm100"), False, False),
        (AcceleratorTarget("hip", "gfx1201"), False, False),
        (AcceleratorTarget("cpu"), False, False),
    ],
)
def test_each_target_selects_at_most_one_load_path(target, tma, async_copy):
    assert policy.uses_tensor_descriptors(target) is tma
    assert policy.uses_async_copies(target) is async_copy
    assert policy.supports_target(target) is (tma or async_copy)


@pytest.mark.parametrize(("head_dim", "max_registers"), [(64, 168), (128, None)])
def test_sm8x_caps_only_d64_registers(head_dim, max_registers):
    assert policy.select_sm8x_max_registers(head_dim) == max_registers


@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("cuda", "sm120"), True),
        *[(AcceleratorTarget("cuda", f"sm{arch}"), False) for arch in (80, 86, 87, 89)],
        (AcceleratorTarget("cuda", "sm121"), False),
        (AcceleratorTarget("cuda", "sm90"), False),
        (AcceleratorTarget("cuda", "sm100"), False),
        (AcceleratorTarget("hip", "gfx1201"), False),
        (AcceleratorTarget("cpu"), False),
    ],
)
def test_minmax_score_kernel_is_limited_to_exact_sm120(target, supported):
    assert policy.supports_minmax_scores(target) is supported


@pytest.mark.parametrize(
    ("query_blocks", "expected"),
    [
        (1, MinmaxScoreSchedule(64, 64, 16)),
        (27, MinmaxScoreSchedule(64, 64, 16)),
        (64, MinmaxScoreSchedule(64, 64, 16)),
        (65, MinmaxScoreSchedule(128, 32, 16)),
        (128, MinmaxScoreSchedule(128, 32, 16)),
        (384, MinmaxScoreSchedule(128, 32, 16)),
    ],
)
def test_sm120_minmax_score_schedule_switches_at_chunk_crossover(query_blocks, expected):
    assert policy.select_sm120_minmax_score_schedule(query_blocks) == expected


@pytest.mark.parametrize(
    ("query_blocks", "expected"),
    [
        (1, MinmaxScoreSchedule(32, 64, 16, lds_only_fences=True)),
        (27, MinmaxScoreSchedule(32, 64, 16, lds_only_fences=True)),
        (64, MinmaxScoreSchedule(32, 64, 16, lds_only_fences=True)),
        (127, MinmaxScoreSchedule(32, 64, 16, lds_only_fences=True)),
        (128, MinmaxScoreSchedule(64, 32, 16, lds_only_fences=True)),
        (384, MinmaxScoreSchedule(64, 32, 16, lds_only_fences=True)),
    ],
)
def test_amd_minmax_score_schedule_switches_at_chunk_crossover(query_blocks, expected):
    assert amd_policy.select_minmax_score_schedule(query_blocks) == expected
