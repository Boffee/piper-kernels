"""Host-only sparse Piper execution-policy checks."""

import pytest

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._nvidia import policy


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("skip_dense_routing", [False, True])
@pytest.mark.parametrize("coarse", [False, True])
def test_attention_schedule_invariants_and_monotonic_regimes(
    head_dim: int, skip_dense_routing: bool, coarse: bool
) -> None:
    lengths = (64, 4096, 8192, 32768, 100000)
    schedules = [
        policy.select_sm120_attention_schedule(
            head_dim,
            length,
            length,
            skip_dense_routing=skip_dense_routing,
            has_coarse_residual=coarse,
            selected_key_rows=2048,
        )
        for length in lengths
    ]
    for schedule in schedules:
        assert schedule.block_m in (64, 128)
        assert schedule.num_warps in (2, 4)
        # D128 and coarse-residual schedules keep the 64-row query granularity.
        if head_dim == 128 or coarse:
            assert schedule.block_m == 64
    block_ms = [schedule.block_m for schedule in schedules]
    assert block_ms == sorted(block_ms)
    if head_dim == 64 and skip_dense_routing and not coarse:
        assert block_ms[-1] > block_ms[0]


def test_short_query_chunk_and_tiny_key_budget_retain_full_warp_occupancy() -> None:
    short_chunk = policy.select_sm120_attention_schedule(
        64,
        64,
        100000,
        skip_dense_routing=True,
        has_coarse_residual=False,
        selected_key_rows=2048,
    )
    assert short_chunk.block_m == 64

    tiny_budget = policy.select_sm120_attention_schedule(
        64,
        8192,
        8192,
        skip_dense_routing=False,
        has_coarse_residual=False,
        selected_key_rows=128,
    )
    large_budget = policy.select_sm120_attention_schedule(
        64,
        8192,
        8192,
        skip_dense_routing=False,
        has_coarse_residual=False,
        selected_key_rows=2048,
    )
    assert tiny_budget.num_warps >= large_budget.num_warps
    assert tiny_budget.num_warps == 4


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


def test_sm8x_caps_only_d64_registers():
    d64_cap = policy.select_sm8x_max_registers(64)
    assert d64_cap is not None
    assert 0 < d64_cap <= 255
    assert policy.select_sm8x_max_registers(128) is None


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
    ("selector", "expect_lds_only_fences"),
    [
        (policy.select_sm120_minmax_score_schedule, False),
        (_backend.select_amd_minmax_score_schedule, True),
    ],
    ids=["sm120", "rdna4"],
)
def test_minmax_score_schedules_have_valid_tiles_and_monotonic_query_regimes(
    selector, expect_lds_only_fences
):
    schedules = [selector(blocks) for blocks in (1, 27, 64, 65, 127, 128, 384)]
    for schedule in schedules:
        assert _is_power_of_two(schedule.block_m)
        assert _is_power_of_two(schedule.block_n)
        assert _is_power_of_two(schedule.block_k)
        assert 128 % schedule.block_k == 0
        assert schedule.lds_only_fences is expect_lds_only_fences
    block_ms = [schedule.block_m for schedule in schedules]
    assert block_ms == sorted(block_ms)
    assert block_ms[-1] > block_ms[0]
