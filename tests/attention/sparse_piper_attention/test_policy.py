"""Host-only sparse Piper execution-policy checks."""

import pytest

from piper_kernels._triton.targets import AcceleratorTarget
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


@pytest.mark.parametrize(
    ("target", "tma", "async_copy"),
    [
        (AcceleratorTarget("cuda", "sm120"), True, False),
        (AcceleratorTarget("cuda", "sm89"), False, True),
        (AcceleratorTarget("cuda", "sm121"), False, False),
        (AcceleratorTarget("cuda", "sm88"), False, False),
        (AcceleratorTarget("cuda", "sm90"), False, False),
        (AcceleratorTarget("hip", "gfx1201"), False, False),
        (AcceleratorTarget("cpu"), False, False),
    ],
)
def test_each_target_selects_at_most_one_load_path(target, tma, async_copy):
    assert policy.uses_tensor_descriptors(target) is tma
    assert policy.uses_async_copies(target) is async_copy
    assert policy.supports_target(target) is (tma or async_copy)


@pytest.mark.parametrize(("head_dim", "max_registers"), [(64, 168), (128, None)])
def test_sm89_caps_only_d64_registers(head_dim, max_registers):
    assert policy.sm89_max_registers(head_dim) == max_registers
