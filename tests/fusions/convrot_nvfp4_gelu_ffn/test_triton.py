"""Tests for bounded mixed standard/ConvRot NVFP4 GELU FFN operations."""

import pytest
import torch

from piper_kernels.fusions.convrot_nvfp4_gelu_ffn.triton import (
    _chunked_gelu_ffn_gated_updates_op,
    _chunked_gelu_ffn_op,
)

from ._helpers import make_operands, materialized


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120"),
]


@pytest.mark.parametrize(
    ("up_group_size", "down_group_size"),
    [(16, 64), (16, None), (None, 64)],
    ids=["convrot", "convrot-standard", "standard-convrot"],
)
@pytest.mark.parametrize(
    ("up_dynamic", "down_dynamic"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["static", "down-dynamic", "up-dynamic", "dynamic"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_ffn_matches_materialized(
    up_group_size: int | None,
    down_group_size: int | None,
    up_dynamic: bool,
    down_dynamic: bool,
    dtype: torch.dtype,
) -> None:
    operands = make_operands(
        rows=385,
        up_dynamic=up_dynamic,
        down_dynamic=down_dynamic,
        up_group_size=up_group_size,
        down_group_size=down_group_size,
        dtype=dtype,
        seed=983 + up_dynamic + 10 * down_dynamic,
    )

    expected = materialized(operands)
    actual = _chunked_gelu_ffn_op(*operands.arguments(128))
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()

    assert actual.dtype is dtype
    assert relative_l2 < (0.1 if down_dynamic else 0.07)


@pytest.mark.parametrize("up_high_first", [False, True])
@pytest.mark.parametrize("down_high_first", [False, True])
def test_static_chunked_ffn_supports_independent_nibble_order(
    up_high_first: bool,
    down_high_first: bool,
) -> None:
    operands = make_operands(
        rows=129,
        up_dynamic=False,
        down_dynamic=False,
        up_high_first=up_high_first,
        down_high_first=down_high_first,
        seed=991,
    )

    expected = materialized(operands)
    actual = _chunked_gelu_ffn_op(*operands.arguments(128))

    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.06


def test_dynamic_chunked_ffn_preserves_all_zero_guard() -> None:
    operands = make_operands(
        rows=129,
        up_dynamic=True,
        down_dynamic=True,
        bias_dtype=None,
        seed=993,
    )
    operands.input.zero_()

    actual = _chunked_gelu_ffn_op(*operands.arguments(128))

    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize(
    ("up_group_size", "down_group_size"),
    [(16, 64), (16, None), (None, 64)],
)
@pytest.mark.parametrize("up_dynamic", [False, True])
@pytest.mark.parametrize("down_dynamic", [False, True])
def test_chunked_ffn_gated_updates_matches_materialized(
    up_group_size: int | None,
    down_group_size: int | None,
    up_dynamic: bool,
    down_dynamic: bool,
) -> None:
    operands = make_operands(
        rows=129,
        up_dynamic=up_dynamic,
        down_dynamic=down_dynamic,
        up_group_size=up_group_size,
        down_group_size=down_group_size,
        seed=997,
    )
    rows, output_features = operands.input.shape[0], operands.down.weight.shape[0]
    base = torch.randn(rows, output_features, device="cuda", dtype=torch.bfloat16)
    reusable_update = torch.randn_like(base)
    update_gate = torch.randn(7, output_features, device="cuda", dtype=torch.bfloat16)
    ffn_gate = torch.randn(7, output_features, device="cuda", dtype=torch.bfloat16)
    gate_indices = torch.randint(0, 7, (rows,), device="cuda", dtype=torch.int64)
    hidden = base.float() + update_gate[gate_indices].float() * reusable_update.float()
    expected = (hidden + ffn_gate[gate_indices].float() * materialized(operands).float()).to(
        base.dtype
    )
    actual = reusable_update.clone()

    result = _chunked_gelu_ffn_gated_updates_op(
        *operands.arguments(128)[:-1],
        base,
        actual,
        update_gate,
        ffn_gate,
        gate_indices,
        True,
        128,
    )

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert result is None
    assert relative_l2 < (0.1 if down_dynamic else 0.06)
