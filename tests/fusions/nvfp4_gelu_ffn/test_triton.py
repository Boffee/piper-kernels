"""Tests for direct bounded standard NVFP4 GELU FFN operations."""

import pytest
import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.fusions.nvfp4_gelu_ffn import _preparation
from piper_kernels.fusions.nvfp4_gelu_ffn.triton import (
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


@pytest.mark.parametrize("rows", [1, 127, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compiled_gelu_scale_matches_fp32_reference(rows: int, dtype: torch.dtype) -> None:
    torch.manual_seed(961)
    projections = torch.randn(rows, 512, device="cuda", dtype=dtype)
    expected = per_tensor_amax_to_scale(
        F.gelu(projections.float(), approximate="tanh").abs().amax()
    )

    torch.testing.assert_close(
        _preparation.dynamic_gelu_scale(projections),
        expected,
        rtol=1e-6,
        atol=0,
    )


@pytest.mark.parametrize("rows", [127, 385], ids=["short", "ragged-multi-chunk"])
@pytest.mark.parametrize("up_dynamic", [False, True], ids=["up-static", "up-dynamic"])
@pytest.mark.parametrize("down_dynamic", [False, True], ids=["down-static", "down-dynamic"])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_ffn_matches_materialized(
    rows: int,
    up_dynamic: bool,
    down_dynamic: bool,
    bias_dtype: torch.dtype | None,
    dtype: torch.dtype,
) -> None:
    operands = make_operands(
        rows=rows,
        up_dynamic=up_dynamic,
        down_dynamic=down_dynamic,
        dtype=dtype,
        bias_dtype=bias_dtype,
        seed=963 + rows + up_dynamic + down_dynamic,
    )

    expected = materialized(operands)
    actual = _chunked_gelu_ffn_op(*operands.arguments(128))
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()

    assert actual.dtype is dtype
    assert relative_l2 < (0.1 if down_dynamic else 0.06)


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
        seed=967,
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
        seed=969,
    )
    operands.input.zero_()

    actual = _chunked_gelu_ffn_op(*operands.arguments(128))

    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("up_dynamic", [False, True])
@pytest.mark.parametrize("down_dynamic", [False, True])
@pytest.mark.parametrize(
    "output_features",
    [384, 640],
    ids=["reused-projection-workspace", "separate-output-workspace"],
)
def test_chunked_ffn_gated_updates_matches_materialized(
    up_dynamic: bool,
    down_dynamic: bool,
    output_features: int,
) -> None:
    operands = make_operands(
        rows=129,
        up_dynamic=up_dynamic,
        down_dynamic=down_dynamic,
        output_features=output_features,
        seed=971,
    )
    rows = operands.input.shape[0]
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
        False,
        128,
    )

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert result is None
    assert relative_l2 < (0.1 if down_dynamic else 0.06)
