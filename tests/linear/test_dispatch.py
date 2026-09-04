"""Tests for shared semantic-linear dispatch helpers."""

import pytest
import torch

from piper_kernels.linear._dispatch import apply_linear_autocast


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_linear_autocast_preserves_ineligible_operands(dtype: torch.dtype) -> None:
    input = torch.ones(2, dtype=dtype)  # noqa: A001
    weight = torch.ones(2, dtype=dtype)
    bias = torch.ones(2, dtype=dtype)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        converted = apply_linear_autocast(input, weight, bias)

    assert all(
        actual is expected
        for actual, expected in zip(converted, (input, weight, bias), strict=True)
    )


def test_linear_autocast_converts_each_eligible_operand() -> None:
    input = torch.ones(2, dtype=torch.float32)  # noqa: A001
    weight = torch.ones(2, dtype=torch.float16)
    bias = torch.ones(2, dtype=torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        converted = apply_linear_autocast(input, weight, bias)

    assert all(tensor is not None and tensor.dtype is torch.bfloat16 for tensor in converted)
