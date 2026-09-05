"""Tests for the bias dtype contract shared by Piper linear operators."""

import pytest
import torch

from piper_kernels.linear import _bias


@pytest.mark.parametrize("dtype", _bias.SUPPORTED_DTYPES)
def test_supported_bias_dtypes(dtype: torch.dtype) -> None:
    bias = torch.empty(3, dtype=dtype)

    _bias.validate_dtype(bias, "test linear")

    assert _bias.is_supported_dtype(dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.complex64])
def test_unsupported_bias_dtypes(dtype: torch.dtype) -> None:
    bias = torch.empty(3, dtype=dtype)

    with pytest.raises(ValueError, match="must be FP16, BF16, or FP32"):
        _bias.validate_dtype(bias, "test linear")

    assert not _bias.is_supported_dtype(dtype)
