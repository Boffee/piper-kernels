"""Tests for direct NVFP4 Triton preparation."""

import pytest
import torch

from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import triton as nvfp4_triton


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("rows", [127, 128, 129])
@pytest.mark.parametrize("activation_fn", [None, "swiglu"])
def test_static_preparation_matches_portable_decomposition(
    rows: int,
    activation_fn: str | None,
) -> None:
    torch.manual_seed(501)
    output_features = 80
    input_features = output_features * (2 if activation_fn == "swiglu" else 1)
    input = torch.randn(  # noqa: A001
        rows,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    per_tensor_scale = torch.tensor(1.0 / 448.0, device="cuda", dtype=torch.float32)

    expected = nvfp4_ops._compiled_prepare_static(input, per_tensor_scale, activation_fn)
    actual = nvfp4_triton.prepare_static(
        input,
        per_tensor_scale,
        swiglu=activation_fn == "swiglu",
    )

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)
