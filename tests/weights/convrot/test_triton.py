"""Tests for the Triton ConvRot rotation shared by weights and activations."""

import pytest
import torch

from piper_kernels.weights.convrot import triton as rotation_triton
from piper_kernels.weights.convrot._rotation import rotate_groups


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_factorized_h4_rotation_matches_gpu_reference(
    group_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(41 + group_size)
    activation = torch.randn(5, 2 * group_size, dtype=dtype, device="cuda")
    actual = torch.empty_like(activation)

    rotation_triton.rotate_input(activation, actual, group_size, num_warps=4)
    expected = rotate_groups(activation, group_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_factorized_h4_rotation_handles_rounding_boundary_values(
    group_size: int,
    dtype: torch.dtype,
) -> None:
    one = torch.tensor(1.0, dtype=dtype, device="cuda")
    positive = torch.nextafter(one, torch.tensor(float("inf"), dtype=dtype, device="cuda"))
    negative = torch.nextafter(one, torch.tensor(float("-inf"), dtype=dtype, device="cuda"))
    pattern = torch.stack((one, positive, negative, -one, -positive, -negative, one, -one))
    width = 2 * group_size
    activation = pattern.repeat((width + pattern.numel() - 1) // pattern.numel())[:width]
    activation = activation.reshape(1, width)
    actual = torch.empty_like(activation)

    rotation_triton.rotate_input(activation, actual, group_size, num_warps=4)
    expected = rotate_groups(activation, group_size)

    torch.testing.assert_close(actual, expected)
