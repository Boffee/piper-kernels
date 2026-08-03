"""Tests for the ConvRot tensor representation."""

import pytest
import torch

from piper_kernels.convrot import ConvRotInt8Tensor
from piper_kernels.convrot._rotation import rotate_groups


def test_dequantize_unrotates_the_stored_weight() -> None:
    qdata = torch.arange(-128, 128, dtype=torch.int8).reshape(16, 16)
    scale = torch.linspace(0.001, 0.016, 16).reshape(16, 1)
    wrapped = ConvRotInt8Tensor.from_packed(
        qdata,
        scale,
        group_size=16,
        dtype=torch.float32,
    )
    expected = rotate_groups(qdata.float() * scale, 16)
    assert torch.equal(wrapped.dequantize(), expected)


def test_meta_tensor_preserves_storage_and_rotation_metadata() -> None:
    wrapped = ConvRotInt8Tensor.from_packed(
        torch.empty(8, 64, dtype=torch.int8, device="meta"),
        torch.empty(8, 1, dtype=torch.float32, device="meta"),
        group_size=64,
    )

    assert wrapped.device.type == "meta"
    assert wrapped.dtype is torch.bfloat16
    assert wrapped.group_size == 64
    assert wrapped.qdata.shape == (8, 64)
    assert wrapped.scale.shape == (8, 1)


@pytest.mark.parametrize("group_size", [15, 32, 128])
def test_rejects_unsupported_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="group size must be one of"):
        ConvRotInt8Tensor.from_packed(
            torch.empty(8, 256, dtype=torch.int8),
            torch.empty(8, 1, dtype=torch.float32),
            group_size=group_size,
        )
