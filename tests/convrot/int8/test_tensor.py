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


def test_from_packed_normalizes_flat_scale_to_column() -> None:
    scale = torch.arange(1, 9, dtype=torch.float32)
    wrapped = ConvRotInt8Tensor.from_packed(
        torch.empty(8, 64, dtype=torch.int8),
        scale,
        group_size=64,
    )

    assert wrapped.scale.shape == (8, 1)
    assert torch.equal(wrapped.scale[:, 0], scale)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_from_hp_rotates_and_quantizes_each_weight_row(dtype: torch.dtype) -> None:
    torch.manual_seed(12)
    weight = torch.randn(7, 32, dtype=dtype)
    rotated = rotate_groups(weight, 16)
    expected_scale = (rotated.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(
        min=1e-30
    )
    expected_qdata = (
        (rotated / expected_scale.to(dtype)).round().clamp(-128, 127).to(torch.int8)
    )

    wrapped = ConvRotInt8Tensor.from_hp(weight, group_size=16)

    assert wrapped.dtype is dtype
    assert wrapped.group_size == 16
    assert wrapped.qdata.dtype is torch.int8
    assert wrapped.scale.dtype is torch.float32
    assert wrapped.scale.shape == (7, 1)
    assert torch.equal(wrapped.qdata, expected_qdata)
    assert torch.equal(wrapped.scale, expected_scale)


def test_from_hp_detaches_quantized_storage_from_autograd() -> None:
    weight = torch.randn(3, 16, requires_grad=True)

    wrapped = ConvRotInt8Tensor.from_hp(weight, group_size=16)

    assert not wrapped.qdata.requires_grad
    assert not wrapped.scale.requires_grad


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_from_hp_quantizes_cuda_weight() -> None:
    weight = torch.randn(9, 64, dtype=torch.bfloat16, device="cuda")

    wrapped = ConvRotInt8Tensor.from_hp(weight, group_size=64)

    assert wrapped.device.type == "cuda"
    assert wrapped.qdata.device.type == "cuda"
    assert wrapped.scale.device.type == "cuda"
    assert wrapped.qdata.shape == weight.shape
    assert wrapped.scale.shape == (weight.shape[0], 1)


@pytest.mark.parametrize(
    ("weight", "message"),
    [
        (torch.empty(2, 3, 16), "must be 2-D"),
        (torch.empty(2, 16, dtype=torch.int32), "must use float16, bfloat16, or float32"),
        (torch.empty(2, 16, device="meta"), "cannot quantize a meta tensor"),
        (torch.empty(2, 24), "is not divisible by group size"),
    ],
)
def test_from_hp_rejects_unsupported_dense_weight(
    weight: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ConvRotInt8Tensor.from_hp(weight, group_size=16)


@pytest.mark.parametrize("scale_shape", [(8,), (1, 8), (2, 4), (7, 1)])
def test_constructor_rejects_noncanonical_scale_shape(
    scale_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match=r"scale must be float32 with shape \(8, 1\)"):
        ConvRotInt8Tensor(
            torch.empty(8, 64, dtype=torch.int8),
            torch.empty(scale_shape, dtype=torch.float32),
            64,
        )


@pytest.mark.parametrize("group_size", [15, 32, 128])
def test_rejects_unsupported_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="group size must be one of"):
        ConvRotInt8Tensor.from_packed(
            torch.empty(8, 256, dtype=torch.int8),
            torch.empty(8, 1, dtype=torch.float32),
            group_size=group_size,
        )
