"""Tests for the ConvRot tensor representation."""

import pytest
import torch

from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor


def test_dequantize_unrotates_the_stored_weight() -> None:
    qdata = torch.arange(-128, 128, dtype=torch.int8).reshape(16, 16)
    scale = torch.linspace(0.001, 0.016, 16).reshape(16, 1)
    wrapped = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=16,
        logical_dtype=torch.float32,
    )
    expected = rotate_groups(qdata.float() * scale, 16)
    assert torch.equal(wrapped.dequantize(), expected)


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dequantize_accepts_an_output_dtype(output_dtype: torch.dtype) -> None:
    qdata = torch.arange(-128, 128, dtype=torch.int8).reshape(16, 16)
    scale = torch.linspace(0.001, 0.016, 16).reshape(16, 1)
    wrapped = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=16,
        logical_dtype=torch.bfloat16,
    )

    actual = wrapped.dequantize(output_dtype)
    expected = rotate_groups(
        qdata.to(output_dtype) * scale.to(output_dtype),
        16,
    )

    assert actual.dtype is output_dtype
    assert torch.equal(actual, expected)


def test_meta_tensor_preserves_storage_and_rotation_metadata() -> None:
    wrapped = ConvRotInt8Tensor.from_quantized(
        torch.empty(8, 64, dtype=torch.int8, device="meta"),
        torch.empty(8, 1, dtype=torch.float32, device="meta"),
        group_size=64,
    )

    assert wrapped.device.type == "meta"
    assert wrapped.dtype is torch.bfloat16
    assert wrapped.group_size == 64
    assert wrapped.qdata.shape == (8, 64)
    assert wrapped.scale.shape == (8, 1)


def test_dtype_copy_changes_only_the_logical_dtype() -> None:
    wrapped = ConvRotInt8Tensor.from_hp(torch.randn(8, 64), group_size=64)

    converted = wrapped.to(dtype=torch.float16)

    assert type(converted) is ConvRotInt8Tensor
    assert converted.dtype is torch.float16
    assert converted.group_size == wrapped.group_size
    assert converted.qdata.dtype is torch.int8
    assert converted.scale.dtype is torch.float32
    assert converted.qdata is wrapped.qdata
    assert converted.scale is wrapped.scale


@pytest.mark.parametrize("positional", [False, True])
def test_explicit_dtype_copy_duplicates_storage(positional: bool) -> None:
    wrapped = ConvRotInt8Tensor.from_hp(torch.randn(8, 64), group_size=64)

    converted = (
        wrapped.to(torch.float16, False, True)
        if positional
        else wrapped.to(dtype=torch.float16, copy=True)
    )

    assert converted.dtype is torch.float16
    assert converted.qdata is not wrapped.qdata
    assert converted.scale is not wrapped.scale
    assert torch.equal(converted.qdata, wrapped.qdata)
    assert torch.equal(converted.scale, wrapped.scale)


def test_from_quantized_canonicalizes_storage_and_names_logical_dtype() -> None:
    qdata = torch.randint(-128, 128, (8, 128), dtype=torch.int8)[:, ::2]
    scale = torch.arange(1, 17, dtype=torch.float32).reshape(16, 1)[::2]
    assert not qdata.is_contiguous()
    assert not scale.is_contiguous()

    wrapped = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=64,
        logical_dtype=torch.float32,
    )

    assert wrapped.dtype is torch.float32
    assert wrapped.qdata.is_contiguous()
    assert wrapped.scale.is_contiguous()
    assert torch.equal(wrapped.qdata, qdata)
    assert torch.equal(wrapped.scale, scale)


@pytest.mark.parametrize("scale_shape", [(1, 8), (2, 4), (8, 1, 1)])
def test_from_quantized_rejects_ambiguous_scale_shapes(
    scale_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="from_quantized scale must have shape"):
        ConvRotInt8Tensor.from_quantized(
            torch.empty(8, 64, dtype=torch.int8),
            torch.empty(scale_shape, dtype=torch.float32),
            group_size=64,
        )


def test_constructor_rejects_noncanonical_storage_layouts() -> None:
    qdata = torch.empty(8, 128, dtype=torch.int8)[:, ::2]
    scale = torch.empty(16, 1, dtype=torch.float32)[::2]

    with pytest.raises(ValueError, match="qdata and scale must be contiguous"):
        ConvRotInt8Tensor(qdata, torch.empty(8, 1), 64)
    with pytest.raises(ValueError, match="qdata and scale must be contiguous"):
        ConvRotInt8Tensor(torch.empty(8, 64, dtype=torch.int8), scale, 64)


@pytest.mark.parametrize("storage_name", ["qdata", "scale"])
def test_dequantize_revalidates_canonical_storage_layout(storage_name: str) -> None:
    wrapped = ConvRotInt8Tensor.from_hp(torch.randn(8, 64), group_size=64)
    if storage_name == "qdata":
        wrapped.qdata = torch.empty(8, 128, dtype=torch.int8)[:, ::2]
    else:
        wrapped.scale = torch.empty(8, 2, dtype=torch.float32)[:, ::2]
    assert not getattr(wrapped, storage_name).is_contiguous()

    with pytest.raises(ValueError, match="qdata and scale must be contiguous"):
        wrapped.dequantize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_from_hp_rotates_and_quantizes_each_weight_row(dtype: torch.dtype) -> None:
    torch.manual_seed(12)
    weight = torch.randn(7, 32, dtype=dtype)
    rotated = rotate_groups(weight.float(), 16)
    expected_scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    expected_qdata = (rotated / expected_scale).round().clamp(-128, 127).to(torch.int8)

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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
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
        ConvRotInt8Tensor.from_quantized(
            torch.empty(8, 256, dtype=torch.int8),
            torch.empty(8, 1, dtype=torch.float32),
            group_size=group_size,
        )
