"""Tests for ConvRot's runtime tensor and execution backends."""

import hashlib

import pytest
import torch
from torch import nn

from piper_kernels.convrot import (
    ConvRotInt8Tensor,
    int8_convrot_linear,
    to_convrot_int8_tensor,
)
from piper_kernels.convrot._reference import (
    build_hadamard,
    dynamic_quantize_rows,
    reference_linear,
    rotate_groups,
)


@pytest.mark.parametrize(
    ("size", "digest"),
    [
        (16, "9b46fb3c57a096bd73a10d9a089ce835fd22c80fc252f10576658e3f958d72ac"),
        (64, "42759176f7fc530ed4a0ae8acbfcf1bb3c4f86fc34ab1b89e3e50a4ed44383af"),
        (256, "de75fba5a830070acfbfec9cf3fba8a70ef4bfc05e9e6e4dc1fa9bb994f3bc5d"),
    ],
)
def test_regular_hadamard_order_matches_comfy_kitchen(size: int, digest: str) -> None:
    """Pin the full sign pattern; orthogonality alone cannot catch a basis reorder."""
    matrix = build_hadamard(size)
    signs = (matrix * size**0.5).to(torch.int8).view(torch.uint8).flatten().tolist()
    assert hashlib.sha256(bytes(signs)).hexdigest() == digest
    assert torch.equal(matrix @ matrix.T, torch.eye(size))


def test_cpu_linear_matches_explicit_w8a8_reference() -> None:
    torch.manual_seed(4)
    group_size, in_features, out_features = 16, 32, 11
    weight = torch.randn(out_features, in_features)
    rotated_weight = rotate_groups(weight, group_size)
    weight_scale = (rotated_weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    weight_qdata = (rotated_weight / weight_scale).round().clamp(-128, 127).to(torch.int8)
    wrapped = to_convrot_int8_tensor(
        weight_qdata,
        weight_scale,
        group_size,
        dtype=torch.float32,
    )
    activation = torch.randn(7, in_features)
    bias = torch.randn(out_features)

    rotated_activation = rotate_groups(activation, group_size)
    activation_qdata, activation_scale = dynamic_quantize_rows(rotated_activation)
    accumulated = activation_qdata.to(torch.int32) @ weight_qdata.T.to(torch.int32)
    expected = accumulated.float() * activation_scale * weight_scale.T + bias

    functional = int8_convrot_linear(activation, weight_qdata, weight_scale, 16, bias)
    assert torch.equal(functional, expected)
    assert torch.equal(torch.nn.functional.linear(activation, wrapped, bias), expected)


def test_dequantize_unrotates_the_stored_weight() -> None:
    qdata = torch.arange(-128, 128, dtype=torch.int8).reshape(16, 16)
    scale = torch.linspace(0.001, 0.016, 16).reshape(16, 1)
    wrapped = to_convrot_int8_tensor(qdata, scale, 16, dtype=torch.float32)
    expected = rotate_groups(qdata.float() * scale, 16)
    assert torch.equal(wrapped.dequantize(), expected)


def test_meta_tensor_preserves_storage_and_rotation_metadata() -> None:
    wrapped = to_convrot_int8_tensor(
        torch.empty(8, 64, dtype=torch.int8, device="meta"),
        torch.empty(8, 1, dtype=torch.float32, device="meta"),
        64,
    )

    assert wrapped.device.type == "meta"
    assert wrapped.dtype is torch.bfloat16
    assert wrapped.group_size == 64
    assert wrapped.qdata.shape == (8, 64)
    assert wrapped.scale.shape == (8, 1)


@pytest.mark.parametrize("group_size", [15, 32, 128])
def test_rejects_unsupported_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="group size must be one of"):
        to_convrot_int8_tensor(
            torch.empty(8, 256, dtype=torch.int8),
            torch.empty(8, 1, dtype=torch.float32),
            group_size,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_triton_linear_matches_gpu_reference(group_size: int) -> None:
    torch.manual_seed(9)
    in_features = 2 * group_size
    qdata = torch.randint(-127, 128, (96, in_features), dtype=torch.int8, device="cuda")
    scale = torch.rand(96, 1, dtype=torch.float32, device="cuda") * 0.01
    wrapped = ConvRotInt8Tensor(qdata, scale, group_size)
    activation = torch.randn(37, in_features, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(96, dtype=torch.bfloat16, device="cuda")

    expected = reference_linear(activation, qdata, scale, group_size, bias)
    actual = torch.nn.functional.linear(activation, wrapped, bias)
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_linear_runs_under_torch_compile() -> None:
    module = nn.Linear(64, 96, bias=True, device="meta", dtype=torch.bfloat16)
    module.weight = nn.Parameter(
        ConvRotInt8Tensor(
            torch.randint(-127, 128, (96, 64), dtype=torch.int8, device="cuda"),
            torch.rand(96, 1, dtype=torch.float32, device="cuda") * 0.01,
            64,
        ),
        requires_grad=False,
    )
    module.bias = nn.Parameter(
        torch.randn(96, dtype=torch.bfloat16, device="cuda"),
        requires_grad=False,
    )
    activation = torch.randn(17, 64, dtype=torch.bfloat16, device="cuda")
    expected = module(activation)
    actual = torch.compile(module, fullgraph=False)(activation)
    assert torch.equal(actual, expected)
