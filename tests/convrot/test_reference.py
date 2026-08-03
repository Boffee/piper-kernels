"""Tests for the portable ConvRot reference implementation."""

import hashlib

import pytest
import torch

from piper_kernels.convrot import ConvRotInt8Tensor
from piper_kernels.convrot._reference import (
    build_hadamard,
    dynamic_quantize_rows,
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
    wrapped = ConvRotInt8Tensor.from_packed(
        weight_qdata,
        weight_scale,
        group_size=group_size,
        dtype=torch.float32,
    )
    activation = torch.randn(7, in_features)
    bias = torch.randn(out_features)

    rotated_activation = rotate_groups(activation, group_size)
    activation_qdata, activation_scale = dynamic_quantize_rows(rotated_activation)
    accumulated = activation_qdata.to(torch.int32) @ weight_qdata.T.to(torch.int32)
    expected = accumulated.float() * activation_scale * weight_scale.T + bias

    assert torch.equal(torch.nn.functional.linear(activation, wrapped, bias), expected)
