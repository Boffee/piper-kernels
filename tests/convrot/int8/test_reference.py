"""Tests for the portable INT8 ConvRot reference implementation."""

import pytest
import torch

from piper_kernels.convrot import ConvRotInt8Tensor
from piper_kernels.convrot._rotation import rotate_groups
from piper_kernels.convrot.int8.reference import (
    dynamic_quantize_rows,
)


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


def test_dynamic_quantize_rows_handles_underflowing_float16_scale() -> None:
    value = torch.zeros(2, 16, dtype=torch.float16)
    value[0, 0] = 1e-6

    qdata, scale = dynamic_quantize_rows(value)

    assert qdata[0, 0] == 127
    assert torch.count_nonzero(qdata[0, 1:]) == 0
    assert torch.count_nonzero(qdata[1]) == 0
    assert scale[0, 0] > 0
    assert scale[0, 0].to(torch.float16) == 0
    assert scale[1, 0] == pytest.approx(1e-30)
