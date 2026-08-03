"""Tests for the Triton ConvRot backend."""

import pytest
import torch
from torch import nn

from piper_kernels.convrot import ConvRotInt8Tensor
from piper_kernels.convrot._reference import reference_linear


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_triton_linear_matches_gpu_reference(group_size: int) -> None:
    torch.manual_seed(9)
    in_features = 2 * group_size
    qdata = torch.randint(-127, 128, (96, in_features), dtype=torch.int8, device="cuda")
    scale = torch.rand(96, 1, dtype=torch.float32, device="cuda") * 0.01
    wrapped = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=group_size)
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
        ConvRotInt8Tensor.from_packed(
            torch.randint(-127, 128, (96, 64), dtype=torch.int8, device="cuda"),
            torch.rand(96, 1, dtype=torch.float32, device="cuda") * 0.01,
            group_size=64,
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
