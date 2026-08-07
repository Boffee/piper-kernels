"""Tests for the Triton ConvRot backend."""

import pytest
import torch
from torch import nn

from piper_kernels.convrot import ConvRotInt8Tensor
from piper_kernels.convrot._rotation import rotate_groups
from piper_kernels.convrot.int8.reference import reference_addmm_, reference_linear


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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("beta", "alpha"), [(0.25, 1.5), (0, 1.5), (0.25, 0)])
def test_triton_addmm_matches_gpu_reference(
    dtype: torch.dtype,
    beta: float,
    alpha: float,
) -> None:
    torch.manual_seed(18)
    weight = torch.randn(96, 64, dtype=dtype, device="cuda")
    mat1 = torch.randn(96, 8, dtype=dtype, device="cuda")
    mat2 = torch.randn(8, 64, dtype=dtype, device="cuda")
    actual = ConvRotInt8Tensor.from_hp(weight, group_size=64)
    expected = actual.clone()

    reference_addmm_(expected.qdata, expected.scale, mat1, mat2, 64, beta, alpha)
    result = actual.addmm_(mat1, mat2, beta=beta, alpha=alpha)

    assert result is actual
    qdata_error = (actual.qdata.to(torch.int16) - expected.qdata.to(torch.int16)).abs()
    assert qdata_error.max().item() <= 2
    assert torch.allclose(
        actual.scale,
        expected.scale,
        rtol=2 * torch.finfo(dtype).eps,
        atol=1e-7,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_addmm_runs_under_torch_compile() -> None:
    torch.manual_seed(30)
    weight = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
    mat1 = torch.randn(32, 4, dtype=torch.bfloat16, device="cuda")
    mat2 = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    expected = ConvRotInt8Tensor.from_hp(weight, group_size=64)
    actual = expected.clone()
    expected.addmm_(mat1, mat2, beta=0.5, alpha=1.25)

    def merge(
        target: ConvRotInt8Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> ConvRotInt8Tensor:
        return target.addmm_(left, right, beta=0.5, alpha=1.25)

    result = torch.compile(merge, fullgraph=True)(actual, mat1, mat2)

    assert result is actual
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_addmm_handles_underflowing_float16_scale() -> None:
    rotated_update = torch.zeros(1, 16, dtype=torch.float16, device="cuda")
    rotated_update[0, 0] = 1e-6
    mat1 = torch.ones(1, 1, dtype=torch.float16, device="cuda")
    mat2 = rotate_groups(rotated_update, 16)
    qdata = torch.zeros(1, 16, dtype=torch.int8, device="cuda")
    scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    actual = ConvRotInt8Tensor.from_packed(
        qdata.clone(),
        scale.clone(),
        group_size=16,
        dtype=torch.float16,
    )
    expected = actual.clone()

    reference_addmm_(expected.qdata, expected.scale, mat1, mat2, 16, 0, 1)
    actual.addmm_(mat1, mat2, beta=0)

    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    assert actual.qdata[0, 0] == 127
    assert torch.count_nonzero(actual.qdata[0, 1:]) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_linear_handles_underflowing_float16_activation_scale() -> None:
    rotated_activation = torch.zeros(1, 16, dtype=torch.float16, device="cuda")
    rotated_activation[0, 0] = 1e-6
    activation = rotate_groups(rotated_activation, 16)
    qdata = torch.arange(-8, 8, dtype=torch.int8, device="cuda").reshape(1, 16)
    scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    weight = ConvRotInt8Tensor.from_packed(
        qdata,
        scale,
        group_size=16,
        dtype=torch.float16,
    )

    expected = reference_linear(activation, qdata, scale, 16)
    actual = torch.nn.functional.linear(activation, weight)

    assert torch.equal(actual, expected)
