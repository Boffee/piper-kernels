"""AMD support boundaries and on-device coverage of the shared INT8 contract."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._input_activations import apply_input_activation
from piper_kernels.linear.convrot import (
    ConvRotInt8Tensor,
    convrot_int8_compile_options,
    convrot_int8_linear,
)
from piper_kernels.linear.convrot.int8 import _backend, _generic, reference
from piper_kernels.linear.convrot.int8._amd import policy
from piper_kernels.linear.convrot.int8._amd import triton as amd
from piper_kernels.linear.convrot.int8._generic import triton as generic_triton

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ROCm support is Linux-only")

_gpu = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires a supported ROCm GPU",
)


@pytest.mark.parametrize("architecture", ["gfx942", "gfx1100", "gfx1151", "gfx1200", "gfx1201"])
def test_amd_selection_and_independent_auxiliary_support(monkeypatch, architecture):
    target = AcceleratorTarget("hip", architecture)
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    monkeypatch.setattr(generic_triton, "supports_device", lambda device: True)
    value = SimpleNamespace(device=torch.device("cuda"))
    assert _backend.select_linear_backend(value) is amd
    assert _backend.select_add(value) is _generic.add_
    assert _backend.select_addmm(value) is _generic.addmm_
    assert _backend.select_gguf_converter(value) is generic_triton.convert_gguf_out
    assert _backend.select_dequantized_mean(value) is None


@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("hip"),
        AcceleratorTarget("hip", "gfx1036"),
        AcceleratorTarget("hip", "gfx9999"),
        AcceleratorTarget("cuda", "sm120"),
        AcceleratorTarget("cpu"),
    ],
)
def test_amd_policy_rejects_unvalidated_targets(target):
    assert not policy.supports_target(target)
    with pytest.raises(ValueError, match="no optimized policy"):
        policy.select_execution_plan(target, in_features=256)


def test_amd_missing_runtime_falls_back(monkeypatch):
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", "gfx1201")
    )
    value = SimpleNamespace(device=torch.device("cuda"))
    assert _backend.select_linear_backend(value) is None
    assert _backend.select_preparation_backend(value) is _generic
    assert _backend.select_add(value) is _generic.add_
    assert _backend.select_addmm(value) is _generic.addmm_


@pytest.mark.parametrize(
    ("width", "blocks", "warps", "fused"),
    [
        (256, (256, 0, 0), 8, True),
        (5376, (2048, 2048, 2048), 4, True),
        (9216, (8192, 1024, 0), 16, True),
        (12288, (4096, 4096, 4096), 16, True),
        (16384, (16384, 0, 0), 32, True),
        (32768, (32768, 0, 0), 32, False),
    ],
)
def test_rdna4_preparation_policy(width, blocks, warps, fused):
    plan = policy.select_execution_plan(AcceleratorTarget("hip", "gfx1201"), in_features=width)
    assert policy.preparation_blocks(width) == blocks
    assert plan.fused_num_warps == warps
    assert plan.fuse_rotation_quantization is fused
    assert (plan.matmul_block_m, plan.matmul_block_n, plan.matmul_block_k) == (128, 256, 64)
    assert plan.matmul_num_stages == 2


def test_optional_amd_compiler_option_compatibility(monkeypatch):
    target = AcceleratorTarget("hip", "gfx1201")
    monkeypatch.setattr(amd.importlib, "import_module", Mock(return_value=object()))
    assert amd._amd_matmul_compiler_options(target) == {}


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("rows", [1023, 1024, 1025, 2049])
@pytest.mark.parametrize("paired", [False, True])
def test_amd_projection_crosses_cache_group_and_tail_boundaries(rows, paired):
    torch.manual_seed(893)
    width, columns = 256, 335
    qdata = torch.randint(-127, 128, (rows, width), device="cuda", dtype=torch.int8)
    scale = torch.rand(rows, device="cuda") * 0.01
    weights = [
        (
            torch.randint(-127, 128, (columns, width), device="cuda", dtype=torch.int8),
            torch.rand(columns, device="cuda") * 0.01,
            # Zero bias keeps this indexing regression bitwise-checkable against
            # separate PyTorch ops (nonzero bias may be fused into an FP32 FMA).
            torch.zeros(columns, device="cuda"),
        )
        for _ in range(2 if paired else 1)
    ]
    expected = []
    for weight, weight_scale, bias in weights:
        padded_weight = torch.nn.functional.pad(weight, (0, 0, 0, (-columns) % 8))
        value = torch._int_mm(qdata, padded_weight.T)[:, :columns].float()
        value = value * scale[:, None] * weight_scale[None, :]
        expected.append((value + bias).bfloat16())
    storage = torch.full(
        (rows, columns * len(weights) + 7), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    output = storage[:, : columns * len(weights)]
    actual = amd.linear_prepared(
        qdata,
        scale,
        weights[0][0],
        weights[0][1],
        weights[0][2],
        torch.bfloat16,
        out=output,
        second_projection=weights[1] if paired else None,
    )
    assert actual is output
    assert torch.equal(actual, torch.cat(expected, dim=-1))
    assert storage[:, columns * len(weights) :].isnan().all()


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [256, 5376, 9216, 12288, 16384, 32768])
@pytest.mark.parametrize("activation", [None, "gelu_tanh", "swiglu"])
def test_amd_preparation_matches_split_and_populates_storage(dtype, width, activation):
    torch.manual_seed(721)
    raw_width = width * (2 if activation == "swiglu" else 1)
    # Noncontiguous input and multidimensional leading shape.
    value = torch.randn(2, 3, raw_width * 2, device="cuda", dtype=dtype)[..., ::2]
    output = (
        torch.empty(2, 3, width, device="cuda", dtype=torch.int8),
        torch.empty(2, 3, device="cuda"),
    )
    implementation = _backend.require_linear_backend(value)
    assert implementation is amd
    actual = implementation.prepare_input(value, 256, activation, out=output)
    plan = replace(
        amd.default_execution_plan(output[0].reshape(6, width)), fuse_rotation_quantization=False
    )
    expected = amd.prepare_input_with_plan(
        value,
        width,
        256,
        activation_fn=activation,
        execution_plan=plan,
        target=AcceleratorTarget.from_device(value.device),
    )
    assert actual is output
    # Activated fused math may differ at quantization boundaries from PyTorch;
    # keep the same one-bin bound used by the CUDA activation tests.
    error = (actual[0].short() - expected[0].short()).abs()
    assert error.max().item() <= (1 if activation else 0)
    if activation is None:
        assert torch.equal(actual[1], expected[1])
    torch.testing.assert_close(actual[1], expected[1], rtol=2 * torch.finfo(dtype).eps, atol=1e-7)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", [None, "gelu_tanh", "swiglu"])
def test_amd_public_linear_matches_reference(group_size, dtype, activation):
    torch.manual_seed(83)
    k, n = 3 * group_size, 79
    qdata = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    scale = torch.rand(n, 1, device="cuda") * 0.01
    bias = torch.randn(n * 2, device="cuda", dtype=torch.float32)[::2]
    value = torch.randn(129, k * (2 if activation == "swiglu" else 1), device="cuda", dtype=dtype)
    weight = ConvRotInt8Tensor.from_quantized(
        qdata, scale, group_size=group_size, logical_dtype=dtype
    )
    actual = convrot_int8_linear(value, weight, bias, activation_fn=activation)
    prepared = amd.prepare_input(value, group_size, activation)
    expected = reference.linear_prepared(*prepared, qdata, scale, dtype, bias)
    torch.testing.assert_close(actual, expected, rtol=2 * torch.finfo(dtype).eps, atol=1e-6)
    portable = reference.prepare_input(apply_input_activation(value, activation), group_size)
    # The factorized H4 and the reference matrix multiply use different FP32
    # reduction orders: a value within a few ULPs of a half-bin can round apart.
    bin_error = (prepared[0].short() - portable[0].short()).abs()
    assert bin_error.max().item() <= (1 if activation or dtype is torch.float32 else 0)
    if activation is None and dtype is torch.float32:
        assert bin_error.count_nonzero().item() <= prepared[0].numel() * 0.001
    torch.testing.assert_close(prepared[1], portable[1], rtol=2 * torch.finfo(dtype).eps, atol=1e-7)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [256, 9216, 16384, 32768])
@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1e-31])
def test_amd_preparation_zero_and_tiny_scales(dtype, width, magnitude):
    value = torch.zeros(2, width, device="cuda", dtype=dtype)
    value[:, 0] = magnitude
    actual = amd.prepare_input(value, 256)
    expected = reference.prepare_input(value, 256)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("width", [256, 9216, 16384, 32768])
def test_amd_scale_matches_pytorch_scalar_reciprocal_rounding(width):
    # 3.25 / 127 and 3.25 * float32(1 / 127) differ by one FP32 ULP.
    # A regular normalized Hadamard leaves this constant input unchanged.
    value = torch.full((2, width), 3.25, dtype=torch.bfloat16, device="cuda")
    actual = amd.prepare_input(value, 256)
    expected = reference.prepare_input(value, 256)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.gpu
@_gpu
def test_amd_compiled_shared_preparation(monkeypatch):
    torch.manual_seed(294)
    weight = ConvRotInt8Tensor.from_hp(
        torch.randn(79, 256, device="cuda", dtype=torch.bfloat16), group_size=256
    )
    other = ConvRotInt8Tensor.from_hp(
        torch.randn(63, 256, device="cuda", dtype=torch.bfloat16), group_size=256
    )
    value = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)

    def project(value):
        return torch.nn.functional.linear(value, weight), torch.nn.functional.linear(value, other)

    expected = project(value)
    prepare = Mock(wraps=amd.prepare_input)
    monkeypatch.setattr(amd, "prepare_input", prepare)
    call = torch.compile(project, fullgraph=True, options=convrot_int8_compile_options())
    call(value)
    prepare.reset_mock()
    actual = call(value)
    assert prepare.call_count == 1
    for result, reference_result in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference_result, rtol=0, atol=0)
