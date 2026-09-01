"""Tests for the semantic ConvRot NVFP4 tensor."""

from __future__ import annotations

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor, convrot_nvfp4_linear
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _quantization(dynamic: bool) -> QuantizeTensorToNVFP4Kwargs:
    return QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )


def _meta_weight(*, group_size: int = 16) -> ConvRotNVFP4Tensor:
    output_features, input_features = 128, 256
    source = TorchAONVFP4Tensor(
        torch.empty(
            output_features,
            input_features // 2,
            dtype=torch.uint8,
            device="meta",
        ),
        torch.empty(
            nvfp4_layout.scale_shape(output_features, input_features),
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        16,
        torch.bfloat16,
        torch.empty((), device="meta"),
        None,
        True,
        False,
        _quantization(True),
    )
    return ConvRotNVFP4Tensor.from_torchao(source, group_size=group_size)


def test_from_torchao_reuses_storage_and_attaches_rotation_metadata() -> None:
    source = _meta_weight()

    wrapped = ConvRotNVFP4Tensor.from_torchao(source, group_size=16)

    assert wrapped is source
    assert wrapped.group_size == 16
    with pytest.raises(ValueError, match="uses group size 16, not 64"):
        ConvRotNVFP4Tensor.from_torchao(source, group_size=64)
    with pytest.raises(TypeError, match="requires a group size"):
        ConvRotNVFP4Tensor.from_torchao(source)


def test_device_and_dtype_copies_preserve_wrapper_and_group_size() -> None:
    source = ConvRotNVFP4Tensor(
        torch.empty(128, 128, dtype=torch.uint8),
        torch.empty(
            nvfp4_layout.scale_shape(128, 256),
            dtype=torch.float8_e4m3fn,
        ),
        16,
        torch.bfloat16,
        64,
        torch.empty(()),
        None,
        True,
        False,
        _quantization(True),
    )

    moved = source.to(device="meta", dtype=torch.float16)

    assert type(moved) is ConvRotNVFP4Tensor
    assert moved.device.type == "meta"
    assert moved.orig_dtype is torch.float16
    assert moved.group_size == 64


def test_tensor_flatten_round_trip_preserves_group_size() -> None:
    source = _meta_weight(group_size=64)
    names, metadata = source.__tensor_flatten__()
    tensors = {name: getattr(source, name) for name in names}

    rebuilt = ConvRotNVFP4Tensor.__tensor_unflatten__(
        tensors,
        metadata,
        source.shape,
        source.stride(),
    )

    assert type(rebuilt) is ConvRotNVFP4Tensor
    assert rebuilt.group_size == 64
    assert rebuilt.qdata is source.qdata
    assert rebuilt.scale is source.scale


def test_stable_hash_distinguishes_rotation_groups() -> None:
    source = _meta_weight(group_size=16)
    other = ConvRotNVFP4Tensor.from_torchao(
        TorchAONVFP4Tensor(
            source.qdata,
            source.scale,
            source.block_size,
            source.orig_dtype,
            source.per_tensor_scale,
            source.act_per_tensor_scale,
            source.is_swizzled_scales,
            source.use_triton_kernel,
            source.act_quant_kwargs,
        ),
        group_size=64,
    )

    assert source._stable_hash_for_caching() != other._stable_hash_for_caching()


def test_dequantize_returns_the_unrotated_logical_weight() -> None:
    torch.manual_seed(611)
    logical_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rotated_weight = rotate_groups(logical_weight, 16)
    source = TorchAONVFP4Tensor.to_nvfp4(rotated_weight)
    weight = ConvRotNVFP4Tensor.from_torchao(source, group_size=16)

    expected = rotate_groups(source.dequantize(), 16)

    assert torch.equal(weight.dequantize(), expected)


def test_meta_linear_supports_functional_and_keyword_forms() -> None:
    input = torch.empty(3, 5, 256, dtype=torch.bfloat16, device="meta")  # noqa: A001
    weight = _meta_weight()
    bias = torch.empty(128, dtype=torch.bfloat16, device="meta")

    functional = F.linear(input=input, weight=weight, bias=bias)
    explicit = convrot_nvfp4_linear(input, weight, bias)

    assert functional.shape == (3, 5, 128)
    assert functional.dtype is torch.bfloat16
    assert explicit.shape == functional.shape


@pytest.mark.parametrize("group_size", [15, 32, 128])
def test_constructor_rejects_unsupported_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="group size must be one of"):
        _meta_weight(group_size=group_size)


def test_constructor_rejects_non_matrix_weight() -> None:
    with pytest.raises(ValueError, match="weight must be two-dimensional"):
        ConvRotNVFP4Tensor(
            torch.empty(2, 128, 128, dtype=torch.uint8, device="meta"),
            torch.empty(2, 32, 64, dtype=torch.float8_e4m3fn, device="meta"),
            16,
            torch.bfloat16,
            16,
        )


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self) -> None:
        self.targets: list[object] = []
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


def _cuda_case(
    dynamic: bool,
    group_size: int,
) -> tuple[torch.Tensor, TorchAONVFP4Tensor, ConvRotNVFP4Tensor, torch.Tensor]:
    input = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    logical_weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    rotated_input = rotate_groups(input, group_size)
    rotated_weight = rotate_groups(logical_weight, group_size)
    activation_scale = None if dynamic else per_tensor_amax_to_scale(rotated_input.abs().amax())
    torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
        rotated_weight,
        per_tensor_scale=per_tensor_amax_to_scale(rotated_weight.abs().amax()),
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(dynamic),
    )
    weight = ConvRotNVFP4Tensor.from_torchao(torchao_weight, group_size=group_size)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    return input, torchao_weight, weight, bias


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_cuda_linear_matches_materialized_rotation(dynamic: bool, group_size: int) -> None:
    torch.manual_seed(612 + group_size + dynamic)
    activation, torchao_weight, weight, bias = _cuda_case(dynamic, group_size)
    rotated_input = rotate_groups(activation, group_size)
    prepared_input = nvfp4_ops._prepare_compiled(
        rotated_input,
        weight.act_per_tensor_scale,
        dynamic,
    )
    expected = nvfp4_ops._execute_prepared(
        *prepared_input,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        bias,
        activation.dtype,
    )
    torchao_reference = F.linear(rotated_input, torchao_weight, bias)

    actual = F.linear(activation, weight, bias)

    assert torch.equal(actual, expected)
    relative_l2 = (
        actual.float() - torchao_reference.float()
    ).norm() / torchao_reference.float().norm()
    assert relative_l2 < 0.02


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dynamic", [False, True])
def test_cuda_compile_preserves_semantic_linear(dynamic: bool) -> None:
    torch.manual_seed(613 + dynamic)
    activation, _torchao_weight, weight, bias = _cuda_case(dynamic, 16)

    def projection(value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weight, bias)

    expected = projection(activation)
    capture = _TargetCapturePass()
    actual = torch.compile(
        projection,
        fullgraph=True,
        options={"post_grad_custom_pre_pass": capture},
    )(activation)

    assert torch.equal(actual, expected)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_nvfp4_linear.default) == 1
