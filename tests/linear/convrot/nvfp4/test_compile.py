"""Tests for ConvRot NVFP4 preparation sharing during compilation."""

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
from piper_kernels.linear.convrot.nvfp4 import (
    ConvRotNVFP4Tensor,
    convrot_nvfp4_compile_options,
)
from piper_kernels.linear.convrot.nvfp4._compile import compile_pass
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout


def _placeholder(graph: torch.fx.Graph, name: str, value: torch.Tensor) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def _linear(
    graph: torch.fx.Graph,
    input: torch.fx.Node,  # noqa: A002 - match linear terminology
    qdata: torch.fx.Node,
    scale: torch.fx.Node,
    global_scale: torch.fx.Node,
    activation_scale: torch.fx.Node | None,
    *,
    dynamic: bool = False,
    group_size: int | bool = 16,
) -> torch.fx.Node:
    node = graph.call_function(
        torch.ops.piper_kernels.convrot_nvfp4_linear.default,
        args=(
            input,
            qdata,
            scale,
            global_scale,
            activation_scale,
            None,
            dynamic,
            group_size,
            False,
        ),
    )
    input_value = input.meta.get("val")
    qdata_value = qdata.meta["val"]
    if isinstance(input_value, torch.Tensor):
        node.meta["val"] = input_value.new_empty((*input_value.shape[:-1], qdata_value.shape[0]))
    return node


def _operands(
    graph: torch.fx.Graph,
    input_value: torch.Tensor,
) -> tuple[torch.fx.Node, ...]:
    input = _placeholder(graph, "input", input_value)  # noqa: A001
    input_features = input_value.shape[-1]
    qdata = _placeholder(
        graph,
        "qdata",
        torch.empty(128, input_features // 2, dtype=torch.uint8, device="meta"),
    )
    scale = _placeholder(
        graph,
        "scale",
        torch.empty(
            nvfp4_layout.scale_shape(128, input_features),
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
    )
    global_scale = _placeholder(graph, "global_scale", torch.empty((), device="meta"))
    activation_scale = _placeholder(
        graph,
        "activation_scale",
        torch.empty((), device="meta"),
    )
    return input, qdata, scale, global_scale, activation_scale


def _run_compile_pass(graph: torch.fx.Graph, *, is_inference: bool = True) -> None:
    torch.fx.GraphModule({}, graph)
    compile_pass(graph, is_inference=is_inference)


def test_pass_shares_compatible_preparation_and_canonicalizes_scale_variants() -> None:
    graph = torch.fx.Graph()
    input, qdata, scale, global_scale, activation_scale = _operands(  # noqa: A001
        graph,
        torch.empty(2, 17, 256, device="meta", dtype=torch.bfloat16),
    )
    other_activation_scale = _placeholder(
        graph,
        "other_activation_scale",
        torch.empty((), device="meta"),
    )
    first = _linear(graph, input, qdata, scale, global_scale, activation_scale)
    second = _linear(graph, input, qdata, scale, global_scale, activation_scale)
    other = _linear(graph, input, qdata, scale, global_scale, other_activation_scale)
    graph.output((first, second, other))

    _run_compile_pass(graph)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default) == 2
    assert targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 3
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    assert targets.count(torch.ops.aten.reshape.default) == 3
    graph.lint()


def test_pass_never_shares_different_rotation_groups() -> None:
    graph = torch.fx.Graph()
    input, qdata, scale, global_scale, activation_scale = _operands(  # noqa: A001
        graph,
        torch.empty(17, 256, device="meta", dtype=torch.bfloat16),
    )
    first = _linear(graph, input, qdata, scale, global_scale, activation_scale)
    second = _linear(graph, input, qdata, scale, global_scale, activation_scale)
    other_group = _linear(
        graph,
        input,
        qdata,
        scale,
        global_scale,
        activation_scale,
        group_size=64,
    )
    graph.output((first, second, other_group))

    _run_compile_pass(graph)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 2
    assert targets.count(torch.ops.piper_kernels.convrot_nvfp4_linear.default) == 1
    graph.lint()


def test_pass_leaves_eligible_singleton_semantic() -> None:
    graph = torch.fx.Graph()
    operands = _operands(
        graph,
        torch.empty(17, 256, device="meta", dtype=torch.bfloat16),
    )
    projected = _linear(graph, *operands)
    graph.output(projected)

    _run_compile_pass(graph)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.piper_kernels.convrot_nvfp4_linear.default) == 1
    assert torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in targets
    graph.lint()


@pytest.mark.parametrize(
    "case",
    [
        "missing-metadata",
        "unsupported-group",
        "boolean-group",
        "wrong-group-width",
        "fp32-input",
    ],
)
def test_pass_fails_closed_for_malformed_semantic_linears(case: str) -> None:
    graph = torch.fx.Graph()
    input_features = 240 if case == "wrong-group-width" else 256
    input_dtype = torch.float32 if case == "fp32-input" else torch.bfloat16
    operands = list(
        _operands(
            graph,
            torch.empty(17, input_features, device="meta", dtype=input_dtype),
        )
    )
    if case == "missing-metadata":
        operands[0].meta.clear()
    group_size: int | bool = 16
    if case == "unsupported-group":
        group_size = 32
    elif case == "boolean-group":
        group_size = True
    elif case == "wrong-group-width":
        group_size = 64
    first = _linear(graph, *operands, group_size=group_size)
    second = _linear(graph, *operands, group_size=group_size)
    graph.output((first, second))
    original = str(graph)

    _run_compile_pass(graph)

    assert str(graph) == original


def test_pass_does_not_rewrite_training_graph() -> None:
    graph = torch.fx.Graph()
    operands = _operands(
        graph,
        torch.empty(17, 256, device="meta", dtype=torch.bfloat16),
    )
    first = _linear(graph, *operands)
    second = _linear(graph, *operands)
    graph.output((first, second))
    original = str(graph)

    _run_compile_pass(graph, is_inference=False)

    assert str(graph) == original


def test_compile_options_install_versioned_idempotent_pass() -> None:
    options = convrot_nvfp4_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options["post_grad_custom_pre_pass"] is compile_pass
    assert convrot_nvfp4_compile_options(options) == options
    assert compile_pass.uuid() == compile_pass.uuid()
    assert compile_pass.uuid()


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self) -> None:
        self.targets: list[object] = []
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


def _quantization(dynamic: bool) -> QuantizeTensorToNVFP4Kwargs:
    return QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(("dynamic", "group_size"), [(False, 16), (True, 64)])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_compile_shares_three_projections(dynamic: bool, group_size: int) -> None:
    torch.manual_seed(621 + group_size + dynamic)
    input = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    rotated_input = rotate_groups(input, group_size)
    activation_scale = None if dynamic else per_tensor_amax_to_scale(rotated_input.abs().amax())
    weights = []
    biases = []
    for _ in range(3):
        dense_weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        rotated_weight = rotate_groups(dense_weight, group_size)
        torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
            rotated_weight,
            per_tensor_scale=per_tensor_amax_to_scale(rotated_weight.abs().amax()),
            act_per_tensor_scale=activation_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=_quantization(dynamic),
        )
        weights.append(ConvRotNVFP4Tensor.from_torchao(torchao_weight, group_size=group_size))
        biases.append(torch.randn(128, device="cuda", dtype=torch.bfloat16))

    def projections(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            F.linear(value, weight, bias) for weight, bias in zip(weights, biases, strict=True)
        )

    expected = projections(input)
    capture = _TargetCapturePass()
    options = convrot_nvfp4_compile_options()
    options["post_grad_custom_pre_pass"] = (
        options["post_grad_custom_pre_pass"],
        capture,
    )
    actual = torch.compile(projections, fullgraph=True, options=options)(input)

    for reference, result in zip(expected, actual, strict=True):
        assert torch.equal(result, reference)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default) == 1
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 3
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in capture.targets
