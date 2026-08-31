"""Tests for automatic NVFP4 preparation sharing during compilation."""

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

from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor, nvfp4_compile_options
from piper_kernels.linear.nvfp4._compile import compile_pass


def _placeholder(graph: torch.fx.Graph, name: str, value: torch.Tensor) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def _linear(
    graph: torch.fx.Graph,
    input: torch.fx.Node,  # noqa: A002
    qdata: torch.fx.Node,
    scale: torch.fx.Node,
    global_scale: torch.fx.Node,
    activation_scale: torch.fx.Node,
) -> torch.fx.Node:
    node = graph.call_function(
        torch.ops.piper_kernels.nvfp4_linear.default,
        args=(input, qdata, scale, global_scale, activation_scale, None, False),
    )
    input_value = input.meta["val"]
    qdata_value = qdata.meta["val"]
    node.meta["val"] = input_value.new_empty((*input_value.shape[:-1], qdata_value.shape[0]))
    return node


def _run_compile_pass(graph: torch.fx.Graph, *, is_inference: bool = True) -> None:
    torch.fx.GraphModule({}, graph)
    compile_pass(graph, is_inference=is_inference)


def test_pass_shares_only_identically_calibrated_compatible_inputs() -> None:
    graph = torch.fx.Graph()
    input = _placeholder(graph, "input", torch.empty(2, 17, 256, device="meta"))  # noqa: A001
    qdata = _placeholder(graph, "qdata", torch.empty(128, 128, dtype=torch.uint8))
    scale = _placeholder(
        graph,
        "scale",
        torch.empty(128, 16, dtype=torch.float8_e4m3fn),
    )
    global_scale = _placeholder(graph, "global_scale", torch.empty(()))
    shared_activation_scale = _placeholder(graph, "shared_activation_scale", torch.empty(()))
    other_activation_scale = _placeholder(graph, "other_activation_scale", torch.empty(()))
    first = _linear(graph, input, qdata, scale, global_scale, shared_activation_scale)
    second = _linear(graph, input, qdata, scale, global_scale, shared_activation_scale)
    other = _linear(graph, input, qdata, scale, global_scale, other_activation_scale)
    graph.output((first, second, other))

    _run_compile_pass(graph)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.piper_kernels.nvfp4_prepare_input.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 2
    assert targets.count(torch.ops.piper_kernels.nvfp4_linear.default) == 1
    assert targets.count(torch.ops.aten.reshape.default) == 2
    graph.lint()


@pytest.mark.parametrize("case", ["missing-metadata", "wrong-dtype", "wrong-width"])
def test_pass_fails_closed_for_malformed_semantic_linears(case: str) -> None:
    graph = torch.fx.Graph()
    input_value = torch.empty(17, 240 if case == "wrong-width" else 256, device="meta")
    input = (  # noqa: A001
        graph.placeholder("input")
        if case == "missing-metadata"
        else _placeholder(graph, "input", input_value)
    )
    qdata_dtype = torch.int8 if case == "wrong-dtype" else torch.uint8
    qdata = _placeholder(graph, "qdata", torch.empty(128, 128, dtype=qdata_dtype))
    scale = _placeholder(
        graph,
        "scale",
        torch.empty(128, 16, dtype=torch.float8_e4m3fn),
    )
    global_scale = _placeholder(graph, "global_scale", torch.empty(()))
    activation_scale = _placeholder(graph, "activation_scale", torch.empty(()))
    first = graph.call_function(
        torch.ops.piper_kernels.nvfp4_linear.default,
        args=(input, qdata, scale, global_scale, activation_scale, None, False),
    )
    second = graph.call_function(
        torch.ops.piper_kernels.nvfp4_linear.default,
        args=(input, qdata, scale, global_scale, activation_scale, None, False),
    )
    graph.output((first, second))
    original = str(graph)

    _run_compile_pass(graph)

    assert str(graph) == original


def test_compile_options_install_versioned_idempotent_pass() -> None:
    options = nvfp4_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options["post_grad_custom_pre_pass"] is compile_pass
    assert nvfp4_compile_options(options) == options
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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_compile_shares_three_static_projections() -> None:
    torch.manual_seed(423)
    input = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    activation_scale = per_tensor_amax_to_scale(input.abs().amax())
    weights = []
    biases = []
    quantization = QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=False,
    )
    for _ in range(3):
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
            weight,
            per_tensor_scale=per_tensor_amax_to_scale(weight.abs().amax()),
            act_per_tensor_scale=activation_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=quantization,
        )
        weights.append(PiperNVFP4Tensor.from_torchao(torchao_weight))
        biases.append(torch.randn(128, device="cuda", dtype=torch.bfloat16))

    def projections(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            F.linear(value, weight, bias) for weight, bias in zip(weights, biases, strict=True)
        )

    expected = projections(input)
    capture = _TargetCapturePass()
    options = nvfp4_compile_options()
    options["post_grad_custom_pre_pass"] = (
        options["post_grad_custom_pre_pass"],
        capture,
    )
    actual = torch.compile(projections, fullgraph=True, options=options)(input)

    for reference, result in zip(expected, actual, strict=True):
        relative_l2 = (reference.float() - result.float()).norm() / reference.float().norm()
        assert relative_l2 < 0.01
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_prepare_input.default) == 1
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 3
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
