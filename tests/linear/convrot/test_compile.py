"""Tests for automatic ConvRot preparation sharing during compilation."""

import operator
import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass

from piper_kernels.linear.convrot import (
    ConvRotInt8Tensor,
    convrot_int8_compile_options,
    convrot_int8_linear,
)
from piper_kernels.linear.convrot.int8._compile import (
    compile_pass,
)


def _placeholder(
    graph: torch.fx.Graph,
    name: str,
    value: torch.Tensor,
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def _run_compile_pass(graph: torch.fx.Graph, *, is_inference: bool) -> None:
    torch.fx.GraphModule({}, graph)
    compile_pass(graph, is_inference=is_inference)


def _linear(
    graph: torch.fx.Graph,
    activation: torch.fx.Node,
    qdata: torch.fx.Node,
    scale: torch.fx.Node,
    bias: torch.fx.Node | None,
    group_size: int,
) -> torch.fx.Node:
    node = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        args=(activation, qdata, scale, bias, group_size),
    )
    activation_value = activation.meta["val"]
    qdata_value = qdata.meta["val"]
    node.meta["val"] = activation_value.new_empty(
        (*activation_value.shape[:-1], qdata_value.shape[0])
    )
    node.meta["eager_input_vals"] = (activation_value, qdata_value)
    return node


def test_pass_rewrites_only_compatible_shared_activation_groups() -> None:
    graph = torch.fx.Graph()
    shared = _placeholder(graph, "shared", torch.empty(2, 17, 64, device="meta"))
    other = _placeholder(graph, "other", torch.empty(2, 17, 64, device="meta"))
    q_qdata = _placeholder(graph, "q_qdata", torch.empty(33, 64, dtype=torch.int8))
    q_scale = _placeholder(graph, "q_scale", torch.empty(33, 1))
    k_qdata = _placeholder(graph, "k_qdata", torch.empty(65, 64, dtype=torch.int8))
    k_scale = _placeholder(graph, "k_scale", torch.empty(65, 1))
    v_qdata = _placeholder(graph, "v_qdata", torch.empty(97, 64, dtype=torch.int8))
    v_scale = _placeholder(graph, "v_scale", torch.empty(97, 1))
    bias = _placeholder(graph, "bias", torch.empty(33))

    query = _linear(graph, shared, q_qdata, q_scale, bias, 16)
    consumer = graph.call_function(operator.add, args=(query, 1))
    consumer.meta["val"] = query.meta["val"]
    key = _linear(graph, shared, k_qdata, k_scale, None, 16)
    different_group = _linear(graph, shared, v_qdata, v_scale, None, 64)
    different_activation = _linear(graph, other, v_qdata, v_scale, None, 16)
    graph.output((query, consumer, key, different_group, different_activation))

    _run_compile_pass(graph, is_inference=True)

    call_targets = [node.target for node in graph.nodes if node.op == "call_function"]
    prepare_target = torch.ops.piper_kernels.convrot_int8_prepare_input.default
    prepared_target = torch.ops.piper_kernels.convrot_int8_linear_prepared.default
    original_target = torch.ops.piper_kernels.convrot_int8_linear.default
    assert call_targets.count(prepare_target) == 1
    assert call_targets.count(prepared_target) == 2
    assert call_targets.count(original_target) == 2

    nodes = list(graph.nodes)
    prepared_linears = [node for node in nodes if node.target == prepared_target]
    assert (
        nodes.index(prepared_linears[0]) < nodes.index(consumer) < nodes.index(prepared_linears[1])
    )
    assert all("eager_input_vals" not in node.meta for node in prepared_linears)
    assert all(shared not in node.args for node in prepared_linears)
    assert prepared_linears[0].meta["val"].shape == (2, 17, 33)
    graph.lint()


def test_pass_keeps_singletons_and_training_graphs_unchanged() -> None:
    graph = torch.fx.Graph()
    activation = _placeholder(graph, "activation", torch.empty(17, 64, device="meta"))
    qdata = _placeholder(graph, "qdata", torch.empty(33, 64, dtype=torch.int8))
    scale = _placeholder(graph, "scale", torch.empty(33, 1))
    result = _linear(graph, activation, qdata, scale, None, 16)
    graph.output(result)
    original = str(graph)

    _run_compile_pass(graph, is_inference=False)
    assert str(graph) == original
    _run_compile_pass(graph, is_inference=True)
    assert str(graph) == original


def test_pass_fails_closed_without_post_aot_tensor_metadata() -> None:
    graph = torch.fx.Graph()
    activation = graph.placeholder("activation")
    qdata = graph.placeholder("qdata")
    scale = graph.placeholder("scale")
    result = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        args=(activation, qdata, scale, None, 16),
    )
    graph.output(result)
    original = str(graph)

    _run_compile_pass(graph, is_inference=True)

    assert str(graph) == original


def test_compile_options_install_the_convrot_pass() -> None:
    options = convrot_int8_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options["post_grad_custom_pre_pass"] is compile_pass


def test_compiler_pass_uuid_is_versioned_and_stable() -> None:
    first = compile_pass.uuid()
    second = compile_pass.uuid()

    assert isinstance(first, bytes)
    assert first
    assert first == second


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_compile_options_fold_gelu_tanh() -> None:
    class CapturePass(CustomInferenceAwareGraphPass):
        def __init__(self) -> None:
            self.targets: list[object] = []
            self._uuid = uuid.uuid4().bytes

        def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
            self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

        def uuid(self) -> bytes:
            return self._uuid

    class GeluProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weight = ConvRotInt8Tensor.from_quantized(
                torch.randint(-127, 128, (96, 512), device="cuda", dtype=torch.int8),
                torch.rand(96, 1, device="cuda", dtype=torch.float32) * 0.01,
                group_size=256,
            )
            self.projection = torch.nn.Linear(
                512,
                96,
                bias=True,
                device="cuda",
                dtype=torch.bfloat16,
            )
            self.projection.weight = torch.nn.Parameter(weight, requires_grad=False)
            assert self.projection.bias is not None
            self.projection.bias.requires_grad_(False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.projection(torch.nn.functional.gelu(value, approximate="tanh"))

    torch.manual_seed(382)
    model = GeluProjection().eval()
    value = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    capture = CapturePass()
    options = convrot_int8_compile_options()
    options["post_grad_custom_pre_pass"] = (
        options["post_grad_custom_pre_pass"],
        capture,
    )
    with torch.no_grad():
        weight = model.projection.weight
        assert isinstance(weight, ConvRotInt8Tensor)
        expected = convrot_int8_linear(
            value,
            weight,
            model.projection.bias,
            activation_fn="gelu_tanh",
        )
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=options)(value)

    assert torch.equal(actual, expected)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_linear_prepared.default) == 1
    assert torch.ops.aten.tanh.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_compile_options_match_unmodified_compiled_linears() -> None:
    class TripleProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            for out_features in (128, 257, 384):
                weight = ConvRotInt8Tensor.from_quantized(
                    torch.randint(
                        -127,
                        128,
                        (out_features, 256),
                        device="cuda",
                        dtype=torch.int8,
                    ),
                    torch.rand(out_features, 1, device="cuda", dtype=torch.float32) * 0.01,
                    group_size=256,
                )
                layer = torch.nn.Linear(
                    256,
                    out_features,
                    bias=True,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                assert layer.bias is not None
                layer.bias.requires_grad_(False)
                layers.append(layer)
            self.query, self.key, self.value = layers

        def forward(self, activation: torch.Tensor) -> tuple[torch.Tensor, ...]:
            shared = torch.nn.functional.layer_norm(
                activation,
                (activation.shape[-1],),
            )
            return self.query(shared), self.key(shared), self.value(shared)

    torch.manual_seed(152)
    model = TripleProjection().eval()
    activation = torch.randn(2, 257, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True)(activation)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=convrot_int8_compile_options(),
        )(activation)

    assert all(
        torch.equal(value, reference) for value, reference in zip(actual, expected, strict=True)
    )
