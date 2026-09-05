"""Tests for MiniMax-H3 video VAE ConvRot graph specialization."""

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._compile import (
    compile_pass as convrot_int8_compile_pass,
)
from piper_kernels.specializations.minimax_h3_vae import (
    minimax_h3_vae_convrot_int8_compile_options,
)
from piper_kernels.specializations.minimax_h3_vae._compile import (
    _schedule_for,
    _specialize_linears,
)
from piper_kernels.specializations.minimax_h3_vae._compile import (
    compile_pass as specialization_compile_pass,
)
from piper_kernels.specializations.minimax_h3_vae._ops import _execution_plan

_SM120 = AcceleratorTarget("cuda", "sm120")


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1_797, 2_048, 2_048), (128, 128, 128, 4, 2)),
        ((1_797, 16_384, 2_048), (128, 128, 64, 8, 3)),
        ((1_797, 2_048, 8_192), (128, 128, 128, 4, 2)),
        ((3_594, 2_048, 2_048), (128, 128, 64, 8, 3)),
        ((3_594, 16_384, 2_048), (128, 128, 64, 8, 3)),
        ((3_594, 2_048, 8_192), (128, 128, 64, 8, 3)),
        ((7_188, 2_048, 2_048), (128, 128, 128, 4, 2)),
        ((7_188, 16_384, 2_048), (128, 128, 64, 8, 3)),
        ((7_188, 2_048, 8_192), (128, 128, 128, 4, 2)),
    ],
)
def test_sm120_schedule_covers_supported_batch_shapes(
    shape: tuple[int, int, int],
    expected: tuple[int, int, int, int, int],
) -> None:
    assert _schedule_for(shape, target=_SM120) == expected


def _placeholder(graph: torch.fx.Graph, name: str, value: torch.Tensor) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def _linear(
    graph: torch.fx.Graph,
    activation: torch.fx.Node,
    qdata: torch.fx.Node,
    scale: torch.fx.Node,
) -> torch.fx.Node:
    node = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        args=(activation, qdata, scale, None, 256),
    )
    activation_value = activation.meta["val"]
    qdata_value = qdata.meta["val"]
    node.meta["val"] = activation_value.new_empty(
        (*activation_value.shape[:-1], qdata_value.shape[0])
    )
    return node


def _run_passes(
    graph: torch.fx.Graph,
    *,
    is_inference: bool = True,
    target: AcceleratorTarget = _SM120,
) -> None:
    torch.fx.GraphModule({}, graph)
    convrot_int8_compile_pass(graph, is_inference=is_inference)
    if is_inference:
        _specialize_linears(graph, target=target)
    else:
        specialization_compile_pass(graph, is_inference=False)


def test_pass_specializes_h3_linears_after_generic_preparation_sharing() -> None:
    graph = torch.fx.Graph()
    activation = _placeholder(
        graph,
        "activation",
        torch.empty(1_797, 2_048, dtype=torch.float16, device="meta"),
    )
    projection_qdata = _placeholder(
        graph,
        "projection_qdata",
        torch.empty(2_048, 2_048, dtype=torch.int8, device="meta"),
    )
    projection_scale = _placeholder(
        graph,
        "projection_scale",
        torch.empty(2_048, 1, device="meta"),
    )
    query = _linear(graph, activation, projection_qdata, projection_scale)
    key = _linear(graph, activation, projection_qdata, projection_scale)
    value = _linear(graph, activation, projection_qdata, projection_scale)
    attention = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        args=(query, key, value),
    )
    attention.meta["val"] = query.meta["val"]
    ffn_activation = _placeholder(
        graph,
        "ffn_activation",
        torch.empty(1_797, 2_048, dtype=torch.float16, device="meta"),
    )
    w1_qdata = _placeholder(
        graph,
        "w1_qdata",
        torch.empty(16_384, 2_048, dtype=torch.int8, device="meta"),
    )
    w1_scale = _placeholder(graph, "w1_scale", torch.empty(16_384, 1, device="meta"))
    ffn = _linear(graph, ffn_activation, w1_qdata, w1_scale)
    graph.output((attention, ffn))

    _run_passes(graph)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 2
    assert (
        targets.count(torch.ops.piper_kernels.minimax_h3_vae_convrot_int8_linear_prepared.default)
        == 4
    )
    assert targets.count(torch.ops.aten.scaled_dot_product_attention.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear_prepared.default not in targets
    specialized_nodes = [
        node
        for node in graph.nodes
        if node.target
        == torch.ops.piper_kernels.minimax_h3_vae_convrot_int8_linear_prepared.default
    ]
    schedules = [node.args[-1] for node in specialized_nodes]
    assert schedules.count([128, 128, 128, 4, 2]) == 3
    assert schedules.count([128, 128, 64, 8, 3]) == 1
    graph.lint()


def test_pass_leaves_unrecognized_shapes_and_training_graphs_unchanged() -> None:
    graph = torch.fx.Graph()
    activation = _placeholder(graph, "activation", torch.empty(17, 64, device="meta"))
    qdata = _placeholder(graph, "qdata", torch.empty(33, 64, dtype=torch.int8, device="meta"))
    scale = _placeholder(graph, "scale", torch.empty(33, 1, device="meta"))
    output = _linear(graph, activation, qdata, scale)
    graph.output(output)
    original = str(graph)

    _run_passes(graph)

    assert str(graph) == original

    training_graph = torch.fx.Graph()
    activation = _placeholder(
        training_graph,
        "activation",
        torch.empty(1_797, 2_048, device="meta"),
    )
    qdata = _placeholder(
        training_graph,
        "qdata",
        torch.empty(16_384, 2_048, dtype=torch.int8, device="meta"),
    )
    scale = _placeholder(training_graph, "scale", torch.empty(16_384, 1, device="meta"))
    output = _linear(training_graph, activation, qdata, scale)
    training_graph.output(output)
    original = str(training_graph)

    _run_passes(training_graph, is_inference=False)

    assert str(training_graph) == original


def test_pass_leaves_exact_h3_shape_unchanged_off_sm120() -> None:
    graph = torch.fx.Graph()
    activation = _placeholder(graph, "activation", torch.empty(1_797, 2_048, device="meta"))
    qdata = _placeholder(
        graph,
        "qdata",
        torch.empty(16_384, 2_048, dtype=torch.int8, device="meta"),
    )
    scale = _placeholder(graph, "scale", torch.empty(16_384, 1, device="meta"))
    output = _linear(graph, activation, qdata, scale)
    graph.output(output)
    original = str(graph)

    _run_passes(graph, target=AcceleratorTarget("hip", "gfx950"))

    assert str(graph) == original


def test_operator_builds_the_emitted_schedule_over_the_portable_plan() -> None:
    weight = torch.empty(16_384, 2_048, dtype=torch.int8, device="meta")

    plan = _execution_plan(weight, [128, 128, 64, 8, 3])

    assert plan.matmul_block_m == 128
    assert plan.matmul_block_n == 128
    assert plan.matmul_block_k == 64
    assert plan.matmul_num_warps == 8
    assert plan.matmul_num_stages == 3
    assert plan.fuse_rotation_quantization


def test_operator_rejects_malformed_schedule() -> None:
    weight = torch.empty(2_048, 2_048, dtype=torch.int8, device="meta")

    with pytest.raises(ValueError, match="exactly five"):
        _execution_plan(weight, [128, 128])


def test_compile_options_install_generic_then_specialized_pass() -> None:
    options = minimax_h3_vae_convrot_int8_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options["post_grad_custom_pre_pass"] == (
        convrot_int8_compile_pass,
        specialization_compile_pass,
    )


def test_compile_options_are_idempotent() -> None:
    options = minimax_h3_vae_convrot_int8_compile_options()
    options = minimax_h3_vae_convrot_int8_compile_options(options)

    assert options["post_grad_custom_pre_pass"] == (
        convrot_int8_compile_pass,
        specialization_compile_pass,
    )


def test_compiler_pass_uuid_is_versioned_and_stable() -> None:
    assert specialization_compile_pass.uuid()
    assert specialization_compile_pass.uuid() == specialization_compile_pass.uuid()
