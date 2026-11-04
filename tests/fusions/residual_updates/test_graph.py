"""Residual-update rewrites address operator arguments by schema name."""

import torch

from piper_kernels.fusions.residual_updates._graph import bind_arguments, call_by_name


def test_bind_arguments_names_positional_and_keyword_values():
    graph = torch.fx.Graph()
    left, right = graph.placeholder("left"), graph.placeholder("right")
    node = graph.call_function(torch.ops.aten.add.Tensor, (left, right), {"alpha": 2})

    assert bind_arguments(node) == {"self": left, "other": right, "alpha": 2}


def test_call_by_name_fills_skipped_positional_defaults():
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    node = call_by_name(graph, torch.ops.aten.clamp.default, {"self": source, "max": 1.0})

    assert node.args == (source, None, 1.0)
    assert node.kwargs == {}


def test_call_by_name_passes_keyword_only_arguments_by_name():
    graph = torch.fx.Graph()
    left, right = graph.placeholder("left"), graph.placeholder("right")
    node = call_by_name(
        graph, torch.ops.aten.add.Tensor, {"self": left, "other": right, "alpha": 3}
    )

    assert node.args == (left, right)
    assert node.kwargs == {"alpha": 3}
