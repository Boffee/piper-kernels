"""Residual-update rewrites address operator arguments by schema name."""

import operator

import torch

from piper_kernels.fusions.residual_updates import _ops  # noqa: F401 - registers residual_add_
from piper_kernels.fusions.residual_updates._graph import (
    bind_arguments,
    call_functional,
    functional_target,
)


def test_bind_arguments_names_positional_and_keyword_values():
    graph = torch.fx.Graph()
    left, right = graph.placeholder("left"), graph.placeholder("right")
    node = graph.call_function(torch.ops.aten.add.Tensor, (left, right), {"alpha": 2})

    assert bind_arguments(node) == {"self": left, "other": right, "alpha": 2}


def test_call_functional_returns_the_new_base_of_a_wrapped_mutation():
    graph = torch.fx.Graph()
    base, update = graph.placeholder("base"), graph.placeholder("update")
    residual_add = torch.ops.piper_kernels.residual_add_.default
    values = {
        "base": base,
        "update": update,
        "gate": update,
        "gate_indices": update,
        "python_indexing": True,
    }

    added = call_functional(graph, residual_add, values)

    (call,) = added.all_input_nodes
    assert added.target is operator.getitem
    assert added.args == (call, 1)
    assert call.target is torch.ops.higher_order.auto_functionalized
    assert call.args == (residual_add,)
    assert call.kwargs == values
    assert functional_target(call) is residual_add
    assert functional_target(base) is None
