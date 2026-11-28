"""Functional residual-add FX graph emission helpers."""

import operator

import torch

from piper_kernels.fusions.residual_updates import _graph


def test_call_functional_returns_the_new_base_of_a_wrapped_mutation():
    graph = torch.fx.Graph()
    base, update = graph.placeholder("base"), graph.placeholder("update")
    residual_add = _graph.RESIDUAL_ADD
    values = {
        "base": base,
        "update": update,
        "gate": update,
        "gate_indices": update,
        "python_indexing": True,
    }

    added = _graph.call_functional(graph, residual_add, values)

    (call,) = added.all_input_nodes
    assert added.target is operator.getitem
    assert added.args == (call, 1)
    assert call.target is torch.ops.higher_order.auto_functionalized
    assert call.args == (residual_add,)
    assert call.kwargs == values
