"""The residual rewrite's matching, on graphs with metadata only and on CPU."""

import pytest
import torch
from _compile_capture import FunctionalTargetCapturePass

from piper_kernels._triton import runtime
from piper_kernels.fusions.residual_updates import _compile as residual_compile
from piper_kernels.fusions.residual_updates import residual_updates_compile_options
from piper_kernels.fusions.residual_updates._graph import RESIDUAL_ADD

_aten = torch.ops.aten
_AUTO_FUNCTIONALIZED = torch.ops.higher_order.auto_functionalized


def _metadata(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, device="meta", dtype=dtype)


def _node(graph, target, args, value):
    node = graph.call_function(target, args)
    node.meta["val"] = value
    return node


def _copy_back_graph(
    updates, *, python_indexing=False, gate_first=True, update_dtype=torch.bfloat16, scaled=False
):
    """``hidden.copy_(...)`` of ``updates`` gated sums, from ``hidden`` unless ``scaled``."""
    graph = torch.fx.Graph()
    names = ("hidden", "gate", "indices", *(f"update{index}" for index in range(updates)))
    values = (
        _metadata(1, 8, 16),
        _metadata(3, 16),
        _metadata(8, dtype=torch.int64),
        *(_metadata(1, 8, 16, dtype=update_dtype) for _ in range(updates)),
    )
    hidden, gate, indices, *update_inputs = (graph.placeholder(name) for name in names)
    for placeholder, value in zip((hidden, gate, indices, *update_inputs), values, strict=True):
        placeholder.meta["val"] = value
    total = _node(graph, _aten.mul.Tensor, (hidden, 2), _metadata(1, 8, 16)) if scaled else hidden
    for update in update_inputs:
        selected = (
            _node(graph, _aten.index.Tensor, (gate, [indices]), _metadata(8, 16))
            if python_indexing
            else _node(graph, _aten.index_select.default, (gate, 0, indices), _metadata(8, 16))
        )
        factors = (selected, update) if gate_first else (update, selected)
        product = _node(graph, _aten.mul.Tensor, factors, _metadata(1, 8, 16))
        total = _node(graph, _aten.add.Tensor, (total, product), _metadata(1, 8, 16))
    graph.call_function(_aten.copy_.default, (hidden, total))
    graph.output(None)
    return graph, hidden


def _adds(graph):
    return [node for node in graph.nodes if node.target is _AUTO_FUNCTIONALIZED]


@pytest.mark.parametrize(
    ("updates", "python_indexing", "gate_first"), [(1, True, True), (3, False, False)]
)
def test_gated_sums_from_the_copied_input_become_chained_adds(
    monkeypatch, updates, python_indexing, gate_first
):
    monkeypatch.setattr(runtime, "supports_device", lambda device: True)
    graph, hidden = _copy_back_graph(
        updates, python_indexing=python_indexing, gate_first=gate_first
    )

    residual_compile._compile_pass(graph, True)

    adds = _adds(graph)
    assert len(adds) == updates
    base = hidden
    for add in adds:
        assert add.args == (RESIDUAL_ADD,)
        assert add.kwargs["base"] is base
        assert add.kwargs["python_indexing"] is python_indexing
        (base,) = add.users
    (copy,) = (node for node in graph.nodes if node.target is _aten.copy_.default)
    assert copy.args == (hidden, base)
    assert not any(node.target is _aten.add.Tensor for node in graph.nodes)


@pytest.mark.parametrize(
    ("supported", "graph_options"),
    [
        (False, {}),
        (True, {"update_dtype": torch.float16}),
        (True, {"scaled": True}),
    ],
    ids=["device-without-triton", "update-dtype-differs", "sums-start-elsewhere"],
)
def test_sums_stay_unless_each_can_add_into_the_copied_input(monkeypatch, supported, graph_options):
    monkeypatch.setattr(runtime, "supports_device", lambda device: supported)
    graph, _ = _copy_back_graph(2, **graph_options)

    residual_compile._compile_pass(graph, True)

    assert not _adds(graph)
    assert sum(node.target is _aten.add.Tensor for node in graph.nodes) == 2


def test_cpu_block_keeps_its_sums_and_runs():
    def block(hidden, update, gate, indices):
        return hidden.copy_(hidden + gate[indices] * update)

    hidden = torch.randn(1, 8, 16, dtype=torch.bfloat16)
    update = torch.randn_like(hidden)
    gate = torch.randn(3, 16, dtype=torch.bfloat16)
    indices = torch.randint(0, 3, (8,))
    expected = hidden + gate[indices] * update
    capture = FunctionalTargetCapturePass()
    options = capture.wrap_options(residual_updates_compile_options())

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(block, fullgraph=True, options=options)(hidden, update, gate, indices)

    torch.testing.assert_close(hidden, expected, rtol=0.02, atol=0.02)
    assert RESIDUAL_ADD not in capture.targets
