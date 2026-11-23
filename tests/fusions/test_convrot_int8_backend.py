"""Shared fusion orchestration needs operations, not vendor launch plans."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _accelerators import forbid_device_probes, mock_cuda_target, mock_rocm_target
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels._input_activations import apply_input_activation
from piper_kernels.fusions.convrot_int8_sparse_piper import output as sparse_output
from piper_kernels.fusions.ffn import _compile as ffn_compile
from piper_kernels.fusions.ffn import _convrot_int8 as ffn
from piper_kernels.linear.convrot.int8 import _backend, reference


@pytest.fixture
def operations(monkeypatch):
    # Deliberately exposes no target, execution plan, or vendor kernel utilities.
    def prepare(input, group_size, activation_fn=None, input_scale=None, *, out=None):  # noqa: A002
        prepared = reference.prepare_input(
            apply_input_activation(input, activation_fn), group_size, input_scale=input_scale
        )
        if out is None:
            return prepared
        for destination, source in zip(out, prepared, strict=True):
            destination.copy_(source)
        return out

    def project(qdata, scale, weight, weight_scale, bias, dtype, *, out, second_projection=None):
        result = reference.linear_prepared(qdata, scale, weight, weight_scale, dtype, bias)
        if second_projection is not None:
            second_weight, second_scale, second_bias = second_projection
            second = reference.linear_prepared(
                qdata, scale, second_weight, second_scale, dtype, second_bias
            )
            result = torch.cat((result, second), dim=-1)
        out.copy_(result)
        return out

    backend = SimpleNamespace(
        prepare_input=Mock(side_effect=prepare),
        linear_prepared=Mock(side_effect=project),
        supports_prepared_swiglu=Mock(return_value=False),
    )
    select = Mock(return_value=backend)
    monkeypatch.setattr(_backend, "require_linear_backend", select)
    return backend, select


def _weight(rows, width, offset=0):
    return (
        ((torch.arange(rows * width).reshape(rows, width) + offset) % 7 - 3).to(torch.int8),
        torch.full((rows, 1), 0.03),
        torch.linspace(-0.1, 0.1, rows),
    )


@pytest.mark.parametrize("chunk_rows", [1, 4, 16])
def test_ffn_uses_operations_with_paired_projection_and_reused_buffers(operations, chunk_rows):
    backend, select = operations
    value = torch.linspace(-1, 1, 160).reshape(2, 5, 16)
    gate, up, down = _weight(32, 16), _weight(32, 16, offset=2), _weight(20, 32)
    actual = ffn.convrot_int8_ffn_op(value, *up, 16, *down, 16, "swiglu", chunk_rows, *gate, 16)
    gate_result = reference.linear(value, *gate[:2], 16, gate[2])
    up_result = reference.linear(value, *up[:2], 16, up[2])
    expected = reference.linear(
        torch.cat((up_result, gate_result), dim=-1), *down[:2], 16, down[2], activation_fn="swiglu"
    )
    # Chunked and full-batch CPU arithmetic can differ by FP32 roundoff.
    # Numerical agreement is approximate; dispatch and storage checks below stay exact.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    select.assert_called_once_with(value)
    chunks = (10 + chunk_rows - 1) // chunk_rows
    assert backend.prepare_input.call_count == backend.linear_prepared.call_count == 2 * chunks
    preparations = backend.prepare_input.call_args_list
    assert [call.kwargs.get("activation_fn") for call in preparations] == [None, "swiglu"] * chunks
    assert len({call.kwargs["out"][0].data_ptr() for call in preparations}) == 1
    assert len({call.kwargs["out"][1].data_ptr() for call in preparations}) == 1
    for index, call in enumerate(backend.linear_prepared.call_args_list):
        if index % 2 == 0:
            assert call.args[2] is up[0]
            assert call.kwargs["second_projection"][0] is gate[0]
            assert call.kwargs["out"].shape[-1] == 64
        else:
            assert call.args[2] is down[0]
            assert "second_projection" not in call.kwargs


@pytest.mark.parametrize("chunk_rows", [1, 4, 16])
def test_ffn_uses_fused_swiglu_operation_and_halves_projection_workspace(operations, chunk_rows):
    backend, select = operations
    value = torch.linspace(-1, 1, 160).reshape(2, 5, 16)
    gate, up, down = _weight(32, 16), _weight(32, 16, offset=2), _weight(20, 32)

    def project_swiglu(qdata, scale, weight, weight_scale, bias, second_projection, dtype, *, out):
        second_weight, second_scale, second_bias = second_projection
        up_result = reference.linear_prepared(qdata, scale, weight, weight_scale, dtype, bias)
        gate_result = reference.linear_prepared(
            qdata, scale, second_weight, second_scale, dtype, second_bias
        )
        paired = torch.cat((up_result, gate_result), dim=-1)
        out.copy_(apply_input_activation(paired, "swiglu").to(dtype))
        return out

    backend.supports_prepared_swiglu = Mock(return_value=True)
    backend.linear_prepared_swiglu = Mock(side_effect=project_swiglu)

    actual = ffn.convrot_int8_ffn_op(value, *up, 16, *down, 16, "swiglu", chunk_rows, *gate, 16)
    gate_result = reference.linear(value, *gate[:2], 16, gate[2])
    up_result = reference.linear(value, *up[:2], 16, up[2])
    expected = reference.linear(
        torch.cat((up_result, gate_result), dim=-1), *down[:2], 16, down[2], activation_fn="swiglu"
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    select.assert_called_once_with(value)
    backend.supports_prepared_swiglu.assert_called_once()
    chunks = (10 + chunk_rows - 1) // chunk_rows
    assert backend.linear_prepared_swiglu.call_count == chunks
    assert backend.linear_prepared.call_count == chunks
    assert [call.kwargs.get("activation_fn") for call in backend.prepare_input.call_args_list] == [
        None,
        None,
    ] * chunks
    assert all(
        call.kwargs["out"].shape[-1] == 32 for call in backend.linear_prepared_swiglu.call_args_list
    )
    assert all(call.args[2] is down[0] for call in backend.linear_prepared.call_args_list)


def test_ffn_rejects_missing_backend_before_allocating_workspace(monkeypatch):
    value = torch.ones(5, 16)
    gate, up, down = _weight(32, 16), _weight(32, 16), _weight(20, 32)
    forbid_device_probes(monkeypatch, "CPU FFN probed GPU hardware")
    allocate = Mock(side_effect=AssertionError("workspace allocated without a backend"))
    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(ValueError, match="optimized linear is unavailable"):
        ffn.convrot_int8_ffn_op(value, *up, 16, *down, 16, "swiglu", 4, *gate, 16)
    allocate.assert_not_called()


@pytest.mark.parametrize("chunk_rows", [1, 4, 16])
def test_ffn_core_supports_one_source_gelu_topology(operations, chunk_rows):
    backend, select = operations
    value = torch.linspace(-1, 1, 160).reshape(2, 5, 16)
    up, down = _weight(32, 16), _weight(20, 32)
    actual = ffn.run_chunked_ffn(
        value,
        (ffn.LinearOperands(*up, 16, None),),
        ffn.LinearOperands(*down, 16, None),
        "gelu_tanh",
        chunk_rows,
    )
    projected = reference.linear(value, *up[:2], 16, up[2])
    expected = reference.linear(
        projected,
        *down[:2],
        16,
        down[2],
        activation_fn="gelu_tanh",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    select.assert_called_once_with(value)
    chunks = (10 + chunk_rows - 1) // chunk_rows
    assert backend.prepare_input.call_count == backend.linear_prepared.call_count == 2 * chunks
    preparations = backend.prepare_input.call_args_list
    assert [call.kwargs.get("activation_fn") for call in preparations] == [
        None,
        "gelu_tanh",
    ] * chunks
    assert all(
        call.kwargs["out"].shape[-1] == 32 for call in backend.linear_prepared.call_args_list[::2]
    )


def _semantic_ffn_graph(device, output_features=20):
    graph = torch.fx.Graph()

    def placeholder(name, shape, dtype):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=dtype, device=device)
        return node

    def linear(name, source, width, columns):
        qdata = placeholder(f"{name}_qdata", (columns, width), torch.int8)
        scale = placeholder(f"{name}_scale", (columns, 1), torch.float32)
        node = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_linear.default,
            (source, qdata, scale, None, 16),
        )
        node.meta["val"] = torch.empty(5, columns, dtype=torch.bfloat16, device=device)
        return node

    value = placeholder("input", (5, 16), torch.bfloat16)
    gate, up = linear("gate", value, 16, 32), linear("up", value, 16, 32)
    activated = graph.call_function(torch.ops.aten.silu.default, (gate,))
    multiplied = graph.call_function(torch.ops.aten.mul.Tensor, (up, activated))
    for node in (activated, multiplied):
        node.meta["val"] = torch.empty(5, 32, dtype=torch.bfloat16, device=device)
    output = linear("down", multiplied, 32, output_features)
    graph.output(output)
    return torch.fx.GraphModule(torch.nn.Module(), graph), value.meta["val"]


@pytest.mark.parametrize(
    ("device", "architecture", "supported"),
    [
        ("cuda:0", "sm120", True),
        ("cuda:0", "gfx1201", True),
        ("cuda:0", "sm90", True),
        ("cuda:0", "sm70", False),
        ("cpu", None, False),
        ("xpu", None, False),
    ],
)
def test_ffn_compiler_uses_backend_support_not_device_family(
    monkeypatch, device, architecture, supported
):
    if architecture == "sm120":
        mock_cuda_target(monkeypatch, (12, 0))
    elif architecture == "gfx1201":
        mock_rocm_target(monkeypatch, "gfx1201")
        monkeypatch.setattr(sys, "platform", "linux")
    elif architecture == "sm90":
        mock_cuda_target(monkeypatch, (9, 0))
    elif architecture == "sm70":
        mock_cuda_target(monkeypatch, (7, 0))
    else:
        forbid_device_probes(monkeypatch, "device probe")
    with FakeTensorMode():
        module, _value = _semantic_ffn_graph(device)
        ffn_compile.fold_ffns(module.graph)
    targets = [node.target for node in module.graph.nodes]
    assert (torch.ops.piper_kernels.convrot_int8_ffn.default in targets) is supported
    assert targets.count(torch.ops.piper_kernels.convrot_int8_linear.default) == (
        0 if supported else 3
    )


@pytest.mark.parametrize("argument_count", [5, 6, 7])
def test_ffn_compiler_preserves_unfused_consumers(monkeypatch, argument_count):
    mock_cuda_target(monkeypatch, (12, 0))
    with FakeTensorMode():
        module, _value = _semantic_ffn_graph("cuda:0", output_features=32)
        graph = module.graph
        output = next(node for node in graph.nodes if node.op == "output")
        down = output.args[0]
        with graph.inserting_before(output):
            consumer = graph.call_function(
                torch.ops.piper_kernels.convrot_int8_linear.default,
                (down, *down.args[1:], *((None,) * (argument_count - 5))),
            )
        consumer.meta = down.meta.copy()
        output.args = (consumer,)
        ffn_compile.fold_ffns(graph)
        graph.lint()
    assert len(consumer.args) == argument_count
    assert consumer.args[0].target == torch.ops.piper_kernels.convrot_int8_ffn.default
    assert down not in graph.nodes


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sparse_output_uses_operations_and_preserves_output_views(operations, dtype):
    backend, select = operations
    sequence = 65
    value = torch.linspace(-1, 1, 2 * sequence * 128).reshape(2, sequence, 128).to(dtype)
    storage = torch.empty(2, 1, 128, 128, dtype=torch.int8)
    weight, scale, bias = _weight(7, 128)
    width, project, prepared = sparse_output._prepare_output_chunk_projector(
        storage,
        sequence,
        weight,
        scale,
        bias,
        16,
        sequence,
        64,
        backend=backend,
        output_dtype=dtype,
    )
    assert width == 7
    backing = torch.full((2, sequence, 9), -999, dtype=dtype)
    output = backing[..., 1:-1]
    chunk = torch.empty(2, 64, 1, 128, dtype=dtype)
    for start, rows in ((0, 64), (64, 1)):
        chunk[:, :rows, 0].copy_(value[:, start : start + rows])
        project(chunk, output, start, rows)
    expected = reference.linear(value, weight, scale, 16, bias)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.all(backing[..., 0] == -999)
    assert torch.all(backing[..., -1] == -999)
    select.assert_not_called()
    assert backend.prepare_input.call_count == backend.linear_prepared.call_count == 4
    for call in backend.prepare_input.call_args_list:
        assert call.kwargs["out"][0].data_ptr() == prepared[0].data_ptr()
        assert call.kwargs["out"][1].data_ptr() == prepared[1].data_ptr()


def test_sparse_gate_uses_selected_projection_operation(operations):
    backend, select = operations
    storage = torch.empty(2, 1, 5, 128, dtype=torch.int8)
    value = torch.linspace(-1, 1, 2 * 5 * 16).reshape(2, 5, 16).to(torch.bfloat16)
    qdata, scale = reference.prepare_input(value, 16)
    weight, weight_scale, bias = _weight(128, 16)
    gate = sparse_output._prepare_gate_projection(
        storage, 5, None, qdata, scale, weight, weight_scale, bias
    )
    output = torch.full((2, 3, 1, 128), -999, dtype=torch.bfloat16)
    gate.project(output, 2, 2)
    expected = reference.linear_prepared(
        qdata[:, 2:4], scale[:, 2:4], weight, weight_scale, torch.bfloat16, bias
    )
    torch.testing.assert_close(output[:, :2, 0], expected, rtol=0, atol=0)
    assert torch.all(output[:, 2] == -999)
    select.assert_called_once_with(qdata)
    assert backend.prepare_input.call_count == 0
    assert backend.linear_prepared.call_count == 2


@pytest.mark.parametrize("static", [False, True])
def test_sparse_gate_prepares_only_requested_rows(operations, static):
    backend, select = operations
    storage = torch.empty(2, 1, 5, 128, dtype=torch.int8)
    value = torch.linspace(-1, 1, 2 * 5 * 16).reshape(2, 5, 16).to(torch.bfloat16)
    input_scale = torch.tensor(0.02) if static else None
    weight, weight_scale, bias = _weight(128, 16)
    gate = sparse_output._prepare_gate_projection(
        storage, 5, None, value, input_scale, weight, weight_scale, bias, 16
    )
    output = torch.full((2, 3, 1, 128), -999, dtype=torch.bfloat16)
    gate.project(output, 2, 2)
    prepared = reference.prepare_input(value[:, 2:4], 16, input_scale=input_scale)
    expected = reference.linear_prepared(*prepared, weight, weight_scale, torch.bfloat16, bias)
    torch.testing.assert_close(output[:, :2, 0], expected, rtol=0, atol=0)
    assert torch.all(output[:, 2] == -999)
    select.assert_called_once_with(value)
    assert backend.prepare_input.call_count == backend.linear_prepared.call_count == 2
    for call in backend.prepare_input.call_args_list:
        assert call.args[0].shape == (2, 16)
        assert call.kwargs["input_scale"] is input_scale
