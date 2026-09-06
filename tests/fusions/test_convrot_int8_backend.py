"""Shared fusion orchestration needs operations, not vendor launch plans."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_sparse_piper import output as sparse_output
from piper_kernels.fusions.convrot_int8_swiglu_ffn import _compile as ffn_compile
from piper_kernels.fusions.convrot_int8_swiglu_ffn import triton as ffn
from piper_kernels.linear._input_activations import apply_input_activation
from piper_kernels.linear.convrot.int8 import _backend, reference


@pytest.fixture
def operations(monkeypatch):
    # Deliberately exposes no target, execution plan, or vendor kernel utilities.
    def prepare(input, group_size, activation_fn=None, *, out):  # noqa: A002
        prepared = reference.prepare_input(apply_input_activation(input, activation_fn), group_size)
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
        prepare_input=Mock(side_effect=prepare), linear_prepared=Mock(side_effect=project)
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
    actual = ffn._run_chunked_swiglu_ffn(value, *gate, 16, *up, 16, *down, 16, chunk_rows)
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
    assert [call.kwargs["activation_fn"] for call in preparations] == [None, "swiglu"] * chunks
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


def test_ffn_rejects_missing_backend_before_allocating_workspace(monkeypatch):
    value = torch.ones(5, 16)
    gate, up, down = _weight(32, 16), _weight(32, 16), _weight(20, 32)
    select = Mock(return_value=None)
    monkeypatch.setattr(_backend, "select_linear_backend", select)
    allocate = Mock(side_effect=AssertionError("workspace allocated without a backend"))
    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(ValueError, match="optimized linear is unavailable"):
        ffn._run_chunked_swiglu_ffn(value, *gate, 16, *up, 16, *down, 16, 4)
    select.assert_called_once_with(value)
    allocate.assert_not_called()


def _semantic_ffn_graph(device):
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
    output = linear("down", multiplied, 32, 20)
    graph.output(output)
    return torch.fx.GraphModule(torch.nn.Module(), graph), value.meta["val"]


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
@pytest.mark.parametrize("supported", [False, True])
def test_ffn_compiler_uses_backend_support_not_device_family(monkeypatch, device, supported):
    select = Mock(return_value=object() if supported else None)
    monkeypatch.setattr(_backend, "select_linear_backend", select)
    with FakeTensorMode():
        module, value = _semantic_ffn_graph(device)
        ffn_compile.compile_pass(module.graph, is_inference=True)
    targets = [node.target for node in module.graph.nodes]
    assert (torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default in targets) is supported
    assert targets.count(torch.ops.piper_kernels.convrot_int8_linear.default) == (
        0 if supported else 3
    )
    assert select.called
    assert all(call.args[0] is value for call in select.call_args_list)


def test_sparse_output_uses_operations_and_preserves_output_views(monkeypatch, operations):
    backend, select = operations
    value = torch.linspace(-1, 1, 2 * 5 * 128).reshape(2, 5, 128).to(torch.bfloat16)
    storage = torch.empty(2, 1, 5, 128, dtype=torch.int8)
    weight, scale, bias = _weight(7, 128)
    monkeypatch.setattr(sparse_output, "_validate_output_projection", lambda *args: (128, 7))
    width, project, prepared = sparse_output._prepare_output_chunk_projector(
        storage, 5, weight, scale, bias, 16, 5, 3
    )
    assert width == 7
    backing = torch.full((2, 5, 9), -999, dtype=torch.bfloat16)
    output = backing[..., 1:-1]
    chunk = torch.empty(2, 3, 1, 128, dtype=torch.bfloat16)
    for start, rows in ((0, 3), (3, 2)):
        chunk[:, :rows, 0].copy_(value[:, start : start + rows])
        project(chunk, output, start, rows)
    expected = reference.linear(value, weight, scale, 16, bias)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.all(backing[..., 0] == -999)
    assert torch.all(backing[..., -1] == -999)
    select.assert_called_once_with(storage)
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


@pytest.mark.parametrize("module", [ffn, ffn_compile, sparse_output])
def test_shared_orchestration_does_not_depend_on_vendor_launch_interfaces(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    forbidden = tuple(
        f"piper_kernels.linear.convrot.int8.{name}"
        for name in ("_nvidia", "_amd", "_plan", "_policy", "triton")
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert not f"{node.module}.{alias.name}".startswith(forbidden)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden)
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "default_execution_plan",
                "prepare_input_with_plan",
                "execute_prepared_linear",
            }
            if module in (ffn, ffn_compile):
                assert node.attr not in {"from_device", "cuda_capability_at_least"}


def test_unvalidated_sparse_fusion_still_rejects_rocm(monkeypatch, operations):
    _implementation, select = operations
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", "gfx1201")
    )
    with FakeTensorMode():
        value = torch.empty(1, 16, device="cuda")
        storage = torch.empty(1, 1, 64, 128, dtype=torch.int8, device="cuda")
        with pytest.raises(ValueError, match="requires exact NVIDIA SM120"):
            sparse_output._prepare_output_chunk_projector(
                storage, 64, value, value, None, 16, 64, 64
            )
    select.assert_not_called()
