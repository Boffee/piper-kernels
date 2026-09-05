"""Shared custom ops dispatch eagerly and after real Inductor rewrites."""

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass

from piper_kernels.linear._input_activations import apply_input_activation
from piper_kernels.linear.convrot import convrot_int8_compile_options
from piper_kernels.linear.convrot.int8 import _backend, _ops, _update, dispatch, reference


class _RecordingBackend:
    """A CPU test implementation using the existing quantized reference contract."""

    def __init__(self):
        self.calls = []

    def linear(self, value, weight_qdata, weight_scale, bias, group_size, activation_fn=None):
        self.calls.append("linear")
        return reference.linear(
            value, weight_qdata, weight_scale, group_size, bias, activation_fn=activation_fn
        )

    def prepare_input(self, value, group_size, activation_fn=None, *, out=None):
        self.calls.append("prepare")
        prepared = reference.prepare_input(apply_input_activation(value, activation_fn), group_size)
        if out is None:
            return prepared
        for output, prepared_value in zip(out, prepared, strict=True):
            output.copy_(prepared_value)
        return out

    def linear_prepared(
        self,
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        logical_dtype,
        *,
        out=None,
        second_projection=None,
    ):
        self.calls.append("project")
        result = reference.linear_prepared(
            input_qdata, input_scale, weight_qdata, weight_scale, logical_dtype, bias
        )
        if second_projection is not None:
            weight, scale, second_bias = second_projection
            second = reference.linear_prepared(
                input_qdata, input_scale, weight, scale, logical_dtype, second_bias
            )
            result = torch.cat((result, second), dim=-1)
        if out is None:
            return result
        out.copy_(result)
        return out


def _operands():
    generator = torch.Generator().manual_seed(83)
    return (
        torch.randn(2, 3, 32, generator=generator),
        torch.randint(-127, 128, (7, 32), dtype=torch.int8, generator=generator),
        torch.rand(7, 1, generator=generator) * 0.01,
        torch.randn(7, generator=generator),
    )


@pytest.mark.parametrize("activation_fn", [None, "gelu_tanh", "swiglu"])
def test_eager_validated_linear_uses_shared_op(monkeypatch, activation_fn):
    implementation = _RecordingBackend()
    monkeypatch.setattr(_backend, "select_linear_backend", lambda value: implementation)
    value, weight, scale, bias = _operands()
    if activation_fn == "swiglu":
        value = torch.cat((value, value / 2), dim=-1)

    actual = dispatch.linear(
        value, weight, scale, value.dtype, 16, bias, activation_fn=activation_fn
    )
    expected = reference.linear(value, weight, scale, 16, bias, activation_fn=activation_fn)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert implementation.calls == ["linear"]


def test_compiled_preparation_sharing_resolves_backend_at_execution(monkeypatch):
    class Capture(CustomInferenceAwareGraphPass):
        def __init__(self):
            self.calls = 0
            self.targets = []
            self.key = uuid.uuid4().bytes

        def __call__(self, graph, is_inference):
            self.calls += 1
            self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

        def uuid(self):
            return self.key

    selected = [_RecordingBackend()]
    monkeypatch.setattr(_backend, "select_linear_backend", lambda value: selected[0])
    value, weight, scale, bias = _operands()

    def two_projections(value, weight, scale, bias):
        return (
            _ops.linear(value, weight, scale, bias, 16),
            _ops.linear(value, weight[:5], scale[:5], None, 16),
        )

    capture = Capture()
    options = convrot_int8_compile_options()
    options["post_grad_custom_pre_pass"] = (options["post_grad_custom_pre_pass"], capture)
    compiled = torch.compile(two_projections, fullgraph=True, options=options)
    with torch.inference_mode():
        expected = two_projections(value, weight, scale, bias)
        selected[0].calls.clear()
        actual = compiled(value, weight, scale, bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert selected[0].calls == ["prepare", "project", "project"]
        assert (
            capture.targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 1
        )
        assert (
            capture.targets.count(torch.ops.piper_kernels.convrot_int8_linear_prepared.default) == 2
        )
        assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets

        previous = selected[0]
        previous.calls.clear()
        selected[0] = _RecordingBackend()
        compile_count = capture.calls
        actual = compiled(value, weight, scale, bias)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert selected[0].calls == ["prepare", "project", "project"]
    assert previous.calls == []
    assert capture.calls == compile_count


def test_fake_schemas_need_no_implementation(monkeypatch):
    def unexpected_selection(value):
        raise AssertionError("fake propagation selected an implementation")

    monkeypatch.setattr(_backend, "select_linear_backend", unexpected_selection)
    monkeypatch.setattr(_backend, "select_add", unexpected_selection)
    monkeypatch.setattr(_backend, "select_addmm", unexpected_selection)
    value = torch.empty(2, 3, 64, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(7, 32, device="meta", dtype=torch.int8)
    scale = torch.empty(7, 1, device="meta")

    result = _ops.linear(value, weight, scale, None, 16, "swiglu")
    qdata, row_scale = _ops.prepare_input(value, 16, "swiglu")
    projected = _ops.linear_prepared(qdata, row_scale, weight, scale, None, value.dtype)
    _ops.add_(weight, scale, torch.empty_like(weight, dtype=value.dtype), 16, 1.0)
    _ops.addmm_(
        weight,
        scale,
        torch.empty(7, 2, device="meta"),
        torch.empty(2, 32, device="meta"),
        16,
        1.0,
        1.0,
    )

    assert result.shape == projected.shape == (2, 3, 7)
    assert result.dtype == projected.dtype == torch.bfloat16
    assert qdata.shape == (2, 3, 32)
    assert qdata.dtype == torch.int8
    assert row_scale.shape == (2, 3)
    assert row_scale.dtype == torch.float32


@pytest.mark.parametrize("operation", ["add", "addmm"])
def test_linear_only_implementation_preserves_weight_update_fallback(monkeypatch, operation):
    implementation = _RecordingBackend()
    monkeypatch.setattr(_backend, "select_linear_backend", lambda value: implementation)
    _value, weight, scale, _bias = _operands()
    expected_weight, expected_scale = weight.clone(), scale.clone()
    delta = torch.ones_like(weight, dtype=torch.float32)
    if operation == "add":
        assert _backend.select_add(weight) is None
        _update.add_(weight, scale, torch.float32, 16, delta)
        reference.add_(expected_weight, expected_scale, delta, 16, 1.0)
    else:
        assert _backend.select_addmm(weight) is None
        mat1, mat2 = torch.ones(7, 2), torch.ones(2, 32)
        _update.addmm_(weight, scale, torch.float32, 16, mat1, mat2)
        reference.addmm_(expected_weight, expected_scale, mat1, mat2, 16, 1.0, 1.0)

    assert torch.equal(weight, expected_weight)
    assert torch.equal(scale, expected_scale)
    assert implementation.calls == []


def test_explicit_optimized_ops_fail_if_no_implementation_exists(monkeypatch):
    monkeypatch.setattr(_backend, "select_linear_backend", lambda value: None)
    value, weight, scale, bias = _operands()
    with pytest.raises(ValueError, match="optimized linear is unavailable"):
        _ops.linear(value, weight, scale, bias, 16)
