"""Implementation selection preserves ConvRot INT8 dispatch and fallback contracts."""

from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import _backend, _update, dispatch
from piper_kernels.linear.convrot.int8._nvidia import triton as nvidia


@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("cuda", "sm70"), False),
        (AcceleratorTarget("cuda", "sm75"), True),
        (AcceleratorTarget("cuda", "sm80"), True),
        (AcceleratorTarget("cuda", "sm89"), True),
        (AcceleratorTarget("cuda", "sm90"), True),
        (AcceleratorTarget("cuda", "sm100"), True),
        (AcceleratorTarget("cuda", "sm120"), True),
        (AcceleratorTarget("cuda", "sm121"), True),
        (AcceleratorTarget("cuda"), False),
        (AcceleratorTarget("hip", "gfx1201"), False),
        (AcceleratorTarget("hip", "gfx942"), False),
        (AcceleratorTarget("cpu"), False),
        (AcceleratorTarget("meta"), False),
    ],
)
def test_select_backend_preserves_supported_targets(monkeypatch, target, supported):
    implementation = ModuleType("test_convrot_implementation")
    monkeypatch.setattr(_backend, "_nvidia_backend", implementation)
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)

    selected = _backend.select_linear_backend(torch.empty(1))

    assert selected is (implementation if supported else None)


def test_missing_triton_uses_reference_without_querying_hardware(monkeypatch):
    resolve_target = Mock(side_effect=AssertionError("unexpected hardware query"))
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    monkeypatch.setattr(AcceleratorTarget, "from_device", resolve_target)

    assert _backend.select_linear_backend(torch.empty(1)) is None
    resolve_target.assert_not_called()


@pytest.mark.parametrize("architecture", ["sm70", "sm75", "sm120"])
def test_auxiliary_operations_keep_their_own_support_rules(monkeypatch, architecture):
    target = AcceleratorTarget("cuda", architecture)
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    value = torch.empty(1)

    assert _backend.select_gguf_converter(value) is nvidia._convert_gguf_out
    assert _backend.select_dequantized_mean(value) is nvidia.dequantized_input_mean
    assert (_backend.select_add(value) is not None) == (architecture != "sm70")
    assert (_backend.select_addmm(value) is not None) == (architecture != "sm70")


def test_nvidia_interface_forwards_preparation_and_projection_buffers(monkeypatch):
    target = AcceleratorTarget("cuda", "sm120")
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    value = torch.empty(2, 64)
    prepared = (torch.empty(2, 32, dtype=torch.int8), torch.empty(2))
    weight, scale = torch.empty(7, 32, dtype=torch.int8), torch.empty(7, 1)
    second = (torch.empty_like(weight), torch.empty_like(scale), torch.empty(7))
    output = torch.empty(2, 18)[:, 2:-2]
    prepare = Mock(return_value=prepared)
    project = Mock(return_value=output)
    monkeypatch.setattr(nvidia, "_prepare_input", prepare)
    monkeypatch.setattr(nvidia, "_execute_prepared_linear", project)
    implementation = _backend.require_linear_backend(value)

    assert implementation.prepare_input(value, 16, "swiglu", out=prepared) is prepared
    assert prepare.call_args.args == (value, 32, 16)
    assert prepare.call_args.kwargs["out"] is prepared
    assert prepare.call_args.kwargs["activation_fn"] == "swiglu"
    assert prepare.call_args.kwargs["target"] == target

    result = implementation.linear_prepared(
        *prepared, weight, scale, None, torch.float32, out=output, second_projection=second
    )
    assert result is output
    assert project.call_args.kwargs == {"out": output, "second_projection": second}


@pytest.mark.parametrize("operation", ["linear", "add_", "addmm_"])
def test_validated_operations_call_the_selected_implementation(monkeypatch, operation):
    qdata = torch.zeros(7, 32, dtype=torch.int8)
    scale = torch.ones(7, 1)
    activation = torch.zeros(3, 64)
    bias = torch.ones(7)
    update = torch.zeros(7, 32)
    mat1, mat2 = torch.zeros(7, 3), torch.zeros(3, 32)
    expected_output = torch.ones(3, 7)
    execute = Mock(return_value=expected_output)
    implementation = ModuleType("test_convrot_implementation")
    monkeypatch.setattr(implementation, operation, execute, raising=False)
    monkeypatch.setattr(_backend, "_nvidia_backend", implementation)
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("cuda", "sm120")
    )

    if operation == "linear":
        actual = dispatch.linear(
            activation, qdata, scale, torch.float32, 16, bias, activation_fn="swiglu"
        )
        assert actual is expected_output
        execute.assert_called_once_with(activation, qdata, scale, bias, 16, "swiglu")
    elif operation == "add_":
        _update.add_(qdata, scale, torch.float32, 16, update, alpha=2, rounding_seed=2**64 - 1)
        execute.assert_called_once_with(qdata, scale, update, 16, 2.0, -1)
    else:
        _update.addmm_(
            qdata, scale, torch.float32, 16, mat1, mat2, beta=3, alpha=2, rounding_seed=2**64 - 1
        )
        execute.assert_called_once_with(qdata, scale, mat1, mat2, 16, 3.0, 2.0, -1)
