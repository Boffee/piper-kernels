"""Device capability and launch context do not depend on the caller's current GPU."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import triton

from piper_kernels._triton import runtime


@pytest.fixture
def devices(monkeypatch):
    state = SimpleNamespace(
        current=0, device_type="cuda", streams={0: "stream-zero", 1: "stream-one"}, queries=[]
    )
    state.targets = {
        0: SimpleNamespace(backend="cuda", arch=60),
        1: SimpleNamespace(backend="cuda", arch=120),
    }

    @contextmanager
    def guard(device):
        previous = state.current
        state.current = previous if device.index is None else device.index
        try:
            yield
        finally:
            state.current = previous

    def target():
        state.queries.append(state.current)
        return state.targets[state.current]

    monkeypatch.setattr(torch, "get_device_module", lambda device: SimpleNamespace(device=guard))
    driver = SimpleNamespace(
        get_active_torch_device=lambda: torch.device(state.device_type, state.current),
        get_current_target=target,
    )
    monkeypatch.setattr(triton.runtime, "driver", SimpleNamespace(active=driver))
    return state


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_host_context_and_probe_do_not_initialize_accelerators(monkeypatch, device):
    unexpected = Mock(side_effect=AssertionError("host context initialized an accelerator"))
    monkeypatch.setattr(torch, "get_device_module", unexpected)
    monkeypatch.setattr(triton.runtime, "driver", None)
    with runtime.device_context(torch.device(device)):
        assert not runtime.supports_device(torch.device(device))
    unexpected.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_nested_context_restores_device_and_preserves_each_current_stream(devices, fail):
    with runtime.device_context(torch.device("cuda:1")):
        assert devices.current == 1
        assert devices.streams[devices.current] == "stream-one"
        error_context = (
            pytest.raises(RuntimeError, match="launch failed") if fail else nullcontext()
        )
        with error_context, runtime.device_context(torch.device("cuda:0")):
            assert devices.current == 0
            assert devices.streams[devices.current] == "stream-zero"
            if fail:
                raise RuntimeError("launch failed")
        assert devices.current == 1
    assert devices.current == 0
    assert devices.streams == {0: "stream-zero", 1: "stream-one"}


@pytest.mark.parametrize("initial", [0, 1])
def test_probe_inspects_requested_gpu_not_current_gpu(devices, initial):
    devices.current = initial
    assert runtime.supports_device(torch.device("cuda:1"))
    assert not runtime.supports_device(torch.device("cuda:0"))
    assert devices.queries == [1, 0]
    assert devices.current == initial


@pytest.mark.parametrize(
    ("backend", "architecture", "device_type", "supported"),
    [
        ("cuda", 60, "cuda", False),
        ("cuda", 70, "cuda", True),
        ("hip", "gfx1036", "cuda", True),
        ("hip", "gfx9999", "cuda", True),
        ("xpu", "future_device", "xpu", True),
    ],
)
def test_probe_uses_compiler_support_not_tuned_model_policy(
    devices, backend, architecture, device_type, supported
):
    devices.device_type = device_type
    devices.targets[1] = SimpleNamespace(backend=backend, arch=architecture)
    assert runtime.supports_device(torch.device(device_type, 1)) is supported
    assert devices.queries == [1]
    assert devices.current == 0


def test_driver_for_other_device_family_is_not_used(devices):
    assert not runtime.supports_device(torch.device("xpu:1"))
    assert devices.queries == []
    assert devices.current == 0


def test_device_without_context_is_not_claimed_supported(monkeypatch, devices):
    monkeypatch.setattr(torch, "get_device_module", lambda device: SimpleNamespace())
    assert not runtime.supports_device(torch.device("cuda:1"))
    assert devices.queries == []


def test_probe_failure_restores_device_without_swallowing_execution_errors(monkeypatch, devices):
    def fail():
        assert devices.current == 1
        raise RuntimeError("query failed")

    monkeypatch.setattr(triton.runtime.driver.active, "get_current_target", fail)
    assert not runtime.supports_device(torch.device("cuda:1"))
    assert devices.current == 0
    with (
        pytest.raises(RuntimeError, match="launch failed"),
        runtime.device_context(torch.device("cuda:1")),
    ):
        raise RuntimeError("launch failed")
    assert devices.current == 0


def test_missing_driver_uses_fallback(monkeypatch):
    driver = SimpleNamespace(get_active_torch_device=Mock(side_effect=RuntimeError("no driver")))
    monkeypatch.setattr(triton.runtime, "driver", SimpleNamespace(active=driver))
    assert not runtime.supports_device(torch.device("cuda:1"))
