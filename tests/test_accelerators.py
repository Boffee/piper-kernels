"""The shared SM120 test gate must exclude ROCm devices that report capability (12, 0)."""

from types import SimpleNamespace

import pytest
import torch
from _accelerators import exact_sm120_available


@pytest.mark.parametrize("hip", [None, "test"])
def test_sm120_gate_distinguishes_rocm_from_cuda(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1201"),
    )
    assert exact_sm120_available() is (hip is None)
