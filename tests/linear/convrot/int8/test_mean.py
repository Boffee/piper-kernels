"""Prepared-input mean is shared math, independent of matrix accelerator support."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import _backend, _ops
from piper_kernels.linear.convrot.int8._generic import mean


@pytest.mark.parametrize("supported", [False, True])
def test_mean_selection_does_not_require_a_matrix_backend(monkeypatch, supported):
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    monkeypatch.setattr(_backend, "_amd_backend", None)
    probe = Mock(return_value=supported)
    monkeypatch.setattr(runtime, "supports_device", probe)
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", Mock(side_effect=AssertionError("target"))
    )
    input = SimpleNamespace(device=torch.device("cuda:1"))  # noqa: A001
    assert _backend.select_dequantized_mean(input) is (
        mean.dequantized_input_mean if supported else None
    )
    probe.assert_called_once_with(input.device)


def test_missing_mean_implementation_does_not_probe_hardware(monkeypatch):
    monkeypatch.setattr(_backend, "_mean_backend", None)
    monkeypatch.setattr(runtime, "supports_device", Mock(side_effect=AssertionError("probe")))
    assert _backend.select_dequantized_mean(torch.empty(1)) is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("shape", [(2, 65, 31), (2, 192, 256), (2, 192, 272), (1, 4097, 1024)])
@pytest.mark.parametrize("padded", [False, True])
def test_prepared_mean_matches_fp64_with_tails_and_internal_padding(shape, padded):
    batch, sequence, width = shape
    storage = (sequence + 63) // 64 * 64 if padded else sequence
    generator = torch.Generator(device="cuda").manual_seed(773)
    qdata = torch.randint(
        -128, 128, (batch, storage, width), device="cuda", dtype=torch.int8, generator=generator
    )
    scale = torch.rand((batch, storage), device="cuda", generator=generator) * 0.1
    lengths = None
    valid = torch.ones(storage, device="cuda", dtype=torch.bool)
    if padded:
        lengths = torch.arange(storage // 64, device="cuda", dtype=torch.int32) % 49 + 16
        lengths[-1] = min(int(lengths[-1]), sequence - storage + 64)
        valid = torch.arange(storage, device="cuda") % 64 < lengths.repeat_interleave(64)
    actual = _ops.dequantized_input_mean(qdata, scale, lengths)
    expected = (qdata.double() * scale.double()[..., None])[:, valid].mean(dim=1)
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_prepared_mean_custom_op_and_compilation():
    qdata = torch.ones((2, 193, 272), dtype=torch.int8, device="cuda")
    scale = torch.full((2, 193), 0.125, device="cuda")
    assert set(torch.library.opcheck(_ops.dequantized_input_mean, (qdata, scale)).values()) == {
        "SUCCESS"
    }
    compiled = torch.compile(_ops.dequantized_input_mean, fullgraph=True)
    torch.testing.assert_close(
        compiled(qdata, scale), torch.full((2, 272), 0.125, device="cuda"), rtol=0, atol=0
    )
