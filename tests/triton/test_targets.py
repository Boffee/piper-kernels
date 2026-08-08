"""Tests for Triton hardware feature predicates."""

import pytest
import torch

from piper_kernels._triton.targets import (
    is_nvidia_cuda,
    supports_fp8_fp16_mma,
    supports_uint8_int8_mma,
)


def test_cpu_supports_no_nvidia_tensor_core_features() -> None:
    device = torch.device("cpu")

    assert not is_nvidia_cuda(device)
    assert not supports_uint8_int8_mma(device)
    assert not supports_fp8_fp16_mma(device)


@pytest.mark.parametrize(
    ("capability", "mixed_int8", "fp8_fp16"),
    [
        ((7, 5), False, False),
        ((8, 0), True, False),
        ((8, 6), True, False),
        ((8, 9), True, True),
        ((9, 0), False, True),
        ((10, 0), False, True),
        ((11, 0), False, True),
        ((12, 0), True, True),
    ],
)
def test_nvidia_capability_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    mixed_int8: bool,
    fp8_fp16: bool,
) -> None:
    device = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    assert is_nvidia_cuda(device)
    assert supports_uint8_int8_mma(device) is mixed_int8
    assert supports_fp8_fp16_mma(device) is fp8_fp16


def test_rocm_never_reports_nvidia_mma_support(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", "7.0")

    def unexpected_capability_query(_device: torch.device) -> tuple[int, int]:
        raise AssertionError("ROCm must be rejected before interpreting its capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_capability_query)

    assert not is_nvidia_cuda(device)
    assert not supports_uint8_int8_mma(device)
    assert not supports_fp8_fp16_mma(device)
