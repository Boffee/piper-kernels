"""Tests for Triton hardware feature predicates."""

from types import SimpleNamespace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget


def test_cpu_supports_no_nvidia_tensor_core_features() -> None:
    device = torch.device("cpu")
    target = AcceleratorTarget.from_device(device)

    assert target == AcceleratorTarget(backend="cpu")
    assert not target.is_nvidia_cuda
    assert not target.supports_uint8_int8_mma
    assert not target.supports_fp8_fp16_mma


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

    target = AcceleratorTarget.from_device(device)

    assert target == AcceleratorTarget(backend="cuda", cuda_capability=capability)
    assert target.is_nvidia_cuda
    assert target.supports_uint8_int8_mma is mixed_int8
    assert target.supports_fp8_fp16_mma is fp8_fp16


def test_cuda_capability_matching_distinguishes_family_and_exact_target() -> None:
    target = AcceleratorTarget(backend="cuda", cuda_capability=(12, 1))

    assert target.is_cuda_capability(12)
    assert target.is_cuda_capability(12, 1)
    assert target.cuda_capability_at_least(12)
    assert target.cuda_capability_at_least(8, 9)
    assert not target.is_cuda_capability(12, 0)
    assert not target.is_cuda_capability(11)
    assert not target.cuda_capability_at_least(12, 2)
    assert target.supports_uint8_int8_mma
    assert target.supports_fp8_fp16_mma
    assert not AcceleratorTarget(backend="hip").is_cuda_capability(12)
    assert not AcceleratorTarget(backend="hip").cuda_capability_at_least(7, 5)


def test_compiler_target_normalizes_integer_cuda_architecture() -> None:
    target = AcceleratorTarget.from_compiler_target(
        SimpleNamespace(backend="cuda", arch=121)
    )

    assert target == AcceleratorTarget(backend="cuda", cuda_capability=(12, 1))
    assert AcceleratorTarget.from_compiler_target(
        SimpleNamespace(backend="hip", arch="gfx1201")
    ) == AcceleratorTarget(backend="hip")


def test_rocm_never_reports_nvidia_mma_support(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", "7.0")

    def unexpected_capability_query(_device: torch.device) -> tuple[int, int]:
        raise AssertionError("ROCm must be rejected before interpreting its capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_capability_query)

    assert AcceleratorTarget.from_device(device) == AcceleratorTarget(backend="hip")
    target = AcceleratorTarget.from_device(device)
    assert not target.is_nvidia_cuda
    assert not target.supports_uint8_int8_mma
    assert not target.supports_fp8_fp16_mma
