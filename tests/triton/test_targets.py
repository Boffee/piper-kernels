"""Tests for Triton hardware feature predicates."""

import sys
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
    assert not target.supports_nvfp4_w4a16
    assert not target.supports_trellis_moe


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

    assert target == AcceleratorTarget(
        backend="cuda",
        architecture=f"sm{capability[0]}{capability[1]}",
    )
    assert target.cuda_capability == capability
    assert target.is_nvidia_cuda
    assert not target.is_amd_hip
    assert target.supports_uint8_int8_mma is mixed_int8
    assert target.supports_fp8_fp16_mma is fp8_fp16


def test_cuda_capability_matching_distinguishes_family_and_exact_target() -> None:
    target = AcceleratorTarget(backend="cuda", architecture="SM121")

    assert target.architecture == "sm121"
    assert target.is_architecture("sm121")
    assert not target.is_architecture("sm120", "gfx1201")
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
    assert not AcceleratorTarget(backend="hip", architecture="gfx942").supports_nvfp4_w4a16
    assert not AcceleratorTarget(backend="hip", architecture="gfx1201").supports_nvfp4_w4a16
    assert not AcceleratorTarget(backend="hip", architecture="gfx90a").supports_trellis_moe
    assert AcceleratorTarget(backend="hip", architecture="gfx1201").supports_trellis_moe is (
        sys.platform in ("linux", "win32")
    )
    for arch, nvfp4_expected, trellis_expected in (
        ("sm70", False, False),
        ("sm75", False, True),
        ("sm80", False, True),
        ("sm89", False, True),
        ("sm90", False, True),
        ("sm100", False, True),
        ("sm120", True, True),
        ("sm121", True, True),
    ):
        cuda_target = AcceleratorTarget(backend="cuda", architecture=arch)
        assert cuda_target.supports_nvfp4_w4a16 is nvfp4_expected
        assert cuda_target.supports_trellis_moe is trellis_expected


def test_compiler_target_normalizes_architecture_identity() -> None:
    target = AcceleratorTarget.from_compiler_target(SimpleNamespace(backend="cuda", arch=121))

    assert target == AcceleratorTarget(backend="cuda", architecture="sm121")
    for architecture in ("gfx1200", "gfx1201"):
        target = AcceleratorTarget.from_compiler_target(
            SimpleNamespace(backend="hip", arch=architecture)
        )
        other_architecture = "gfx1201" if architecture == "gfx1200" else "gfx1200"
        assert target == AcceleratorTarget(backend="hip", architecture=architecture)
        assert target.is_architecture(architecture)
        assert not target.is_architecture(other_architecture)


def test_rocm_preserves_normalized_architecture_without_nvidia_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeviceProperties:
        gcnArchName = "gfx1201:sramecc+:xnack-"  # noqa: N815

    device = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: DeviceProperties())

    def unexpected_capability_query(_device: torch.device) -> tuple[int, int]:
        raise AssertionError("ROCm must be rejected before interpreting its capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_capability_query)

    target = AcceleratorTarget.from_device(device)
    assert target == AcceleratorTarget(backend="hip", architecture="gfx1201")
    assert target.is_amd_hip
    assert target.is_architecture("gfx1200", "gfx1201")
    assert target.is_hip_architecture("gfx1200", "gfx1201")
    assert target.is_hip_architecture("gfx1201")
    assert not target.is_hip_architecture("gfx1200")
    assert not AcceleratorTarget(backend="hip", architecture="gfx1250").is_hip_architecture(
        "gfx1200", "gfx1201"
    )
    assert not AcceleratorTarget(backend="cuda", architecture="gfx1201").is_hip_architecture(
        "gfx1200", "gfx1201"
    )
    assert not target.is_nvidia_cuda
    assert target.cuda_capability is None
    assert not target.supports_uint8_int8_mma
    assert not target.supports_fp8_fp16_mma
    assert not target.supports_nvfp4_w4a16
