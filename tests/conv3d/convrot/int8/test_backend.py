"""Convolution dispatch and vendor-owned launch policy contracts."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.conv3d.convrot.int8 import _backend, _ops, reference
from piper_kernels.conv3d.convrot.int8._amd import policy as amd
from piper_kernels.conv3d.convrot.int8._nvidia import policy as nvidia
from piper_kernels.specializations.minimax_h3_vae.conv3d import _compile


@pytest.mark.parametrize(
    ("target", "vendor"),
    [
        (AcceleratorTarget("cuda", "sm120"), "nvidia"),
        (AcceleratorTarget("cuda", "sm100"), None),
        (AcceleratorTarget("hip", "gfx1200"), "amd"),
        (AcceleratorTarget("hip", "gfx1201"), "amd"),
        (AcceleratorTarget("hip", "gfx1100"), None),
        (AcceleratorTarget("hip", "gfx942"), None),
        (AcceleratorTarget("hip", "gfx9999"), None),
        (AcceleratorTarget("cpu", "cpu"), None),
    ],
)
@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_select_backend_uses_target_policy(monkeypatch, target, vendor, platform):
    monkeypatch.setattr(sys, "platform", platform)
    if vendor == "amd" and platform not in ("linux", "win32"):
        vendor = None
    implementations = {"nvidia": object(), "amd": object(), None: None}
    monkeypatch.setattr(_backend, "_nvidia_backend", implementations["nvidia"])
    monkeypatch.setattr(_backend, "_amd_backend", implementations["amd"])
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda device: target)
    assert _backend.select_backend(torch.empty(0)) is implementations[vendor]


def test_missing_triton_uses_reference(monkeypatch):
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    monkeypatch.setattr(_backend, "_amd_backend", None)
    assert _backend.select_backend(torch.empty(0)) is None


@pytest.mark.parametrize("fused", [False, True])
def test_custom_op_uses_selected_backend_or_reference(monkeypatch, fused):
    activation = torch.zeros(1, 64, 1, 3, 3, dtype=torch.float16)
    tail = (
        torch.zeros(4, 3, 3, 3, 64, dtype=torch.int8),
        torch.ones(4, 1),
        None,
        64,
        torch.tensor(0.02),
        [1, 1, 1],
        True,
        False,
        None,
    )
    args = (
        (activation, torch.ones(64), torch.zeros(64), 8, 1e-6, *tail)
        if fused
        else (activation, *tail)
    )
    name = "group_norm_silu_conv3d" if fused else "conv3d"
    op = _ops._group_norm_silu_conv3d_op if fused else _ops._conv3d_op
    expected = torch.zeros(1, 4, 1, 3, 3, dtype=torch.float16)
    implementation = Mock()
    getattr(implementation, name).return_value = expected
    select = Mock(return_value=implementation)
    monkeypatch.setattr(_backend, "select_backend", select)
    torch.testing.assert_close(op(*args), expected)
    select.assert_called_once_with(activation)
    getattr(implementation, name).assert_called_once()
    select.return_value = None
    fallback = Mock(return_value=expected)
    monkeypatch.setattr(reference, name, fallback)
    torch.testing.assert_close(op(*args), expected)
    fallback.assert_called_once()


@pytest.mark.parametrize(
    ("channels", "outputs", "rows", "expected"),
    [
        (128, 128, 1024, (128, 128, 64, 4, 3)),
        (256, 128, 200_000, (128, 128, 64, 4, 4)),
        (256, 128, 30_000, (128, 128, 64, 4, 2)),
        (256, 256, 1024, (64, 128, 128, 4, 3)),
        (256, 512, 1024, (128, 128, 128, 8, 3)),
        (512, 512, 5000, (128, 128, 128, 8, 3)),
        (512, 1024, 1024, (64, 128, 128, 4, 3)),
        (512, 512, 1024, (32, 128, 256, 8, 3)),
        (64, 32, 1024, (64, 64, 64, 4, 3)),
        (1024, 128, 1024, (64, 128, 128, 4, 3)),
    ],
)
def test_nvidia_convolution_policy_preserves_existing_tiles(channels, outputs, rows, expected):
    assert nvidia.convolution_plan(channels, outputs, rows) == expected


@pytest.mark.parametrize(
    ("channels", "rows", "group_norm", "expected"),
    [
        (128, 1024, False, (64, 4)),
        (256, 200_000, True, (32, 4)),
        (256, 1024, True, (16, 4)),
        (256, 200_000, False, (64, 4)),
        (256, 1024, False, (32, 8)),
        (512, 5000, True, (16, 8)),
        (512, 5000, False, (32, 8)),
        (1024, 1024, False, (8, 8)),
    ],
)
def test_nvidia_preparation_policy_preserves_existing_tiles(channels, rows, group_norm, expected):
    assert nvidia.preparation_plan(channels, rows, group_norm=group_norm) == expected


@pytest.mark.parametrize(
    ("channels", "outputs", "height", "aligned", "expected"),
    [
        (128, 128, 256, True, True),
        (128, 256, 16, True, True),
        (128, 256, 16, False, False),
        (128, 65, 256, True, False),
        (128, 128, 16, True, False),
        (256, 256, 256, True, False),
    ],
)
def test_nvidia_descriptor_policy(channels, outputs, height, aligned, expected):
    assert nvidia.use_weight_descriptor(channels, outputs, height, 128, aligned=aligned) is expected


def test_compiler_cache_tracks_backend_sources(monkeypatch):
    files = _backend.source_files()
    root = Path(_backend.__file__).parent
    assert {
        str(root / path)
        for path in (
            "_backend.py",
            "_interfaces.py",
            "_plan.py",
            "_nvidia/policy.py",
            "_amd/policy.py",
        )
    } <= set(files)
    if _backend._shared is not None:
        assert {
            str(root / path) for path in ("triton.py", "_nvidia/triton.py", "_amd/triton.py")
        } <= set(files)
    capture = Mock(return_value=b"cache-key")
    monkeypatch.setattr(_compile, "get_hash_for_files", capture)
    assert _compile.compile_pass.uuid() == b"cache-key"
    assert set(files) <= set(capture.call_args.args[0])


def test_amd_policy_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setattr(amd.sys, "platform", "darwin")
    assert not amd.supports_target(AcceleratorTarget("hip", "gfx1201"))


@pytest.mark.parametrize("channels", [64, 128, 256, 512, 1024, 2048, 4096])
def test_amd_preparation_bounds_rotation_tile_and_never_uses_descriptors(channels):
    for fused in (False, True):
        plan = amd.preparation_plan(channels, 32768, group_norm=fused)
        assert plan.block_m * channels <= 4096
        assert plan.block_m > 0
        assert plan == amd.preparation_plan(channels, 16, group_norm=fused)
    assert not amd.use_weight_descriptor(channels, 256, 256, 128, aligned=True)


@pytest.mark.parametrize(
    ("channels", "outputs", "rows", "expected"),
    [
        (128, 128, 8192, (128, 128, 128, 8, 2)),
        (256, 256, 3072, (64, 128, 128, 4, 2)),
        (512, 512, 768, (64, 64, 64, 4, 2)),
        (64, 7, 18, (32, 64, 64, 4, 2)),
    ],
)
def test_amd_convolution_uses_measured_tiles(channels, outputs, rows, expected):
    assert amd.convolution_plan(channels, outputs, rows) == expected
