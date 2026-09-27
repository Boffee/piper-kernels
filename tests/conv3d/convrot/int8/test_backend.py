"""Convolution dispatch and vendor-owned launch policy contracts."""

import sys
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.conv3d.convrot.int8 import _backend, _ops, reference
from piper_kernels.conv3d.convrot.int8._amd import policy as amd
from piper_kernels.conv3d.convrot.int8._dispatch import default_execution_plan
from piper_kernels.conv3d.convrot.int8._nvidia import policy as nvidia
from piper_kernels.conv3d.convrot.int8._plan import ConvolutionSchedule
from piper_kernels.specializations.minimax_h3_vae.conv3d import _compile

_nvidia_plan = partial(
    nvidia.select_execution_plan,
    channels=128,
    outputs=128,
    input_rows=1024,
    output_rows=1024,
    output_height=16,
    weight_aligned=True,
    group_norm=False,
)
SM120 = partial(_nvidia_plan, AcceleratorTarget("cuda", "sm120"))
SM8X = partial(_nvidia_plan, AcceleratorTarget("cuda", "sm89"))
RDNA4 = partial(
    amd.select_execution_plan,
    AcceleratorTarget("hip", "gfx1201"),
    channels=128,
    outputs=128,
    input_rows=1024,
    output_rows=1024,
    output_height=16,
    weight_aligned=True,
    group_norm=False,
)


@pytest.mark.parametrize(
    ("target", "vendor"),
    [
        (AcceleratorTarget("cuda", "sm120"), "nvidia"),
        (AcceleratorTarget("cuda", "sm89"), "nvidia"),
        (AcceleratorTarget("cuda", "sm86"), "nvidia"),
        (AcceleratorTarget("cuda", "sm80"), "nvidia"),
        (AcceleratorTarget("cuda", "sm90"), None),
        (AcceleratorTarget("cuda", "sm100"), None),
        (AcceleratorTarget("cuda", "sm75"), None),
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
    assert SM120(channels=channels, outputs=outputs, output_rows=rows).convolution == expected


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
    assert SM120(channels=channels, input_rows=rows, group_norm=group_norm).preparation == expected
    assert SM8X(channels=channels, input_rows=rows, group_norm=group_norm).preparation == expected


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
    assert (
        SM120(
            channels=channels, outputs=outputs, output_height=height, weight_aligned=aligned
        ).use_weight_descriptor
        is expected
    )
    assert not SM8X(
        channels=channels, outputs=outputs, output_height=height, weight_aligned=aligned
    ).use_weight_descriptor


@pytest.mark.parametrize("architecture", ["sm80", "sm86", "sm87", "sm89"])
def test_nvidia_selects_sm8x_policy_for_the_family(architecture):
    assert _nvidia_plan(AcceleratorTarget("cuda", architecture)) == SM8X()


@pytest.mark.parametrize("target", [AcceleratorTarget("cuda", "sm90"), AcceleratorTarget("cpu")])
def test_nvidia_policy_rejects_unsupported_targets(target):
    with pytest.raises(ValueError, match="no NVIDIA policy"):
        _nvidia_plan(target)


@pytest.mark.parametrize(
    ("channels", "outputs", "rows", "expected"),
    [
        (128, 128, 1_114_112, (64, 128, 128, 4, 3)),
        (256, 512, 5120, (64, 128, 128, 4, 3)),
        (1024, 1024, 2048, (64, 128, 128, 4, 3)),
        (512, 512, 2047, (64, 64, 128, 4, 3)),
        (1024, 1024, 1280, (64, 64, 128, 4, 3)),
        (1024, 65, 1280, (64, 64, 128, 4, 3)),
        (1024, 64, 1280, (32, 32, 128, 2, 3)),
        (64, 7, 1_000_000, (32, 32, 128, 2, 3)),
    ],
)
def test_sm8x_convolution_policy_uses_measured_tiles(channels, outputs, rows, expected):
    assert SM8X(channels=channels, outputs=outputs, output_rows=rows).convolution == expected


@pytest.mark.parametrize("fused", [False, True])
def test_strided_execution_selects_preparation_from_input_rows(fused):
    # 200,000 input rows cross the preparation threshold; 12,500 output rows do not.
    activation = torch.empty(1, 256, 1, 1, 1).expand(1, 256, 1, 400, 500)
    weight = torch.empty(256, 3, 3, 3, 256, dtype=torch.int8)
    plan = default_execution_plan(
        activation,
        weight,
        (1, 4, 4),
        policy=nvidia,
        target=AcceleratorTarget("cuda", "sm120"),
        group_norm=fused,
        symmetric_spatial_padding=True,
        right_spatial_padding=False,
    )
    assert plan.preparation == ((32, 4) if fused else (64, 4))
    assert plan.convolution == (64, 128, 128, 4, 3)


@pytest.mark.parametrize("architecture", ["sm120", "sm89"])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("block_n", [64, 128, 256])
def test_tuning_tile_recomputes_descriptor_eligibility(architecture, aligned, block_n):
    activation = torch.empty(1, 128, 1, 4, 4)
    storage = torch.empty(192 * 27 * 128 + 1, dtype=torch.int8)
    weight = (storage[:-1] if aligned else storage[1:]).view(192, 3, 3, 3, 128)
    select = partial(
        default_execution_plan,
        activation,
        weight,
        (1, 1, 1),
        policy=nvidia,
        target=AcceleratorTarget("cuda", architecture),
        group_norm=False,
        symmetric_spatial_padding=True,
        right_spatial_padding=False,
    )
    production = select()
    tile = ConvolutionSchedule(64, block_n, 128, 4, 3)
    candidate = select(convolution_schedule=tile)
    assert candidate.convolution == tile
    assert candidate.preparation == production.preparation
    assert not production.use_weight_descriptor
    assert candidate.use_weight_descriptor is (
        architecture == "sm120" and aligned and block_n == 64
    )


def test_compiler_cache_tracks_backend_sources(monkeypatch):
    files = _backend.source_files()
    root = Path(_backend.__file__).parent
    assert {
        str(root / path)
        for path in (
            "_backend.py",
            "_interfaces.py",
            "_dispatch.py",
            "_plan.py",
            "_nvidia/policy.py",
            "_amd/policy.py",
        )
    } <= set(files)
    if _backend._shared is not None:
        assert {
            str(root / path) for path in ("triton.py", "_nvidia/dispatch.py", "_amd/dispatch.py")
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
        schedule = amd._preparation_schedule(channels, 32768, group_norm=fused)
        assert schedule.block_m * channels <= 4096
        assert schedule.block_m > 0
        assert schedule == amd._preparation_schedule(channels, 16, group_norm=fused)
    assert not RDNA4(channels=channels, outputs=256, output_height=256).use_weight_descriptor


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
    assert RDNA4(channels=channels, outputs=outputs, output_rows=rows).convolution == expected
