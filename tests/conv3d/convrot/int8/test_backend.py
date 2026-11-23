"""Convolution dispatch and unified target launch policy contracts."""

import builtins
import math
import sys
from functools import partial
from unittest.mock import Mock

import pytest
import torch
from _accelerators import mock_cuda_target
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.conv3d.convrot.int8 import _ops, _policy, reference

_plan = partial(
    _policy.select_execution_plan,
    channels=128,
    outputs=128,
    input_rows=1024,
    output_rows=1024,
    output_height=16,
    weight_aligned=True,
    group_norm=False,
)
SM120 = partial(_plan, AcceleratorTarget("cuda", "sm120"))
SM8X = partial(_plan, AcceleratorTarget("cuda", "sm89"))
RDNA4 = partial(_plan, AcceleratorTarget("hip", "gfx1201"))


@pytest.mark.parametrize(
    ("target", "supported"),
    [
        (AcceleratorTarget("cuda", "sm120"), True),
        (AcceleratorTarget("cuda", "sm89"), True),
        (AcceleratorTarget("cuda", "sm86"), True),
        (AcceleratorTarget("cuda", "sm80"), True),
        (AcceleratorTarget("cuda", "sm90"), False),
        (AcceleratorTarget("cuda", "sm100"), False),
        (AcceleratorTarget("cuda", "sm75"), False),
        (AcceleratorTarget("hip", "gfx1200"), True),
        (AcceleratorTarget("hip", "gfx1201"), True),
        (AcceleratorTarget("hip", "gfx1100"), False),
        (AcceleratorTarget("hip", "gfx942"), False),
        (AcceleratorTarget("hip", "gfx9999"), False),
        (AcceleratorTarget("cpu", "cpu"), False),
    ],
)
@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_target_policy_selection(monkeypatch, target, supported, platform):
    monkeypatch.setattr(sys, "platform", platform)
    expected = supported and not (target.is_amd_hip and platform not in ("linux", "win32"))
    assert _policy.supports_target(target) is expected


def test_missing_triton_uses_reference(monkeypatch):
    mock_cuda_target(monkeypatch, (12, 0))
    with FakeTensorMode():
        operand = torch.empty(0, device="cuda:0")
    monkeypatch.delitem(sys.modules, "piper_kernels.conv3d.convrot.int8.triton", raising=False)
    original_import = builtins.__import__

    def without_triton(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ModuleNotFoundError("Triton intentionally unavailable", name="triton")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_triton)
    assert _ops._select_native_backend(operand) is None


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
        [1, 1, 1, 1, 2, 0],
        True,
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
    select = Mock(return_value=(implementation, AcceleratorTarget("cuda", "sm120")))
    monkeypatch.setattr(_ops, "_select_native_backend", select)
    torch.testing.assert_close(op(*args), expected)
    select.assert_called_once_with(activation)
    getattr(implementation, name).assert_called_once()
    select.return_value = None
    fallback = Mock(return_value=expected)
    monkeypatch.setattr(reference, name, fallback)
    torch.testing.assert_close(op(*args), expected)
    fallback.assert_called_once()


def test_sm120_convolution_schedule_contracts():
    for channels in (64, 128, 256, 512, 1024, 2048):
        for outputs in (32, 64, 128, 256, 384, 512, 1024):
            for rows in (16, 128, 1280, 5040, 20160, 293760):
                schedule = SM120(channels=channels, outputs=outputs, output_rows=rows).convolution
                assert (
                    schedule == SM120(channels=128, outputs=outputs, output_rows=rows).convolution
                )
                expected_n = 64 if outputs <= 64 else 128
                assert schedule.block_n == expected_n
                full_grid = rows >= 128 and (
                    math.ceil(rows / 128) * math.ceil(outputs / expected_n) >= 128
                )
                if full_grid:
                    assert schedule.block_m == 128
                    assert schedule.block_k == 64
                else:
                    assert schedule.block_m == 64
                    assert schedule.block_k == (64 if outputs <= 64 else 128)


def test_nvidia_preparation_schedule_contracts():
    for channels in (64, 128, 256, 512, 1024, 2048, 4096):
        for rows in (64, 1024, 5000, 200_000):
            for group_norm in (False, True):
                schedule = SM120(
                    channels=channels, input_rows=rows, group_norm=group_norm
                ).preparation
                assert (
                    schedule
                    == SM8X(channels=channels, input_rows=rows, group_norm=group_norm).preparation
                )
                assert schedule.block_m > 0
                assert schedule.block_m * channels <= 32768
                assert schedule.num_warps in (4, 8)
                if channels <= 128:
                    wide = SM120(channels=1024, input_rows=rows, group_norm=group_norm).preparation
                    assert schedule.block_m > wide.block_m
                if channels == 512 and rows >= 5000:
                    assert (
                        schedule
                        == SM120(
                            channels=channels, input_rows=rows, group_norm=not group_norm
                        ).preparation
                    )


@pytest.mark.parametrize(
    ("channels", "outputs", "height", "aligned", "expected"),
    [
        (128, 128, 256, True, True),
        (128, 256, 16, True, True),
        (128, 256, 16, False, False),
        (128, 65, 256, True, False),
        (128, 128, 16, True, False),
        (256, 256, 256, True, True),
        (256, 512, 68, True, True),
        (512, 512, 256, True, False),
        (64, 64, 256, True, False),
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
    assert _plan(AcceleratorTarget("cuda", architecture)) == SM8X()


@pytest.mark.parametrize("target", [AcceleratorTarget("cuda", "sm90"), AcceleratorTarget("cpu")])
def test_policy_rejects_unsupported_targets(target):
    with pytest.raises(ValueError, match="no native policy"):
        _plan(target)


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
    # 200,000 input rows cross the preparation threshold; 3,150 output rows (< 128 tiles) do not.
    activation = torch.empty(1, 256, 1, 1, 1).expand(1, 256, 1, 400, 500)
    weight = torch.empty(256, 3, 3, 3, 256, dtype=torch.int8)
    plan = _policy.default_execution_plan(
        activation,
        weight,
        (1, 8, 8),
        target=AcceleratorTarget("cuda", "sm120"),
        group_norm=fused,
        padding=(1, 1, 1, 1, 2, 0),
    )
    assert plan.preparation == ((32, 4) if fused else (64, 4))
    assert plan.convolution.block_m == 64
    assert plan.convolution != SM120(channels=256, outputs=256, output_rows=200_000).convolution


@pytest.mark.parametrize("architecture", ["sm120", "sm89"])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("block_n", [64, 128, 256])
def test_tuning_tile_recomputes_descriptor_eligibility(architecture, aligned, block_n):
    activation = torch.empty(1, 128, 1, 4, 4)
    storage = torch.empty(192 * 27 * 128 + 1, dtype=torch.int8)
    weight = (storage[:-1] if aligned else storage[1:]).view(192, 3, 3, 3, 128)
    select = partial(
        _policy.default_execution_plan,
        activation,
        weight,
        (1, 1, 1),
        target=AcceleratorTarget("cuda", architecture),
        group_norm=False,
        padding=(1, 1, 1, 1, 2, 0),
    )
    production = select()
    tile = _policy.ConvolutionSchedule(64, block_n, 128, 4, 3)
    candidate = select(convolution_schedule=tile)
    assert candidate.convolution == tile
    assert candidate.preparation == production.preparation
    assert not production.use_weight_descriptor
    assert candidate.use_weight_descriptor is (
        architecture == "sm120" and aligned and block_n == 64
    )


def test_amd_policy_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setattr(_policy.sys, "platform", "darwin")
    assert not _policy.supports_target(AcceleratorTarget("hip", "gfx1201"))


@pytest.mark.parametrize("channels", [64, 128, 256, 512, 1024, 2048, 4096])
def test_amd_preparation_bounds_rotation_tile_and_never_uses_descriptors(channels):
    for fused in (False, True):
        schedule = RDNA4(channels=channels, input_rows=32768, group_norm=fused).preparation
        assert schedule.block_m * channels <= 4096
        assert schedule.block_m > 0
        assert schedule == RDNA4(channels=channels, input_rows=16, group_norm=fused).preparation
    assert not RDNA4(channels=channels, outputs=256, output_height=256).use_weight_descriptor


def test_amd_convolution_schedule_contracts():
    for channels in (64, 128, 256, 512, 1024):
        for outputs in (32, 64, 128, 256, 384, 512, 1024, 16384):
            for rows in (18, 64, 192, 1024, 3072, 8192, 40960):
                plan = RDNA4(channels=channels, outputs=outputs, output_rows=rows)
                schedule = plan.convolution
                assert plan.lds_only_fences
                assert not plan.use_weight_descriptor
                assert schedule.num_stages == 2
                assert schedule.num_warps in (4, 8)
                if schedule.block_n == 256:
                    assert outputs % 256 == 0
                if schedule.block_m == 128:
                    assert (
                        math.ceil(rows / schedule.block_m) * math.ceil(outputs / schedule.block_n)
                        >= 64
                    )
                if rows < 128:
                    assert schedule.block_m <= 64
                if outputs < 64 or rows < 64:
                    assert schedule.block_n == 64


def test_convolution_accumulator_seed_contracts():
    gfx1201 = AcceleratorTarget("hip", "gfx1201")
    gfx1200 = AcceleratorTarget("hip", "gfx1200")
    tile_64 = _policy.ConvolutionSchedule(64, 64, 64, 4, 2)
    wide_tile = _policy.ConvolutionSchedule(128, 256, 64, 8, 2)
    # 64-tile minimum and 4096-element reduction minimum checked with explicit schedules.
    assert (
        _plan(
            gfx1201, channels=256, outputs=64, output_rows=4032, convolution_schedule=tile_64
        ).accumulator_seed
        == 0
    )
    assert (
        _plan(
            gfx1201, channels=256, outputs=64, output_rows=4096, convolution_schedule=tile_64
        ).accumulator_seed
        == 0x155555
    )
    assert (
        _plan(
            gfx1201, channels=256, outputs=256, output_rows=4096, convolution_schedule=wide_tile
        ).accumulator_seed
        == 0
    )
    assert (
        _plan(
            gfx1201, channels=256, outputs=256, output_rows=8192, convolution_schedule=wide_tile
        ).accumulator_seed
        == 0x155555
    )
    assert (
        _plan(
            gfx1201, channels=128, outputs=256, output_rows=8192, convolution_schedule=wide_tile
        ).accumulator_seed
        == 0
    )
    assert (
        _plan(
            gfx1201, channels=152, outputs=256, output_rows=8192, convolution_schedule=wide_tile
        ).accumulator_seed
        == 0x155555
    )
    # Only gfx1201 uses the zero-crossing seed.
    assert _plan(gfx1200, channels=256, outputs=256, output_rows=8192).accumulator_seed == 0
    assert (
        _plan(
            gfx1200, channels=256, outputs=256, output_rows=8192, convolution_schedule=wide_tile
        ).accumulator_seed
        == 0
    )
    assert (
        SM120(
            channels=256, outputs=256, output_rows=8192, convolution_schedule=wide_tile
        ).accumulator_seed
        == 0
    )
    assert (
        SM8X(
            channels=256, outputs=256, output_rows=8192, convolution_schedule=wide_tile
        ).accumulator_seed
        == 0
    )
