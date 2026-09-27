"""Shared argument, measurement, and baseline contracts for convolution benchmarks."""

import argparse
from unittest.mock import Mock

import benchmark_convrot_int8_conv3d as benchmark
import pytest
import torch
from lib.timing import DeviceTimings
from torch.nn import functional

from piper_kernels._triton.targets import AcceleratorTarget


def test_default_arguments():
    args = benchmark._parse_args([])
    assert args.shape is None
    assert args.dtype == "float16"
    assert args.measurement_time_ms == 100
    assert not args.tune
    assert not args.vendor_benchmark
    assert not args.skip_reference_timing


def test_shape_and_dtype_arguments():
    args = benchmark._parse_args(
        [
            "--shape",
            "1,128,5,64,64,128",
            "--shape",
            "2,4096,1,3,3,7",
            "--dtype",
            "float32",
            "--rep-ms",
            "25",
            "--tune",
            "--vendor-benchmark",
            "--skip-reference-timing",
        ]
    )
    assert args.shape == [(1, 128, 5, 64, 64, 128), (2, 4096, 1, 3, 3, 7)]
    assert args.dtype == "float32"
    assert args.measurement_time_ms == 25
    assert args.tune
    assert args.vendor_benchmark
    assert args.skip_reference_timing


@pytest.mark.parametrize("skip_reference_timing", [False, True])
def test_measurements_always_check_outputs_before_timing(monkeypatch, skip_reference_timing):
    # FP16 may differ from INT8: only the two implementations of each are compared.
    implementations = {
        name: Mock(return_value=torch.tensor(value))
        for name, value in (
            ("native", 1.0),
            ("reference", 1.0),
            ("fp16", 2.0),
            ("fp16_channels_last", 2.0),
        )
    }

    def measured(operation, **kwargs):
        assert all(implementation.call_count == 1 for implementation in implementations.values())
        return DeviceTimings(25, 25, (1.25,), (0.75,))

    timer = Mock(side_effect=measured)
    monkeypatch.setattr(benchmark, "measure_device", timer)
    timings = benchmark._measure_implementations(
        implementations,
        warmup_ms=25,
        measurement_time_ms=25,
        samples=1,
        skip_reference_timing=skip_reference_timing,
    )
    timed = {
        name: operation
        for name, operation in implementations.items()
        if name != "reference" or not skip_reference_timing
    }
    assert timings == {name: DeviceTimings(25, 25, (1.25,), (0.75,)) for name in timed}
    assert timer.call_count == len(timed)
    for operation in timed.values():
        timer.assert_any_call(operation, warmup_ms=25, measurement_time_ms=25, samples=1)


@pytest.mark.parametrize("skip_reference_timing", [False, True])
@pytest.mark.parametrize("incorrect", ["reference", "fp16_channels_last"])
def test_incorrect_outputs_fail_before_timing(monkeypatch, skip_reference_timing, incorrect):
    implementations = {
        name: Mock(return_value=torch.tensor(1.0))
        for name in ("native", "reference", "fp16", "fp16_channels_last")
    }
    implementations[incorrect].return_value = torch.tensor(2.0)
    timer = Mock()
    monkeypatch.setattr(benchmark, "measure_device", timer)
    with pytest.raises(AssertionError):
        benchmark._measure_implementations(
            implementations,
            warmup_ms=25,
            measurement_time_ms=25,
            samples=1,
            skip_reference_timing=skip_reference_timing,
        )
    timer.assert_not_called()


@pytest.mark.parametrize("arch", ["sm120", "sm89"])
def test_convolution_selects_the_production_nvidia_policy(arch):
    target = AcceleratorTarget("cuda", arch)
    assert benchmark._convolution_policy(target) is benchmark.nvidia_policy


@pytest.mark.parametrize("arch", ["gfx1200", "gfx1201"])
def test_convolution_selects_the_production_amd_policy(arch):
    assert benchmark._convolution_policy(AcceleratorTarget("hip", arch)) is benchmark.amd_policy


@pytest.mark.parametrize(("backend", "arch"), [("cuda", "sm90"), ("hip", "gfx9999"), ("cpu", None)])
def test_unsupported_targets_cannot_benchmark_a_portable_fallback(backend, arch):
    with pytest.raises(ValueError, match="no optimized backend"):
        benchmark._convolution_policy(AcceleratorTarget(backend, arch))


@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last_3d])
@pytest.mark.parametrize("fused", [False, True])
def test_standard_baseline_padding_and_framewise_normalization(memory_format, fused):
    torch.manual_seed(123)
    activation = torch.randn(2, 64, 3, 4, 5).contiguous(memory_format=memory_format)
    weight = torch.randn(7, 64, 3, 3, 3).contiguous(memory_format=memory_format)
    norm_weight, norm_bias = torch.randn(64), torch.randn(64)
    expected_input = activation
    if fused:
        # Independent per-frame execution catches normalization across time.
        expected_input = torch.stack(
            [
                functional.silu(
                    functional.group_norm(activation[:, :, frame], 32, norm_weight, norm_bias, 1e-6)
                )
                for frame in range(3)
            ],
            dim=2,
        )
        actual = benchmark._standard_group_norm_silu_conv3d(
            activation, weight, norm_weight, norm_bias
        )
    else:
        actual = benchmark._standard_conv3d(activation, weight)
    spatial = functional.pad(expected_input, (1, 1, 1, 1, 0, 0), mode="reflect")
    padded = torch.cat([torch.zeros_like(spatial[:, :, :2]), spatial], dim=2)
    expected = functional.conv3d(padded, weight)
    assert actual.shape == (2, 7, 3, 4, 5)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [
        "bad",
        "1,128,3,4,4",
        "0,128,3,4,4,128",
        "1,32,3,4,4,128",
        "1,192,3,4,4,128",
        "1,8192,3,4,4,128",
        "1,128,3,1,4,128",
    ],
)
def test_invalid_shapes_are_rejected(shape):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._shape(shape)


@pytest.mark.parametrize("duration", ["0", "-1"])
def test_nonpositive_measurement_duration_is_rejected(duration):
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--rep-ms", duration])
