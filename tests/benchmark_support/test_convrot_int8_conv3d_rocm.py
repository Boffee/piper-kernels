"""Argument, measurement, and baseline contracts for the ROCm convolution benchmark."""

import argparse
from unittest.mock import Mock

import benchmark_convrot_int8_conv3d_rocm as benchmark
import pytest
import torch
from torch.nn import functional


def test_default_arguments():
    args = benchmark._parse_args([])
    assert args.shape is None
    assert args.dtype == "float16"
    assert args.rep_ms == 100
    assert not args.tune
    assert not args.miopen_benchmark
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
            "--miopen-benchmark",
            "--skip-reference-timing",
        ]
    )
    assert args.shape == [(1, 128, 5, 64, 64, 128), (2, 4096, 1, 3, 3, 7)]
    assert args.dtype == "float32"
    assert args.rep_ms == 25
    assert args.tune
    assert args.miopen_benchmark
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
        return 1.25 if "warmup" in kwargs else 0.75

    eager_timer = Mock(side_effect=measured)
    graph_timer = Mock(side_effect=measured)
    monkeypatch.setattr(benchmark, "do_bench", eager_timer)
    monkeypatch.setattr(benchmark, "do_bench_cudagraph", graph_timer)
    timings = benchmark._measure_implementations(
        implementations, rep_ms=25, skip_reference_timing=skip_reference_timing
    )
    timed = {
        name: operation
        for name, operation in implementations.items()
        if name != "reference" or not skip_reference_timing
    }
    assert timings == {
        f"{name}{suffix}": elapsed
        for name in timed
        for suffix, elapsed in (("_ms", 1.25), ("_graph_ms", 0.75))
    }
    assert eager_timer.call_count == graph_timer.call_count == len(timed)
    for operation in timed.values():
        eager_timer.assert_any_call(operation, warmup=25, rep=25, return_mode="median")
        graph_timer.assert_any_call(operation, rep=25, return_mode="median")


@pytest.mark.parametrize("skip_reference_timing", [False, True])
@pytest.mark.parametrize("incorrect", ["reference", "fp16_channels_last"])
def test_incorrect_outputs_fail_before_timing(monkeypatch, skip_reference_timing, incorrect):
    implementations = {
        name: Mock(return_value=torch.tensor(1.0))
        for name in ("native", "reference", "fp16", "fp16_channels_last")
    }
    implementations[incorrect].return_value = torch.tensor(2.0)
    eager_timer, graph_timer = Mock(), Mock()
    monkeypatch.setattr(benchmark, "do_bench", eager_timer)
    monkeypatch.setattr(benchmark, "do_bench_cudagraph", graph_timer)
    with pytest.raises(AssertionError):
        benchmark._measure_implementations(
            implementations, rep_ms=25, skip_reference_timing=skip_reference_timing
        )
    eager_timer.assert_not_called()
    graph_timer.assert_not_called()


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
