"""Argument contracts for the reproducible ROCm convolution benchmark."""

import argparse

import benchmark_convrot_int8_conv3d_rocm as benchmark
import pytest


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
        ]
    )
    assert args.shape == [(1, 128, 5, 64, 64, 128), (2, 4096, 1, 3, 3, 7)]
    assert args.dtype == "float32"
    assert args.rep_ms == 25
    assert args.tune


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
