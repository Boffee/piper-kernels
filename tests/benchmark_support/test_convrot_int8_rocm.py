"""CPU coverage of ROCm benchmark input validation and timing semantics."""

import argparse

import benchmark_convrot_int8_rocm as benchmark
import pytest


@pytest.mark.parametrize("shape", ["1,256,1", "8192,6144,4096", "129,512,300"])
def test_shape_parser(shape):
    assert benchmark._shape(shape) == tuple(map(int, shape.split(",")))


@pytest.mark.parametrize("shape", ["1,2", "1,2,3,4", "x,256,3", "0,256,3", "1,255,3"])
def test_invalid_shapes(shape):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._shape(shape)


@pytest.mark.parametrize("option", ["--repeats", "--rep-ms", "--warmup-ms", "--graph-rep-ms"])
def test_nonpositive_timing_arguments(option):
    with pytest.raises(SystemExit):
        benchmark._parse_args([option, "0"])


def test_timing_keeps_cache_flushed_and_graph_samples_separate(monkeypatch):
    samples = iter([3.0, 1.0, 2.0])
    calls = []

    def cold(operation, **kwargs):
        calls.append((operation, kwargs))
        return next(samples)

    def graph(operation, **kwargs):
        calls.append((operation, kwargs))
        return 4.0

    monkeypatch.setattr(benchmark, "do_bench", cold)
    monkeypatch.setattr(benchmark, "do_bench_cudagraph", graph)
    args = benchmark._parse_args([])
    operation = object
    assert benchmark._measure(operation, args) == {
        "cache_flushed_median_ms": 2.0,
        "cache_flushed_samples_ms": [3.0, 1.0, 2.0],
        "graph_median_ms": 4.0,
    }
    assert len(calls) == 4
    assert all(call[0] is operation for call in calls)
    assert calls[-1][1] == {"rep": args.graph_rep_ms, "return_mode": "median"}
