"""Guard FFN benchmark timing, configuration, and experiment isolation."""

import os
from types import SimpleNamespace

import benchmark_nvfp4_ffn as benchmark
import pytest


@pytest.mark.parametrize(
    "arguments",
    [
        ["--shape", "0", "256", "512"],
        ["--shape", "127", "256", "384"],
        ["--chunk-rows", "129"],
        ["--calls-per-graph", "0"],
        ["--rotated-workspace-mib", "-1"],
        ["--format", "nvfp4", "--rotated-workspace-mib", "0"],
    ],
)
def test_invalid_benchmark_configuration_fails_before_cuda(arguments):
    with pytest.raises(SystemExit, match="2"):
        benchmark._parse_args(arguments)


def test_graph_measurement_excludes_warmup_and_records_call_counts(monkeypatch):
    graph = object()
    events = []
    measurements = iter([1.0, 91.0, 92.0, 2.0, 3.0, 4.0])

    def elapsed(measured_graph, replays, calls):
        events.append((measured_graph, replays, calls))
        return next(measurements)

    monkeypatch.setattr(benchmark, "_elapsed_ms", elapsed)
    config = benchmark.GraphTiming(calls_per_graph=4, warmup_rounds=2, samples=3, sample_ms=16)
    samples, iterations = benchmark._measure_graph(graph, config)

    assert events == [(graph, 5, 4), *[(graph, 4, 4)] * 5]
    assert samples == [2.0, 3.0, 4.0]
    assert iterations == 16


def test_workspace_override_is_restored_after_failure(monkeypatch):
    monkeypatch.setattr(benchmark.rotated_preparation, "_ROTATED_WORKSPACE_BYTES", 99)

    def fail_measurement():
        with benchmark._workspace_limit(0):
            assert benchmark.rotated_preparation._ROTATED_WORKSPACE_BYTES == 0
            raise RuntimeError("measurement failed")

    with pytest.raises(RuntimeError, match="measurement failed"):
        fail_measurement()
    assert benchmark.rotated_preparation._ROTATED_WORKSPACE_BYTES == 99


def test_exclusivity_checks_only_the_selected_gpu(monkeypatch):
    monkeypatch.setattr(benchmark.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        benchmark.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid="GPU-selected"),
    )
    query = SimpleNamespace(stdout=f"{os.getpid()}, GPU-selected\n123, GPU-other\n\n")
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: query)
    benchmark._require_exclusive_gpu()
    query.stdout += f"{os.getpid() + 1}, GPU-selected\n"
    with pytest.raises(RuntimeError, match="GPU is shared"):
        benchmark._require_exclusive_gpu()
