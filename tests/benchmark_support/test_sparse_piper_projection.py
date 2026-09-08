"""Shape validation and transparent timing/accounting for fused projections."""

from unittest.mock import Mock

import benchmark_sparse_piper_projection as benchmark
import pytest
import torch
from lib.timing import ClockDomain, Timing
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize(
    "option", ["--sequence", "--batch", "--heads", "--input-features", "--rep-ms", "--samples"]
)
def test_nonpositive_counts_are_rejected(option):
    with pytest.raises(SystemExit):
        benchmark._parse_args([option, "0"])


@pytest.mark.parametrize("arguments", [["--sequence", "63"], ["--input-features", "272"]])
def test_invalid_preparation_shapes_are_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark._parse_args(arguments)


def test_h3_defaults_and_long_ragged_sequence():
    args = benchmark._parse_args(["--sequence", "100000"])
    assert args.sequence == [100000]
    assert args.input_features == 5376
    assert args.heads == 56
    assert args.batch == 1
    assert args.samples == 3


def test_unsupported_backend_rejects_before_large_allocations_or_input_preparation(monkeypatch):
    select = Mock(side_effect=ValueError("sparse projections are unavailable"))
    monkeypatch.setattr(benchmark._backend, "require_projection_backend", select)
    monkeypatch.setattr(torch, "Generator", Mock(side_effect=AssertionError("created RNG")))
    monkeypatch.setattr(torch, "randn", Mock(side_effect=AssertionError("allocated inputs")))
    monkeypatch.setattr(
        benchmark, "select_preparation_backend", Mock(side_effect=AssertionError("prepared input"))
    )
    args = benchmark._parse_args(["--device", "1", "--sequence", "100000"])
    with FakeTensorMode(), pytest.raises(ValueError, match="sparse projections are unavailable"):
        benchmark._benchmark(args, args.sequence[0])
    select.assert_called_once()
    probe = select.call_args.args[0]
    assert probe.device == torch.device("cuda:1")
    assert probe.numel() == 0


@pytest.mark.parametrize("operations", [0, 6_000_000_000])
def test_measure_reports_medians_and_does_not_credit_mean_reduction_with_integer_ops(
    monkeypatch, operations
):
    cold = Mock(side_effect=[3.0, 2.0, 1.0])
    graph = Mock(side_effect=[4.0, 3.0, 2.0])
    wall = Timing(5.0, 4.0, 6.0, ClockDomain.SYNCHRONIZED_WALL)
    monkeypatch.setattr(benchmark, "do_bench", cold)
    monkeypatch.setattr(benchmark, "do_bench_cudagraph", graph)
    monkeypatch.setattr(benchmark, "synchronized_wall_benchmark", Mock(return_value=wall))
    result = benchmark._measure(lambda: None, operations, benchmark._parse_args([]))
    assert result["cache_flushed_samples_ms"] == [3.0, 2.0, 1.0]
    assert result["graph_samples_ms"] == [4.0, 3.0, 2.0]
    assert result["graph_median_ms"] == 3.0
    assert result["integer_operations"] == operations
    assert result["cache_flushed_effective_tops"] == (3.0 if operations else None)
    assert result["graph_effective_tops"] == (2.0 if operations else None)
    assert result["wall"] == wall.as_dict()
