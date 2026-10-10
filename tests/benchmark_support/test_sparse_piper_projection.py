"""Shape validation and transparent timing/accounting for fused projections."""

from unittest.mock import Mock

import benchmark_sparse_piper_projection as benchmark
import pytest
import torch
from lib.cases import PipelineCase, named_case
from lib.timing import ClockDomain, DeviceTimings, Timing
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize("option", ["--rep-ms", "--samples"])
def test_nonpositive_counts_are_rejected(option):
    with pytest.raises(SystemExit):
        benchmark._parse_args([option, "0"])


@pytest.mark.parametrize("identity", ["missing", "linear-small", "attention-small"])
def test_requires_a_pipeline_case(identity):
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--case", identity])


def test_projection_diagnostic_inherits_shared_gqa_workload():
    args = benchmark._parse_args(["--case", "pipeline-small"])
    case = named_case(args.case)
    assert isinstance(case, PipelineCase)
    assert case.sequence == 257
    assert case.heads == 4
    assert case.kv_heads == 2
    assert case.head_dim == 64
    assert args.samples == 3


def test_unsupported_backend_rejects_before_large_allocations_or_input_preparation(monkeypatch):
    select = Mock(side_effect=ValueError("sparse projections are unavailable"))
    monkeypatch.setattr(benchmark._backend, "require_projection_backend", select)
    monkeypatch.setattr(torch, "Generator", Mock(side_effect=AssertionError("created RNG")))
    monkeypatch.setattr(torch, "randn", Mock(side_effect=AssertionError("allocated inputs")))
    monkeypatch.setattr(
        benchmark, "select_preparation_backend", Mock(side_effect=AssertionError("prepared input"))
    )
    args = benchmark._parse_args(["--device", "1", "--case", "pipeline-small"])
    case = named_case(args.case)
    with FakeTensorMode(), pytest.raises(ValueError, match="sparse projections are unavailable"):
        benchmark._benchmark(args, case, Mock())
    select.assert_called_once()
    probe = select.call_args.args[0]
    assert probe.device == torch.device("cuda:1")
    assert probe.numel() == 0
    assert select.call_args.kwargs == {"head_dim": case.head_dim}


@pytest.mark.parametrize("operations", [0, 6_000_000_000])
def test_measure_reports_medians_and_does_not_credit_mean_reduction_with_integer_ops(
    monkeypatch, operations
):
    timing = DeviceTimings(60, 100, (3.0, 2.0, 1.0), (4.0, 3.0, 2.0))
    wall = Timing(5.0, 4.0, 6.0, ClockDomain.SYNCHRONIZED_WALL)
    timer = Mock(return_value=timing)
    monkeypatch.setattr(benchmark, "measure_device", timer)
    monkeypatch.setattr(benchmark, "synchronized_wall_benchmark", Mock(return_value=wall))
    measured, result = benchmark._measure(lambda: None, operations, benchmark._parse_args([]))
    assert measured is timing
    assert timer.call_args.kwargs == {"warmup_ms": 60, "measurement_time_ms": 100, "samples": 3}
    assert result["integer_operations"] == operations
    assert result["cache_flushed_effective_tops"] == (3.0 if operations else None)
    assert result["graph_effective_tops"] == (2.0 if operations else None)
    assert result["wall"] == wall.as_dict()
