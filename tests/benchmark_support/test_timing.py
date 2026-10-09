import time
from unittest.mock import Mock

import pytest
from lib import timing as timing_module
from lib.timing import (
    ClockDomain,
    DeviceTimings,
    SampleTimings,
    Timing,
    measure_device,
    synchronized_wall_benchmark,
)


def test_device_timing_preserves_cache_and_graph_measurements(monkeypatch):
    import triton.testing  # noqa: PLC0415

    cold = Mock(
        side_effect=[
            Timing(value, value, value, ClockDomain.DEVICE_EVENT) for value in (3.0, 1.0, 2.0)
        ]
    )
    graph = Mock(side_effect=[6.0, 4.0, 5.0])
    monkeypatch.setattr(timing_module, "triton_benchmark", cold)
    monkeypatch.setattr(triton.testing, "do_bench_cudagraph", graph)
    operation = Mock()
    result = measure_device(operation, warmup_ms=10, measurement_time_ms=20, samples=3)
    assert result.cache_flushed_samples_ms == (3.0, 1.0, 2.0)
    assert result.graph_samples_ms == (6.0, 4.0, 5.0)
    assert result.cache_flushed.median_ms == 2.0
    assert result.graph.median_ms == 5.0
    assert result.as_dict()["sample_statistic"] == "median"
    assert result.cache_flushed.clock is ClockDomain.DEVICE_EVENT
    assert result.graph.clock is ClockDomain.GRAPH_DEVICE_EVENT
    assert cold.call_count == graph.call_count == 3
    cold.assert_called_with(operation, 10, 20)
    graph.assert_called_with(operation, rep=20, return_mode="median")


@pytest.mark.parametrize(("warmup", "duration", "samples"), [(-1, 1, 1), (0, 0, 1), (0, 1, 0)])
def test_device_timing_rejects_invalid_windows_before_launch(
    monkeypatch, warmup, duration, samples
):
    timer = Mock()
    monkeypatch.setattr(timing_module, "triton_benchmark", timer)
    with pytest.raises(ValueError, match="requires positive"):
        measure_device(Mock(), warmup_ms=warmup, measurement_time_ms=duration, samples=samples)
    timer.assert_not_called()


def test_device_timings_reject_mismatched_or_empty_samples():
    with pytest.raises(ValueError, match="equal sample counts"):
        DeviceTimings(0, 1, (1.0,), (1.0, 2.0))
    with pytest.raises(ValueError, match="latency samples"):
        DeviceTimings(0, 1, (), ())


@pytest.mark.parametrize("clock", list(ClockDomain))
def test_sample_quantiles_preserve_the_clock_and_input_order(clock):
    samples = [5.0, 1.0, 3.0]
    timing = Timing.from_samples(samples, clock)
    assert samples == [5.0, 1.0, 3.0]
    assert timing.as_dict() == {
        "median_ms": 3.0,
        "p20_ms": 1.8,
        "p80_ms": 4.2,
        "clock": clock.value,
        "sample_count": 3,
    }


@pytest.mark.parametrize("samples", [[], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_samples_are_rejected(samples):
    with pytest.raises(ValueError, match="latency samples"):
        Timing.from_samples(samples, ClockDomain.DEVICE_EVENT)
    with pytest.raises(ValueError, match="latency samples"):
        SampleTimings(warmup_calls=1, samples_ms=tuple(samples))


def test_fixed_count_timings_report_only_measured_phases():
    timing = SampleTimings(warmup_calls=1, samples_ms=(2.0,))
    assert timing.as_dict() == {
        "warmup_calls": 1,
        "sample_count": 1,
        "operator_end_to_end": {
            "median_ms": 2.0,
            "p20_ms": 2.0,
            "p80_ms": 2.0,
            "clock": "synchronized_wall",
            "sample_count": 1,
        },
        "samples_ms": [2.0],
    }
    with pytest.raises(ValueError, match="warmup calls"):
        SampleTimings(warmup_calls=-1, samples_ms=(2.0,))


def test_synchronized_wall_benchmark_captures_host_work() -> None:
    calls = 0
    synchronizations = 0

    def host_work() -> None:
        nonlocal calls
        calls += 1
        time.sleep(0.001)

    def synchronize() -> None:
        nonlocal synchronizations
        synchronizations += 1

    timing = synchronized_wall_benchmark(
        host_work,
        warmup_ms=0,
        measurement_time_ms=3,
        synchronize=synchronize,
    )

    assert timing.clock is ClockDomain.SYNCHRONIZED_WALL
    assert timing.p20_ms >= 0.8
    assert timing.p20_ms <= timing.median_ms <= timing.p80_ms
    assert synchronizations == calls + 1
    assert timing.sample_count == calls


@pytest.mark.parametrize(
    ("warmup_ms", "measurement_time_ms"),
    [(-1, 1), (0, 0)],
)
def test_synchronized_wall_benchmark_validates_time_windows(
    warmup_ms: int,
    measurement_time_ms: int,
) -> None:
    with pytest.raises(ValueError, match=r"warmup.*measurement time"):
        synchronized_wall_benchmark(
            lambda: None,
            warmup_ms=warmup_ms,
            measurement_time_ms=measurement_time_ms,
        )
