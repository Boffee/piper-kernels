import time

import pytest
from lib.timing import ClockDomain, SampleTimings, Timing, synchronized_wall_benchmark


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
