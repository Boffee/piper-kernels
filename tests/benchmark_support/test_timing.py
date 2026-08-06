import time

import pytest
from _lib.timing import ClockDomain, synchronized_wall_benchmark


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
