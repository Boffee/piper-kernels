from collections.abc import Callable

from lib.providers import BenchmarkProvider, measure_provider
from lib.timing import ClockDomain, Timing


def _fake_device_timer(
    function: Callable[[], object], warmup_ms: int, measurement_time_ms: int
) -> Timing:
    assert warmup_ms == 2
    assert measurement_time_ms == 5
    function()
    return Timing(
        median_ms=1.0,
        p20_ms=0.8,
        p80_ms=1.2,
        clock=ClockDomain.DEVICE_EVENT,
    )


def _fake_wall_timer(
    function: Callable[[], object], warmup_ms: int, measurement_time_ms: int
) -> Timing:
    assert warmup_ms == 2
    assert measurement_time_ms == 5
    function()
    return Timing(
        median_ms=2.0,
        p20_ms=1.8,
        p80_ms=2.2,
        clock=ClockDomain.SYNCHRONIZED_WALL,
    )


def test_measure_provider_exercises_explicit_phases() -> None:
    prepared_values: list[int] = []
    synchronized: list[None] = []

    def prepare() -> int:
        value = len(prepared_values) + 1
        prepared_values.append(value)
        return value

    provider = BenchmarkProvider(
        name="test",
        prepare=prepare,
        run=lambda prepared: prepared * 2,
        synchronize=lambda: synchronized.append(None),
        configuration={"block": 64},
    )

    measurement = measure_provider(
        provider,
        warmup_ms=2,
        measurement_time_ms=5,
        device_timer=_fake_device_timer,
        wall_timer=_fake_wall_timer,
    )

    assert measurement.provider == "test"
    assert measurement.output > 0
    assert measurement.configuration == {"block": 64}
    assert measurement.timings.first_call_ms is not None
    assert measurement.timings.preparation == Timing(
        2.0, 1.8, 2.2, ClockDomain.SYNCHRONIZED_WALL
    )
    assert measurement.timings.prepared_execution == Timing(
        1.0, 0.8, 1.2, ClockDomain.DEVICE_EVENT
    )
    assert measurement.timings.operator_end_to_end == Timing(
        2.0, 1.8, 2.2, ClockDomain.SYNCHRONIZED_WALL
    )
    assert len(prepared_values) == 4
    assert len(synchronized) == 3


def test_measure_provider_can_skip_inapplicable_phases() -> None:
    measurement = measure_provider(
        BenchmarkProvider(name="kernel", prepare=lambda: 3, run=lambda value: value + 1),
        warmup_ms=2,
        measurement_time_ms=5,
        device_timer=_fake_device_timer,
        wall_timer=_fake_wall_timer,
        measure_first_call=False,
        measure_preparation=False,
        measure_operator_end_to_end=False,
    )

    assert measurement.output == 4
    assert measurement.timings.first_call_ms is None
    assert measurement.timings.preparation is None
    assert measurement.timings.operator_end_to_end is None
