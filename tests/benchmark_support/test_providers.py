from collections.abc import Callable

from _lib.providers import BenchmarkProvider, measure_provider
from _lib.timing import Timing


def _fake_timer(function: Callable[[], object], warmup_ms: int, repeat_ms: int) -> Timing:
    assert warmup_ms == 2
    assert repeat_ms == 5
    function()
    return Timing(median_ms=1.0, p20_ms=0.8, p80_ms=1.2)


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
        repeat_ms=5,
        timer=_fake_timer,
    )

    assert measurement.provider == "test"
    assert measurement.output > 0
    assert measurement.configuration == {"block": 64}
    assert measurement.timings.compilation_ms is not None
    assert measurement.timings.preparation == Timing(1.0, 0.8, 1.2)
    assert measurement.timings.kernel == Timing(1.0, 0.8, 1.2)
    assert measurement.timings.complete == Timing(1.0, 0.8, 1.2)
    assert len(prepared_values) == 4
    assert len(synchronized) == 3


def test_measure_provider_can_skip_inapplicable_phases() -> None:
    measurement = measure_provider(
        BenchmarkProvider(name="kernel", prepare=lambda: 3, run=lambda value: value + 1),
        warmup_ms=2,
        repeat_ms=5,
        timer=_fake_timer,
        measure_compilation=False,
        measure_preparation=False,
        measure_complete=False,
    )

    assert measurement.output == 4
    assert measurement.timings.compilation_ms is None
    assert measurement.timings.preparation is None
    assert measurement.timings.complete is None
