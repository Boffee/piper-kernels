from collections.abc import Callable

import pytest
from lib.environment import EnvironmentInfo
from lib.providers import BenchmarkProvider, ProviderPhase
from lib.quality import QualityMetrics
from lib.timing import ClockDomain, Timing
from lib.tuning import (
    TuningCandidate,
    TuningStatus,
    UnsupportedTuningCandidateError,
    boolean_tuning_axis,
    meets_minimum_sqnr,
    optional_integer_tuning_axis,
    tune_candidates,
    tuning_axis,
    tuning_candidate_count,
)
from triton.runtime.errors import OutOfResources


def _environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        captured_at_utc="2026-08-08T00:00:00+00:00",
        python_version="3.14.0",
        platform="test",
        torch_version="2.12.0",
        triton_version="3.7.1",
        accelerator_backend="cuda",
        accelerator_runtime_version="13.0",
        accelerator_driver_version="580.0",
        gpu_name="test GPU",
        gpu_architecture="SM120",
        gpu_index=0,
        git_revision="a" * 40,
        git_dirty=False,
    )


def _quality(sqnr_db: float, *, nonfinite_mismatch_count: int = 0) -> QualityMetrics:
    return QualityMetrics(
        mean_absolute_error=0.0,
        max_absolute_error=0.0,
        relative_l1_error=0.0,
        relative_l2_error=0.0,
        sqnr_db=sqnr_db,
        cosine_similarity=1.0,
        actual_nonfinite_count=0,
        reference_nonfinite_count=0,
        nonfinite_mismatch_count=nonfinite_mismatch_count,
    )


def _timer(
    function: Callable[[], object],
    warmup_ms: int,
    measurement_time_ms: int,
) -> Timing:
    assert warmup_ms == 2
    assert measurement_time_ms == 5
    latency = float(function())
    return Timing(latency, latency, latency, ClockDomain.DEVICE_EVENT)


def _candidate(name: str, latency: int) -> TuningCandidate[int, int]:
    return TuningCandidate(
        name=name,
        configuration={"block_m": latency * 64},
        make_provider=lambda: BenchmarkProvider(
            name=name,
            prepare=lambda: latency,
            run=lambda prepared: prepared,
            configuration={"resolved": True},
        ),
    )


def test_shared_tuning_axes_default_and_deduplicate_explicit_values() -> None:
    assert tuning_axis(None, 64) == (64,)
    assert tuning_axis([32, 64, 32], 128) == (32, 64)
    assert boolean_tuning_axis(None, True) == (True,)
    assert boolean_tuning_axis(False, True) == (False,)
    assert optional_integer_tuning_axis(None, 3) == (3,)
    assert optional_integer_tuning_axis(["none", "2", "none"], 3) == (None, 2)
    assert tuning_candidate_count(((32, 64), (True,), (1, 2, 3))) == 6


def test_shared_sqnr_gate_rejects_nonfinite_mismatches() -> None:
    assert meets_minimum_sqnr(_quality(20.0), 20.0)
    assert not meets_minimum_sqnr(_quality(100.0, nonfinite_mismatch_count=1), 20.0)


def test_tuning_selects_fastest_candidate_and_records_every_result() -> None:
    run = tune_candidates(
        (_candidate("slow", 2), _candidate("fast", 1)),
        tuning="attention",
        shape={"sequence": 128},
        environment=_environment(),
        warmup_ms=2,
        measurement_time_ms=5,
        device_timer=_timer,
    )

    assert [record.candidate for record in run.records] == ["slow", "fast"]
    assert run.winner is not None
    assert run.winner.candidate == "fast"
    assert [record.selected for record in run.records] == [False, True]
    assert all(record.status is TuningStatus.MEASURED for record in run.records)
    assert run.winner.configuration["resolved"] is True
    value = run.winner.as_dict()
    assert value["warmup_ms"] == 2
    assert value["measurement_time_ms"] == 5
    assert value["environment"]["gpu_architecture"] == "SM120"


def test_tuning_rejects_quality_before_spending_measurement_time() -> None:
    timer_calls = 0

    def timer(
        function: Callable[[], object],
        _warmup_ms: int,
        _measurement_time_ms: int,
    ) -> Timing:
        nonlocal timer_calls
        timer_calls += 1
        function()
        return Timing(1.0, 1.0, 1.0, ClockDomain.DEVICE_EVENT)

    run = tune_candidates(
        (_candidate("bad", 1), _candidate("good", 2)),
        tuning="attention",
        shape={},
        environment=_environment(),
        measure_candidate_quality=lambda output: _quality(float(output) * 10),
        quality_gate=lambda quality: quality.sqnr_db >= 20,
        device_timer=timer,
    )

    assert timer_calls == 1
    assert run.records[0].status is TuningStatus.QUALITY_REJECTED
    assert run.records[0].timing is None
    assert run.winner is not None
    assert run.winner.candidate == "good"


@pytest.mark.parametrize(
    "error",
    [
        UnsupportedTuningCandidateError("unsupported schedule"),
        OutOfResources(256, 128, "registers"),
    ],
)
def test_tuning_records_expected_candidate_failures(error: Exception) -> None:
    def make_provider() -> BenchmarkProvider[None, None]:
        raise error

    run = tune_candidates(
        (TuningCandidate("unsupported", {}, make_provider),),
        tuning="attention",
        shape={},
        environment=_environment(),
    )

    assert run.winner is None
    assert run.records[0].status is TuningStatus.SKIPPED
    assert type(error).__name__ in (run.records[0].reason or "")


def test_tuning_propagates_unexpected_failures() -> None:
    def make_provider() -> BenchmarkProvider[None, None]:
        raise RuntimeError("implementation bug")

    with pytest.raises(RuntimeError, match="implementation bug"):
        tune_candidates(
            (TuningCandidate("broken", {}, make_provider),),
            tuning="attention",
            shape={},
            environment=_environment(),
        )


def test_end_to_end_phase_uses_wall_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def wall_timer(
        function: Callable[[], object],
        warmup_ms: int,
        measurement_time_ms: int,
        *,
        synchronize: Callable[[], None] | None = None,
    ) -> Timing:
        assert warmup_ms == 2
        assert measurement_time_ms == 5
        function()
        return Timing(1.0, 1.0, 1.0, ClockDomain.SYNCHRONIZED_WALL)

    def make_provider() -> BenchmarkProvider[int, int]:
        return BenchmarkProvider(
            name="end-to-end",
            prepare=lambda: calls.append("prepare") or 1,
            run=lambda value: calls.append("run") or value,
            synchronize=lambda: calls.append("synchronize"),
        )

    monkeypatch.setattr("lib.tuning.synchronized_wall_benchmark", wall_timer)
    run = tune_candidates(
        (TuningCandidate("end-to-end", {}, make_provider),),
        tuning="attention",
        shape={},
        environment=_environment(),
        phase=ProviderPhase.OPERATOR_END_TO_END,
        warmup_ms=2,
        measurement_time_ms=5,
    )

    assert run.winner is not None
    assert run.winner.timing is not None
    assert run.winner.timing.clock is ClockDomain.SYNCHRONIZED_WALL
    assert calls == ["prepare", "run", "synchronize", "prepare", "run"]


@pytest.mark.parametrize(
    ("candidates", "error"),
    [
        ((), "at least one"),
        ((_candidate("same", 1), _candidate("same", 2)), "unique"),
    ],
)
def test_tuning_validates_candidate_lists(
    candidates: tuple[TuningCandidate[int, int], ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        tune_candidates(
            candidates,
            tuning="attention",
            shape={},
            environment=_environment(),
        )
