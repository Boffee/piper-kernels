from collections.abc import Callable

import pytest
from lib.cases import CATALOG_VERSION
from lib.providers import BenchmarkProvider, Implementation, ProviderPhase
from lib.quality import QualityMetrics
from lib.reporting import OutputFormat, OutputTarget
from lib.timing import ClockDomain, Timing
from lib.tuning import (
    TuningStatus,
    UnsupportedTuningCandidateError,
    boolean_tuning_axis,
    meets_minimum_sqnr,
    parse_optional_integer,
    report_tuning_run,
    tune_candidates,
    tuning_axis,
    validate_tuning_candidate_count,
)
from triton.runtime.errors import OutOfResources


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


def _candidate(name: str, latency: int) -> Implementation[BenchmarkProvider[int, int]]:
    return Implementation(
        name=name,
        configuration={"block_m": latency * 64},
        build=lambda: BenchmarkProvider(
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
    assert parse_optional_integer("0") is None
    assert parse_optional_integer("2") == 2

    with pytest.raises(SystemExit, match="search expands to 6 candidates"):
        validate_tuning_candidate_count(((32, 64), (True,), (1, 2, 3)), 5)


def test_shared_sqnr_gate_rejects_nonfinite_mismatches() -> None:
    assert meets_minimum_sqnr(_quality(20.0), 20.0)
    assert not meets_minimum_sqnr(_quality(100.0, nonfinite_mismatch_count=1), 20.0)


def test_tuning_selects_fastest_candidate_and_records_every_result(environment) -> None:
    run = tune_candidates(
        (_candidate("slow", 2), _candidate("fast", 1)),
        tuning="attention",
        shape={"sequence": 128},
        case_id="attention-small",
        environment=environment,
        warmup_ms=2,
        measurement_time_ms=5,
        device_timer=_timer,
    )

    assert all(record.as_dict()["case_id"] == "attention-small" for record in run.records)
    assert all(record.as_dict()["catalog_version"] == CATALOG_VERSION for record in run.records)
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


def test_tuning_rejects_quality_before_spending_measurement_time(environment) -> None:
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
        environment=environment,
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
        None,
    ],
)
def test_tuning_records_expected_candidate_failures(environment, error: Exception | None) -> None:
    def make_provider() -> BenchmarkProvider[None, None]:
        assert error is not None, "unsupported factory must not run"
        raise error

    run = tune_candidates(
        (Implementation("unsupported", make_provider, "unavailable" if error is None else None),),
        tuning="attention",
        shape={},
        environment=environment,
    )

    assert run.winner is None
    assert run.records[0].status is TuningStatus.SKIPPED
    assert (type(error).__name__ if error is not None else "unavailable") in run.records[0].reason


def test_tuning_propagates_unexpected_failures(environment) -> None:
    def make_provider() -> BenchmarkProvider[None, None]:
        raise RuntimeError("implementation bug")

    with pytest.raises(RuntimeError, match="implementation bug"):
        tune_candidates(
            (Implementation("broken", make_provider),),
            tuning="attention",
            shape={},
            environment=environment,
        )


def test_end_to_end_phase_uses_wall_timer(environment, monkeypatch: pytest.MonkeyPatch) -> None:
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
        (Implementation("end-to-end", make_provider),),
        tuning="attention",
        shape={},
        environment=environment,
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
    environment,
    candidates: tuple[Implementation[BenchmarkProvider[int, int]], ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        tune_candidates(
            candidates,
            tuning="attention",
            shape={},
            environment=environment,
        )


def test_report_tuning_run_prints_and_writes_selected_candidate(
    environment,
    tmp_path,
    capsys,
) -> None:
    run = tune_candidates(
        (_candidate("winner", 1),),
        tuning="attention",
        shape={},
        environment=environment,
        warmup_ms=2,
        measurement_time_ms=5,
        device_timer=_timer,
    )
    path = tmp_path / "tuning.json"

    report_tuning_run(run, OutputTarget(path, OutputFormat.JSON))

    assert "selected: winner" in capsys.readouterr().out
    assert '"candidate": "winner"' in path.read_text()
