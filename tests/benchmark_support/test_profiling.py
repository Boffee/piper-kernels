import pytest
from _lib.profiling import CudaProfilerController, ProfilePhase, profile_provider
from _lib.providers import BenchmarkProvider


class FakeCapture:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def push_range(self, name: str) -> None:
        self.events.append(f"push:{name}")

    def pop_range(self) -> None:
        self.events.append("pop")


def test_cuda_capture_rejects_unavailable_and_rocm_backends(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available NVIDIA GPU"):
        CudaProfilerController()

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.version.hip", "7.0")
    with pytest.raises(RuntimeError, match="ROCTracer/ROCTx"):
        CudaProfilerController()


def test_cuda_capture_uses_checked_public_profiler_api(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.version.hip", None)
    monkeypatch.setattr("torch.version.cuda", "13.0")
    monkeypatch.setattr("torch.cuda.profiler.start", lambda: events.append("start"))
    monkeypatch.setattr("torch.cuda.profiler.stop", lambda: events.append("stop"))

    controller = CudaProfilerController()
    controller.start()
    controller.stop()

    assert events == ["start", "stop"]


def _provider(events: list[str]) -> BenchmarkProvider[int, int]:
    def prepare() -> int:
        events.append("prepare")
        return 3

    def run(prepared: int) -> int:
        events.append(f"run:{prepared}")
        return prepared + 1

    return BenchmarkProvider(
        name="test-provider",
        prepare=prepare,
        run=run,
        synchronize=lambda: events.append("synchronize"),
    )


def test_profile_excludes_compilation_and_warmup_by_default() -> None:
    events: list[str] = []

    result = profile_provider(
        _provider(events),
        iterations=2,
        warmup_iterations=1,
        controller=FakeCapture(events),
    )

    start_index = events.index("start")
    assert events[:start_index] == [
        "prepare",
        "run:3",
        "prepare",
        "run:3",
        "synchronize",
    ]
    assert events[start_index:] == [
        "start",
        "push:profile",
        "prepare",
        "run:3",
        "prepare",
        "run:3",
        "synchronize",
        "pop",
        "stop",
    ]
    assert result.phase is ProfilePhase.OPERATOR_END_TO_END
    assert not result.include_setup


def test_profile_can_capture_setup_in_a_distinct_range() -> None:
    events: list[str] = []

    result = profile_provider(
        _provider(events),
        iterations=1,
        warmup_iterations=1,
        range_name="attention",
        include_setup=True,
        controller=FakeCapture(events),
    )

    assert events == [
        "start",
        "push:attention/setup",
        "prepare",
        "run:3",
        "prepare",
        "run:3",
        "synchronize",
        "pop",
        "push:attention",
        "prepare",
        "run:3",
        "synchronize",
        "pop",
        "stop",
    ]
    assert result.include_setup


def test_prepared_execution_prepares_once_outside_capture() -> None:
    events: list[str] = []

    result = profile_provider(
        _provider(events),
        iterations=2,
        warmup_iterations=0,
        phase=ProfilePhase.PREPARED_EXECUTION,
        controller=FakeCapture(events),
    )

    assert events.count("prepare") == 1
    assert events.count("run:3") == 3
    assert events.index("prepare") < events.index("start")
    assert result.phase is ProfilePhase.PREPARED_EXECUTION


def test_capture_is_stopped_when_measured_launch_fails() -> None:
    events: list[str] = []
    launches = 0

    def run(_prepared: None) -> None:
        nonlocal launches
        launches += 1
        events.append("run")
        if launches == 2:
            raise RuntimeError("kernel failed")

    provider = BenchmarkProvider(
        name="failing",
        prepare=lambda: None,
        run=run,
        synchronize=lambda: events.append("synchronize"),
    )

    with pytest.raises(RuntimeError, match="kernel failed"):
        profile_provider(
            provider,
            iterations=1,
            warmup_iterations=0,
            controller=FakeCapture(events),
        )

    assert events[-3:] == ["synchronize", "pop", "stop"]
