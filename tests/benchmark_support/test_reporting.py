import argparse
import json
from pathlib import Path

from lib.environment import EnvironmentInfo, capture_environment
from lib.quality import QualityMetrics
from lib.reporting import (
    BenchmarkRecord,
    OutputFormat,
    OutputTarget,
    add_output_arguments,
    output_target,
    write_records,
)
from lib.timing import ClockDomain, PhaseTimings, Timing


def _environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        captured_at_utc="2026-08-06T00:00:00+00:00",
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


def _record() -> BenchmarkRecord:
    wall_timing = Timing(1.0, 0.8, 1.2, ClockDomain.SYNCHRONIZED_WALL)
    device_timing = Timing(1.0, 0.8, 1.2, ClockDomain.DEVICE_EVENT)
    quality = QualityMetrics(
        mean_absolute_error=0.0,
        max_absolute_error=0.0,
        relative_l1_error=0.0,
        relative_l2_error=0.0,
        sqnr_db=float("inf"),
        cosine_similarity=1.0,
        actual_nonfinite_count=0,
        reference_nonfinite_count=0,
        nonfinite_mismatch_count=0,
    )
    return BenchmarkRecord(
        benchmark="attention",
        provider="test",
        shape={"sequence": 128},
        configuration={"dtype": "float16"},
        timings=PhaseTimings(
            warmup_ms=100,
            measurement_time_ms=500,
            first_call_ms=12.0,
            preparation=wall_timing,
            prepared_execution=device_timing,
            operator_end_to_end=wall_timing,
        ),
        quality=quality,
        environment=_environment(),
    )


def test_json_output_is_versioned_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "results.json"

    write_records([_record()], OutputTarget(path, OutputFormat.JSON))

    values = json.loads(path.read_text())
    assert values[0]["schema_version"] == 1
    assert values[0]["timings"]["warmup_ms"] == 100
    assert values[0]["timings"]["measurement_time_ms"] == 500
    assert values[0]["timings"]["first_call_clock"] == "synchronized_wall"
    assert values[0]["timings"]["prepared_execution"]["median_ms"] == 1.0
    assert values[0]["timings"]["prepared_execution"]["clock"] == "device_event"
    assert values[0]["timings"]["operator_end_to_end"]["clock"] == "synchronized_wall"
    assert values[0]["quality"]["sqnr_db"] is None
    assert values[0]["environment"]["gpu_architecture"] == "SM120"


def test_jsonl_output_has_one_record_per_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"

    write_records([_record(), _record()], OutputTarget(path, OutputFormat.JSONL))

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["provider"] == "test" for line in lines)


def test_output_arguments_are_optional_and_mutually_exclusive(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_output_arguments(parser)

    assert output_target(parser.parse_args([])) is None
    assert output_target(parser.parse_args(["--json", str(tmp_path / "a.json")])) == OutputTarget(
        tmp_path / "a.json",
        OutputFormat.JSON,
    )


def test_output_arguments_support_namespaced_record_types(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_output_arguments(parser, option_prefix="compiler", record_name="compiler report")
    path = tmp_path / "compiler.jsonl"

    arguments = parser.parse_args(["--compiler-jsonl", str(path)])

    assert output_target(arguments, option_prefix="compiler") == OutputTarget(
        path,
        OutputFormat.JSONL,
    )


def test_environment_capture_does_not_require_cuda(monkeypatch, tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setenv("GIT_DIR", str(repository / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(repository))
    monkeypatch.setenv("GIT_INDEX_FILE", str(repository / ".git" / "index"))

    environment = capture_environment(tmp_path)

    assert environment.gpu_name is None
    assert environment.gpu_architecture is None
    assert environment.git_revision is None


def test_environment_capture_identifies_rocm(monkeypatch, tmp_path: Path) -> None:
    class DeviceProperties:
        gcnArchName = "gfx1201:sramecc+:xnack-"  # noqa: N815

    monkeypatch.setattr("torch.version.hip", "7.0")
    monkeypatch.setattr("torch.version.cuda", None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda _index: "AMD Radeon")
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda _index: DeviceProperties(),
    )

    environment = capture_environment(tmp_path)

    assert environment.accelerator_backend == "rocm"
    assert environment.accelerator_runtime_version == "7.0"
    assert environment.accelerator_driver_version is None
    assert environment.gpu_name == "AMD Radeon"
    assert environment.gpu_architecture == "gfx1201"
