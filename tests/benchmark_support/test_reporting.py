import argparse
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import lib.environment as environment_module
import pytest
from lib.cases import CATALOG_VERSION
from lib.environment import capture_environment
from lib.quality import QualityMetrics
from lib.reporting import (
    BenchmarkRecord,
    OutputFormat,
    OutputTarget,
    add_output_arguments,
    output_target,
    write_records,
)
from lib.timing import ClockDomain, Timing


def _record(environment) -> BenchmarkRecord:
    wall_timing = Timing(1.0, 0.8, 1.2, ClockDomain.SYNCHRONIZED_WALL, sample_count=3)
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
        timings=wall_timing,
        quality=quality,
        environment=environment,
    )


@pytest.mark.parametrize("output_format", [OutputFormat.JSON, OutputFormat.JSONL])
def test_output_is_versioned_strict_and_preserves_catalog_identity(
    environment, tmp_path, output_format
):
    path = tmp_path / f"results.{output_format.value}"
    custom = _record(environment)
    named = replace(custom, case_id="attention-small")
    write_records([named, custom], OutputTarget(path, output_format))
    content = path.read_text()
    values = (
        json.loads(content)
        if output_format is OutputFormat.JSON
        else [json.loads(line) for line in content.splitlines()]
    )
    assert len(values) == 2
    assert [(value["case_id"], value["catalog_version"]) for value in values] == [
        ("attention-small", CATALOG_VERSION),
        (None, None),
    ]
    for value in values:
        assert value["schema_version"] == 1
        assert value["provider"] == "test"
        assert value["timings"]["median_ms"] == 1.0
        assert value["timings"]["clock"] == "synchronized_wall"
        assert value["timings"]["sample_count"] == 3
        assert value["quality"]["sqnr_db"] is None
        assert value["environment"]["gpu_architecture"] == "SM120"


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


def test_triton_version_accepts_windows_distribution(monkeypatch) -> None:
    def package_version(name: str) -> str:
        if name == "triton-windows":
            return "3.7.1.post27"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(environment_module.importlib.metadata, "version", package_version)

    assert environment_module._triton_version() == "3.7.1.post27"


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
