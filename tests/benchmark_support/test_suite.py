"""Contract tests for identical workloads, measurement scope, and explicit outcomes."""

import argparse

import benchmark
import pytest
import torch
from lib import suite
from lib.case_cli import apply_case
from lib.cases import CATALOG_VERSION, diagnostic_cases, named_case, select_cases, standard_cases
from lib.providers import Implementation, Operation
from lib.quality import QualityCheck, measure_quality
from lib.suite import Measurement, run_implementation
from lib.timing import ClockDomain, Timing


def _quality(output):
    return QualityCheck(measure_quality(output, torch.ones_like(output)), "ones", 4, 4, 0.01)


def test_fixed_catalog_has_production_bounds_and_no_hardware_dependency(monkeypatch):
    baseline = standard_cases()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert baseline == standard_cases()
    assert len({case.id for case in baseline}) == len(baseline)
    assert {case.family for case in baseline} == {
        "attention",
        "sparse_attention",
        "linear",
        "ffn",
        "conv3d",
        "pipeline",
    }
    assert named_case("attention-video-high").sequence == 110592
    assert named_case("attention-image-low").sequence == 2816
    assert named_case("attention-image-high").kv_heads == 12
    assert named_case("conv3d-high-stage1").height == 192
    assert not {case.id for case in diagnostic_cases()} & {case.id for case in baseline}


def test_selection_preserves_cases_and_rejects_typos():
    selected = select_cases(("*video-high*",), families=("attention",))
    assert selected == (named_case("attention-video-high"),)
    assert select_cases(("linear-small",)) == (named_case("linear-small"),)
    with pytest.raises(ValueError, match="no benchmark case"):
        select_cases(("attention-typo",))
    with pytest.raises(ValueError, match="no benchmark cases selected"):
        select_cases(("linear-small",), families=("attention",))


def test_complete_call_timing_excludes_setup_and_reference(monkeypatch, environment):
    events = []

    def build():
        events.append("build")

        def run():
            events.extend(("prepare", "execute"))
            return torch.ones(4)

        def check(output):
            events.append("quality")
            return _quality(output)

        return Operation(run, check)

    def timer(run, warmup, measurement, *, synchronize):
        assert events == ["build", "prepare", "execute", "quality"]
        assert (warmup, measurement) == (2, 3)
        run()
        assert events[-2:] == ["prepare", "execute"]
        return Timing.from_samples([1, 1, 1], ClockDomain.SYNCHRONIZED_WALL)

    monkeypatch.setattr(suite, "synchronized_wall_benchmark", timer)
    record = run_implementation(
        named_case("linear-small"),
        Implementation("test", build),
        device=torch.device("cpu"),
        environment=environment,
        measurement=Measurement(2, 3),
    )
    assert record.status == "ok"
    assert record.as_dict()["case_id"] == "linear-small"
    assert record.as_dict()["catalog_version"] == CATALOG_VERSION
    assert record.as_dict()["measurement"]["scope"] == "operator_end_to_end"
    assert record.as_dict()["quality"]["metrics"]["actual_nonfinite_count"] == 0
    assert record.as_dict()["timings"]["operator_end_to_end"]["sample_count"] == 3
    assert record.configuration["execution_device"] == "cpu"


@pytest.mark.parametrize(
    ("exception", "status"),
    [
        (torch.OutOfMemoryError("capacity"), "oom"),
        (RuntimeError("kernel fault"), "failed"),
    ],
)
def test_failures_preserve_original_workload(environment, exception, status):
    case = named_case("attention-video-high")

    def build():
        raise exception

    record = run_implementation(
        case,
        Implementation("test", build),
        device=torch.device("cpu"),
        environment=environment,
        measurement=Measurement(0, 1),
    )
    assert record.status == status
    assert record.case_id == case.id
    assert record.shape == case.as_dict()
    assert record.timings is None
    assert record.stage == "setup"


def test_unsupported_does_not_allocate(environment):
    def build():
        pytest.fail("unsupported implementation must not build")

    record = run_implementation(
        named_case("linear-small"),
        Implementation("test", build, "requires another architecture"),
        device=torch.device("cpu"),
        environment=environment,
        measurement=Measurement(),
    )
    assert record.status == "unsupported"
    assert record.reason == "requires another architecture"


@pytest.mark.parametrize(
    ("index", "value", "reason"),
    [(0, float("nan"), "nonfinite"), (100, float("inf"), "nonfinite"), (0, 0.0, "relative L2")],
    ids=["nonfinite-sample", "nonfinite-outside-sample", "inaccurate-sample"],
)
def test_numerical_failure_prevents_measurement(environment, monkeypatch, index, value, reason):
    output = torch.ones(128)
    output[index] = value
    monkeypatch.setattr(
        suite, "synchronized_wall_benchmark", lambda *a, **k: pytest.fail("timed invalid output")
    )
    record = run_implementation(
        named_case("linear-small"),
        Implementation(
            "test", lambda: Operation(lambda: output, lambda value: _quality(value[:4]))
        ),
        device=torch.device("cpu"),
        environment=environment,
        measurement=Measurement(0, 1),
    )
    assert record.status == "failed"
    assert record.stage == "validation"
    assert record.timings is None
    assert reason in record.reason


def test_finiteness_validation_bounds_temporary_memory_for_strided_output(monkeypatch):
    original = torch.isfinite
    sizes = []

    def bounded(value):
        sizes.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch, "isfinite", bounded)
    output = torch.ones(4096, 2048).T
    assert not output.is_contiguous()
    suite._check_finite(output)
    assert max(sizes) <= 1 << 22
    assert sum(sizes) == output.numel()


def test_tuning_case_selection_cannot_silently_override_workload():
    args = argparse.Namespace(case="attention-image-low")
    result = apply_case(args, ["--case", args.case], attention=True)
    assert (result.heads, result.kv_heads, result.sequence) == (48, 12, 2816)
    with pytest.raises(SystemExit, match="overrides"):
        apply_case(args, ["--case", args.case, "--heads=16"], attention=True)
    with pytest.raises(SystemExit, match="overrides"):
        apply_case(argparse.Namespace(case="linear-small"), ["--case", "linear-small", "--no-bias"])


def test_case_list_does_not_import_native_implementations(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "implementations", lambda *_args: pytest.fail("native import"))
    assert benchmark.main(["--list", "--case", "*video-high*", "--family", "attention"]) == 0
    assert "110592" in capsys.readouterr().out
