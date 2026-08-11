import json
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import pytest
from lib.environment import EnvironmentInfo
from lib.providers import BenchmarkProvider
from lib.reporting import OutputFormat, OutputTarget, write_records
from lib.triton_inspection import (
    DeviceResourceLimits,
    NvdisasmUnavailableError,
    TritonArtifactUnavailableError,
    TritonCompatibilityError,
    TritonInspectionError,
    compiled_artifact,
    discover_compiled_specializations,
    find_nvdisasm,
    inspect_provider,
    resource_residency_ceiling,
    summarize_ptx,
    summarize_sass,
)


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


def _jit_kernel(*, backend: str = "cuda", num_ctas: int = 1) -> SimpleNamespace:
    assembly: dict[str, object] = {
        "ptx": """
            mov.u32 %r1, %tid.x;
            mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%r2}, {%r3}, {%r4}, {%r2};
            ret;
        """,
    }
    if backend == "cuda":
        assembly["cubin"] = b"fake-cubin"
    metadata = SimpleNamespace(
        backend_name=backend,
        arch="sm120" if backend == "cuda" else "gfx1201",
        shared=16_384,
        num_warps=4,
        num_stages=3,
        num_ctas=num_ctas,
    )
    compiled = SimpleNamespace(
        metadata=metadata,
        n_regs=64,
        n_spills=2,
        asm=assembly,
    )
    return SimpleNamespace(device_caches={0: ({"specialization": compiled}, {}, None)})


def _limits() -> DeviceResourceLimits:
    return DeviceResourceLimits(
        registers_per_compute_unit=65_536,
        shared_memory_bytes_per_compute_unit=98_304,
        max_threads_per_compute_unit=1_536,
        warp_size=32,
    )


def _provider(kernel: object) -> BenchmarkProvider[None, None]:
    return BenchmarkProvider(
        name="test-provider",
        prepare=lambda: None,
        run=lambda _prepared: None,
        configuration={"block_m": 64},
        triton_jit_functions={"test-kernel": kernel},
    )


def test_instruction_summaries_detect_mma_and_families() -> None:
    ptx = summarize_ptx(
        """
        mov.u32 %r1, %tid.x;
        @%p1 bra label;
        mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%r2}, {%r3}, {%r4}, {%r2};
        ret;
        """
    )
    sass = summarize_sass(
        """
        /*0000*/ IMMA.16832.S8.S8.SAT R4, R12.ROW, R16.COL, RZ ;
        /*0010*/ @P0 IADD3 R4, R4, 0x1, RZ ;
        /*0020*/ HMMA.16816.F32 R8, R12, R16, R8 ;
        """
    )

    assert ptx.instruction_count == 4
    assert ptx.families == {"bra": 1, "mma": 1, "mov": 1, "ret": 1}
    assert list(ptx.mma_opcodes.values()) == [1]
    assert sass.instruction_count == 3
    assert sass.families == {"HMMA": 1, "IADD3": 1, "IMMA": 1}
    assert sass.mma_opcodes == {"HMMA.16816.F32": 1, "IMMA.16832.S8.S8.SAT": 1}


def test_resource_ceiling_reports_each_constraint_and_limiter() -> None:
    ceiling = resource_residency_ceiling(
        registers_per_thread=64,
        shared_memory_bytes_per_workgroup=16_384,
        warps_per_workgroup=4,
        limits=_limits(),
    )

    assert ceiling.workgroup_limits == {"registers": 8, "shared_memory": 6, "threads": 12}
    assert ceiling.resident_workgroups_per_compute_unit == 6
    assert ceiling.resident_warps_per_compute_unit == 24
    assert ceiling.limiting_resources == ("shared_memory",)


def test_provider_report_covers_resources_ptx_sass_and_json(tmp_path: Path) -> None:
    sass = """
        /*0000*/ IMMA.16832.S8.S8.SAT R4, R12.ROW, R16.COL, RZ ;
        /*0010*/ I2FP.F32.S32 R8, R4 ;
    """
    report = inspect_provider(
        _provider(_jit_kernel()),
        _environment(),
        limits=_limits(),
        sass_disassembler=lambda cubin: sass if cubin == b"fake-cubin" else "",
    )
    specialization = report.specializations[0]

    assert specialization.kernel == "test-kernel"
    assert specialization.device_index == 0
    assert specialization.compiler_backend == "cuda"
    assert specialization.target_architecture == "sm120"
    assert specialization.ctas_per_cluster == 1
    assert specialization.registers_per_thread == 64
    assert specialization.spills == 2
    assert specialization.sass is not None
    assert specialization.sass.families == {"I2FP": 1, "IMMA": 1}
    assert specialization.ptx is not None
    assert specialization.ptx.mma_opcodes

    path = tmp_path / "compiler.json"
    write_records([report], OutputTarget(path, OutputFormat.JSON))
    value = json.loads(path.read_text())[0]
    assert value["schema_version"] == 1
    assert value["record_type"] == "triton_compiler"
    assert value["provider"] == "test-provider"
    assert value["configuration"] == {"block_m": 64}
    assert value["environment"]["git_revision"] == "a" * 40
    assert (
        value["specializations"][0]["residency_ceiling"]["resident_workgroups_per_compute_unit"]
        == 6
    )


def test_clustered_cuda_report_keeps_resources_and_residency_workgroup_scoped() -> None:
    report = inspect_provider(
        _provider(_jit_kernel(num_ctas=2)),
        _environment(),
        include_sass=False,
        limits=_limits(),
    )

    specialization = report.specializations[0]
    assert specialization.ctas_per_cluster == 2
    assert specialization.shared_memory_bytes_per_workgroup == 16_384
    assert specialization.warps_per_workgroup == 4
    assert specialization.residency_ceiling.resident_workgroups_per_compute_unit == 6
    assert specialization.residency_ceiling.resident_warps_per_compute_unit == 24


def test_compiler_json_normalizes_provider_configuration(tmp_path: Path) -> None:
    class Choice(StrEnum):
        TEST = "test"

    provider = _provider(_jit_kernel())
    provider.configuration = {
        "path": tmp_path,
        "choice": Choice.TEST,
        "nonfinite": float("inf"),
    }
    report = inspect_provider(
        provider,
        _environment(),
        include_sass=False,
        limits=_limits(),
    )
    path = tmp_path / "compiler.json"

    write_records([report], OutputTarget(path, OutputFormat.JSON))

    assert json.loads(path.read_text())[0]["configuration"] == {
        "path": str(tmp_path),
        "choice": "test",
        "nonfinite": None,
    }


def test_specialization_discovery_and_artifact_access() -> None:
    kernel = _jit_kernel()

    specializations = discover_compiled_specializations(kernel)

    assert len(specializations) == 1
    assert specializations[0].device_index == 0
    assert specializations[0].specialization_key == "specialization"
    assert "mma.sync" in compiled_artifact(kernel, "ptx")
    with pytest.raises(TritonArtifactUnavailableError, match="available"):
        compiled_artifact(kernel, "amdgcn")


def test_artifact_access_requires_selection_for_multiple_specializations() -> None:
    kernel = _jit_kernel()
    compiled_cache = kernel.device_caches[0][0]
    compiled_cache["another"] = compiled_cache["specialization"]

    with pytest.raises(TritonInspectionError, match="select specialization_index"):
        compiled_artifact(kernel, "ptx")
    assert "mma.sync" in compiled_artifact(kernel, "ptx", specialization_index=1)


def test_provider_report_rejects_ambiguous_process_wide_jit_cache() -> None:
    kernel = _jit_kernel()
    compiled_cache = kernel.device_caches[0][0]
    compiled_cache["another"] = compiled_cache["specialization"]

    with pytest.raises(TritonInspectionError, match="isolated one-provider process"):
        inspect_provider(
            _provider(kernel),
            _environment(),
            include_sass=False,
            limits=_limits(),
        )

    report = inspect_provider(
        _provider(kernel),
        _environment(),
        include_sass=False,
        limits=_limits(),
        require_isolated_jit_cache=False,
    )
    assert len(report.specializations) == 2


def test_non_cuda_report_skips_sass_automatically() -> None:
    report = inspect_provider(
        _provider(_jit_kernel(backend="hip")),
        _environment(),
        limits=_limits(),
    )

    assert report.specializations[0].compiler_backend == "hip"
    assert report.specializations[0].ctas_per_cluster is None
    assert report.specializations[0].sass is None


def test_non_cuda_report_rejects_explicit_sass() -> None:
    with pytest.raises(TritonArtifactUnavailableError, match="CUDA backend"):
        inspect_provider(
            _provider(_jit_kernel(backend="hip")),
            _environment(),
            include_sass=True,
            limits=_limits(),
        )


def test_missing_nvdisasm_has_actionable_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr("lib.triton_inspection.shutil.which", lambda _name: None)

    with pytest.raises(NvdisasmUnavailableError, match="disable SASS inspection"):
        find_nvdisasm()


def test_uncompiled_and_incompatible_kernels_fail_at_boundary() -> None:
    with pytest.raises(TritonInspectionError, match="no compiled Triton"):
        inspect_provider(
            _provider(SimpleNamespace(device_caches={})),
            _environment(),
            limits=_limits(),
        )
    with pytest.raises(TritonCompatibilityError, match="device_caches"):
        inspect_provider(_provider(object()), _environment(), limits=_limits())
