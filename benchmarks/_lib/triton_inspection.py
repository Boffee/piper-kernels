"""Triton compiler-resource and NVIDIA assembly inspection.

This module is the sole compatibility boundary around Triton's compiled-kernel
caches and metadata.  Callers should register JIT functions on a
``BenchmarkProvider`` and use :func:`inspect_provider` instead of traversing
Triton runtime internals themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch

from .environment import EnvironmentInfo
from .reporting import add_output_arguments

COMPILER_REPORT_SCHEMA_VERSION = 1
_AttributeT = TypeVar("_AttributeT")


class TritonInspectionError(RuntimeError):
    """Base error for compiler inspection failures."""


class TritonCompatibilityError(TritonInspectionError):
    """The installed Triton no longer exposes the expected compiler state."""


class NvdisasmUnavailableError(TritonInspectionError):
    """NVIDIA SASS inspection was requested without an available disassembler."""


class TritonArtifactUnavailableError(TritonInspectionError):
    """A requested compiler artifact is unavailable for the active backend."""


class _InspectableProvider(Protocol):
    """Provider fields needed for compiler inspection."""

    name: str
    configuration: Mapping[str, Any]
    triton_jit_functions: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DeviceResourceLimits:
    """Hardware limits used for a resource-only residency estimate."""

    registers_per_compute_unit: int | None
    shared_memory_bytes_per_compute_unit: int | None
    max_threads_per_compute_unit: int | None
    warp_size: int


@dataclass(frozen=True, slots=True)
class ResidencyCeiling:
    """Workgroup residency ceiling from resources and exposed device limits.

    This is not an achieved-occupancy prediction.  It intentionally omits
    allocation granularities and architectural limits that PyTorch does not
    expose consistently across CUDA and ROCm.  For a clustered CUDA launch,
    this remains a per-CTA ceiling and is not a cluster-occupancy estimate.
    """

    resident_workgroups_per_compute_unit: int | None
    resident_warps_per_compute_unit: int | None
    limiting_resources: tuple[str, ...]
    workgroup_limits: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        """Return stable machine-readable fields."""
        return {
            "resident_workgroups_per_compute_unit": self.resident_workgroups_per_compute_unit,
            "resident_warps_per_compute_unit": self.resident_warps_per_compute_unit,
            "limiting_resources": list(self.limiting_resources),
            "workgroup_limits": dict(self.workgroup_limits),
        }


@dataclass(frozen=True, slots=True)
class InstructionSummary:
    """Static instruction counts from one compiler artifact."""

    instruction_count: int
    families: Mapping[str, int]
    mma_opcodes: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        """Return stable machine-readable fields."""
        return {
            "instruction_count": self.instruction_count,
            "families": dict(self.families),
            "mma_opcodes": dict(self.mma_opcodes),
        }


@dataclass(frozen=True, slots=True)
class TritonSpecializationReport:
    """Resources and static assembly properties for one specialization."""

    kernel: str
    specialization_index: int
    specialization_fingerprint: str
    specialization_key: str
    device_index: int
    compiler_backend: str
    target_architecture: str
    registers_per_thread: int
    spills: int
    shared_memory_bytes_per_workgroup: int
    warps_per_workgroup: int
    stages: int | None
    ctas_per_cluster: int | None
    residency_ceiling: ResidencyCeiling
    ptx: InstructionSummary | None
    sass: InstructionSummary | None

    def as_dict(self) -> dict[str, object]:
        """Return stable machine-readable fields."""
        return {
            "kernel": self.kernel,
            "specialization_index": self.specialization_index,
            "specialization_fingerprint": self.specialization_fingerprint,
            "specialization_key": self.specialization_key,
            "device_index": self.device_index,
            "compiler_backend": self.compiler_backend,
            "target_architecture": self.target_architecture,
            "registers_per_thread": self.registers_per_thread,
            "spills": self.spills,
            "shared_memory_bytes_per_workgroup": self.shared_memory_bytes_per_workgroup,
            "warps_per_workgroup": self.warps_per_workgroup,
            "stages": self.stages,
            "ctas_per_cluster": self.ctas_per_cluster,
            "residency_ceiling": self.residency_ceiling.as_dict(),
            "ptx": None if self.ptx is None else self.ptx.as_dict(),
            "sass": None if self.sass is None else self.sass.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TritonCompilerRecord:
    """Versioned compiler report for every kernel registered by a provider."""

    provider: str
    configuration: Mapping[str, Any]
    environment: EnvironmentInfo
    specializations: tuple[TritonSpecializationReport, ...]
    schema_version: int = COMPILER_REPORT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        """Return the stable compiler-report schema."""
        return {
            "schema_version": self.schema_version,
            "record_type": "triton_compiler",
            "provider": self.provider,
            "configuration": dict(self.configuration),
            "environment": self.environment.as_dict(),
            "specializations": [report.as_dict() for report in self.specializations],
        }


@dataclass(frozen=True, slots=True)
class TritonCompiledSpecialization:
    """One discovered Triton specialization with its compiler cache identity."""

    device_index: int
    specialization_key: str
    _compiled_kernel: object = field(repr=False, compare=False)


_SASS_INSTRUCTION_PATTERN = re.compile(
    r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)"
)
_PTX_INSTRUCTION_PATTERN = re.compile(
    r"^\s*(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]*)(?=\s|;)",
    re.MULTILINE,
)


def _is_mma_opcode(opcode: str) -> bool:
    family = opcode.split(".", maxsplit=1)[0]
    return "mma" in family.lower() or opcode.lower().startswith("tcgen05.mma")


def _summarize_opcodes(opcodes: Sequence[str]) -> InstructionSummary:
    full = Counter(opcodes)
    families: Counter[str] = Counter()
    for opcode, count in full.items():
        families[opcode.split(".", maxsplit=1)[0]] += count
    return InstructionSummary(
        instruction_count=sum(full.values()),
        families=dict(sorted(families.items())),
        mma_opcodes={
            opcode: count
            for opcode, count in sorted(full.items())
            if _is_mma_opcode(opcode)
        },
    )


def summarize_ptx(ptx: str) -> InstructionSummary:
    """Count static PTX instructions, families, and MMA opcode forms."""
    return _summarize_opcodes(_PTX_INSTRUCTION_PATTERN.findall(ptx))


def summarize_sass(sass: str) -> InstructionSummary:
    """Count static SASS instructions, families, and MMA opcode forms."""
    return _summarize_opcodes(_SASS_INSTRUCTION_PATTERN.findall(sass))


def find_nvdisasm(explicit_path: Path | None = None) -> Path:
    """Resolve ``nvdisasm`` or fail with an actionable diagnostic."""
    if explicit_path is not None:
        candidate = explicit_path.expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        raise NvdisasmUnavailableError(
            f"an executable nvdisasm was not found at {candidate}; install the NVIDIA "
            "CUDA Toolkit or pass the path to an existing executable"
        )
    discovered = shutil.which("nvdisasm")
    if discovered is not None:
        return Path(discovered).resolve()
    raise NvdisasmUnavailableError(
        "SASS inspection requires nvdisasm, but it was not found on PATH; "
        "install the NVIDIA CUDA Toolkit, add its bin directory to PATH, "
        "or disable SASS inspection"
    )


def disassemble_cubin(cubin: bytes, nvdisasm: Path | None = None) -> str:
    """Disassemble one NVIDIA cubin with ``nvdisasm --print-code``."""
    executable = find_nvdisasm(nvdisasm)
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin)
        cubin_file.flush()
        try:
            result = subprocess.run(
                [str(executable), "--print-code", cubin_file.name],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            if isinstance(error, subprocess.CalledProcessError):
                diagnostic = error.stderr.strip() or error.stdout.strip() or "no diagnostic"
                exit_description = f"exit code {error.returncode}"
            else:
                diagnostic = str(error)
                exit_description = type(error).__name__
            raise TritonInspectionError(
                f"nvdisasm failed with {exit_description}: {diagnostic}"
            ) from error
    return result.stdout


def current_device_resource_limits(device_index: int | None = None) -> DeviceResourceLimits:
    """Read the portable subset of accelerator limits exposed by PyTorch."""
    if not torch.cuda.is_available():
        raise TritonInspectionError(
            "compiler resource inspection requires a Triton-supported accelerator"
        )
    resolved_index = torch.cuda.current_device() if device_index is None else device_index
    properties = torch.cuda.get_device_properties(resolved_index)

    def optional_integer(name: str) -> int | None:
        value = getattr(properties, name, None)
        return int(value) if value is not None else None

    warp_size = optional_integer("warp_size")
    if warp_size is None:
        raise TritonCompatibilityError(
            "PyTorch device properties do not expose warp_size for this accelerator"
        )
    return DeviceResourceLimits(
        registers_per_compute_unit=optional_integer("regs_per_multiprocessor"),
        shared_memory_bytes_per_compute_unit=optional_integer(
            "shared_memory_per_multiprocessor"
        ),
        max_threads_per_compute_unit=optional_integer("max_threads_per_multi_processor"),
        warp_size=warp_size,
    )


def resource_residency_ceiling(
    *,
    registers_per_thread: int,
    shared_memory_bytes_per_workgroup: int,
    warps_per_workgroup: int,
    limits: DeviceResourceLimits,
) -> ResidencyCeiling:
    """Estimate a workgroup ceiling from registers, shared memory, and threads."""
    if (
        registers_per_thread < 0
        or shared_memory_bytes_per_workgroup < 0
        or warps_per_workgroup <= 0
    ):
        raise ValueError("compiler resource values must be non-negative and warps positive")
    threads_per_workgroup = warps_per_workgroup * limits.warp_size
    workgroup_limits: dict[str, int] = {}
    if limits.registers_per_compute_unit is not None and registers_per_thread:
        workgroup_limits["registers"] = limits.registers_per_compute_unit // (
            registers_per_thread * threads_per_workgroup
        )
    if (
        limits.shared_memory_bytes_per_compute_unit is not None
        and shared_memory_bytes_per_workgroup
    ):
        workgroup_limits["shared_memory"] = (
            limits.shared_memory_bytes_per_compute_unit // shared_memory_bytes_per_workgroup
        )
    if limits.max_threads_per_compute_unit is not None:
        workgroup_limits["threads"] = (
            limits.max_threads_per_compute_unit // threads_per_workgroup
        )

    resident_workgroups = min(workgroup_limits.values()) if workgroup_limits else None
    limiting_resources = (
        tuple(
            sorted(
                name for name, value in workgroup_limits.items() if value == resident_workgroups
            )
        )
        if resident_workgroups is not None
        else ()
    )
    return ResidencyCeiling(
        resident_workgroups_per_compute_unit=resident_workgroups,
        resident_warps_per_compute_unit=(
            None
            if resident_workgroups is None
            else resident_workgroups * warps_per_workgroup
        ),
        limiting_resources=limiting_resources,
        workgroup_limits=workgroup_limits,
    )


def discover_compiled_specializations(
    jit_kernel: object,
) -> tuple[TritonCompiledSpecialization, ...]:
    """Return cached compiled kernels from Triton's current JIT cache layout."""
    device_caches = getattr(jit_kernel, "device_caches", None)
    if not isinstance(device_caches, Mapping):
        raise TritonCompatibilityError(
            "the registered object does not expose Triton's device_caches mapping; "
            "the Triton inspection adapter may need updating for this version"
        )

    specializations: list[TritonCompiledSpecialization] = []
    for raw_device_index, device_cache in sorted(device_caches.items()):
        if not isinstance(raw_device_index, int):
            raise TritonCompatibilityError(
                "Triton's device cache used a non-integer device key; "
                "the inspection adapter may need updating"
            )
        if not isinstance(device_cache, tuple) or not device_cache:
            raise TritonCompatibilityError(
                "Triton's device cache layout is unsupported; expected a tuple whose "
                "first item is the compiled-specialization mapping"
            )
        kernel_cache = device_cache[0]
        if not isinstance(kernel_cache, Mapping):
            raise TritonCompatibilityError(
                "Triton's compiled-specialization cache is not a mapping; "
                "the inspection adapter may need updating"
            )
        specializations.extend(
            TritonCompiledSpecialization(raw_device_index, str(key), compiled_kernel)
            for key, compiled_kernel in sorted(kernel_cache.items(), key=lambda item: str(item[0]))
        )
    if not specializations:
        raise TritonInspectionError(
            "no compiled Triton specializations were found; run the provider at least "
            "once before requesting a compiler report"
        )
    return tuple(specializations)


def compiled_artifact(
    jit_kernel: object,
    artifact: str,
    *,
    specialization_index: int | None = None,
) -> str | bytes:
    """Return one TTIR/TTGIR/LLVM/PTX/cubin artifact.

    A kernel with multiple cached specializations requires an explicit index so
    callers cannot silently inspect a different specialization after a new
    launch configuration is added.
    """
    specializations = discover_compiled_specializations(jit_kernel)
    if specialization_index is None:
        if len(specializations) != 1:
            raise TritonInspectionError(
                f"expected one compiled specialization, found {len(specializations)}; "
                "select specialization_index explicitly"
            )
        specialization_index = 0
    try:
        specialization = specializations[specialization_index]
    except IndexError as error:
        raise TritonInspectionError(
            f"specialization index {specialization_index} is out of range for "
            f"{len(specializations)} compiled specializations"
        ) from error
    assembly = _assembly_mapping(specialization._compiled_kernel)
    value = assembly.get(artifact)
    if not isinstance(value, (str, bytes)):
        available = ", ".join(sorted(assembly)) or "none"
        raise TritonArtifactUnavailableError(
            f"compiled specialization has no {artifact!r} artifact; available: {available}"
        )
    return value


def _required_attribute(
    value: object,
    name: str,
    expected_type: type[_AttributeT],
) -> _AttributeT:
    try:
        result = getattr(value, name)
    except AttributeError as error:
        raise TritonCompatibilityError(
            f"the compiled Triton kernel does not expose {name!r}; "
            "the inspection adapter may need updating for this version"
        ) from error
    if not isinstance(result, expected_type):
        raise TritonCompatibilityError(
            f"the compiled Triton kernel attribute {name!r} has type "
            f"{type(result).__name__}, expected {expected_type.__name__}; "
            "the inspection adapter may need updating for this version"
        )
    return result


def _optional_integer(value: object, name: str) -> int | None:
    result = getattr(value, name, None)
    return None if result is None else int(result)


def _assembly_mapping(compiled_kernel: object) -> Mapping[str, object]:
    return _required_attribute(compiled_kernel, "asm", Mapping)


def _inspect_specialization(
    specialization: TritonCompiledSpecialization,
    *,
    kernel_name: str,
    specialization_index: int,
    include_sass: bool | None,
    limits: DeviceResourceLimits | None,
    sass_disassembler: Callable[[bytes], str] | None,
    nvdisasm: Path | None,
) -> TritonSpecializationReport:
    compiled = specialization._compiled_kernel
    metadata = _required_attribute(compiled, "metadata", object)
    backend = _required_attribute(metadata, "backend_name", str)
    architecture = _required_attribute(metadata, "arch", str)
    registers = _required_attribute(compiled, "n_regs", int)
    spills = _required_attribute(compiled, "n_spills", int)
    shared_memory = _required_attribute(metadata, "shared", int)
    warps = _required_attribute(metadata, "num_warps", int)
    resolved_limits = limits or current_device_resource_limits(specialization.device_index)
    residency = resource_residency_ceiling(
        registers_per_thread=registers,
        shared_memory_bytes_per_workgroup=shared_memory,
        warps_per_workgroup=warps,
        limits=resolved_limits,
    )

    assembly = _assembly_mapping(compiled)
    ptx_summary = None
    ptx = assembly.get("ptx")
    if ptx is not None:
        if not isinstance(ptx, str):
            raise TritonCompatibilityError("Triton's PTX artifact is not text")
        ptx_summary = summarize_ptx(ptx)

    should_include_sass = backend == "cuda" if include_sass is None else include_sass
    sass_summary = None
    if should_include_sass:
        if backend != "cuda":
            raise TritonArtifactUnavailableError(
                f"SASS inspection requires Triton's CUDA backend, not {backend!r}; "
                "disable SASS inspection for this target"
            )
        cubin = assembly.get("cubin")
        if not isinstance(cubin, bytes):
            raise TritonArtifactUnavailableError(
                "the compiled CUDA specialization does not contain a cubin artifact"
            )
        if sass_disassembler is not None:
            sass = sass_disassembler(cubin)
        else:
            sass = disassemble_cubin(cubin, nvdisasm)
        sass_summary = summarize_sass(sass)

    key_text = specialization.specialization_key
    specialization_identity = f"{backend}\0{architecture}\0{key_text}"
    specialization_fingerprint = hashlib.sha256(specialization_identity.encode()).hexdigest()[:16]
    return TritonSpecializationReport(
        kernel=kernel_name,
        specialization_index=specialization_index,
        specialization_fingerprint=specialization_fingerprint,
        specialization_key=key_text,
        device_index=specialization.device_index,
        compiler_backend=backend,
        target_architecture=architecture,
        registers_per_thread=registers,
        spills=spills,
        shared_memory_bytes_per_workgroup=shared_memory,
        warps_per_workgroup=warps,
        stages=_optional_integer(metadata, "num_stages"),
        ctas_per_cluster=(
            _optional_integer(metadata, "num_ctas") if backend == "cuda" else None
        ),
        residency_ceiling=residency,
        ptx=ptx_summary,
        sass=sass_summary,
    )


def inspect_provider(
    provider: _InspectableProvider,
    environment: EnvironmentInfo,
    *,
    include_sass: bool | None = None,
    limits: DeviceResourceLimits | None = None,
    sass_disassembler: Callable[[bytes], str] | None = None,
    nvdisasm: Path | None = None,
    require_isolated_jit_cache: bool = True,
) -> TritonCompilerRecord:
    """Inspect every compiled specialization registered by a provider.

    ``include_sass=None`` enables SASS automatically for CUDA specializations
    and leaves it disabled for other Triton backends. By default, each
    registered JIT function must have exactly one process-wide specialization
    so a report cannot silently attribute another provider's compilation.
    """
    if not provider.triton_jit_functions:
        raise TritonInspectionError(
            f"provider {provider.name!r} has no registered Triton JIT functions"
        )
    if sass_disassembler is not None and nvdisasm is not None:
        raise ValueError("pass either sass_disassembler or nvdisasm, not both")
    reports: list[TritonSpecializationReport] = []
    for kernel_name, jit_kernel in provider.triton_jit_functions.items():
        specializations = discover_compiled_specializations(jit_kernel)
        if require_isolated_jit_cache and len(specializations) != 1:
            raise TritonInspectionError(
                f"provider {provider.name!r} registered kernel {kernel_name!r}, but its "
                f"process-wide Triton JIT cache contains {len(specializations)} "
                "specializations; run compiler reporting in an isolated one-provider "
                "process or explicitly disable the isolation requirement"
            )
        for index, specialization in enumerate(specializations):
            reports.append(
                _inspect_specialization(
                    specialization,
                    kernel_name=kernel_name,
                    specialization_index=index,
                    include_sass=include_sass,
                    limits=limits,
                    sass_disassembler=sass_disassembler,
                    nvdisasm=nvdisasm,
                )
            )
    return TritonCompilerRecord(
        provider=provider.name,
        configuration=dict(provider.configuration),
        environment=environment,
        specializations=tuple(reports),
    )


def format_compiler_report(report: TritonCompilerRecord) -> str:
    """Format a concise terminal summary without implying achieved occupancy."""
    lines: list[str] = []
    for specialization in report.specializations:
        residency = specialization.residency_ceiling
        workgroups = residency.resident_workgroups_per_compute_unit
        warps = residency.resident_warps_per_compute_unit
        limiting = ",".join(residency.limiting_resources) or "unknown"
        lines.append(
            f"kernel={specialization.kernel} "
            f"specialization_fingerprint={specialization.specialization_fingerprint} "
            f"device={specialization.device_index} "
            f"compiler_backend={specialization.compiler_backend} "
            f"target_architecture={specialization.target_architecture} "
            f"registers/thread={specialization.registers_per_thread} "
            f"spills={specialization.spills} "
            f"shared_bytes/workgroup={specialization.shared_memory_bytes_per_workgroup} "
            f"warps/workgroup={specialization.warps_per_workgroup} "
            f"ctas/cluster={specialization.ctas_per_cluster} "
            f"resource_workgroups/cu={workgroups} resource_warps/cu={warps} "
            f"limiting={limiting}"
        )
        if specialization.ptx is not None and specialization.ptx.mma_opcodes:
            lines.append(f"  PTX MMA: {dict(specialization.ptx.mma_opcodes)}")
        if specialization.sass is not None:
            lines.append(
                f"  SASS instructions={specialization.sass.instruction_count} "
                f"MMA={dict(specialization.sass.mma_opcodes)}"
            )
    return "\n".join(lines)


def add_compiler_inspection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional terminal and JSON compiler-report arguments."""
    parser.add_argument(
        "--compiler-report",
        action="store_true",
        help="print Triton resources plus static PTX/SASS instruction summaries",
    )
    parser.add_argument(
        "--sass",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "enable or disable nvdisasm SASS inspection; by default it is enabled "
            "for CUDA and disabled for other Triton backends"
        ),
    )
    parser.add_argument(
        "--nvdisasm",
        type=Path,
        metavar="PATH",
        help="path to nvdisasm when it is not available on PATH",
    )
    add_output_arguments(
        parser,
        option_prefix="compiler",
        record_name="compiler report",
    )
