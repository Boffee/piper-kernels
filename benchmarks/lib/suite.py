"""Common execution, validation, and reporting for fixed benchmark cases."""

from __future__ import annotations

import gc
from dataclasses import dataclass, replace
from typing import Literal

import torch

from .cases import AttentionCase, BenchmarkCase, PipelineCase
from .environment import EnvironmentInfo
from .providers import Implementation
from .quality import QualityCheck
from .reporting import BenchmarkRecord
from .timing import Timing, synchronized_wall_benchmark

# Backends are optional and loaded only for the requested operation family.
# ruff: noqa: PLC0415


@dataclass(frozen=True)
class Measurement:
    """One complete-call timing protocol, independent of the accelerator."""

    warmup_ms: int = 100
    measurement_ms: int = 500

    def as_dict(self) -> dict[str, object]:
        return {
            "scope": "operator_end_to_end",
            "clock": "synchronized_wall",
            "cache_policy": "warm",
            "warmup_ms": self.warmup_ms,
            "measurement_ms": self.measurement_ms,
            "input_recipe": "cpu_normal_fp32_cast_v1",
        }


@dataclass(frozen=True, kw_only=True)
class SuiteRecord(BenchmarkRecord[Timing | None, QualityCheck]):
    """A benchmark result with the suite's protocol and explicit outcome."""

    measurement: Measurement
    status: Literal["ok", "unsupported", "oom", "failed"] = "failed"
    peak_extra_bytes: int | None = None
    reason: str | None = None
    stage: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            **super().as_dict(),
            "record_type": "operator_suite",
            "timings": {
                "operator_end_to_end": None if self.timings is None else self.timings.as_dict()
            },
            "measurement": self.measurement.as_dict(),
            "status": self.status,
            "peak_extra_bytes": self.peak_extra_bytes,
            "reason": self.reason,
            "stage": self.stage,
        }


def implementations(case: BenchmarkCase, device: torch.device) -> list[Implementation]:
    """Import only the selected family; listing workloads needs no accelerator backend."""
    if isinstance(case, AttentionCase):
        from .suite_attention import implementations as attention_implementations

        return attention_implementations(case, device)
    if isinstance(case, PipelineCase):
        from .suite_pipeline import implementations as pipeline_implementations

        return pipeline_implementations(case, device)
    from .suite_weights import implementations as weight_implementations

    return weight_implementations(case, device)


def _check_finite(output: torch.Tensor) -> None:
    maximum_elements = 1 << 22
    if output.numel() <= maximum_elements:
        if not bool(torch.isfinite(output).all()):
            raise ValueError("operator returned nonfinite output")
        return
    dimension = max(range(output.ndim), key=lambda index: output.shape[index])
    step = max(1, maximum_elements * output.shape[dimension] // output.numel())
    for chunk in output.split(step, dim=dimension):
        _check_finite(chunk)


@torch.inference_mode()
def run_implementation(
    case: BenchmarkCase,
    implementation: Implementation,
    *,
    device: torch.device,
    environment: EnvironmentInfo,
    measurement: Measurement,
) -> SuiteRecord:
    """Compile and validate first; measure the same complete operation on every device."""
    record = SuiteRecord(
        benchmark=case.family,
        provider=implementation.name,
        case_id=case.id,
        shape=case.as_dict(),
        configuration={"execution_device": str(device), **implementation.configuration},
        environment=environment,
        measurement=measurement,
        timings=None,
        stage="setup",
    )
    if implementation.unsupported_reason is not None:
        return replace(
            record,
            status="unsupported",
            stage="capability",
            reason=implementation.unsupported_reason,
        )
    operation = output = None
    try:
        operation = implementation.build()
        record = replace(record, configuration={**record.configuration, **operation.configuration})
        output = operation.run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = replace(record, stage="validation")
        _check_finite(output)
        check = operation.check(output)
        record = replace(record, quality=check)
        check.metrics.validate(check.relative_l2_limit, check.reference)
        del output
        output = None
        record = replace(record, stage="measurement")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            resident_bytes = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            resident_bytes = 0
        timing = synchronized_wall_benchmark(
            operation.run,
            measurement.warmup_ms,
            measurement.measurement_ms,
            synchronize=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None,
        )
        extra_bytes = (
            max(0, torch.cuda.max_memory_allocated(device) - resident_bytes)
            if device.type == "cuda"
            else None
        )
        return replace(
            record,
            status="ok",
            stage=None,
            timings=timing,
            peak_extra_bytes=extra_bytes,
        )
    except Exception as error:
        return replace(
            record,
            status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            reason=f"{type(error).__name__}: {error}",
        )
    finally:
        del output, operation
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
