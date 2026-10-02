"""Common execution, validation, and reporting for fixed benchmark cases."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .cases import CATALOG_VERSION, AttentionCase, BenchmarkCase, PipelineCase
from .environment import EnvironmentInfo
from .suite_types import Implementation, QualityCheck
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


@dataclass(frozen=True)
class SuiteRecord:
    """Every requested case/provider has an outcome, including unavailable implementations."""

    case: BenchmarkCase
    provider: str
    status: Literal["ok", "unsupported", "oom", "failed"]
    environment: EnvironmentInfo
    measurement: Measurement
    timing: Timing | None = None
    quality: QualityCheck | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    peak_extra_bytes: int | None = None
    reason: str | None = None
    stage: str | None = None

    def as_dict(self) -> dict[str, object]:
        check = self.quality
        return {
            "schema_version": 1,
            "record_type": "operator_suite",
            "catalog_version": CATALOG_VERSION,
            "case_id": self.case.id,
            "benchmark": self.case.family,
            "case": self.case.as_dict(),
            "provider": self.provider,
            "status": self.status,
            "configuration": self.configuration,
            "measurement": self.measurement.as_dict(),
            "timings": {
                "operator_end_to_end": None if self.timing is None else self.timing.as_dict()
            },
            "quality": None
            if check is None
            else {
                "reference": check.reference,
                "sample_count": check.sample_count,
                "total_count": check.total_count,
                "relative_l2_limit": check.relative_l2_limit,
                "metrics": check.metrics.as_dict(),
                "comparisons": {name: value.as_dict() for name, value in check.comparisons.items()},
                "full_output_finite": True,
            },
            "peak_extra_bytes": self.peak_extra_bytes,
            "reason": self.reason,
            "stage": self.stage,
            "environment": self.environment.as_dict(),
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


def _validate_quality(check: QualityCheck) -> None:
    metrics = check.metrics
    if metrics.actual_nonfinite_count or metrics.reference_nonfinite_count:
        raise ValueError("quality comparison contains nonfinite values")
    if not math.isfinite(metrics.relative_l2_error) or (
        metrics.relative_l2_error > check.relative_l2_limit
    ):
        raise ValueError(
            f"relative L2 error {metrics.relative_l2_error:.6g} exceeds "
            f"{check.relative_l2_limit:.6g} against {check.reference}",
        )


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
    if implementation.unsupported_reason is not None:
        return SuiteRecord(
            case,
            implementation.name,
            "unsupported",
            environment,
            measurement,
            reason=implementation.unsupported_reason,
            stage="capability",
        )
    operation = None
    output = None
    configuration: dict[str, Any] = {"execution_device": str(device)}
    check = None
    stage = "setup"
    try:
        operation = implementation.build()
        configuration.update(operation.configuration)
        output = operation.run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stage = "validation"
        _check_finite(output)
        check = operation.check(output)
        _validate_quality(check)
        del output
        output = None
        stage = "measurement"
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
        return SuiteRecord(
            case,
            implementation.name,
            "ok",
            environment,
            measurement,
            timing=timing,
            quality=check,
            configuration=configuration,
            peak_extra_bytes=extra_bytes,
        )
    except torch.OutOfMemoryError as error:
        return SuiteRecord(
            case,
            implementation.name,
            "oom",
            environment,
            measurement,
            configuration=configuration,
            reason=str(error),
            stage=stage,
        )
    except Exception as error:
        return SuiteRecord(
            case,
            implementation.name,
            "failed",
            environment,
            measurement,
            quality=check,
            configuration=configuration,
            reason=f"{type(error).__name__}: {error}",
            stage=stage,
        )
    finally:
        del output, operation
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
