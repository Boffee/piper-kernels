"""Numerical quality and quantizer saturation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class QuantizerSaturation:
    """Counts at the minimum and maximum values of a quantized representation."""

    minimum_count: int
    maximum_count: int
    total_count: int

    @property
    def count(self) -> int:
        """Return the combined number of saturated values."""
        return self.minimum_count + self.maximum_count

    @property
    def fraction(self) -> float:
        """Return the fraction of values at either endpoint."""
        return self.count / self.total_count if self.total_count else 0.0

    def as_dict(self) -> dict[str, int | float]:
        """Return stable machine-readable field names."""
        return {
            "minimum_count": self.minimum_count,
            "maximum_count": self.maximum_count,
            "count": self.count,
            "total_count": self.total_count,
            "fraction": self.fraction,
        }


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Model-neutral comparison metrics for an actual and reference tensor."""

    mean_absolute_error: float
    max_absolute_error: float
    relative_l1: float
    relative_l2: float
    sqnr_db: float
    cosine_similarity: float
    actual_nonfinite_count: int
    reference_nonfinite_count: int
    nonfinite_mismatch_count: int
    saturation: Mapping[str, QuantizerSaturation] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | dict[str, dict[str, int | float]]]:
        """Return stable machine-readable field names."""
        return {
            "mean_absolute_error": self.mean_absolute_error,
            "max_absolute_error": self.max_absolute_error,
            "relative_l1": self.relative_l1,
            "relative_l2": self.relative_l2,
            "sqnr_db": self.sqnr_db,
            "cosine_similarity": self.cosine_similarity,
            "actual_nonfinite_count": self.actual_nonfinite_count,
            "reference_nonfinite_count": self.reference_nonfinite_count,
            "nonfinite_mismatch_count": self.nonfinite_mismatch_count,
            "saturation": {name: value.as_dict() for name, value in self.saturation.items()},
        }


def measure_saturation(
    values: torch.Tensor,
    minimum: int | float,
    maximum: int | float,
) -> QuantizerSaturation:
    """Count quantized values equal to the representation endpoints."""
    return QuantizerSaturation(
        minimum_count=int(torch.count_nonzero(values == minimum).item()),
        maximum_count=int(torch.count_nonzero(values == maximum).item()),
        total_count=values.numel(),
    )


def _zero_safe_cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_norm = torch.linalg.vector_norm(actual)
    reference_norm = torch.linalg.vector_norm(reference)
    if actual_norm == 0 or reference_norm == 0:
        return 1.0 if actual_norm == 0 and reference_norm == 0 else 0.0
    return float(torch.dot(actual.flatten(), reference.flatten()) / (actual_norm * reference_norm))


def measure_quality(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    saturation: Mapping[str, QuantizerSaturation] | None = None,
) -> QualityMetrics:
    """Compare tensors using finite pairs and report non-finite values separately."""
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: actual {actual.shape}, reference {reference.shape}")

    actual_float = actual.detach().to(torch.float32)
    reference_float = reference.detach().to(torch.float32)
    actual_finite = torch.isfinite(actual_float)
    reference_finite = torch.isfinite(reference_float)
    finite_pairs = actual_finite & reference_finite
    actual_nonfinite = int(torch.count_nonzero(~actual_finite).item())
    reference_nonfinite = int(torch.count_nonzero(~reference_finite).item())
    nonfinite_mismatch = int(torch.count_nonzero(actual_finite ^ reference_finite).item())

    if not torch.any(finite_pairs):
        nan = math.nan
        return QualityMetrics(
            mean_absolute_error=nan,
            max_absolute_error=nan,
            relative_l1=nan,
            relative_l2=nan,
            sqnr_db=nan,
            cosine_similarity=nan,
            actual_nonfinite_count=actual_nonfinite,
            reference_nonfinite_count=reference_nonfinite,
            nonfinite_mismatch_count=nonfinite_mismatch,
            saturation=dict(saturation or {}),
        )

    actual_values = actual_float[finite_pairs]
    reference_values = reference_float[finite_pairs]
    error = actual_values - reference_values
    absolute_error = error.abs()
    epsilon = torch.finfo(torch.float32).tiny
    reference_l1 = reference_values.abs().sum()
    reference_l2_squared = reference_values.square().sum()
    error_l2_squared = error.square().sum()
    sqnr = 10 * torch.log10(reference_l2_squared / error_l2_squared)

    return QualityMetrics(
        mean_absolute_error=float(absolute_error.mean()),
        max_absolute_error=float(absolute_error.max()),
        relative_l1=float(absolute_error.sum() / reference_l1.clamp_min(epsilon)),
        relative_l2=float(
            torch.sqrt(error_l2_squared) / torch.sqrt(reference_l2_squared).clamp_min(epsilon)
        ),
        sqnr_db=float(sqnr),
        cosine_similarity=_zero_safe_cosine(actual_values, reference_values),
        actual_nonfinite_count=actual_nonfinite,
        reference_nonfinite_count=reference_nonfinite,
        nonfinite_mismatch_count=nonfinite_mismatch,
        saturation=dict(saturation or {}),
    )
