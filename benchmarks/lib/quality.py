"""Numerical quality and quantizer-saturation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

_FULL_WIDTH_INTEGER_DTYPES = (torch.int64, torch.uint64)
_NONFINITE_COUNT_CHUNK_ELEMENTS = 1 << 26


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
    relative_l1_error: float
    relative_l2_error: float
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
            "relative_l1_error": self.relative_l1_error,
            "relative_l2_error": self.relative_l2_error,
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
    cosine = torch.dot(actual.flatten(), reference.flatten()) / (actual_norm * reference_norm)
    return float(cosine.clamp(-1, 1))


def _comparison_dtype(actual: torch.Tensor, reference: torch.Tensor) -> torch.dtype:
    """Choose a floating dtype that preserves the supported input precision."""
    if (
        torch.float64 in (actual.dtype, reference.dtype)
        or not actual.is_floating_point()
        or not reference.is_floating_point()
    ):
        return torch.float64
    return torch.float32


def _nonfinite_count(value: torch.Tensor) -> int:
    flattened = value.flatten()
    count = 0
    for start in range(0, flattened.numel(), _NONFINITE_COUNT_CHUNK_ELEMENTS):
        chunk = flattened[start : start + _NONFINITE_COUNT_CHUNK_ELEMENTS]
        nonfinite = torch.isfinite(chunk).logical_not_()
        count += int(nonfinite.sum(dtype=torch.int32).item())
    return count


def _measure_finite_floating_quality(
    actual: torch.Tensor,
    reference: torch.Tensor,
    comparison_dtype: torch.dtype,
    saturation: Mapping[str, QuantizerSaturation] | None,
) -> QualityMetrics:
    """Measure finite floating tensors with fused promoted norm reductions."""
    error = actual - reference
    absolute_error_sum = torch.linalg.vector_norm(
        error,
        ord=1,
        dtype=comparison_dtype,
    )
    error_l2 = torch.linalg.vector_norm(error, dtype=comparison_dtype)
    mean_absolute_error = absolute_error_sum / actual.numel()
    error.abs_()
    max_absolute_error = error.max()
    del error

    reference_l1 = torch.linalg.vector_norm(
        reference,
        ord=1,
        dtype=comparison_dtype,
    )
    reference_l2 = torch.linalg.vector_norm(reference, dtype=comparison_dtype)
    actual_l2 = torch.linalg.vector_norm(actual, dtype=comparison_dtype)

    epsilon = torch.finfo(comparison_dtype).tiny
    sqnr = 20 * torch.log10(reference_l2 / error_l2)
    if actual_l2 == 0 or reference_l2 == 0:
        cosine_similarity = 1.0 if actual_l2 == 0 and reference_l2 == 0 else 0.0
    else:
        dot = (actual_l2.square() + reference_l2.square() - error_l2.square()) / 2
        cosine = dot / (actual_l2 * reference_l2)
        cosine_similarity = float(cosine.clamp(-1, 1))

    return QualityMetrics(
        mean_absolute_error=float(mean_absolute_error),
        max_absolute_error=float(max_absolute_error),
        relative_l1_error=float(absolute_error_sum / reference_l1.clamp_min(epsilon)),
        relative_l2_error=float(error_l2 / reference_l2.clamp_min(epsilon)),
        sqnr_db=float(sqnr),
        cosine_similarity=cosine_similarity,
        actual_nonfinite_count=0,
        reference_nonfinite_count=0,
        nonfinite_mismatch_count=0,
        saturation=dict(saturation or {}),
    )


def measure_quality(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    saturation: Mapping[str, QuantizerSaturation] | None = None,
) -> QualityMetrics:
    """Compare tensors using finite pairs and report non-finite values separately."""
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: actual {actual.shape}, reference {reference.shape}")
    if actual.dtype in _FULL_WIDTH_INTEGER_DTYPES or reference.dtype in _FULL_WIDTH_INTEGER_DTYPES:
        raise ValueError(
            "int64 and uint64 quality inputs are unsupported because float64 cannot "
            "preserve every full-width integer value"
        )

    actual_value = actual.detach()
    reference_value = reference.detach()
    actual_nonfinite = _nonfinite_count(actual_value)
    reference_nonfinite = _nonfinite_count(reference_value)
    comparison_dtype = _comparison_dtype(actual_value, reference_value)
    if (
        actual_value.is_floating_point()
        and reference_value.is_floating_point()
        and not actual_nonfinite
        and not reference_nonfinite
    ):
        return _measure_finite_floating_quality(
            actual_value,
            reference_value,
            comparison_dtype,
            saturation,
        )

    actual_float = actual_value.to(comparison_dtype)
    reference_float = reference_value.to(comparison_dtype)
    actual_finite = torch.isfinite(actual_float)
    reference_finite = torch.isfinite(reference_float)
    finite_pairs = actual_finite & reference_finite
    nonfinite_positions = ~actual_finite | ~reference_finite
    matching_nonfinite = (torch.isnan(actual_float) & torch.isnan(reference_float)) | (
        actual_float == reference_float
    )
    nonfinite_mismatch = int(torch.count_nonzero(nonfinite_positions & ~matching_nonfinite).item())

    if not torch.any(finite_pairs):
        nan = math.nan
        return QualityMetrics(
            mean_absolute_error=nan,
            max_absolute_error=nan,
            relative_l1_error=nan,
            relative_l2_error=nan,
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
    epsilon = torch.finfo(comparison_dtype).tiny
    reference_l1 = reference_values.abs().sum()
    reference_l2_squared = reference_values.square().sum()
    error_l2_squared = error.square().sum()
    sqnr = 10 * torch.log10(reference_l2_squared / error_l2_squared)

    return QualityMetrics(
        mean_absolute_error=float(absolute_error.mean()),
        max_absolute_error=float(absolute_error.max()),
        relative_l1_error=float(absolute_error.sum() / reference_l1.clamp_min(epsilon)),
        relative_l2_error=float(
            torch.sqrt(error_l2_squared) / torch.sqrt(reference_l2_squared).clamp_min(epsilon)
        ),
        sqnr_db=float(sqnr),
        cosine_similarity=_zero_safe_cosine(actual_values, reference_values),
        actual_nonfinite_count=actual_nonfinite,
        reference_nonfinite_count=reference_nonfinite,
        nonfinite_mismatch_count=nonfinite_mismatch,
        saturation=dict(saturation or {}),
    )
