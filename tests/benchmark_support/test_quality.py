import math

import pytest
import torch
from _lib.quality import measure_quality, measure_saturation


def test_quality_metrics_have_expected_definitions() -> None:
    reference = torch.tensor([1.0, 2.0, 0.0])
    actual = torch.tensor([1.0, 1.0, 0.0])

    quality = measure_quality(actual, reference)

    assert quality.mean_absolute_error == pytest.approx(1 / 3)
    assert quality.max_absolute_error == 1.0
    assert quality.relative_l1_error == pytest.approx(1 / 3)
    assert quality.relative_l2_error == pytest.approx(1 / math.sqrt(5))
    assert quality.sqnr_db == pytest.approx(10 * math.log10(5))
    assert quality.cosine_similarity == pytest.approx(3 / math.sqrt(10))
    assert quality.actual_nonfinite_count == 0
    assert quality.reference_nonfinite_count == 0
    assert quality.nonfinite_mismatch_count == 0


def test_quality_clamps_cosine_roundoff_to_its_mathematical_range() -> None:
    values = torch.arange(1000, dtype=torch.int32) * 1000

    quality = measure_quality(values, values)

    assert quality.cosine_similarity == 1.0


def test_quality_counts_nonfinite_values_and_uses_finite_pairs() -> None:
    reference = torch.tensor([torch.nan, 2.0, 1.0])
    actual = torch.tensor([torch.nan, torch.inf, 1.0])

    quality = measure_quality(actual, reference)

    assert quality.mean_absolute_error == 0
    assert math.isinf(quality.sqnr_db)
    assert quality.actual_nonfinite_count == 2
    assert quality.reference_nonfinite_count == 1
    assert quality.nonfinite_mismatch_count == 1


def test_quality_distinguishes_nonfinite_values() -> None:
    reference = torch.tensor([torch.nan, torch.inf, -torch.inf, torch.nan, torch.inf, -torch.inf])
    actual = torch.tensor([torch.nan, torch.inf, -torch.inf, torch.inf, torch.nan, torch.inf])

    quality = measure_quality(actual, reference)

    assert quality.actual_nonfinite_count == 6
    assert quality.reference_nonfinite_count == 6
    assert quality.nonfinite_mismatch_count == 3


def test_saturation_counts_both_quantizer_endpoints() -> None:
    saturation = measure_saturation(torch.tensor([-128, -1, 0, 127, 127]), -128, 127)

    assert saturation.minimum_count == 1
    assert saturation.maximum_count == 2
    assert saturation.count == 3
    assert saturation.total_count == 5
    assert saturation.fraction == pytest.approx(0.6)


def test_quality_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        measure_quality(torch.zeros(2), torch.zeros(3))
