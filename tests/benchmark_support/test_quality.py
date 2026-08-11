import math

import pytest
import torch
from lib.quality import measure_quality, measure_saturation


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


def test_quality_preserves_integer_precision_beyond_float32() -> None:
    reference = torch.tensor([16_777_216], dtype=torch.int32)
    actual = torch.tensor([16_777_217], dtype=torch.int32)

    quality = measure_quality(actual, reference)

    assert quality.mean_absolute_error == 1.0
    assert quality.max_absolute_error == 1.0
    assert math.isfinite(quality.sqnr_db)


def test_quality_preserves_float64_precision() -> None:
    reference = torch.tensor([1.0], dtype=torch.float64)
    actual = torch.nextafter(reference, torch.tensor([2.0], dtype=torch.float64))

    quality = measure_quality(actual, reference)

    assert quality.mean_absolute_error == 2**-52
    assert quality.max_absolute_error == 2**-52
    assert math.isfinite(quality.sqnr_db)


def test_quality_uses_fp32_norms_without_promoting_full_inputs() -> None:
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn((256, 128), dtype=torch.bfloat16, generator=generator)
    actual = (
        reference.float()
        + 0.03 * torch.randn(reference.shape, dtype=torch.float32, generator=generator)
    ).to(torch.bfloat16)

    quality = measure_quality(actual, reference)
    fp32_error = actual.float() - reference.float()
    fp32_sqnr = 20 * torch.log10(
        torch.linalg.vector_norm(reference.float()) / torch.linalg.vector_norm(fp32_error)
    )

    assert quality.sqnr_db == pytest.approx(float(fp32_sqnr), abs=0.01)


def test_quality_fp16_norms_do_not_overflow() -> None:
    reference = torch.full((1024,), 300.0, dtype=torch.float16)
    actual = torch.full((1024,), 299.0, dtype=torch.float16)

    quality = measure_quality(actual, reference)

    assert quality.relative_l2_error == pytest.approx(1 / 300)
    assert quality.sqnr_db == pytest.approx(20 * math.log10(300))


def test_quality_fp16_error_norm_does_not_underflow() -> None:
    reference = torch.tensor([0.1], dtype=torch.float16)
    actual = torch.nextafter(reference, torch.zeros_like(reference))

    quality = measure_quality(actual, reference)
    expected_sqnr = 20 * torch.log10(
        torch.linalg.vector_norm(reference.float())
        / torch.linalg.vector_norm(actual.float() - reference.float())
    )

    assert math.isfinite(quality.sqnr_db)
    assert quality.sqnr_db == pytest.approx(float(expected_sqnr))


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
def test_quality_rejects_full_width_integer_inputs(dtype: torch.dtype) -> None:
    actual = torch.tensor([9_007_199_254_740_993], dtype=dtype)
    reference = torch.tensor([9_007_199_254_740_992], dtype=dtype)

    with pytest.raises(ValueError, match="int64 and uint64 quality inputs are unsupported"):
        measure_quality(actual, reference)


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
