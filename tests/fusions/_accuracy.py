"""Numerical comparisons across quantized sparse-attention fusion boundaries."""

import torch


def assert_fusion_output_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Enforce a 0.1% relative-L2 pipeline accuracy bound.

    FP32 reduction/FMA changes can cross quantization boundaries even when the
    final activation dtype is FP32. Primitive tests check accumulation and bias
    precision separately; composed paths need numerical agreement, not bit identity.
    """
    assert actual.shape == expected.shape
    assert actual.dtype is expected.dtype
    assert actual.device == expected.device
    assert bool(torch.isfinite(actual).all())
    assert bool(torch.isfinite(expected).all())
    actual_values, expected_values = actual.double(), expected.double()
    error = (actual_values - expected_values).norm()
    reference_norm = expected_values.norm()
    assert bool(error <= 1e-3 * reference_norm), (
        f"fusion output error norm {error.item():.6g} exceeds "
        f"0.001 * reference norm {reference_norm.item():.6g}"
    )
