"""Tests for the experimental INT4-range ConvRot Sage2++ path."""

import pytest
import torch

from piper_kernels.attention._sage2pp.experiments import (
    triton_sage_attention_int4_convrot,
)
from piper_kernels.attention._sage2pp.reference import reference_sage_attention


def _consumer_gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability == (8, 9) or capability[0] == 12


@pytest.mark.parametrize("head_dim", [64, 128])
def test_int4_convrot_reference_is_finite(head_dim: int) -> None:
    torch.manual_seed(73)
    query = torch.randn(1, 2, 65, head_dim, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual = reference_sage_attention(
        query,
        key,
        value,
        head_dim**-0.5,
        False,
        qk_bits=4,
        rotation_group=64,
    )

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not _consumer_gpu_available(),
    reason="requires consumer Ada SM89 or Blackwell SM12x",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_triton_int4_convrot_matches_reference(head_dim: int, is_causal: bool) -> None:
    torch.manual_seed(74)
    query = torch.randn(1, 2, 193, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_int4_convrot(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
        )
        expected = reference_sage_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            qk_bits=4,
            rotation_group=64,
        )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.08
