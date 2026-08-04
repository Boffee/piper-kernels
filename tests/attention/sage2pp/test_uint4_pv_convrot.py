"""Tests for the experimental UINT4-P plus ConvRot-INT4-V Sage2++ path."""

import pytest
import torch

from piper_kernels.attention._sage2pp.experiments import (
    reference_sage_attention_uint4_pv_convrot,
    triton_sage_attention_uint4_pv_convrot,
    triton_sage_attention_uint4_pv_paired_convrot,
)
from piper_kernels.attention._sage2pp.experiments.uint4_pv_convrot import (
    _quantize_probability_uint4,
)


def _consumer_gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability == (8, 9) or capability[0] == 12


def test_probability_quantizer_uses_full_uint4_range_per_row() -> None:
    probability = torch.tensor([[0.0, 0.25, 0.5, 1.0], [0.0, 0.025, 0.05, 0.1]])

    quantized, scale = _quantize_probability_uint4(probability)

    assert quantized.dtype is torch.int8
    assert quantized.min().item() == 0
    assert torch.equal(quantized.amax(dim=-1), torch.tensor([15, 15], dtype=torch.int8))
    reconstructed = quantized.float() * scale[:, None]
    assert torch.allclose(reconstructed, probability, atol=0.035, rtol=0)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("rotation_group", [0, 64])
def test_uint4_pv_convrot_reference_is_close_to_exact_attention(
    head_dim: int,
    is_causal: bool,
    rotation_group: int,
) -> None:
    torch.manual_seed(75)
    query = torch.randn(1, 2, 65, head_dim, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual = reference_sage_attention_uint4_pv_convrot(
        query,
        key,
        value,
        head_dim**-0.5,
        is_causal,
        rotation_group=rotation_group,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=is_causal
    )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.04
    assert error.max().item() < 0.5


def test_uint4_pv_paired_convrot_reference_is_finite() -> None:
    torch.manual_seed(751)
    query = torch.randn(1, 2, 65, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual = reference_sage_attention_uint4_pv_convrot(
        query,
        key,
        value,
        64**-0.5,
        False,
        paired_rotation=True,
    )

    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not _consumer_gpu_available(),
    reason="requires consumer Ada SM89 or Blackwell SM12x",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("paired_rotation", [False, True])
def test_triton_uint4_pv_convrot_matches_reference(
    head_dim: int,
    is_causal: bool,
    paired_rotation: bool,
) -> None:
    torch.manual_seed(76)
    query = torch.randn(1, 2, 193, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        implementation = (
            triton_sage_attention_uint4_pv_paired_convrot
            if paired_rotation
            else triton_sage_attention_uint4_pv_convrot
        )
        actual = implementation(query, key, value, head_dim**-0.5, is_causal)
        expected = reference_sage_attention_uint4_pv_convrot(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            paired_rotation=paired_rotation,
        )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.005
    assert error.max().item() < (0.4 if paired_rotation else 0.12)
