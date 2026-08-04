"""GPU tests for the experimental ConvRot integer-attention kernel."""

import pytest
import torch

from piper_kernels.attention._int8_convrot.backends.triton import (
    triton_int8_convrot_attention,
)
from piper_kernels.attention._int8_convrot.reference import reference_int8_convrot_attention


def _consumer_gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability == (8, 9) or capability[0] == 12


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _consumer_gpu_available(),
        reason="requires consumer Ada SM89 or Blackwell SM12x",
    ),
]


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_triton_matches_integer_reference(head_dim: int, is_causal: bool) -> None:
    torch.manual_seed(53)
    query = torch.randn(1, 2, 193, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_int8_convrot_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
        )
        expected = reference_int8_convrot_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
        )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.002
    assert error.max().item() < 0.05
