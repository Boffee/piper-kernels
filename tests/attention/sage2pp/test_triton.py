"""GPU tests for the pure-Triton SageAttention2++ backend."""

import pytest
import torch

from piper_kernels.attention import sage_attention
from piper_kernels.attention._sage2pp.reference import reference_sage_attention


def _consumer_fp8_available() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability == (8, 9) or capability[0] == 12


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _consumer_fp8_available(),
        reason="requires consumer Ada SM89 or Blackwell SM12x",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_triton_matches_quantized_reference(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
) -> None:
    torch.manual_seed(43)
    query = torch.randn(1, 2, 193, head_dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = sage_attention(query, key, value, is_causal=is_causal)
        expected = reference_sage_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
        )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


def test_triton_supports_rectangular_and_strided_inputs() -> None:
    torch.manual_seed(44)
    query_storage = torch.randn(2, 2, 194, 64, device="cuda", dtype=torch.float16)
    key_storage = torch.randn(2, 2, 286, 64, device="cuda", dtype=torch.float16)
    value_storage = torch.randn_like(key_storage)
    query = query_storage[:, :, ::2]
    key = key_storage[:, :, ::2]
    value = value_storage[:, :, ::2]

    with torch.no_grad():
        actual = sage_attention(query, key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.01
    assert error.max().item() < 0.2


def test_triton_runs_under_torch_compile() -> None:
    torch.manual_seed(45)
    query = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        expected = sage_attention(query, key, value)
        actual = torch.compile(sage_attention, fullgraph=True)(query, key, value)

    assert torch.equal(actual, expected)
