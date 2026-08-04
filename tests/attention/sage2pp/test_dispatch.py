"""Public API and validation tests for SageAttention2++."""

import pytest
import torch

import piper_kernels.attention
from piper_kernels.attention import sage_attention


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    return query, torch.randn_like(query), torch.randn_like(query)


def test_attention_exports_sage_attention() -> None:
    assert piper_kernels.attention.sage_attention is sage_attention
    assert piper_kernels.attention.__all__ == ["sage_attention"]


def test_sage_attention_rejects_unsupported_head_dimension() -> None:
    query = torch.randn(1, 1, 8, 32, dtype=torch.float16)
    with pytest.raises(ValueError, match="head dimensions 64 and 128"):
        sage_attention(query, query, query)


def test_sage_attention_rejects_mismatched_key_value_lengths() -> None:
    query, key, value = _inputs()
    with pytest.raises(ValueError, match="key/value lengths"):
        sage_attention(query, key, value[:, :, :-1])


def test_sage_attention_rejects_rectangular_causal_inputs() -> None:
    query, key, value = _inputs()
    with pytest.raises(ValueError, match="equal query and key lengths"):
        sage_attention(query[:, :, :-1], key, value, is_causal=True)


def test_sage_attention_rejects_autograd() -> None:
    query, key, value = _inputs()
    query.requires_grad_(True)
    with pytest.raises(RuntimeError, match="inference-only"):
        sage_attention(query, key, value)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf")])
def test_sage_attention_rejects_invalid_scale(scale: float) -> None:
    query, key, value = _inputs()
    with pytest.raises(ValueError, match="finite and positive"):
        sage_attention(query, key, value, scale=scale)
