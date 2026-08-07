"""Public API and validation tests for SageAttention2++."""

import pytest
import torch

import piper_kernels.attention
from piper_kernels.attention import sage_attention_2pp


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    return query, torch.randn_like(query), torch.randn_like(query)


def test_attention_exports_only_sage_attention_2pp() -> None:
    assert piper_kernels.attention.sage_attention_2pp is sage_attention_2pp
    assert piper_kernels.attention.__all__ == ["sage_attention_2pp"]


def test_public_api_uses_portable_reference_on_cpu() -> None:
    torch.manual_seed(40)
    query = torch.randn(1, 1, 9, 64, dtype=torch.float16)
    key = torch.randn(1, 1, 11, 64, dtype=torch.float16)
    value = torch.randn_like(key)

    with torch.no_grad():
        output = sage_attention_2pp(query, key, value, scale=0.2)

    assert output.shape == query.shape
    assert output.dtype is query.dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_rejects_unsupported_dtype(dtype: torch.dtype) -> None:
    query = torch.zeros((1, 1, 8, 64), dtype=dtype)

    with pytest.raises(ValueError, match="float16 or bfloat16"):
        sage_attention_2pp(query, query, query)


def test_rejects_mismatched_dtypes() -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="share a dtype"):
        sage_attention_2pp(query, key.to(torch.bfloat16), value)


def test_rejects_unsupported_head_dimension() -> None:
    query = torch.randn(1, 1, 8, 32, dtype=torch.float16)

    with pytest.raises(ValueError, match="head dimensions 64 and 128"):
        sage_attention_2pp(query, query, query)


def test_rejects_mismatched_key_value_lengths() -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="key/value lengths"):
        sage_attention_2pp(query, key, value[:, :, :-1])


def test_rejects_grouped_query_attention() -> None:
    query = torch.randn(1, 4, 16, 64, dtype=torch.float16)
    key = torch.randn(1, 2, 16, 64, dtype=torch.float16)

    with pytest.raises(ValueError, match="equal batch and head dimensions"):
        sage_attention_2pp(query, key, key)


def test_rejects_rectangular_causal_inputs() -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="equal query and key lengths"):
        sage_attention_2pp(query[:, :, :-1], key, value, is_causal=True)


@pytest.mark.parametrize(("query_length", "key_length"), [(0, 8), (8, 0)])
def test_rejects_empty_sequences(query_length: int, key_length: int) -> None:
    query = torch.empty((1, 1, query_length, 64), dtype=torch.float16)
    key = torch.empty((1, 1, key_length, 64), dtype=torch.float16)

    with pytest.raises(ValueError, match="does not accept empty"):
        sage_attention_2pp(query, key, key)


def test_rejects_noncontiguous_head_dimension() -> None:
    storage = torch.randn(1, 1, 8, 128, dtype=torch.float16)
    query = storage[..., ::2]

    with pytest.raises(ValueError, match="head dimension must be contiguous"):
        sage_attention_2pp(query, query, query)


def test_rejects_autograd() -> None:
    query, key, value = _inputs()
    query.requires_grad_(True)

    with pytest.raises(RuntimeError, match="inference-only"):
        sage_attention_2pp(query, key, value)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_scale(scale: float) -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="finite and positive"):
        sage_attention_2pp(query, key, value, scale=scale)
