"""Public contracts shared by dense Piper Attention and SageAttention2++."""

import pytest
import torch

import piper_kernels
from piper_kernels.attention.piper_attention import piper_attention
from piper_kernels.attention.piper_attention.reference import reference_piper_attention
from piper_kernels.attention.sage_attention_2pp import sage_attention_2pp
from piper_kernels.attention.sage_attention_2pp.reference import reference_sage_attention_2pp


@pytest.fixture(params=[piper_attention, sage_attention_2pp], ids=["piper", "sage"])
def attention(request: pytest.FixtureRequest):
    return request.param


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(1, 2, 16, 64, dtype=torch.float16)
    return query, torch.randn_like(query), torch.randn_like(query)


@pytest.mark.parametrize(
    ("name", "attention"),
    [("piper_attention", piper_attention), ("sage_attention_2pp", sage_attention_2pp)],
)
def test_package_root_exports_attention(name: str, attention) -> None:
    assert getattr(piper_kernels, name) is attention
    assert name in piper_kernels.__all__


@pytest.mark.parametrize(
    ("attention", "reference"),
    [
        (piper_attention, reference_piper_attention),
        (sage_attention_2pp, reference_sage_attention_2pp),
    ],
    ids=["piper", "sage"],
)
def test_public_api_uses_portable_reference_on_cpu(attention, reference) -> None:
    torch.manual_seed(50)
    query = torch.randn(1, 1, 9, 64, dtype=torch.float16)
    key = torch.randn(1, 1, 11, 64, dtype=torch.float16)
    value = torch.randn_like(key)

    with torch.no_grad():
        output = attention(query, key, value, scale=0.2)

    assert output.shape == query.shape
    assert output.dtype is query.dtype
    expected = reference(query, key, value, scale=0.2, is_causal=False)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_rejects_unsupported_dtype(attention, dtype: torch.dtype) -> None:
    query = torch.zeros((1, 1, 8, 64), dtype=dtype)

    with pytest.raises(ValueError, match="float16 or bfloat16"):
        attention(query, query, query)


def test_rejects_mismatched_dtypes(attention) -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="share a dtype"):
        attention(query, key.to(torch.bfloat16), value)


def test_rejects_unsupported_head_dimension(attention) -> None:
    query = torch.randn(1, 1, 8, 32, dtype=torch.float16)

    with pytest.raises(ValueError, match="head dimensions 64 and 128"):
        attention(query, query, query)


def test_rejects_mismatched_key_value_lengths(attention) -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="key/value lengths"):
        attention(query, key, value[:, :, :-1])


def test_rejects_rectangular_causal_inputs(attention) -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="equal query and key lengths"):
        attention(query[:, :, :-1], key, value, is_causal=True)


@pytest.mark.parametrize(("query_length", "key_length"), [(0, 8), (8, 0)])
def test_rejects_empty_sequences(attention, query_length: int, key_length: int) -> None:
    query = torch.empty((1, 1, query_length, 64), dtype=torch.float16)
    key = torch.empty((1, 1, key_length, 64), dtype=torch.float16)

    with pytest.raises(ValueError, match="does not accept empty"):
        attention(query, key, key)


def test_rejects_noncontiguous_head_dimension(attention) -> None:
    storage = torch.randn(1, 1, 8, 128, dtype=torch.float16)
    query = storage[..., ::2]

    with pytest.raises(ValueError, match="head dimension must be contiguous"):
        attention(query, query, query)


def test_rejects_autograd(attention) -> None:
    query, key, value = _inputs()
    query.requires_grad_(True)

    with pytest.raises(RuntimeError, match="inference-only"):
        attention(query, key, value)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_scale(attention, scale: float) -> None:
    query, key, value = _inputs()

    with pytest.raises(ValueError, match="finite and positive"):
        attention(query, key, value, scale=scale)
