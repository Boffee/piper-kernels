"""GPU tests for the pure-Triton Piper Attention backend."""

from typing import Literal

import pytest
import torch

from piper_kernels._triton.targets import supports_uint8_int8_mma
from piper_kernels.attention import piper_attention
from piper_kernels.attention.piper.reference import reference_piper_attention
from piper_kernels.attention.piper.triton import (
    _launch_piper_attention,
    _prepare_piper_attention,
    _run_piper_attention,
)


def _piper_gpu_available() -> bool:
    return torch.cuda.is_available() and supports_uint8_int8_mma(torch.device("cuda"))


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def _qk_quantization() -> Literal["per_thread", "per_warp"]:
    return "per_warp" if torch.cuda.get_device_capability()[0] == 12 else "per_thread"


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _piper_gpu_available(),
        reason="requires NVIDIA SM8x or consumer Blackwell SM12x mixed-sign MMAv2",
    ),
]


@pytest.mark.parametrize("center_value", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_triton_matches_quantized_reference(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    center_value: bool,
) -> None:
    torch.manual_seed(54)
    query = torch.randn(1, 2, 193, head_dim, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = piper_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            center_value=center_value,
        )
        expected = reference_piper_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            center_value=center_value,
            qk_quantization=_qk_quantization(),
        )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


@pytest.mark.parametrize("sequence", [193, 1024])
def test_affine_fallback_matches_native_uint8(sequence: int) -> None:
    torch.manual_seed(55 + sequence)
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False, True)

    with torch.no_grad():
        native = _run_piper_attention(
            *arguments,
            native_uint8=True,
            sort_value_rows=False,
            use_tensor_descriptors=False,
        )
        affine = _run_piper_attention(
            *arguments,
            native_uint8=False,
            sort_value_rows=False,
            use_tensor_descriptors=False,
        )

    assert torch.equal(native, affine)


def test_centered_value_fusion_restores_constant_value() -> None:
    torch.manual_seed(56)
    query = torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value_row = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16)
    value = value_row.expand_as(query).contiguous()

    with torch.no_grad():
        actual = piper_attention(query, key, value, center_value=True)

    torch.testing.assert_close(actual, value, atol=0.0, rtol=0.0)


def test_large_value_scale_multiplier_remains_finite() -> None:
    query = torch.ones((1, 1, 64, 64), device="cuda", dtype=torch.float16)
    key = torch.ones_like(query)
    key[:, :, 0] = -1
    value = torch.ones_like(query)
    value[:, :, 0] = 40000

    with torch.no_grad():
        prepared = _prepare_piper_attention(
            query,
            key,
            value,
            64**-0.5,
            False,
            False,
        )
        actual = _launch_piper_attention(prepared)
        expected = reference_piper_attention(
            query,
            key,
            value,
            64**-0.5,
            False,
            center_value=False,
            qk_quantization=_qk_quantization(),
        )

    assert prepared.value_scale_multiplier.dtype is torch.float32
    assert torch.isfinite(prepared.value_scale_multiplier).all()
    torch.testing.assert_close(actual, expected, atol=2**-9, rtol=0.0)


@pytest.mark.skipif(not _sm120_available(), reason="centered long path is tuned for SM12x")
def test_centering_improves_biased_value_quality() -> None:
    torch.manual_seed(57)
    sequence = 1024
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    offset = torch.linspace(-8, 8, 128, device="cuda").reshape(1, 1, 1, 128)
    value = (offset + torch.randn_like(query.float()) * 0.25).to(torch.bfloat16)

    with torch.no_grad():
        uncentered = piper_attention(query, key, value, center_value=False)
        centered = piper_attention(query, key, value, center_value=True)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    uncentered_mse = (uncentered.float() - expected.float()).square().mean()
    centered_mse = (centered.float() - expected.float()).square().mean()
    assert centered_mse < uncentered_mse * 0.2


@pytest.mark.skipif(not _sm120_available(), reason="ordered grouped-QK path targets SM12x")
def test_sorted_ragged_path_matches_quantized_reference() -> None:
    torch.manual_seed(58)
    query = torch.randn(1, 2, 257, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = _run_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            False,
            True,
            native_uint8=True,
            sort_value_rows=True,
            use_tensor_descriptors=False,
        )
        expected = reference_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            False,
            center_value=True,
            qk_quantization="per_warp",
            sort_value_rows=True,
        )
    error = (actual.float() - expected.float()).abs()

    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


@pytest.mark.skipif(not _sm120_available(), reason="tensor descriptors target SM12x")
def test_long_descriptor_path_matches_pointer_path() -> None:
    torch.manual_seed(59)
    sequence = 8192
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False, True)

    with torch.no_grad():
        descriptor = _run_piper_attention(
            *arguments,
            native_uint8=True,
            sort_value_rows=False,
            use_tensor_descriptors=True,
        )
        pointer = _run_piper_attention(
            *arguments,
            native_uint8=True,
            sort_value_rows=False,
            use_tensor_descriptors=False,
        )

    torch.testing.assert_close(descriptor, pointer, atol=2**-9, rtol=0.0)


def test_triton_runs_under_torch_compile() -> None:
    torch.manual_seed(60)
    query_storage = torch.randn(3, 2, 128, 64, device="cuda", dtype=torch.float16)
    key_storage = torch.randn_like(query_storage)
    value_storage = torch.randn_like(query_storage)
    query = query_storage.permute(1, 0, 2, 3)
    key = key_storage.permute(1, 0, 2, 3)
    value = value_storage.permute(1, 0, 2, 3)

    def consumer(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return -piper_attention(query, key, value, center_value=True)

    with torch.no_grad():
        expected = consumer(query, key, value)
        actual = torch.compile(consumer, fullgraph=True)(query, key, value)

    assert torch.equal(actual, expected)
