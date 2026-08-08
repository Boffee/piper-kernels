"""GPU tests for the pure-Triton SageAttention2++ backend."""

from typing import Literal

import pytest
import torch

import piper_kernels.attention.sage2pp.triton as sage_backend
from piper_kernels._triton.targets import supports_fp8_fp16_mma
from piper_kernels.attention import sage_attention_2pp
from piper_kernels.attention.sage2pp.reference import reference_sage_attention_2pp
from piper_kernels.attention.sage2pp.triton import _run_sage_attention_2pp


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def _fp8_gpu_available() -> bool:
    return torch.cuda.is_available() and supports_fp8_fp16_mma(torch.device("cuda"))


def _qk_quantization() -> Literal["per_thread", "per_warp"]:
    return "per_warp" if torch.cuda.get_device_capability()[0] == 12 else "per_thread"


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _fp8_gpu_available(),
        reason="requires NVIDIA FP8 tensor cores with FP16 accumulation",
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
        actual = sage_attention_2pp(query, key, value, is_causal=is_causal)
        expected = reference_sage_attention_2pp(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            qk_quantization=_qk_quantization(),
        )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


@pytest.mark.skipif(
    not _sm120_available(),
    reason="raw-score recurrence is selected for grouped SM12x quantization",
)
@pytest.mark.parametrize(
    ("is_causal", "threshold_name"),
    [
        (False, "_NONCAUSAL_RAW_SCORE_MIN_KEY_LENGTH"),
        (True, "_CAUSAL_RAW_SCORE_MIN_KEY_LENGTH"),
    ],
)
def test_raw_score_recurrence_matches_quantized_reference(
    monkeypatch: pytest.MonkeyPatch,
    is_causal: bool,
    threshold_name: str,
) -> None:
    monkeypatch.setattr(sage_backend, threshold_name, 0)
    assert sage_backend._should_use_raw_score_recurrence(True, is_causal, 193, 128)
    assert not sage_backend._should_use_raw_score_recurrence(True, is_causal, 193, 64)
    assert sage_backend._should_inline_query_quantization(True, is_causal, 193, 128)
    assert not sage_backend._should_inline_query_quantization(False, is_causal, 193, 128)
    torch.manual_seed(432)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = _run_sage_attention_2pp(
            query,
            key,
            value,
            128**-0.5,
            is_causal,
        )
        expected = reference_sage_attention_2pp(
            query,
            key,
            value,
            128**-0.5,
            is_causal,
            qk_quantization="per_warp",
        )
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


def test_triton_ragged_offset_key_matches_quantized_reference() -> None:
    torch.manual_seed(431)
    query = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.float16)
    key = 100.0 + torch.randn(1, 1, 17, 64, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)

    with torch.no_grad():
        actual = sage_attention_2pp(query, key, value)
        expected = reference_sage_attention_2pp(
            query,
            key,
            value,
            64**-0.5,
            False,
            qk_quantization=_qk_quantization(),
        )
    error = (actual.float() - expected.float()).abs()

    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_triton_supports_rectangular_and_strided_inputs(
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    torch.manual_seed(44)
    query_storage = torch.randn(2, 2, 194, head_dim, device="cuda", dtype=dtype)
    key_storage = torch.randn(2, 2, 286, head_dim, device="cuda", dtype=dtype)
    value_storage = torch.randn_like(key_storage)
    query = query_storage[:, :, ::2]
    key = key_storage[:, :, ::2]
    value = value_storage[:, :, ::2]

    with torch.no_grad():
        actual = sage_attention_2pp(query, key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.01
    assert error.max().item() < 0.2


@pytest.mark.skipif(
    not _sm120_available(),
    reason="tensor-descriptor padding is currently selected on SM12x",
)
def test_triton_ragged_descriptor_storage_matches_pointer_path() -> None:
    torch.manual_seed(441)
    query = torch.randn(1, 24, 1025, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = sage_attention_2pp(query, key, value)
        expected = _run_sage_attention_2pp(
            query,
            key,
            value,
            128**-0.5,
            False,
            use_tensor_descriptors=False,
        )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_runs_under_torch_compile(dtype: torch.dtype) -> None:
    torch.manual_seed(45)
    query = torch.randn(1, 1, 128, 64, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        expected = sage_attention_2pp(query, key, value)
        actual = torch.compile(sage_attention_2pp, fullgraph=True)(query, key, value)

    assert torch.equal(actual, expected)


def test_triton_torch_compile_supports_permuted_batch_head_strides() -> None:
    torch.manual_seed(451)
    query_storage = torch.randn(3, 2, 32, 64, device="cuda", dtype=torch.float16)
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
        return -sage_attention_2pp(query, key, value)

    with torch.no_grad():
        expected = consumer(query, key, value)
        actual = torch.compile(consumer, fullgraph=True)(query, key, value)

    assert torch.equal(actual, expected)
