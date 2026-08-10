"""GPU tests for the pure-Triton Piper Attention backend."""

from dataclasses import replace
from typing import Literal

import pytest
import torch

from piper_kernels import piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention.reference import reference_piper_attention
from piper_kernels.attention.piper_attention.triton import (
    _default_piper_attention_execution_plan,
    _launch_piper_attention,
    _prepare_piper_attention,
    _run_piper_attention,
)


def _piper_gpu_available() -> bool:
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).supports_uint8_int8_mma
    )


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_triton_matches_quantized_reference(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
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
        )
        expected = reference_piper_attention(
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


@pytest.mark.parametrize("sequence", [193, 1024])
def test_affine_fallback_matches_native_uint8(sequence: int) -> None:
    torch.manual_seed(55 + sequence)
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False)
    pointer_plan = replace(
        _default_piper_attention_execution_plan(query, key, False),
        use_tensor_descriptors=False,
        num_stages=3,
    )

    with torch.no_grad():
        native = _run_piper_attention(
            *arguments,
            execution_plan=replace(pointer_plan, native_uint8=True),
        )
        affine = _run_piper_attention(
            *arguments,
            execution_plan=replace(pointer_plan, native_uint8=False),
        )

    assert torch.equal(native, affine)


def test_centered_value_fusion_restores_constant_value() -> None:
    torch.manual_seed(56)
    query = torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value_row = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16)
    value = value_row.expand_as(query).contiguous()

    with torch.no_grad():
        actual = piper_attention(query, key, value)

    torch.testing.assert_close(actual, value, atol=0.0, rtol=0.0)


def test_causal_triton_is_independent_of_future_value_rows() -> None:
    torch.manual_seed(62)
    query = torch.randn(1, 1, 65, 64, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    changed_value = value.clone()
    changed_value[:, :, 32:] = torch.randn_like(changed_value[:, :, 32:]) * 32

    with torch.no_grad():
        original = piper_attention(query, key, value, is_causal=True)
        changed = piper_attention(query, key, changed_value, is_causal=True)

    torch.testing.assert_close(
        original[:, :, :32],
        changed[:, :, :32],
        atol=0.0,
        rtol=0.0,
    )


def test_large_value_scale_multiplier_remains_finite() -> None:
    query = torch.ones((1, 1, 64, 64), device="cuda", dtype=torch.float16)
    key = torch.ones_like(query)
    key[:, :, 0] = -1
    value = torch.ones_like(query)
    value[:, :, 0] = 40000
    plan = _default_piper_attention_execution_plan(query, key, False)

    with torch.no_grad():
        prepared = _prepare_piper_attention(
            query,
            key,
            value,
            64**-0.5,
            False,
            execution_plan=plan,
        )
        actual = _launch_piper_attention(prepared)

    assert prepared.value_scale_multiplier.dtype is torch.float32
    assert torch.isfinite(prepared.value_scale_multiplier).all()
    assert torch.isfinite(actual).all()


def test_biased_value_quality() -> None:
    torch.manual_seed(57)
    sequence = 1024
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    offset = torch.linspace(-8, 8, 128, device="cuda").reshape(1, 1, 1, 128)
    value = (offset + torch.randn_like(query.float()) * 0.25).to(torch.bfloat16)

    with torch.no_grad():
        actual = piper_attention(query, key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    mse = (actual.float() - expected.float()).square().mean()
    assert mse < 1e-3


@pytest.mark.skipif(not _sm120_available(), reason="tensor descriptors target SM12x")
def test_long_descriptor_path_matches_pointer_path() -> None:
    torch.manual_seed(59)
    sequence = 8192
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False)
    descriptor_plan = replace(
        _default_piper_attention_execution_plan(query, key, False),
        native_uint8=True,
    )
    pointer_plan = replace(
        descriptor_plan,
        use_tensor_descriptors=False,
        num_stages=3,
    )

    with torch.no_grad():
        descriptor = _run_piper_attention(
            *arguments,
            execution_plan=descriptor_plan,
        )
        pointer = _run_piper_attention(
            *arguments,
            execution_plan=pointer_plan,
        )

    torch.testing.assert_close(descriptor, pointer, atol=2**-9, rtol=0.0)


def test_explicit_execution_plan_runs_native_loop_controls() -> None:
    torch.manual_seed(61)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    production_plan = _default_piper_attention_execution_plan(query, key, True)
    alternate_plan = replace(
        production_plan,
        block_m=32,
        num_stages=2,
        use_tensor_descriptors=False,
        reverse_causal_blocks=True,
        loop_num_stages=2,
        disable_loop_licm=False,
    )

    with torch.no_grad():
        actual = _run_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            True,
            execution_plan=alternate_plan,
        )
        expected = reference_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            True,
            qk_quantization=_qk_quantization(),
        )
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


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
        return -piper_attention(query, key, value)

    with torch.no_grad():
        expected = consumer(query, key, value)
        actual = torch.compile(consumer, fullgraph=True)(query, key, value)

    assert torch.equal(actual, expected)
