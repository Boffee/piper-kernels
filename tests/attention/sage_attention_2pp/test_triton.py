"""GPU tests for the pure-Triton SageAttention2++ backend."""

from dataclasses import replace
from typing import Literal

import pytest
import torch
import triton
import triton.language as tl
from lib.triton_inspection import compiled_artifact

import piper_kernels.attention.sage_attention_2pp.triton as sage_attention_2pp_backend
from piper_kernels import sage_attention_2pp
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage_attention_2pp import _policy as sage_attention_2pp_policy
from piper_kernels.attention.sage_attention_2pp.reference import reference_sage_attention_2pp
from piper_kernels.attention.sage_attention_2pp.triton import (
    _ptx_float32_to_e4m3x4,
    _run_sage_attention_2pp,
)


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _fp8_gpu_available() -> bool:
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).supports_fp8_fp16_mma
    )


def _qk_quantization() -> Literal["per_thread", "per_warp"]:
    return "per_warp" if torch.cuda.get_device_capability()[0] == 12 else "per_thread"


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _fp8_gpu_available(),
        reason="requires NVIDIA FP8 tensor cores with FP16 accumulation",
    ),
]


@triton.jit
def _stock_fp8_conversion_kernel(input_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, values.to(tl.float8e4nv))


@triton.jit
def _packed_fp8_conversion_kernel(input_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, _ptx_float32_to_e4m3x4(values))


def test_packed_fp8_conversion_matches_stock_triton() -> None:
    values = torch.linspace(-448.0, 448.0, 256, device="cuda", dtype=torch.float32)
    edge_values = torch.tensor(
        [
            -448.0,
            -447.0,
            -256.0,
            -1.0625,
            -1.0,
            -0.9375,
            -0.015625,
            -0.001953125,
            -0.0,
            0.0,
            0.001953125,
            0.015625,
            0.9375,
            1.0,
            1.0625,
            256.0,
            447.0,
            448.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    values[: edge_values.numel()] = edge_values
    stock = torch.empty(256, device="cuda", dtype=torch.float8_e4m3fn)
    packed = torch.empty_like(stock)

    _stock_fp8_conversion_kernel[(1,)](values, stock, num_warps=4)
    _packed_fp8_conversion_kernel[(1,)](values, packed, num_warps=4)
    torch.cuda.synchronize()

    assert torch.equal(packed.view(torch.uint8), stock.view(torch.uint8))
    ptx = compiled_artifact(_packed_fp8_conversion_kernel, "ptx")
    assert ptx.count("cvt.rn.satfinite.e4m3x2.f32") == 2


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
    reason="raw-score reduction before scaling is tuned for SM120",
)
@pytest.mark.parametrize(
    ("head_dim", "is_causal", "fuse_query"),
    [(64, False, False), (64, True, False), (128, False, True), (128, True, True)],
)
def test_intrinsic_grouped_raw_score_reduction_matches_quantized_reference(
    head_dim: int,
    is_causal: bool,
    fuse_query: bool,
) -> None:
    plan = sage_attention_2pp_policy.select_execution_plan(
        AcceleratorTarget(backend="cuda", architecture="sm120"),
        candidate_block_m=64,
        head_dim=head_dim,
        is_causal=is_causal,
    )
    assert plan.grouped_qk
    assert plan.fuse_query_quantization is fuse_query
    torch.manual_seed(432)
    query = torch.randn(1, 2, 193, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = _run_sage_attention_2pp(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
        )
        expected = reference_sage_attention_2pp(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            qk_quantization="per_warp",
        )
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


@pytest.mark.parametrize("is_causal", [False, True])
def test_sm89_d128_schedule_runs_at_short_sequence(is_causal: bool) -> None:
    plan = sage_attention_2pp_policy.select_execution_plan(
        AcceleratorTarget(backend="cuda", architecture="sm89"),
        candidate_block_m=128,
        head_dim=128,
        is_causal=is_causal,
    )
    torch.manual_seed(433 + is_causal)
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
            execution_plan=plan,
        )
        expected = reference_sage_attention_2pp(
            query,
            key,
            value,
            128**-0.5,
            is_causal,
            qk_quantization="per_thread",
        )
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


@pytest.mark.skipif(
    not _sm120_available(),
    reason="alternate execution-plan candidates are validated on SM120",
)
def test_explicit_execution_plan_runs_alternate_tuning_candidate() -> None:
    torch.manual_seed(433)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    production_plan = sage_attention_2pp_backend._default_sage_attention_2pp_execution_plan(
        query, False
    )
    alternate_plan = replace(
        production_plan,
        block_m=64,
        num_stages=2,
        use_tensor_descriptors=False,
        fuse_query_quantization=False,
        loop_num_stages=3,
        loop_licm=True,
    )

    with torch.no_grad():
        actual = _run_sage_attention_2pp(
            query,
            key,
            value,
            128**-0.5,
            False,
            execution_plan=alternate_plan,
        )
        expected = reference_sage_attention_2pp(
            query,
            key,
            value,
            128**-0.5,
            False,
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
    reason="tensor-descriptor schedule is tuned for SM120",
)
def test_triton_ragged_descriptor_storage_matches_pointer_path() -> None:
    torch.manual_seed(441)
    query = torch.randn(1, 24, 1025, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    pointer_plan = replace(
        sage_attention_2pp_backend._default_sage_attention_2pp_execution_plan(
            query,
            False,
        ),
        use_tensor_descriptors=False,
    )

    with torch.no_grad():
        actual = sage_attention_2pp(query, key, value)
        expected = _run_sage_attention_2pp(
            query,
            key,
            value,
            128**-0.5,
            False,
            execution_plan=pointer_plan,
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
