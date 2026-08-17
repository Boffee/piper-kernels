"""GPU tests for the pure-Triton Piper Attention backend."""

import math
from dataclasses import replace
from typing import Literal

import pytest
import torch
import triton
import triton.language as tl
from lib.triton_inspection import compiled_artifact
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels import piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _policy
from piper_kernels.attention.piper_attention.reference import reference_piper_attention
from piper_kernels.attention.piper_attention.triton import (
    _conservative_value_log_scale_bound,
    _default_piper_attention_execution_plan,
    _launch_piper_attention,
    _prepare_piper_attention,
    _ptx_float32_to_uint8x4,
    _run_piper_attention,
)


def _piper_gpu_available() -> bool:
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).supports_uint8_int8_mma
    )


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _qk_quantization() -> Literal["per_thread", "per_warp"]:
    return "per_warp" if torch.cuda.get_device_capability()[0] == 12 else "per_thread"


def _sqnr_db(actual: torch.Tensor, reference: torch.Tensor) -> float:
    reference_float = reference.float()
    error_float = actual.float() - reference_float
    signal_energy = torch.sum(reference_float.square())
    error_energy = torch.sum(error_float.square())
    return float(10 * torch.log10(signal_energy / error_energy))


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _piper_gpu_available(),
        reason="requires NVIDIA SM8x or consumer Blackwell SM12x mixed-sign MMAv2",
    ),
]


@triton.jit
def _stock_uint8_conversion_kernel(input_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    values = tl.load(input_ptr + offsets)
    codes = tl.minimum(255.0, values).to(tl.int32)
    tl.store(output_ptr + offsets, codes.to(tl.uint8))


@triton.jit
def _packed_uint8_conversion_kernel(input_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, _ptx_float32_to_uint8x4(values))


@triton.jit
def _value_log_scale_bound_kernel(input_ptr, output_ptr, elements: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < elements
    multipliers = tl.load(input_ptr + offsets, mask=mask)
    bounds = _conservative_value_log_scale_bound(multipliers)
    tl.store(output_ptr + offsets, bounds, mask=mask)


def test_derived_value_log_scale_is_conservative_for_all_fp32_mantissas() -> None:
    mantissa_count = 1 << 23
    mantissas = torch.arange(mantissa_count, device="cuda", dtype=torch.int32)
    for exponent in (1, 2, 126, 127, 128, 191, 253, 254):
        multipliers = (mantissas | (exponent << 23)).view(torch.float32)
        bounds = torch.empty_like(multipliers)

        _value_log_scale_bound_kernel[(triton.cdiv(mantissa_count, 256),)](
            multipliers,
            bounds,
            elements=mantissa_count,
            num_warps=4,
        )
        # The multiplier is the authoritative rounded FP32 metadata consumed
        # by the kernel, so this is the exact scale it can safely reconstruct.
        exact = torch.log2(multipliers) - math.log2(255.0)
        gap = bounds - exact

        assert gap.min().item() >= 0.0
        assert gap.max().item() < 0.087


@pytest.mark.parametrize("round_probability_codes", [False, True])
def test_packed_uint8_conversion_matches_stock_triton(
    round_probability_codes: bool,
) -> None:
    values = torch.linspace(0.0, 300.0, 256, device="cuda", dtype=torch.float32)
    edge_values = torch.tensor(
        [
            0.0,
            0.49999997,
            0.5,
            0.99999994,
            1.0,
            1.4999999,
            1.5,
            127.49999,
            127.5,
            254.49998,
            254.5,
            254.99998,
            255.0,
            255.49998,
            256.0,
            300.0,
            # PTX clamps finite FP32-to-S32 overflow before the saturated pack.
            2147483648.0,
            4294967296.0,
            1.0e20,
            torch.finfo(torch.float32).max,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    values[: edge_values.numel()] = edge_values
    if round_probability_codes:
        values += 0.5
    stock = torch.empty(256, device="cuda", dtype=torch.uint8)
    packed = torch.empty_like(stock)

    _stock_uint8_conversion_kernel[(1,)](values, stock, num_warps=4)
    _packed_uint8_conversion_kernel[(1,)](values, packed, num_warps=4)
    torch.cuda.synchronize()

    assert torch.equal(packed, stock)
    ptx = compiled_artifact(_packed_uint8_conversion_kernel, "ptx")
    assert ptx.count("cvt.rzi.s32.f32") == 4
    assert ptx.count("cvt.pack.sat.u8.s32.b32") == 2


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


@pytest.mark.parametrize("head_dim", [64, 128])
def test_aligned_rectangular_key_tiles_do_not_require_square_attention(head_dim: int) -> None:
    torch.manual_seed(75 + head_dim)
    query = torch.randn(1, 2, 128, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 64, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)

    with torch.no_grad():
        actual = piper_attention(query, key, value)
        expected = reference_piper_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            False,
            qk_quantization=_qk_quantization(),
        )
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.003
    assert error.max().item() < 0.12


@pytest.mark.parametrize("is_causal", [False, True])
def test_packed_probability_conversion_matches_stock_attention(
    is_causal: bool,
) -> None:
    torch.manual_seed(63 + is_causal)
    query = torch.randn(1, 1, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    plan = replace(
        _default_piper_attention_execution_plan(query, is_causal),
        use_tensor_descriptors=False,
        num_stages=3,
    )
    arguments = (query, key, value, 128**-0.5, is_causal)

    with torch.no_grad():
        stock = _run_piper_attention(
            *arguments,
            execution_plan=replace(
                plan,
                use_packed_probability_conversion=False,
            ),
        )
        packed = _run_piper_attention(
            *arguments,
            execution_plan=replace(
                plan,
                use_packed_probability_conversion=True,
            ),
        )

    assert torch.equal(packed, stock)


@pytest.mark.parametrize("sequence", [63, 64, 65, 127, 128, 129, 193])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("block_m", [64, 128])
def test_optimized_causal_traversal_matches_fully_masked_loop(
    sequence: int,
    head_dim: int,
    block_m: int,
) -> None:
    torch.manual_seed(74 + sequence + head_dim)
    query = torch.randn(1, 1, sequence, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    base_plan = replace(
        _default_piper_attention_execution_plan(query, True),
        block_m=block_m,
    )
    arguments = (query, key, value, head_dim**-0.5, True)

    with torch.no_grad():
        masked = _run_piper_attention(
            *arguments,
            execution_plan=replace(base_plan, optimize_causal_traversal=False),
        )
        partitioned = _run_piper_attention(
            *arguments,
            execution_plan=replace(base_plan, optimize_causal_traversal=True),
        )

    # The two traversal orders can straddle one final BF16 rounding boundary.
    torch.testing.assert_close(partitioned, masked, atol=2**-9, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sequence", [193, 8192, 32768])
def test_sm89_d128_measured_schedule_clears_relative_quality_gate(
    sequence: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(68)
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    target = AcceleratorTarget(backend="cuda", architecture="sm89")
    specialized_plan = _default_piper_attention_execution_plan(
        query,
        False,
        target=target,
    )
    generic_plan = _policy._generic_execution_plan(
        target,
        head_dim=128,
        is_causal=False,
    )

    assert specialized_plan.split_pv_head_dim
    assert specialized_plan.use_packed_probability_conversion
    with torch.no_grad():
        specialized = _run_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            False,
            execution_plan=specialized_plan,
        )
        generic = _run_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            False,
            execution_plan=generic_plan,
        )
        reference = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    assert torch.isfinite(specialized).all()
    assert _sqnr_db(specialized, reference) >= _sqnr_db(generic, reference) - 0.5


def test_derived_value_log_bound_skips_log_metadata() -> None:
    torch.manual_seed(71)
    query = torch.randn(1, 1, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 1, 257, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    target = AcceleratorTarget(backend="cuda", architecture="sm89")
    base_plan = _default_piper_attention_execution_plan(
        query,
        False,
        target=target,
    )
    derived_plan = replace(base_plan, derive_value_log_bound=True)

    with torch.no_grad():
        prepared = _prepare_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            False,
            execution_plan=derived_plan,
        )
        output = _launch_piper_attention(prepared).clone()

    assert prepared.value_log_scale.numel() == 1
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_derived_value_log_bound_runs_all_common_modes(
    dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
) -> None:
    torch.manual_seed(72)
    query_length = 193
    key_length = query_length if is_causal else 257
    query = torch.randn(1, 1, query_length, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(1, 1, key_length, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    base_plan = _default_piper_attention_execution_plan(query, is_causal)
    stored_plan = replace(base_plan, derive_value_log_bound=False)
    derived_plan = replace(base_plan, derive_value_log_bound=True)

    with torch.no_grad():
        stored_prepared = _prepare_piper_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            execution_plan=stored_plan,
        )
        derived_prepared = _prepare_piper_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            execution_plan=derived_plan,
        )
        stored = _launch_piper_attention(stored_prepared).clone()
        derived = _launch_piper_attention(derived_prepared).clone()
        reference = reference_piper_attention(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            qk_quantization=_qk_quantization(),
        )

    assert stored_prepared.value_log_scale.numel() == key_length
    assert derived_prepared.value_log_scale.numel() == 1
    assert torch.isfinite(derived).all()
    assert _sqnr_db(derived, reference) >= 40.0
    assert _sqnr_db(derived, stored) >= 35.0


def test_centered_value_fusion_restores_constant_value() -> None:
    torch.manual_seed(56)
    query = torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value_row = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.bfloat16)
    value = value_row.expand_as(query).contiguous()

    with torch.no_grad():
        actual = piper_attention(query, key, value)

    torch.testing.assert_close(actual, value, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("optimize_causal_traversal", [False, True])
def test_causal_triton_is_independent_of_future_value_rows(
    optimize_causal_traversal: bool,
) -> None:
    torch.manual_seed(62)
    query = torch.randn(1, 1, 65, 64, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    changed_value = value.clone()
    changed_value[:, :, 32:] = torch.randn_like(changed_value[:, :, 32:]) * 32
    plan = replace(
        _default_piper_attention_execution_plan(query, True),
        optimize_causal_traversal=optimize_causal_traversal,
    )

    with torch.no_grad():
        original = _run_piper_attention(
            query,
            key,
            value,
            64**-0.5,
            True,
            execution_plan=plan,
        )
        changed = _run_piper_attention(
            query,
            key,
            changed_value,
            64**-0.5,
            True,
            execution_plan=plan,
        )

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
    plan = _default_piper_attention_execution_plan(query, False)

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


@pytest.mark.skipif(not _sm120_available(), reason="test requires exact SM120 policy")
@pytest.mark.parametrize("derive_value_log_bound", [False, True])
def test_long_descriptor_path_matches_pointer_path(
    derive_value_log_bound: bool,
) -> None:
    torch.manual_seed(59)
    sequence = 8192
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False)
    descriptor_plan = replace(
        _default_piper_attention_execution_plan(query, False),
        derive_value_log_bound=derive_value_log_bound,
    )
    pointer_plan = replace(
        descriptor_plan,
        use_tensor_descriptors=False,
        num_stages=3,
    )

    with torch.no_grad():
        descriptor_prepared = _prepare_piper_attention(
            *arguments,
            execution_plan=descriptor_plan,
        )
        descriptor = _launch_piper_attention(descriptor_prepared)
        pointer_prepared = _prepare_piper_attention(
            *arguments,
            execution_plan=pointer_plan,
        )
        pointer = _launch_piper_attention(pointer_prepared)

    assert isinstance(descriptor_prepared.query, torch.Tensor)
    assert isinstance(descriptor_prepared.query_descriptor, TensorDescriptor)
    assert descriptor_prepared.query_descriptor.block_shape == [1, 128, 128]
    assert pointer_prepared.query_descriptor is None
    torch.testing.assert_close(descriptor, pointer, atol=2**-9, rtol=0.0)


@pytest.mark.skipif(not _sm120_available(), reason="test requires exact SM120 policy")
def test_ragged_query_tail_falls_back_to_masked_pointer_load() -> None:
    torch.manual_seed(76)
    sequence = 129
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    plan = _default_piper_attention_execution_plan(query, False)

    with torch.no_grad():
        prepared = _prepare_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            False,
            execution_plan=plan,
        )
        descriptor = _launch_piper_attention(prepared).clone()
        pointer = _launch_piper_attention(replace(prepared, query_descriptor=None))

    assert isinstance(prepared.query_descriptor, TensorDescriptor)
    assert prepared.query_descriptor.shape == [1, sequence, 128]
    assert torch.isfinite(descriptor).all()
    torch.testing.assert_close(descriptor, pointer, atol=2**-9, rtol=0.0)


def test_explicit_execution_plan_runs_native_loop_controls() -> None:
    torch.manual_seed(61)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    production_plan = _default_piper_attention_execution_plan(query, True)
    alternate_plan = replace(
        production_plan,
        block_m=64,
        num_stages=2,
        use_tensor_descriptors=False,
        optimize_causal_traversal=True,
        loop_num_stages=2,
        loop_licm=True,
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
