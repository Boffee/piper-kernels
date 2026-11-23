"""Tests for exact dynamic ConvRot NVFP4 preparation."""

import pytest
import torch
from _accelerators import exact_sm120_available
from torch.nn import functional as F  # noqa: N812
from torch.utils._python_dispatch import TorchDispatchMode
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)

from piper_kernels._input_activations import apply_input_activation
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.weights.convrot import triton as convrot_backend
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.nvfp4 import triton as convrot_nvfp4_primitives
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout


def _materialized_optimized_preparation(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare fused preparation with separate optimized rotation and packing."""
    flattened = input.reshape(-1, input.shape[-1]).contiguous()
    rotated = torch.empty_like(flattened, dtype=torch.float32)
    convrot_backend.rotate_input(
        flattened,
        rotated,
        group_size,
        num_warps=4,
    )
    return nvfp4_ops._compiled_prepare_dynamic(rotated, None)


@pytest.mark.parametrize(
    ("input_features", "expected"),
    [
        (5_376, (4_096, 1_024, 256)),
        (13_824, (8_192, 8_192, 0)),
        (15_360, (8_192, 8_192, 0)),
        (16_640, (16_384, 256, 0)),
        (49_152, (16_384, 16_384, 16_384)),
    ],
)
def test_rotation_chunk_sizes_support_general_aligned_widths(
    input_features: int,
    expected: tuple[int, int, int],
) -> None:
    assert convrot_nvfp4_primitives._rotation_chunk_sizes(input_features, 16) == expected


def test_rotation_chunk_sizes_reject_widths_beyond_three_chunks() -> None:
    with pytest.raises(ValueError, match="exceeds three"):
        convrot_nvfp4_primitives._rotation_chunk_sizes(49_168, 16)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("group_size", "input_features", "dtype"),
    [
        (16, 5_376, torch.float16),
        (16, 5_376, torch.float32),
        (256, 5_376, torch.float32),
        (16, 5_376, torch.bfloat16),
        (64, 5_376, torch.bfloat16),
        (256, 5_376, torch.bfloat16),
        (256, 16_640, torch.bfloat16),
    ],
)
def test_dynamic_preparation_matches_materialized_rotation_exactly(
    group_size: int,
    input_features: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(601 + group_size + input_features)
    input = torch.randn(  # noqa: A001
        (2, 65, input_features),
        device="cuda",
        dtype=dtype,
    )

    expected = _materialized_optimized_preparation(input, group_size)
    actual = convrot_nvfp4.prepare_dynamic(input, group_size)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("input_features", [13_824, 15_360])
def test_padded_chunks_match_materialized_rotation_exactly(input_features: int) -> None:
    torch.manual_seed(604 + input_features)
    input = torch.randn(  # noqa: A001
        (3, input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )

    expected = _materialized_optimized_preparation(input, 16)
    actual = convrot_nvfp4.prepare_dynamic(input, 16)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_scale_and_static_stages_match_dynamic_preparation_with_reusable_output() -> None:
    torch.manual_seed(605)
    rows, input_features = 130, 5_376
    input = torch.randn(  # noqa: A001
        (rows, input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    scale_out = torch.empty((), device="cuda", dtype=torch.float32)
    qdata_storage = torch.empty(
        (256, input_features // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    scale_storage = torch.empty(
        nvfp4_layout.scale_shape(256, input_features),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )

    expected = convrot_nvfp4.prepare_dynamic(input, 16)
    per_tensor_scale = convrot_nvfp4.dynamic_scale(input, 16, out=scale_out)
    qdata, scale = convrot_nvfp4.prepare_static_out(
        input,
        per_tensor_scale,
        16,
        (qdata_storage, scale_storage),
    )

    assert per_tensor_scale is scale_out
    assert qdata.data_ptr() == qdata_storage.data_ptr()
    assert scale.data_ptr() == scale_storage.data_ptr()
    for actual_tensor, expected_tensor in zip(
        (qdata, scale, per_tensor_scale),
        expected,
        strict=True,
    ):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_preparation_preserves_noncontiguous_logical_order() -> None:
    torch.manual_seed(602)
    input = torch.randn(  # noqa: A001
        (2, 3, 5_376),
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(0, 1)
    assert not input.is_contiguous()

    expected = _materialized_optimized_preparation(input, 16)
    actual = convrot_nvfp4.prepare_dynamic(input, 16)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_preparation_matches_materialized_zero_input() -> None:
    input = torch.zeros((130, 256), device="cuda", dtype=torch.bfloat16)  # noqa: A001

    expected = _materialized_optimized_preparation(input, 16)
    actual = convrot_nvfp4.prepare_dynamic(input, 16)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_high_first_dynamic_preparation_swaps_only_packed_pairs() -> None:
    torch.manual_seed(606)
    input = torch.randn((65, 5_376), device="cuda", dtype=torch.bfloat16)  # noqa: A001

    low_qdata, low_scale, low_global = convrot_nvfp4.prepare_dynamic(input, 16)
    high_qdata, high_scale, high_global = convrot_nvfp4.prepare_dynamic(
        input,
        16,
        high_first=True,
    )

    assert torch.equal(high_qdata, ((low_qdata & 0x0F) << 4) | (low_qdata >> 4))
    assert torch.equal(high_scale, low_scale)
    assert torch.equal(high_global, low_global)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_preparation_runs_existing_nvfp4_gemm() -> None:
    torch.manual_seed(603)
    input = torch.randn((257, 256), device="cuda", dtype=torch.bfloat16)  # noqa: A001
    weight = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    rotated_weight = rotate_groups(weight, 16)
    weight_qdata, weight_scale, weight_per_tensor_scale = nvfp4_ops._compiled_prepare_dynamic(
        rotated_weight, None
    )

    expected = nvfp4_ops._execute_prepared(
        *_materialized_optimized_preparation(input, 16),
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        None,
        input.dtype,
    )
    actual = nvfp4_ops._execute_prepared(
        *convrot_nvfp4.prepare_dynamic(input, 16),
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        None,
        input.dtype,
    )

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_swiglu_preparation_matches_materialized_activation(
    group_size: int,
    dynamic: bool,
) -> None:
    torch._dynamo.reset()
    torch.manual_seed(607 + group_size + dynamic)
    input = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    activated = apply_input_activation(input, "swiglu")
    rotated = torch.empty_like(activated)
    convrot_backend.rotate_input(activated, rotated, group_size, num_warps=4)

    if dynamic:
        expected = nvfp4_ops._compiled_prepare_dynamic(rotated, None)
        actual = convrot_nvfp4.prepare_dynamic(input, group_size, "swiglu")
    else:
        per_tensor_scale = torch.tensor(1.0 / 448.0, device="cuda", dtype=torch.float32)
        expected = nvfp4_reference.prepare_input(
            rotated,
            per_tensor_scale,
            False,
        )
        actual = convrot_nvfp4.prepare_static(
            input,
            per_tensor_scale,
            group_size,
            "swiglu",
        )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        if actual_tensor.dtype is torch.float32:
            # FP32 SiLU implementations can differ by one ULP in the global amax.
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=3e-7, atol=0)
        else:
            assert torch.equal(actual_tensor, expected_tensor)


def _assert_prepared_equal(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for left, right in zip(actual[:2], expected[:2], strict=True):
        assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-6, atol=0)


class _RejectCopies(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func not in (torch.ops.aten.clone.default, torch.ops.aten._to_copy.default)
        return func(*args, **(kwargs or {}))


def _prepare_convrot_nvfp4_for_test(
    projections: torch.Tensor,
    activation: str,
    per_tensor_scale: torch.Tensor | None,
    dynamic: bool,
    group: int,
    high_first: bool,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dynamic:
        return convrot_nvfp4.prepare_dynamic(
            projections,
            group,
            activation,
            out=out,
            high_first=high_first,
        )
    assert per_tensor_scale is not None
    if out is None:
        return convrot_nvfp4.prepare_static(
            projections,
            per_tensor_scale,
            group,
            activation,
            high_first,
        )
    qdata, scale = convrot_nvfp4.prepare_static_out(
        projections,
        per_tensor_scale,
        group,
        out,
        high_first=high_first,
        activation_fn=activation,
    )
    return qdata, scale, per_tensor_scale


def _dequantize_prepared_nvfp4(
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    high_first: bool,
) -> torch.Tensor:
    qdata, scale, per_tensor_scale = prepared
    if high_first:
        qdata = nvfp4_layout.swap_packed_pairs(qdata)
    return (
        TorchAONVFP4Tensor(qdata, scale, 16, torch.float32, is_swizzled_scales=True).dequantize(
            torch.float32
        )
        * per_tensor_scale
    )


def _assert_prepared_matches_reference(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    high_first: bool,
) -> None:
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-5, atol=0)
    assert actual[0].shape == expected[0].shape
    assert actual[1].shape == expected[1].shape
    assert (actual[0] != expected[0]).float().mean().item() < 1e-3
    assert (
        actual[1].view(torch.uint8) != expected[1].view(torch.uint8)
    ).float().mean().item() < 1e-3
    actual_dq = _dequantize_prepared_nvfp4(actual, high_first)
    expected_dq = _dequantize_prepared_nvfp4(expected, high_first)
    assert ((actual_dq - expected_dq).norm() / expected_dq.norm()).item() < 5e-3


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("shape", "group", "high_first"),
    [
        ((2, 17, 256), 256, False),
        ((129, 80), 16, False),
        ((129, 5_376), 16, True),
        ((3, 13_824), 64, False),
        ((1_797, 8_192), 256, True),
    ],
)
def test_activation_preparation_matches_fp32_reference(
    activation: str,
    dtype: torch.dtype,
    dynamic: bool,
    shape: tuple[int, ...],
    group: int,
    high_first: bool,
) -> None:
    torch.manual_seed(941)
    if activation == "swiglu":
        gate = torch.randn(shape, device="cuda", dtype=dtype) * 3
        value = torch.randn_like(gate)
        projections = torch.cat((value, gate), dim=-1)
    else:
        projections = torch.randn(shape, device="cuda", dtype=dtype) * 3
    per_tensor_scale = None if dynamic else torch.tensor(0.01, device="cuda")
    with torch.inference_mode():
        expected = nvfp4_reference.prepare_input(
            projections,
            per_tensor_scale,
            dynamic,
            activation,
            high_first,
            group_size=group,
        )
        actual = _prepare_convrot_nvfp4_for_test(
            projections,
            activation,
            per_tensor_scale,
            dynamic,
            group,
            high_first,
        )
    _assert_prepared_matches_reference(actual, expected, high_first)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_activation_preparation_zero_copy_and_overflow_safe(
    activation: str, zero: bool, dynamic: bool
) -> None:
    torch.manual_seed(943)
    width = 1024 if activation == "swiglu" else 512
    workspace = torch.randn(128, width, device="cuda", dtype=torch.float16)
    projections = workspace[:17]
    if zero:
        if activation == "swiglu":
            projections[:, 512:].zero_()
        else:
            projections.zero_()
    before = workspace.clone()
    per_tensor_scale = None if dynamic else torch.tensor(0.01, device="cuda")
    with torch.inference_mode():
        expected = nvfp4_reference.prepare_input(
            projections, per_tensor_scale, dynamic, activation, group_size=256
        )
        with _RejectCopies():
            if dynamic:
                actual = convrot_nvfp4.prepare_dynamic(projections, 256, activation)
            else:
                assert per_tensor_scale is not None
                out = (
                    torch.empty((17, 256), device="cuda", dtype=torch.uint8),
                    torch.empty((32, 128), device="cuda", dtype=torch.float8_e4m3fn),
                )
                qdata, scale = convrot_nvfp4.prepare_static_out(
                    projections,
                    per_tensor_scale,
                    256,
                    out,
                    activation_fn=activation,
                )
                actual = (qdata, scale, per_tensor_scale)
    if zero:
        _assert_prepared_equal(actual, expected)
    else:
        _assert_prepared_matches_reference(actual, expected, False)
    assert torch.equal(workspace, before)
    assert torch.isfinite(actual[2])

    # Verify FP32 intermediates prevent FP16 product overflow across tile counts.
    if activation == "swiglu" and not zero and dynamic:
        for rows, cols in ((17, 256), (1_025, 8_192)):
            gate = torch.full((rows, cols), 512.0, dtype=torch.float16, device="cuda")
            value = torch.full_like(gate, 512.0)
            assert not torch.isfinite(F.silu(gate) * value).all()
            with torch.inference_mode():
                cat = torch.cat((value, gate), dim=-1)
                exp = nvfp4_reference.prepare_input(cat, None, True, "swiglu", group_size=256)
                act = convrot_nvfp4.prepare_dynamic(cat, 256, "swiglu")
            _assert_prepared_equal(act, exp)
            assert torch.isfinite(act[1].float()).all()
            assert torch.isfinite(act[2])
            assert act[2] > 0


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("shape", "group", "high_first"),
    [
        ((17, 80), 16, False),  # non-64-aligned width & padded rows
        ((1_025, 8_192), 256, True),  # multi-tile width & padded rows
    ],
)
def test_activation_preparation_reuses_dirty_out_storage(
    activation: str,
    dynamic: bool,
    shape: tuple[int, int],
    group: int,
    high_first: bool,
) -> None:
    torch.manual_seed(945)
    rows, features = shape
    input_width = features * (2 if activation == "swiglu" else 1)
    projections = torch.randn(rows, input_width, device="cuda", dtype=torch.bfloat16)
    per_tensor_scale = None if dynamic else torch.tensor(0.01, device="cuda")
    scale_elements = ((rows + 127) // 128 * 32) * ((features + 63) // 64 * 16)
    qdata_buffer = torch.full(
        (rows + 8, features // 2),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    scale_buffer = torch.full(
        (scale_elements + 32,),
        0x55,
        device="cuda",
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)

    with torch.inference_mode():
        expected = _prepare_convrot_nvfp4_for_test(
            projections,
            activation,
            per_tensor_scale,
            dynamic,
            group,
            high_first,
        )
        actual = _prepare_convrot_nvfp4_for_test(
            projections,
            activation,
            per_tensor_scale,
            dynamic,
            group,
            high_first,
            out=(qdata_buffer, scale_buffer),
        )
    assert actual[0].data_ptr() == qdata_buffer.data_ptr()
    assert actual[1].data_ptr() == scale_buffer.data_ptr()
    _assert_prepared_equal(actual, expected)

    bad_qdata = torch.empty((rows, features // 2 + 1), device="cuda", dtype=torch.uint8)
    untouched_scale = torch.full(
        (scale_elements,),
        0x55,
        device="cuda",
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="NVFP4 activation storage is incompatible"):
        _prepare_convrot_nvfp4_for_test(
            projections,
            activation,
            per_tensor_scale,
            dynamic,
            group,
            high_first,
            out=(bad_qdata, untouched_scale),
        )
    assert torch.all(untouched_scale.view(torch.uint8) == 0x55)

    if not dynamic:
        bad_scale = torch.empty((1,), device="cuda", dtype=torch.float32)
        with pytest.raises(ValueError, match="contiguous FP32 scalar"):
            _prepare_convrot_nvfp4_for_test(
                projections,
                activation,
                bad_scale,
                False,
                group,
                high_first,
                out=(qdata_buffer, untouched_scale),
            )
        assert torch.all(untouched_scale.view(torch.uint8) == 0x55)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("activation_fn", [None, "gelu_tanh", "swiglu"])
@pytest.mark.parametrize("group_size", [16, 256])
def test_dynamic_scale_with_activation_matches_fp32_reference(
    activation_fn: str | None,
    group_size: int,
) -> None:
    torch.manual_seed(609 + group_size)
    width = 1024 if activation_fn == "swiglu" else 512
    input = torch.randn(17, width, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    expected = nvfp4_reference.prepare_input(
        input,
        None,
        True,
        activation_fn,
        group_size=group_size,
    )[2]
    actual = convrot_nvfp4.dynamic_scale(input, group_size, activation_fn=activation_fn)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=0)

    bad_out = torch.empty((1,), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous FP32 scalar"):
        convrot_nvfp4.dynamic_scale(input, group_size, activation_fn=activation_fn, out=bad_out)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("activation_fn", [None, "gelu_tanh"])
def test_dense_preparation_supports_widths_beyond_three_gguf_chunks(
    activation_fn: str | None,
) -> None:
    torch.manual_seed(610)
    input = torch.randn((2, 57_344), device="cuda", dtype=torch.bfloat16)  # noqa: A001
    actual = convrot_nvfp4.prepare_dynamic(input, 256, activation_fn)
    if activation_fn is None:
        expected = _materialized_optimized_preparation(input, 256)
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            assert torch.equal(actual_tensor, expected_tensor)
    else:
        expected = nvfp4_reference.prepare_input(
            input,
            None,
            True,
            activation_fn,
            group_size=256,
        )
        _assert_prepared_matches_reference(actual, expected, False)


@pytest.mark.parametrize("group_size", [15, 32])
def test_dynamic_preparation_rejects_unsupported_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="group size"):
        convrot_nvfp4.prepare_dynamic(
            torch.empty((1, 64), dtype=torch.bfloat16),
            group_size,
        )


def test_dynamic_preparation_rejects_empty_feature_dimension() -> None:
    with pytest.raises(ValueError, match="nonempty feature dimension"):
        convrot_nvfp4.prepare_dynamic(
            torch.empty((1, 0), dtype=torch.bfloat16),
            16,
        )
