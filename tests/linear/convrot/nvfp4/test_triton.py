"""Tests for exact dynamic ConvRot NVFP4 preparation."""

import pytest
import torch

from piper_kernels.linear._input_activations import apply_input_activation
from piper_kernels.linear.convrot import triton as convrot_backend
from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _materialized_reference(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flattened = input.reshape(-1, input.shape[-1]).contiguous()
    rotated = torch.empty_like(flattened)
    convrot_backend.rotate_input(
        flattened,
        rotated,
        group_size,
        num_warps=4,
    )
    return nvfp4_ops._compiled_prepare_dynamic(rotated, None)


def _materialized_projected_swiglu(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    source_global_scale: torch.Tensor,
    source_bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    projected = (
        nvfp4_ops._compiled_scale_result(input, source_global_scale)
        if source_bias is None
        else nvfp4_ops._compiled_scale_result_and_add_bias(
            input,
            source_global_scale,
            source_bias,
        )
    )
    activated = apply_input_activation(projected, "swiglu")
    flattened = activated.reshape(-1, activated.shape[-1]).contiguous()
    rotated = torch.empty_like(flattened)
    convrot_backend.rotate_input(flattened, rotated, group_size, num_warps=4)
    return rotated


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
    assert convrot_nvfp4._rotation_chunk_sizes(input_features, 16) == expected


def test_rotation_chunk_sizes_reject_widths_beyond_three_chunks() -> None:
    with pytest.raises(ValueError, match="exceeds three"):
        convrot_nvfp4._rotation_chunk_sizes(49_168, 16)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("group_size", "input_features", "dtype"),
    [
        (16, 5_376, torch.float16),
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

    expected = _materialized_reference(input, group_size)
    actual = convrot_nvfp4.prepare_dynamic(input, group_size)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("input_features", [13_824, 15_360])
def test_padded_chunks_match_materialized_rotation_exactly(input_features: int) -> None:
    torch.manual_seed(604 + input_features)
    input = torch.randn(  # noqa: A001
        (3, input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )

    expected = _materialized_reference(input, 16)
    actual = convrot_nvfp4.prepare_dynamic(input, 16)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
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
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_preparation_preserves_noncontiguous_logical_order() -> None:
    torch.manual_seed(602)
    input = torch.randn(  # noqa: A001
        (2, 3, 5_376),
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(0, 1)
    assert not input.is_contiguous()

    expected = _materialized_reference(input, 16)
    actual = convrot_nvfp4.prepare_dynamic(input, 16)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_preparation_matches_zero_input_reference() -> None:
    input = torch.zeros((130, 256), device="cuda", dtype=torch.bfloat16)  # noqa: A001

    expected = _materialized_reference(input, 16)
    actual = convrot_nvfp4.prepare_dynamic(input, 16)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
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
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dynamic_preparation_runs_existing_nvfp4_gemm() -> None:
    torch.manual_seed(603)
    input = torch.randn((257, 256), device="cuda", dtype=torch.bfloat16)  # noqa: A001
    weight = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    rotated_weight = rotate_groups(weight, 16)
    weight_qdata, weight_scale, weight_per_tensor_scale = nvfp4_ops._compiled_prepare_dynamic(
        rotated_weight, None
    )

    expected = nvfp4_ops._execute_prepared(
        *_materialized_reference(input, 16),
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
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_projected_swiglu_preparation_matches_materialized_reference(
    group_size: int,
    with_bias: bool,
) -> None:
    torch._dynamo.reset()
    torch.manual_seed(606 + group_size + with_bias)
    rows, intermediate_features = 17, 256
    input = torch.randn(  # noqa: A001
        (rows, 2 * intermediate_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_global_scale = torch.tensor(0.01, device="cuda", dtype=torch.float32)
    source_bias = (
        torch.randn(2 * intermediate_features, device="cuda", dtype=input.dtype)
        if with_bias
        else None
    )
    rotated = _materialized_projected_swiglu(
        input,
        source_global_scale,
        source_bias,
        group_size,
    )

    expected_dynamic = nvfp4_ops._compiled_prepare_dynamic(rotated, None)
    actual_per_tensor_scale = convrot_nvfp4.projected_swiglu_dynamic_scale(
        input,
        source_global_scale,
        source_bias,
        group_size,
    )
    actual_qdata, actual_scale = convrot_nvfp4.prepare_static_projected_swiglu(
        input,
        actual_per_tensor_scale,
        source_global_scale,
        source_bias,
        group_size,
    )
    for actual_tensor, expected_tensor in zip(
        (actual_qdata, actual_scale, actual_per_tensor_scale),
        expected_dynamic,
        strict=True,
    ):
        assert torch.equal(actual_tensor, expected_tensor)

    static_scale = torch.tensor(1.0 / 448.0, device="cuda", dtype=torch.float32)
    expected_static = nvfp4_ops._compiled_prepare_static(rotated, static_scale, None)
    actual_static = convrot_nvfp4.prepare_static_projected_swiglu(
        input,
        static_scale,
        source_global_scale,
        source_bias,
        group_size,
    )
    for actual_tensor, expected_tensor in zip(
        actual_static,
        expected_static[:2],
        strict=True,
    ):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
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
        expected = nvfp4_ops._compiled_prepare_static(rotated, per_tensor_scale, None)
        actual = convrot_nvfp4.prepare_static(
            input,
            per_tensor_scale,
            group_size,
            "swiglu",
        )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


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
