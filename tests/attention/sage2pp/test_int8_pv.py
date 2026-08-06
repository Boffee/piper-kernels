"""GPU tests for the fixed- and block-scaled signed-INT8 PV baselines."""

from collections.abc import Callable

import pytest
import torch

from piper_kernels.attention._sage2pp.experiments import (
    triton_sage_attention_int8_pv,
    triton_sage_attention_int8_pv_block_scaled,
    triton_sage_attention_int8_pv_convrot_rms,
    triton_sage_attention_int8_pv_per_key_log,
    triton_sage_attention_uint8_pv_bucketed_grouped,
    triton_sage_attention_uint8_pv_feature_convrot,
    triton_sage_attention_uint8_pv_int32_recurrence,
)
from piper_kernels.attention._sage2pp.experiments.int8_pv import (
    _launch_uint8_grouped_output_pv_attention,
    _prepare_uint8_grouped_output_pv_inputs,
)
from piper_kernels.attention._sage2pp.experiments.uint8_pv_feature_convrot import (
    _launch_uint8_pv_feature_convrot_attention,
    _prepare_uint8_pv_feature_convrot_inputs,
)

Int8Attention = Callable[..., torch.Tensor]


def _consumer_gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability == (8, 9) or capability[0] == 12


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _consumer_gpu_available(),
        reason="requires consumer Ada SM89 or Blackwell SM12x",
    ),
]


@pytest.mark.parametrize(
    "implementation",
    [triton_sage_attention_int8_pv, triton_sage_attention_int8_pv_block_scaled],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_int8_pv_baseline_is_close_to_exact_attention(
    implementation: Int8Attention,
    head_dim: int,
    is_causal: bool,
) -> None:
    torch.manual_seed(81)
    query = torch.randn(1, 2, 193, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = implementation(
            query,
            key,
            value,
            head_dim**-0.5,
            is_causal,
            grouped_qk=False,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
        )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.005
    assert error.max().item() < 0.1


@pytest.mark.parametrize("is_causal", [False, True])
def test_fixed_int8_split_pv_is_close_to_exact_attention(is_causal: bool) -> None:
    torch.manual_seed(811)
    query = torch.randn(1, 2, 192, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    with torch.no_grad():
        actual = triton_sage_attention_int8_pv(
            query,
            key,
            value,
            128**-0.5,
            is_causal,
            grouped_qk=False,
            split_pv_head_dim=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
        )
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.005
    assert error.max().item() < 0.1


@pytest.mark.skipif(
    not _sm120_available(),
    reason="predicate-free M128 split path is currently tuned for SM12x",
)
def test_fixed_int8_full_tile_long_context_is_close_to_exact_attention() -> None:
    torch.manual_seed(812)
    query = torch.randn(1, 1, 8192, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_int8_pv(
            query,
            key,
            value,
            128**-0.5,
            False,
            grouped_qk=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
        )
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.005
    assert error.max().item() < 0.1


@pytest.mark.skipif(
    not _sm120_available(),
    reason="native mixed-sign fixed-PV path is currently tuned for SM12x",
)
def test_fixed_native_uint8_probability_path_is_close_to_exact_attention() -> None:
    """Exercise native UINT8 P with fixed-scale signed-INT8 V."""
    torch.manual_seed(813)
    query = torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_int8_pv(
            query,
            key,
            value,
            128**-0.5,
            False,
            grouped_qk=True,
            native_unsigned_probability=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.005
    assert error.max().item() < 0.1


@pytest.mark.parametrize("value_rms_group_tiles", [1, 2])
def test_int8_pv_convrot_rms_is_close_to_exact_attention(
    value_rms_group_tiles: int,
) -> None:
    torch.manual_seed(82)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_int8_pv_convrot_rms(
            query,
            key,
            value,
            128**-0.5,
            False,
            grouped_qk=False,
            value_rms_group_tiles=value_rms_group_tiles,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.006
    assert error.max().item() < 0.2


@pytest.mark.parametrize(
    ("value_scale_axis", "rotation_group", "probability_scale_mode", "value_scale_floor"),
    [
        ("feature", 0, "dynamic", 0.0),
        ("feature", 64, "dynamic", 0.0),
        ("key", 0, "dynamic", 0.0),
        ("key", 0, "log", 0.0),
        ("key", 0, "tile", 0.0),
        ("key", 0, "tile", 0.125),
        ("key", 16, "dynamic", 0.0),
        ("key", 64, "dynamic", 0.0),
        ("key", 64, "log", 0.0),
        ("key", 64, "tile", 0.0),
        ("key", 64, "tile", 0.125),
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_uint8_pv_feature_convrot_is_close_to_exact_attention(
    value_scale_axis: str,
    rotation_group: int,
    probability_scale_mode: str,
    value_scale_floor: float,
    is_causal: bool,
) -> None:
    torch.manual_seed(83)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_feature_convrot(
            query,
            key,
            value,
            128**-0.5,
            is_causal,
            grouped_qk=False,
            rotation_group=rotation_group,
            value_scale_axis=value_scale_axis,
            probability_scale_mode=probability_scale_mode,
            value_scale_floor=value_scale_floor,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
        )
    error = (actual.float() - expected.float()).abs()

    assert actual.shape == query.shape
    assert actual.dtype is query.dtype
    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.005
    assert error.max().item() < 0.1


@pytest.mark.skipif(
    not _sm120_available(),
    reason="native UINT8 MMA path is currently tuned for SM12x",
)
def test_native_per_key_full_tile_path_is_close_to_exact_attention() -> None:
    """Exercise the selected predicate-free SM12x per-key kernel."""
    torch.manual_seed(834)
    query = torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_feature_convrot(
            query,
            key,
            value,
            128**-0.5,
            False,
            native_uint8_mma=True,
            split_pv_head_dim=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.005
    assert error.max().item() < 0.1


@pytest.mark.skipif(
    not _sm120_available(),
    reason="native UINT8 MMA path is currently tuned for SM12x",
)
@pytest.mark.parametrize("sequence", [1025, 16385])
def test_native_per_key_ragged_descriptor_path_is_close_to_exact_attention(
    sequence: int,
) -> None:
    """Cover padded descriptor storage and the long-context query-tail split."""
    torch.manual_seed(835 + sequence)
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_feature_convrot(
            query,
            key,
            value,
            128**-0.5,
            False,
            native_uint8_mma=True,
            split_pv_head_dim=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.005
    assert error.max().item() < 0.1


@pytest.mark.parametrize("is_causal", [False, True])
def test_log_probability_scaling_matches_dynamic(is_causal: bool) -> None:
    torch.manual_seed(84)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, is_causal)

    with torch.no_grad():
        dynamic = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="dynamic",
        )
        log_domain = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
        )
    difference = (dynamic.float() - log_domain.float()).abs()

    assert difference.mean().item() < 0.0005
    assert difference.max().item() < 0.01


@pytest.mark.parametrize("is_causal", [False, True])
def test_scale_forward_recurrence_matches_weighted_log_recurrence(
    is_causal: bool,
) -> None:
    torch.manual_seed(842)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, is_causal)

    with torch.no_grad():
        weighted = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            split_pv_head_dim=True,
        )
        scale_forward = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            split_pv_head_dim=True,
            scale_forward_log_recurrence=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
        )

    weighted_mse = (weighted.float() - expected.float()).square().mean()
    scale_forward_mse = (scale_forward.float() - expected.float()).square().mean()
    assert scale_forward_mse <= weighted_mse * 1.05
    assert (scale_forward.float() - weighted.float()).abs().max().item() < 0.02


def test_scale_forward_automatic_selection_matches_sm120_policy() -> None:
    torch.manual_seed(843)
    query = torch.randn(1, 1, 1024, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False)

    with torch.no_grad():
        automatic = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            split_pv_head_dim=True,
        )
        policy = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            split_pv_head_dim=True,
            scale_forward_log_recurrence=torch.cuda.get_device_capability()[0] == 12,
        )

    torch.testing.assert_close(automatic, policy, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "denominator_options",
    [
        {"tile_common_log_denominator": True},
        {"narrow_int8_log_denominator": True},
    ],
)
def test_approximate_log_denominator_ablation_is_finite(
    denominator_options: dict[str, bool],
) -> None:
    torch.manual_seed(841)
    query = torch.randn(1, 2, 192, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_feature_convrot(
            query,
            key,
            value,
            128**-0.5,
            False,
            grouped_qk=False,
            probability_scale_mode="log",
            **denominator_options,
        )

    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()


def test_signed_log_probability_descriptor_path_is_close_to_exact_attention() -> None:
    torch.manual_seed(85)
    query = torch.randn(1, 24, 1152, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_int8_pv_per_key_log(
            query,
            key,
            value,
            128**-0.5,
            False,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.006
    assert error.max().item() < 0.15


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("tile_exponent", [False, True])
def test_int32_output_recurrence_is_close_to_exact_attention(
    is_causal: bool,
    tile_exponent: bool,
) -> None:
    torch.manual_seed(86)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_int32_recurrence(
            query,
            key,
            value,
            128**-0.5,
            is_causal,
            grouped_qk=False,
            tile_exponent=tile_exponent,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
        )
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.01
    assert error.max().item() < 0.2


@pytest.mark.skipif(
    not _sm120_available(),
    reason="native mixed-sign profiler path is currently tuned for SM12x",
)
def test_dithered_predot_alignment_is_finite_and_close_to_exact_attention() -> None:
    torch.manual_seed(861)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_feature_convrot(
            query,
            key,
            value,
            128**-0.5,
            False,
            grouped_qk=False,
            probability_scale_mode="log",
            native_uint8_mma=True,
            integer_tile_exponent_recurrence=True,
            predot_exponent_alignment=True,
            dithered_predot_alignment=True,
            split_pv_head_dim=True,
            scale_forward_log_recurrence=True,
            optimize_pv_scaling=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.015
    assert error.max().item() < 0.3


@pytest.mark.skipif(
    not _sm120_available(),
    reason="native mixed-sign grouped-output path is currently tuned for SM12x",
)
@pytest.mark.parametrize(("block_n", "feature_group"), [(64, 32), (128, 8)])
def test_grouped_output_scale_kernel_is_close_to_exact_attention(
    block_n: int,
    feature_group: int,
) -> None:
    torch.manual_seed(862)
    query = torch.randn(1, 2, 256, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    prepared = _prepare_uint8_grouped_output_pv_inputs(
        query,
        key,
        value,
        128**-0.5,
        feature_group=feature_group,
        block_n=block_n,
    )
    actual = torch.empty_like(query)

    with torch.no_grad():
        _launch_uint8_grouped_output_pv_attention(
            prepared,
            actual,
            256,
            256,
            feature_group=feature_group,
            block_n=block_n,
            block_m=64,
            num_warps=4,
            num_stages=2,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.004
    assert error.max().item() < 0.05


@pytest.mark.skipif(
    not _sm120_available(),
    reason="bucketed grouped-output path is currently tuned for SM12x",
)
@pytest.mark.parametrize(
    ("integer_output_recurrence", "common_feature_exponent"),
    [(False, False), (True, False), (True, True)],
)
def test_bucketed_grouped_output_attention_is_close_to_exact_attention(
    integer_output_recurrence: bool,
    common_feature_exponent: bool,
) -> None:
    torch.manual_seed(863)
    query = torch.randn(1, 2, 256, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_bucketed_grouped(
            query,
            key,
            value,
            128**-0.5,
            False,
            feature_group=4,
            integer_output_recurrence=integer_output_recurrence,
            common_feature_exponent=common_feature_exponent,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.004
    assert error.max().item() < 0.05


@pytest.mark.skipif(
    not _sm120_available(),
    reason="bucketed grouped-output path is currently tuned for SM12x",
)
@pytest.mark.parametrize(("feature_group", "maxnreg"), [(4, None), (16, 224)])
def test_bucketed_grouped_output_scale_runs_are_close_to_exact_attention(
    feature_group: int,
    maxnreg: int | None,
) -> None:
    torch.manual_seed(864)
    query = torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_bucketed_grouped(
            query,
            key,
            value,
            128**-0.5,
            False,
            feature_group=feature_group,
            scale_run_n=512,
            maxnreg=maxnreg,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.004
    assert error.max().item() < 0.05


@pytest.mark.skipif(
    not _sm120_available(),
    reason="scaled FP16 numerator path is currently tuned for SM12x",
)
def test_bucketed_scaled_fp16_numerator_is_close_to_exact_attention() -> None:
    torch.manual_seed(865)
    query = torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        actual = triton_sage_attention_uint8_pv_bucketed_grouped(
            query,
            key,
            value,
            128**-0.5,
            False,
            feature_group=4,
            block_n=64,
            scale_run_n=512,
            scaled_fp16_numerator=True,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    error = (actual.float() - expected.float()).abs()

    assert torch.isfinite(actual).all()
    assert error.mean().item() < 0.004
    assert error.max().item() < 0.05


@pytest.mark.parametrize("is_causal", [False, True])
def test_split_pv_head_dim_matches_unsplit(is_causal: bool) -> None:
    torch.manual_seed(87)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, is_causal)

    with torch.no_grad():
        unsplit = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
        )
        split = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            split_pv_head_dim=True,
        )

    torch.testing.assert_close(split, unsplit, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("is_causal", [False, True])
def test_paired_int32_tiles_match_single_tiles(is_causal: bool) -> None:
    torch.manual_seed(88)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, is_causal)

    with torch.no_grad():
        single = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
        )
        paired = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            paired_int32_tiles=True,
        )

    torch.testing.assert_close(paired, single, atol=0.004, rtol=0.007)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "variant_options",
    [
        {"probability_fp16": True},
        {"normalized_fp16_recurrence": True},
    ],
)
def test_fp16_recurrence_ablations_match_fp32(
    is_causal: bool,
    variant_options: dict[str, bool],
) -> None:
    torch.manual_seed(89)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, is_causal)

    with torch.no_grad():
        fp32 = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
        )
        reduced = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            **variant_options,
        )

    torch.testing.assert_close(reduced, fp32, atol=0.004, rtol=0.01)


@pytest.mark.skipif(
    not _sm120_available(),
    reason="native mixed-sign FP16-numerator path is currently tuned for SM12x",
)
@pytest.mark.parametrize("scaled_fp16_denominator", [False, True])
@pytest.mark.parametrize("split_pv_head_dim", [False, True])
def test_key_scaled_fp16_numerator_is_close_to_fp32(
    scaled_fp16_denominator: bool,
    split_pv_head_dim: bool,
) -> None:
    torch.manual_seed(891)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False)

    with torch.no_grad():
        fp32 = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            affine_probability=True,
            native_uint8_mma=True,
            split_pv_head_dim=split_pv_head_dim,
            scale_forward_log_recurrence=True,
            optimize_pv_scaling=split_pv_head_dim,
        )
        reduced = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            affine_probability=True,
            native_uint8_mma=True,
            split_pv_head_dim=True,
            scale_forward_log_recurrence=True,
            optimize_pv_scaling=True,
            scaled_fp16_numerator=True,
            scaled_fp16_denominator=scaled_fp16_denominator,
        )
        explicit_metadata_policy = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            probability_scale_mode="log",
            affine_probability=True,
            native_uint8_mma=True,
            split_pv_head_dim=True,
            scale_forward_log_recurrence=True,
            optimize_pv_scaling=True,
            fp32_pv_scale_metadata=False,
            scaled_fp16_numerator=True,
            scaled_fp16_denominator=scaled_fp16_denominator,
        )

    torch.testing.assert_close(reduced, fp32, atol=0.005, rtol=0.012)
    torch.testing.assert_close(reduced, explicit_metadata_policy, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not _sm120_available(),
    reason="affine scaled-FP16 correction path is currently tuned for SM12x",
)
def test_affine_fp16_correction_is_close_to_int32_correction() -> None:
    torch.manual_seed(893)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False)
    common_options = {
        "grouped_qk": False,
        "native_uint8_mma": False,
        "split_pv_head_dim": True,
        "scale_forward_log_recurrence": True,
        "optimize_pv_scaling": True,
        "scaled_fp16_numerator": True,
    }

    with torch.no_grad():
        exact = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            **common_options,
        )
        reduced = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            scaled_fp16_correction=True,
            **common_options,
        )

    torch.testing.assert_close(reduced, exact, atol=0.002, rtol=0.004)


@pytest.mark.skipif(
    not _sm120_available(),
    reason="signed scaled-FP16 numerator path is currently tuned for SM12x",
)
def test_signed_key_scaled_fp16_numerator_is_close_to_fp32() -> None:
    torch.manual_seed(895)
    query = torch.randn(1, 2, 193, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False)
    common_options = {
        "grouped_qk": False,
        "affine_probability": False,
        "split_pv_head_dim": True,
        "scale_forward_log_recurrence": True,
        "optimize_pv_scaling": True,
    }

    with torch.no_grad():
        fp32 = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            **common_options,
        )
        reduced = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            scaled_fp16_numerator=True,
            **common_options,
        )

    torch.testing.assert_close(reduced, fp32, atol=0.004, rtol=0.008)


@pytest.mark.skipif(
    not _sm120_available(),
    reason="delayed affine correction path is currently tuned for SM12x",
)
@pytest.mark.parametrize("correction_group", [8, 16])
def test_delayed_correction_is_close_to_per_tile_correction(
    correction_group: int,
) -> None:
    torch.manual_seed(894)
    sequence = 1024
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    prepared = _prepare_uint8_pv_feature_convrot_inputs(
        query,
        key,
        value,
        128**-0.5,
        grouped_qk=True,
        rotation_group=0,
        value_scale_axis="key",
        value_scale_floor=0.0,
        probability_scale_mode="log",
        affine_probability=True,
        native_uint8_mma=False,
        scaled_fp16_correction=True,
        scale_forward_log_recurrence=True,
        precompute_pv_multiplier=True,
    )
    common_options = {
        "grouped_qk": True,
        "rotation_group": 0,
        "value_scale_axis": "key",
        "probability_scale_mode": "log",
        "fuse_output_rotation": True,
        "block_m": 128,
        "num_warps": 4,
        "num_stages": 2,
        "affine_probability": True,
        "native_uint8_mma": False,
        "factored_pv_scaling": True,
        "precomputed_pv_multiplier": True,
        "scaled_fp16_numerator": True,
        "scaled_fp16_correction": True,
        "split_pv_head_dim": True,
        "scale_forward_log_recurrence": True,
        "unmasked_self_attention": True,
        "use_tensor_descriptors": True,
    }

    with torch.no_grad():
        per_tile_output = torch.empty_like(query)
        per_tile = _launch_uint8_pv_feature_convrot_attention(
            prepared,
            per_tile_output,
            per_tile_output,
            sequence,
            sequence,
            False,
            **common_options,
        )
        delayed_output = torch.empty_like(query)
        delayed = _launch_uint8_pv_feature_convrot_attention(
            prepared,
            delayed_output,
            delayed_output,
            sequence,
            sequence,
            False,
            delayed_fp16_correction_group=correction_group,
            **common_options,
        )

    torch.testing.assert_close(delayed, per_tile, atol=0.003, rtol=0.006)


@pytest.mark.skipif(
    not _sm120_available(),
    reason="M128 split-PV descriptors are currently selected on SM12x",
)
def test_key_scaled_m128_descriptor_matches_pointer_path() -> None:
    torch.manual_seed(892)
    sequence = 8192
    query = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    arguments = (query, key, value, 128**-0.5, False)

    with torch.no_grad():
        descriptor = triton_sage_attention_uint8_pv_feature_convrot(
            *arguments,
            grouped_qk=False,
            native_uint8_mma=True,
            split_pv_head_dim=True,
            scale_forward_log_recurrence=True,
            optimize_pv_scaling=True,
            scaled_fp16_numerator=True,
        )
        prepared = _prepare_uint8_pv_feature_convrot_inputs(
            query,
            key,
            value,
            128**-0.5,
            grouped_qk=False,
            rotation_group=0,
            value_scale_axis="key",
            value_scale_floor=0.0,
            probability_scale_mode="log",
            value_transposed=True,
            affine_probability=True,
            native_uint8_mma=True,
            scale_forward_log_recurrence=True,
            fp32_scale_forward_metadata=False,
            precompute_pv_multiplier=True,
        )
        pointer_output = torch.empty_like(query)
        pointer = _launch_uint8_pv_feature_convrot_attention(
            prepared,
            pointer_output,
            pointer_output,
            sequence,
            sequence,
            False,
            grouped_qk=False,
            rotation_group=0,
            value_scale_axis="key",
            probability_scale_mode="log",
            fuse_output_rotation=True,
            block_m=128,
            num_warps=4,
            num_stages=2,
            affine_probability=True,
            native_uint8_mma=True,
            factored_pv_scaling=True,
            precomputed_pv_multiplier=True,
            scaled_fp16_numerator=True,
            split_pv_head_dim=True,
            scale_forward_log_recurrence=True,
            unmasked_self_attention=True,
            use_tensor_descriptors=False,
        )

    torch.testing.assert_close(descriptor, pointer, atol=0.0, rtol=0.0)
