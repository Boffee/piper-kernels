"""GPU tests for the fixed- and block-scaled signed-INT8 PV baselines."""

from collections.abc import Callable

import pytest
import torch

from piper_kernels.attention._sage2pp.experiments import (
    triton_sage_attention_int8_pv,
    triton_sage_attention_int8_pv_block_scaled,
    triton_sage_attention_int8_pv_convrot_rms,
    triton_sage_attention_int8_pv_per_key_log,
    triton_sage_attention_uint8_pv_feature_convrot,
)

Int8Attention = Callable[..., torch.Tensor]


def _consumer_gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability == (8, 9) or capability[0] == 12


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
