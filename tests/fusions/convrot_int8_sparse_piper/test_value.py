"""Tests for projection-aware sparse Piper V preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from lib.sparse_piper import reference_value_storage

from piper_kernels.attention.kernels.sparse_piper.layout import (
    ValueFormat,
    padded_sequence_length,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import value as value_fusion
from piper_kernels.linear.convrot.int8 import _ops as int8_ops

from ._helpers import projection_available
from ._reference import composed_value_projection


@dataclass(frozen=True, slots=True)
class _Operands:
    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return self.input_qdata, self.input_scale, self.weight_qdata, self.weight_scale


def _random_operands(
    *,
    batch: int = 1,
    sequence_length: int = 64,
    input_features: int = 272,
    heads: int = 2,
    head_dim: int = 128,
) -> _Operands:
    torch.manual_seed(193)
    input_qdata = torch.randint(
        -127,
        128,
        (batch, sequence_length, input_features),
        device="cuda",
        dtype=torch.int8,
    )
    input_scale = (
        torch.rand(
            (batch, sequence_length),
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.01)
        .add_(0.001)
    )
    weight_qdata = torch.randint(
        -127,
        128,
        (heads * head_dim, input_features),
        device="cuda",
        dtype=torch.int8,
    )
    weight_scale = (
        torch.rand(
            (heads * head_dim, 1),
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.01)
        .add_(0.001)
    )
    return _Operands(input_qdata, input_scale, weight_qdata, weight_scale)


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
@pytest.mark.parametrize("sequence_length", [64, 65])
@pytest.mark.parametrize(
    "head_dim",
    [
        pytest.param(
            64,
            marks=pytest.mark.skipif(
                not projection_available(64), reason="requires D64 fused sparse projection support"
            ),
        ),
        128,
    ],
)
@pytest.mark.parametrize("bias_dtype", [None, torch.float16, torch.bfloat16, torch.float32])
# NVIDIA projects 256 input features with Gluon and 272 with the shared Triton launchers.
@pytest.mark.parametrize("input_features", [256, 272])
@pytest.mark.parametrize("value_format", list(ValueFormat))
def test_fused_value_projection_matches_the_fp32_composed_contract(
    input_features: int,
    bias_dtype: torch.dtype | None,
    sequence_length: int,
    head_dim: int,
    value_format: ValueFormat,
) -> None:
    operands = _random_operands(
        sequence_length=sequence_length, input_features=input_features, head_dim=head_dim
    )
    bias = (
        torch.randn(2 * head_dim, device="cuda", dtype=bias_dtype)
        if bias_dtype is not None
        else None
    )

    actual_value, actual_scale, actual_mean = value_fusion._project_value_op(
        *operands.as_tuple(),
        head_dim=head_dim,
        bias=bias,
        value_format=int(value_format),
    )
    expected = composed_value_projection(
        *operands.as_tuple(),
        head_dim=head_dim,
        bias=bias,
    )
    storage_length = padded_sequence_length(sequence_length)

    assert actual_value.shape == value_format.shape(1, 2, storage_length, head_dim)
    assert actual_value.dtype is torch.int8
    assert actual_scale.shape == (1, 2, storage_length // 64, 1)
    assert actual_scale.dtype is torch.float32
    assert actual_mean.shape == (1, 2, storage_length // 64, head_dim)
    assert actual_mean.dtype is torch.bfloat16
    assert (
        int(
            (actual_value.to(torch.int16) - reference_value_storage(expected.value, value_format))
            .abs()
            .max()
        )
        <= 1
    )
    torch.testing.assert_close(
        actual_scale,
        expected.value_scale_multiplier,
        atol=2e-4,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        actual_mean.float(), expected.value_mean.float(), atol=2e-4, rtol=1.6e-2
    )


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
@pytest.mark.parametrize("sequence_length", [64, 65, 192])
def test_fused_value_projection_emits_valid_prefix_tile_means(
    sequence_length: int,
) -> None:
    operands = _random_operands(sequence_length=sequence_length)

    actual = value_fusion._project_value_op(*operands.as_tuple())
    expected = composed_value_projection(*operands.as_tuple())

    assert int((actual[0].to(torch.int16) - expected.value).abs().max()) <= 1
    torch.testing.assert_close(actual[1], expected.value_scale_multiplier, atol=2e-4, rtol=2e-5)
    assert actual[2].shape == expected.value_mean.shape
    torch.testing.assert_close(
        actual[2].float(), expected.value_mean.float(), atol=2e-4, rtol=1.6e-2
    )


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
@pytest.mark.parametrize("input_features", [256, 272])
def test_fused_value_tile_means_respect_internal_block_lengths(input_features: int) -> None:
    operands = _random_operands(sequence_length=128, input_features=input_features)
    block_lengths = torch.tensor([64, 17], device="cuda", dtype=torch.int32)

    actual = value_fusion._project_value_op(*operands.as_tuple(), block_lengths)
    projected = int8_ops.linear_prepared(
        *operands.as_tuple(),
        None,
        torch.float32,
    ).view(1, 128, 2, 128)
    expected = torch.stack(
        (projected[:, :64].mean(dim=1), projected[:, 64:81].mean(dim=1)),
        dim=2,
    )

    torch.testing.assert_close(actual[2].float(), expected, atol=2e-4, rtol=1.6e-2)
    assert not bool(actual[0][..., 81:].any())


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
def test_fused_value_projection_supports_full_blocks_batches_and_odd_heads() -> None:
    operands = _random_operands(
        batch=2,
        sequence_length=192,
        input_features=256,
        heads=3,
    )
    value, value_scale_multiplier, value_mean = value_fusion._project_value_op(*operands.as_tuple())

    assert value.shape == (2, 3, 128, 192)
    assert value_scale_multiplier.shape == (2, 3, 3, 1)
    assert value_mean.shape == (2, 3, 3, 128)
    assert value_mean.dtype is torch.bfloat16


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
def test_fused_value_mean_stays_below_the_tile_int8_error_floor() -> None:
    operands = _random_operands(sequence_length=192, input_features=256, heads=3)
    _value, value_scale_multiplier, value_mean = value_fusion._project_value_op(
        *operands.as_tuple()
    )
    materialized = int8_ops.linear_prepared(
        *operands.as_tuple(),
        None,
        torch.bfloat16,
    ).view(1, 192, 3, 128)
    exact_mean = materialized.float().unflatten(1, (3, 64)).mean(dim=2).permute(0, 2, 1, 3)
    tile_scale = value_scale_multiplier[..., 0] / 255.0

    assert bool(((value_mean.float() - exact_mean).abs() <= tile_scale[..., None] * 0.05).all())


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
@pytest.mark.parametrize("with_bias", [False, True])
def test_value_projection_custom_ops_pass_opcheck(with_bias: bool) -> None:
    operands = _random_operands()
    bias = torch.randn(256, device="cuda") if with_bias else None

    value_result = torch.library.opcheck(
        value_fusion._project_value_op,
        operands.as_tuple(),
        kwargs={"bias": bias},
    )
    masked_value_result = torch.library.opcheck(
        value_fusion._project_value_op,
        (*operands.as_tuple(), torch.tensor([64], device="cuda", dtype=torch.int32)),
        kwargs={"bias": bias},
    )

    assert set(value_result.values()) == {"SUCCESS"}
    assert set(masked_value_result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
def test_value_projection_runs_under_fullgraph_compile() -> None:
    operands = _random_operands()

    def prepare(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return value_fusion._project_value_op(*args)

    expected = prepare(*operands.as_tuple())
    compiled = torch.compile(prepare, backend="eager", fullgraph=True)
    actual = compiled(*operands.as_tuple())

    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("value_format", list(ValueFormat))
@pytest.mark.parametrize("head_dim", [64, 128])
def test_value_projection_fake_kernels_propagate_shapes(value_format, head_dim) -> None:
    input_qdata = torch.empty((2, 128, 272), device="meta", dtype=torch.int8)
    input_scale = torch.empty((2, 128), device="meta", dtype=torch.float32)
    weight_qdata = torch.empty((3 * head_dim, 272), device="meta", dtype=torch.int8)
    weight_scale = torch.empty((3 * head_dim, 1), device="meta", dtype=torch.float32)

    value, value_scale_multiplier, value_mean = value_fusion._project_value_op(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        head_dim=head_dim,
        value_format=int(value_format),
    )

    assert value.shape == value_format.shape(2, 3, 128, head_dim)
    assert value.dtype is torch.int8
    assert value_scale_multiplier.shape == (2, 3, 2, 1)
    assert value_scale_multiplier.dtype is torch.float32
    assert value_mean.shape == (2, 3, 2, head_dim)
    assert value_mean.dtype is torch.bfloat16


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
@pytest.mark.parametrize("input_features", [256, 272])
@pytest.mark.parametrize("constant_value", [2.0**100, -(2.0**100)])
def test_fused_value_projection_bounds_large_constant_tile_scale(
    input_features: int,
    constant_value: float,
) -> None:
    input_qdata = torch.zeros((1, 128, input_features), device="cuda", dtype=torch.int8)
    input_scale = torch.ones((1, 128), device="cuda", dtype=torch.float32)
    weight_qdata = torch.zeros((2 * 128, input_features), device="cuda", dtype=torch.int8)
    weight_scale = torch.ones((2 * 128, 1), device="cuda", dtype=torch.float32)
    bias = torch.full((2 * 128,), constant_value, device="cuda", dtype=torch.bfloat16)

    value, value_scale_multiplier, value_mean = value_fusion._project_value_op(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias=bias,
    )

    assert not bool(value.any())
    torch.testing.assert_close(
        value_mean, torch.full_like(value_mean, constant_value), atol=0, rtol=0
    )
    assert bool((value_scale_multiplier >= abs(constant_value) * (2.0**-80) * 255.0).all())
