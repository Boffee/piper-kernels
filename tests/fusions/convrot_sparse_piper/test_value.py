"""Tests for projection-aware sparse Piper V preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from piper_kernels.fusions.convrot_sparse_piper import value as value_fusion
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

from ._reference import composed_value_projection


@dataclass(frozen=True, slots=True)
class _Operands:
    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return self.input_qdata, self.input_scale, self.weight_qdata, self.weight_scale


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _random_operands(
    *,
    batch: int = 1,
    sequence_length: int = 64,
    input_features: int = 272,
    heads: int = 2,
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
        (heads * 128, input_features),
        device="cuda",
        dtype=torch.int8,
    )
    weight_scale = (
        torch.rand(
            (heads * 128, 1),
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.01)
        .add_(0.001)
    )
    return _Operands(input_qdata, input_scale, weight_qdata, weight_scale)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("valid_sequence_length", [64, 59])
def test_fused_value_projection_matches_the_fp32_composed_contract(
    valid_sequence_length: int,
) -> None:
    operands = _random_operands()
    input_mean = convrot_backend.dequantized_input_mean(
        operands.input_qdata,
        operands.input_scale,
        valid_sequence_length,
    )

    actual_value, actual_scale, actual_mean = value_fusion._project_value_op(
        *operands.as_tuple()[:2],
        input_mean,
        *operands.as_tuple()[2:],
        valid_sequence_length,
    )
    expected = composed_value_projection(
        *operands.as_tuple()[:2],
        input_mean,
        *operands.as_tuple()[2:],
        valid_sequence_length=valid_sequence_length,
    )

    assert actual_value.shape == (1, 2, 128, 64)
    assert actual_value.dtype is torch.int8
    assert actual_scale.shape == (1, 2, 1, 1)
    assert actual_scale.dtype is torch.float32
    assert actual_mean.shape == (1, 2, 128)
    assert actual_mean.dtype is torch.float32
    assert int((actual_value.to(torch.int16) - expected.value).abs().max()) <= 1
    torch.testing.assert_close(
        actual_scale,
        expected.value_scale_multiplier,
        atol=2e-4,
        rtol=2e-5,
    )
    torch.testing.assert_close(actual_mean, expected.value_mean, atol=2e-6, rtol=2e-6)
    assert bool((actual_value[..., valid_sequence_length:] == 0).all())


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_fused_value_projection_supports_full_blocks_batches_and_odd_heads() -> None:
    operands = _random_operands(
        batch=2,
        sequence_length=192,
        input_features=256,
        heads=3,
    )
    input_mean = convrot_backend.dequantized_input_mean(
        operands.input_qdata,
        operands.input_scale,
        181,
    )
    value, value_scale, value_mean = value_fusion._project_value_op(
        *operands.as_tuple()[:2],
        input_mean,
        *operands.as_tuple()[2:],
        181,
    )

    assert value.shape == (2, 3, 128, 192)
    assert value_scale.shape == (2, 3, 3, 1)
    assert value_mean.shape == (2, 3, 128)
    assert bool((value[..., 181:] == 0).all())


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_fused_value_mean_stays_below_the_tile_int8_error_floor() -> None:
    operands = _random_operands(sequence_length=192, input_features=256, heads=3)
    input_mean = convrot_backend.dequantized_input_mean(
        operands.input_qdata,
        operands.input_scale,
        181,
    )
    _value, value_scale, value_mean = value_fusion._project_value_op(
        *operands.as_tuple()[:2],
        input_mean,
        *operands.as_tuple()[2:],
        181,
    )
    materialized = convrot_backend.linear_prepared(
        *operands.as_tuple(),
        None,
        torch.bfloat16,
    ).view(1, 192, 3, 128)
    exact_mean = materialized[:, :181].float().mean(dim=1)
    smallest_tile_scale = value_scale[..., 0].amin(dim=-1) / 255.0

    assert bool(((value_mean - exact_mean).abs() <= smallest_tile_scale[:, :, None] * 0.05).all())


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_dequantized_input_mean_matches_the_represented_activation() -> None:
    operands = _random_operands(batch=2, sequence_length=192, input_features=256)

    actual = convrot_backend.dequantized_input_mean(
        operands.input_qdata,
        operands.input_scale,
        181,
    )
    expected = (operands.input_qdata[:, :181].float() * operands.input_scale[:, :181, None]).mean(
        dim=1
    )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_value_projection_custom_ops_pass_opcheck() -> None:
    operands = _random_operands()
    input_mean = convrot_backend.dequantized_input_mean(
        operands.input_qdata,
        operands.input_scale,
        59,
    )

    mean_result = torch.library.opcheck(
        convrot_backend.dequantized_input_mean,
        (operands.input_qdata, operands.input_scale, 59),
    )
    value_result = torch.library.opcheck(
        value_fusion._project_value_op,
        (
            operands.input_qdata,
            operands.input_scale,
            input_mean,
            operands.weight_qdata,
            operands.weight_scale,
            59,
        ),
    )

    assert set(mean_result.values()) == {"SUCCESS"}
    assert set(value_result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_value_projection_runs_under_fullgraph_compile() -> None:
    operands = _random_operands()

    def prepare(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        input_mean = convrot_backend.dequantized_input_mean(args[0], args[1], 59)
        return value_fusion._project_value_op(
            args[0],
            args[1],
            input_mean,
            args[2],
            args[3],
            59,
        )

    expected = prepare(*operands.as_tuple())
    compiled = torch.compile(prepare, backend="eager", fullgraph=True)
    actual = compiled(*operands.as_tuple())

    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_value_projection_fake_kernels_propagate_shapes() -> None:
    input_qdata = torch.empty((2, 128, 272), device="meta", dtype=torch.int8)
    input_scale = torch.empty((2, 128), device="meta", dtype=torch.float32)
    weight_qdata = torch.empty((3 * 128, 272), device="meta", dtype=torch.int8)
    weight_scale = torch.empty((3 * 128, 1), device="meta", dtype=torch.float32)

    input_mean = convrot_backend.dequantized_input_mean(input_qdata, input_scale, 113)
    value, value_scale, value_mean = value_fusion._project_value_op(
        input_qdata,
        input_scale,
        input_mean,
        weight_qdata,
        weight_scale,
        113,
    )

    assert input_mean.shape == (2, 272)
    assert input_mean.dtype is torch.float32
    assert value.shape == (2, 3, 128, 128)
    assert value.dtype is torch.int8
    assert value_scale.shape == (2, 3, 2, 1)
    assert value_scale.dtype is torch.float32
    assert value_mean.shape == (2, 3, 128)
    assert value_mean.dtype is torch.float32
