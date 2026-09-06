"""Tests for fused ConvRot INT8 sparse-Piper key preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import key as key_fusion
from piper_kernels.fusions.convrot_int8_sparse_piper._layout import padded_sequence_length

from ._reference import composed_key_projection, composed_mean_pool_summary


@dataclass(frozen=True, slots=True)
class _Operands:
    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    norm_weight: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_qdata,
            self.input_scale,
            self.weight_qdata,
            self.weight_scale,
            self.norm_weight,
            self.cos,
            self.sin,
        )


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _random_operands(
    *,
    batch: int = 1,
    sequence_length: int = 64,
    input_features: int = 272,
    heads: int = 2,
) -> _Operands:
    torch.manual_seed(197)
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
    norm_weight = torch.rand((128,), device="cuda", dtype=torch.float32).add_(0.5).bfloat16()
    angles = torch.rand(
        (sequence_length, 96),
        device="cuda",
        dtype=torch.float32,
    ).mul_(2 * torch.pi)
    return _Operands(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        angles.cos().contiguous(),
        angles.sin().contiguous(),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("sequence_length", [64, 65])
def test_fused_key_projection_matches_the_fp32_composed_contract(sequence_length: int) -> None:
    operands = _random_operands(sequence_length=sequence_length)
    options = {
        "norm_epsilon": 1e-5,
    }

    actual_key, actual_scale, actual_max, actual_min = key_fusion._project_key_op(
        *operands.as_tuple(),
        options["norm_epsilon"],
        _MINMAX_ROUTING,
    )
    expected = composed_key_projection(*operands.as_tuple(), **options)
    storage_length = padded_sequence_length(sequence_length)

    assert actual_key.shape == (1, 2, storage_length, 128)
    assert actual_key.dtype is torch.int8
    assert actual_scale.shape == (1, 2, storage_length // 64)
    assert actual_scale.dtype is torch.float32
    assert actual_max.shape == (1, 2, storage_length // 64, 128)
    assert actual_min.shape == actual_max.shape
    assert int((actual_key.to(torch.int16) - expected.key).abs().max()) <= 1
    torch.testing.assert_close(actual_scale, expected.key_scale, atol=1e-4, rtol=3e-3)
    torch.testing.assert_close(actual_max, expected.key_max, atol=0.125, rtol=0.01)
    torch.testing.assert_close(actual_min, expected.key_min, atol=0.125, rtol=0.01)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("sequence_length", [64, 65])
def test_mean_pool_key_projection_emits_exact_valid_prefix_means(
    sequence_length: int,
) -> None:
    operands = _random_operands(sequence_length=sequence_length)

    actual_key, actual_scale, actual_mean, actual_aux = key_fusion._project_key_op(
        *operands.as_tuple(),
        1e-5,
        _MEAN_ROUTING,
    )
    expected = composed_mean_pool_summary(
        *operands.as_tuple(),
        norm_epsilon=1e-5,
    )

    assert actual_key.dtype is torch.int8
    assert actual_scale.dtype is torch.float32
    assert actual_aux.numel() == 0
    torch.testing.assert_close(actual_mean, expected, atol=0.015625, rtol=0.01)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_fused_key_projection_supports_k64_tail_batches_and_odd_heads() -> None:
    # S192 is K64-aligned but leaves a half-M128 tail whose nonexistent rows
    # must not read beyond the physical RoPE buffers.
    operands = _random_operands(
        batch=2,
        sequence_length=192,
        input_features=256,
        heads=3,
    )

    key, key_scale, key_max, key_min = key_fusion._project_key_op(
        *operands.as_tuple(),
        1e-5,
        _MINMAX_ROUTING,
    )

    assert key.shape == (2, 3, 192, 128)
    assert key_scale.shape == (2, 3, 3)
    assert key_max.shape == (2, 3, 3, 128)
    assert key_min.shape == key_max.shape


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("routing_mode", [_MEAN_ROUTING, _MINMAX_ROUTING])
def test_fused_key_projection_ignores_internal_padding(routing_mode: int) -> None:
    operands = _random_operands(sequence_length=192)
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    valid_rows = torch.arange(192, device="cuda") % 64
    valid_rows = valid_rows < block_lengths.repeat_interleave(64)
    corrupted_qdata = operands.input_qdata.clone()
    corrupted_scale = operands.input_scale.clone()
    corrupted_qdata[:, ~valid_rows] = 127
    corrupted_scale[:, ~valid_rows] = 100
    arguments = (*operands.as_tuple()[2:], 1e-5, routing_mode, block_lengths)

    expected = key_fusion._project_key_op(
        operands.input_qdata,
        operands.input_scale,
        *arguments,
    )
    actual = key_fusion._project_key_op(
        corrupted_qdata,
        corrupted_scale,
        *arguments,
    )

    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_key_projection_custom_op_passes_opcheck() -> None:
    operands = _random_operands()
    result = torch.library.opcheck(
        key_fusion._project_key_op,
        (*operands.as_tuple(), 1e-5, _MINMAX_ROUTING),
    )

    assert set(result.values()) == {"SUCCESS"}

    mean_result = torch.library.opcheck(
        key_fusion._project_key_op,
        (*operands.as_tuple(), 1e-5, _MEAN_ROUTING),
    )
    assert set(mean_result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_key_projection_runs_under_fullgraph_compile() -> None:
    operands = _random_operands()

    def prepare(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return key_fusion._project_key_op(*args, 1e-5, _MINMAX_ROUTING)

    expected = prepare(*operands.as_tuple())
    compiled = torch.compile(prepare, backend="eager", fullgraph=True)
    actual = compiled(*operands.as_tuple())

    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_key_projection_fake_kernels_propagate_shapes() -> None:
    key, key_scale, key_max, key_min = key_fusion._project_key_op(
        torch.empty((2, 128, 272), device="meta", dtype=torch.int8),
        torch.empty((2, 128), device="meta", dtype=torch.float32),
        torch.empty((3 * 128, 272), device="meta", dtype=torch.int8),
        torch.empty((3 * 128, 1), device="meta", dtype=torch.float32),
        torch.empty((128,), device="meta", dtype=torch.bfloat16),
        torch.empty((128, 96), device="meta", dtype=torch.float32),
        torch.empty((128, 96), device="meta", dtype=torch.float32),
        1e-5,
        _MINMAX_ROUTING,
    )

    assert key.shape == (2, 3, 128, 128)
    assert key.dtype is torch.int8
    assert key_scale.shape == (2, 3, 2)
    assert key_scale.dtype is torch.float32
    assert key_max.shape == (2, 3, 2, 128)
    assert key_max.dtype is torch.float32
    assert key_min.shape == key_max.shape
