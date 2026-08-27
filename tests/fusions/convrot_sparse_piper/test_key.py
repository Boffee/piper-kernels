"""Tests for fused ConvRot sparse-Piper key preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from piper_kernels.fusions.convrot_sparse_piper import key as key_fusion

from ._reference import composed_key_projection


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
@pytest.mark.parametrize("valid_sequence_length", [64, 59])
def test_fused_key_projection_matches_the_composed_contract(
    valid_sequence_length: int,
) -> None:
    operands = _random_operands()
    options = {
        "valid_sequence_length": valid_sequence_length,
        "norm_epsilon": 1e-5,
    }

    actual_key, actual_scale, actual_max, actual_min = key_fusion._project_key_op(
        *operands.as_tuple(),
        options["valid_sequence_length"],
        options["norm_epsilon"],
    )
    expected = composed_key_projection(*operands.as_tuple(), **options)

    assert actual_key.shape == (1, 2, 64, 128)
    assert actual_key.dtype is torch.int8
    assert actual_scale.shape == (1, 2, 1)
    assert actual_scale.dtype is torch.float32
    assert actual_max.shape == (1, 2, 1, 128)
    assert actual_min.shape == actual_max.shape
    assert int((actual_key.to(torch.int16) - expected.key).abs().max()) <= 1
    torch.testing.assert_close(actual_scale, expected.key_scale, atol=1e-4, rtol=3e-3)
    torch.testing.assert_close(actual_max, expected.key_max, atol=0.125, rtol=0.01)
    torch.testing.assert_close(actual_min, expected.key_min, atol=0.125, rtol=0.01)
    assert bool((actual_key[:, :, valid_sequence_length:] == 0).all())


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_fused_key_projection_supports_full_blocks_batches_and_odd_heads() -> None:
    operands = _random_operands(
        batch=2,
        sequence_length=192,
        input_features=256,
        heads=3,
    )

    key, key_scale, key_max, key_min = key_fusion._project_key_op(
        *operands.as_tuple(),
        181,
        1e-5,
    )

    assert key.shape == (2, 3, 192, 128)
    assert key_scale.shape == (2, 3, 3)
    assert key_max.shape == (2, 3, 3, 128)
    assert key_min.shape == key_max.shape
    assert bool((key[:, :, 181:] == 0).all())


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_key_projection_custom_op_passes_opcheck() -> None:
    operands = _random_operands()
    result = torch.library.opcheck(
        key_fusion._project_key_op,
        (*operands.as_tuple(), 59, 1e-5),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_key_projection_runs_under_fullgraph_compile() -> None:
    operands = _random_operands()

    def prepare(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return key_fusion._project_key_op(*args, 59, 1e-5)

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
        113,
        1e-5,
    )

    assert key.shape == (2, 3, 128, 128)
    assert key.dtype is torch.int8
    assert key_scale.shape == (2, 3, 2)
    assert key_scale.dtype is torch.float32
    assert key_max.shape == (2, 3, 2, 128)
    assert key_max.dtype is torch.float32
    assert key_min.shape == key_max.shape
