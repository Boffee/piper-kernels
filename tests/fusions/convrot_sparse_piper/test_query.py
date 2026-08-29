"""Tests for fused ConvRot sparse-Piper query preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from piper_kernels.fusions.convrot_sparse_piper import query as query_fusion
from piper_kernels.fusions.convrot_sparse_piper._layout import padded_sequence_length

from ._reference import composed_query_projection


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
    sequence_length: int = 64,
    input_features: int = 272,
    heads: int = 2,
) -> _Operands:
    torch.manual_seed(191)
    input_qdata = torch.randint(
        -127,
        128,
        (1, sequence_length, input_features),
        device="cuda",
        dtype=torch.int8,
    )
    input_scale = (
        torch.rand(
            (1, sequence_length),
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
def test_fused_query_projection_matches_the_fp32_composed_contract(
    sequence_length: int,
) -> None:
    operands = _random_operands(sequence_length=sequence_length)
    options = {
        "norm_epsilon": 1e-6,
        "softmax_scale": 128**-0.5,
    }

    actual_query, actual_scale, actual_summary = query_fusion._project_query_op(
        *operands.as_tuple(),
        options["norm_epsilon"],
        options["softmax_scale"],
    )
    expected = composed_query_projection(*operands.as_tuple(), **options)
    storage_length = padded_sequence_length(sequence_length)

    assert actual_query.shape == (1, 2, storage_length, 128)
    assert actual_query.dtype is torch.int8
    assert actual_scale.shape == (1, 2, storage_length // 32)
    assert actual_scale.dtype is torch.float32
    assert actual_summary.shape == (1, 2, storage_length // 64, 128)
    assert actual_summary.dtype is torch.float32
    assert int((actual_query.to(torch.int16) - expected.query).abs().max()) <= 1
    torch.testing.assert_close(
        actual_scale,
        expected.query_scale,
        atol=1e-5,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        actual_summary,
        expected.query_summary,
        atol=0.125,
        rtol=0.01,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_fused_query_projection_supports_multiple_q64_blocks() -> None:
    operands = _random_operands(sequence_length=128, input_features=256, heads=3)

    query, query_scale, query_summary = query_fusion._project_query_op(
        *operands.as_tuple(),
        1e-6,
        128**-0.5,
    )

    assert query.shape == (1, 3, 128, 128)
    assert query_scale.shape == (1, 3, 4)
    assert query_summary.shape == (1, 3, 2, 128)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_fused_query_projection_custom_op_passes_opcheck() -> None:
    operands = _random_operands()

    result = torch.library.opcheck(
        query_fusion._project_query_op,
        (*operands.as_tuple(), 1e-6, 128**-0.5),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_fused_query_projection_is_a_fullgraph_compile_boundary() -> None:
    operands = _random_operands()

    def prepare(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return query_fusion._project_query_op(*args, 1e-6, 128**-0.5)

    expected = prepare(*operands.as_tuple())
    compiled = torch.compile(prepare, backend="eager", fullgraph=True)
    actual = compiled(*operands.as_tuple())

    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_fused_query_projection_fake_kernel_propagates_shapes() -> None:
    query, query_scale, query_summary = query_fusion._project_query_op(
        torch.empty((2, 128, 272), device="meta", dtype=torch.int8),
        torch.empty((2, 128), device="meta", dtype=torch.float32),
        torch.empty((3 * 128, 272), device="meta", dtype=torch.int8),
        torch.empty((3 * 128, 1), device="meta", dtype=torch.float32),
        torch.empty((128,), device="meta", dtype=torch.bfloat16),
        torch.empty((128, 96), device="meta", dtype=torch.float32),
        torch.empty((128, 96), device="meta", dtype=torch.float32),
        1e-6,
        128**-0.5,
    )

    assert query.shape == (2, 3, 128, 128)
    assert query.dtype is torch.int8
    assert query_scale.shape == (2, 3, 4)
    assert query_scale.dtype is torch.float32
    assert query_summary.shape == (2, 3, 2, 128)
    assert query_summary.dtype is torch.float32
