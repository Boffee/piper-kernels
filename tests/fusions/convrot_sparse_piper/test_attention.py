"""End-to-end ConvRot-to-sparse-Piper fusion tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels import (
    SparsePiperAttentionPlan,
    prepare_sparse_piper_attention_plan,
    sparse_piper_attention,
)
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sm120_sparse_piper_attention_from_quantized,
)
from piper_kernels.fusions.convrot_sparse_piper import key, query, value
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

_BATCH = 1
_SEQUENCE = 192
_INPUT_FEATURES = 256
_HEADS = 2
_HEAD_DIM = 128
_ROTARY_DIM = 96
_VALID_SEQUENCE = 181
_SUFFIX_START = 128

type _QuantizedQueryOperands = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
type _QuantizedKeyOperands = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
type _QuantizedValueOperands = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _Operands:
    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    query_weight: torch.Tensor
    key_weight: torch.Tensor
    value_weight: torch.Tensor
    query_weight_scale: torch.Tensor
    key_weight_scale: torch.Tensor
    value_weight_scale: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _operands() -> _Operands:
    torch.manual_seed(199)
    input_qdata = torch.randint(
        -127,
        128,
        (_BATCH, _SEQUENCE, _INPUT_FEATURES),
        device="cuda",
        dtype=torch.int8,
    )
    input_scale = (
        torch.rand(
            (_BATCH, _SEQUENCE),
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.01)
        .add_(0.001)
    )
    weights = tuple(
        torch.randint(
            -127,
            128,
            (_HEADS * _HEAD_DIM, _INPUT_FEATURES),
            device="cuda",
            dtype=torch.int8,
        )
        for _ in range(3)
    )
    weight_scales = tuple(
        torch.rand(
            (_HEADS * _HEAD_DIM, 1),
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.01)
        .add_(0.001)
        for _ in range(3)
    )
    norms = tuple(
        torch.rand((_HEAD_DIM,), device="cuda", dtype=torch.float32).add_(0.5).bfloat16()
        for _ in range(2)
    )
    angles = torch.rand(
        (_SEQUENCE, _ROTARY_DIM),
        device="cuda",
        dtype=torch.float32,
    ).mul_(2 * torch.pi)
    return _Operands(
        input_qdata,
        input_scale,
        *weights,
        *weight_scales,
        *norms,
        angles.cos().contiguous(),
        angles.sin().contiguous(),
    )


def _prepare(
    operands: _Operands,
) -> tuple[_QuantizedQueryOperands, _QuantizedKeyOperands, _QuantizedValueOperands]:
    prepared_query = query._project_query_op(
        operands.input_qdata,
        operands.input_scale,
        operands.query_weight,
        operands.query_weight_scale,
        operands.query_norm,
        operands.cos,
        operands.sin,
        _VALID_SEQUENCE,
        1e-5,
        _HEAD_DIM**-0.5,
    )
    prepared_key = key._project_key_op(
        operands.input_qdata,
        operands.input_scale,
        operands.key_weight,
        operands.key_weight_scale,
        operands.key_norm,
        operands.cos,
        operands.sin,
        _VALID_SEQUENCE,
        1e-5,
    )
    input_mean = convrot_backend.dequantized_input_mean(
        operands.input_qdata,
        operands.input_scale,
        _VALID_SEQUENCE,
    )
    prepared_value = value._project_value_op(
        operands.input_qdata,
        operands.input_scale,
        input_mean,
        operands.value_weight,
        operands.value_weight_scale,
        _VALID_SEQUENCE,
    )
    return prepared_query, prepared_key, prepared_value


def _run_sparse_piper_attention_from_quantized(
    prepared_query: _QuantizedQueryOperands,
    prepared_key: _QuantizedKeyOperands,
    prepared_value: _QuantizedValueOperands,
    plan: SparsePiperAttentionPlan,
) -> torch.Tensor:
    return _sm120_sparse_piper_attention_from_quantized(
        *prepared_query,
        *prepared_key,
        *prepared_value,
        plan.keep_blocks,
        plan.head_offsets,
        _SUFFIX_START,
        _VALID_SEQUENCE,
        plan.routes_per_query,
        plan.query_chunk_blocks,
    )


def _materialize_qk(
    operands: _Operands,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    norm: torch.Tensor,
) -> torch.Tensor:
    projected = convrot_backend.linear_prepared(
        operands.input_qdata,
        operands.input_scale,
        weight,
        weight_scale,
        None,
        torch.bfloat16,
    ).view(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)
    normalized = F.rms_norm(projected, (_HEAD_DIM,), norm, 1e-5)
    rotary = normalized[..., :_ROTARY_DIM]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos = operands.cos.to(torch.bfloat16)[None, :, None, :]
    sin = operands.sin.to(torch.bfloat16)[None, :, None, :]
    rotary = rotary * cos + rotated * sin
    return torch.cat((rotary, normalized[..., _ROTARY_DIM:]), dim=-1).contiguous()


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_quantized_sparse_piper_writes_engine_layout() -> None:
    operands = _operands()
    query, key, value = _prepare(operands)
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        sparse_key_blocks=2,
    )

    with torch.no_grad():
        output = _run_sparse_piper_attention_from_quantized(query, key, value, plan)

    assert output.shape == (_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)
    assert output.dtype is torch.bfloat16
    assert output.is_contiguous()
    assert bool(torch.isfinite(output).all())


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_quantized_sparse_piper_matches_the_materialized_path() -> None:
    operands = _operands()
    query, key, value = _prepare(operands)
    materialized_query = _materialize_qk(
        operands,
        operands.query_weight,
        operands.query_weight_scale,
        operands.query_norm,
    )
    materialized_key = _materialize_qk(
        operands,
        operands.key_weight,
        operands.key_weight_scale,
        operands.key_norm,
    )
    materialized_value = convrot_backend.linear_prepared(
        operands.input_qdata,
        operands.input_scale,
        operands.value_weight,
        operands.value_weight_scale,
        None,
        torch.bfloat16,
    ).view(_BATCH, _SEQUENCE, _HEADS, _HEAD_DIM)
    keep = torch.full((_HEADS,), 2, device="cuda", dtype=torch.int32)
    plan = prepare_sparse_piper_attention_plan(keep, sparse_key_blocks=2)

    with torch.no_grad():
        expected = sparse_piper_attention(
            materialized_query,
            materialized_key,
            materialized_value,
            plan,
            suffix_start=_SUFFIX_START,
            valid_sequence_length=_VALID_SEQUENCE,
        )
        actual = _run_sparse_piper_attention_from_quantized(query, key, value, plan)

    difference = actual[:, :_VALID_SEQUENCE].float() - expected[:, :_VALID_SEQUENCE].float()
    relative_l2 = difference.norm() / expected[:, :_VALID_SEQUENCE].float().norm()
    assert relative_l2 < 0.025


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_full_fused_sparse_piper_pipeline_compiles_as_one_graph() -> None:
    operands = _operands()
    plan = prepare_sparse_piper_attention_plan(
        torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        sparse_key_blocks=2,
    )

    def run(input_qdata: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        dynamic_operands = _Operands(
            input_qdata,
            input_scale,
            operands.query_weight,
            operands.key_weight,
            operands.value_weight,
            operands.query_weight_scale,
            operands.key_weight_scale,
            operands.value_weight_scale,
            operands.query_norm,
            operands.key_norm,
            operands.cos,
            operands.sin,
        )
        query, key, value = _prepare(dynamic_operands)
        return _run_sparse_piper_attention_from_quantized(query, key, value, plan)

    compiled = torch.compile(run, fullgraph=True)
    with torch.no_grad():
        expected = run(operands.input_qdata, operands.input_scale)
        actual = compiled(operands.input_qdata, operands.input_scale)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
