"""End-to-end ConvRot-to-sparse-Piper fusion tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
)
from piper_kernels.attention.sparse_piper_attention._routes import _MINMAX_ROUTING
from piper_kernels.fusions.convrot_sparse_piper import key, query, value
from piper_kernels.linear.convrot.int8 import triton as convrot_backend

_BATCH = 1
_SEQUENCE = 192
_INPUT_FEATURES = 256
_HEADS = 2
_HEAD_DIM = 128
_ROTARY_DIM = 96
_SPARSE_KEY_BLOCKS = 2

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


def _operands(*, batch: int = _BATCH, sequence_length: int = _SEQUENCE) -> _Operands:
    torch.manual_seed(199)
    input_qdata = torch.randint(
        -127,
        128,
        (batch, sequence_length, _INPUT_FEATURES),
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
        (sequence_length, _ROTARY_DIM),
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
        1e-5,
        _HEAD_DIM**-0.5,
        _MINMAX_ROUTING,
    )
    prepared_key = key._project_key_op(
        operands.input_qdata,
        operands.input_scale,
        operands.key_weight,
        operands.key_weight_scale,
        operands.key_norm,
        operands.cos,
        operands.sin,
        1e-5,
        _MINMAX_ROUTING,
    )
    input_mean = convrot_backend.dequantized_input_mean(
        operands.input_qdata,
        operands.input_scale,
    )
    prepared_value = value._project_value_op(
        operands.input_qdata,
        operands.input_scale,
        input_mean,
        operands.value_weight,
        operands.value_weight_scale,
    )
    return prepared_query, prepared_key, prepared_value


def _run_sparse_piper_attention_from_quantized(
    prepared_query: _QuantizedQueryOperands,
    prepared_key: _QuantizedKeyOperands,
    prepared_value: _QuantizedValueOperands,
    attention: SparsePiperAttention,
    *,
    logical_sequence_length: int,
    sparse_key_blocks: int,
) -> torch.Tensor:
    return _sparse_piper_attention_from_quantized_op(
        *prepared_query,
        *prepared_key,
        *prepared_value,
        list(attention._head_keep_ratio_units),
        sparse_key_blocks,
        logical_sequence_length,
        attention._routing_mode,
    )


def _materialize_qk(
    operands: _Operands,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    norm: torch.Tensor,
) -> torch.Tensor:
    batch, sequence_length, _input_features = operands.input_qdata.shape
    heads = weight.shape[0] // _HEAD_DIM
    projected = convrot_backend.linear_prepared(
        operands.input_qdata,
        operands.input_scale,
        weight,
        weight_scale,
        None,
        torch.bfloat16,
    ).view(batch, sequence_length, heads, _HEAD_DIM)
    normalized = F.rms_norm(projected, (_HEAD_DIM,), norm, 1e-5)
    rotary_dim = operands.cos.shape[1]
    rotary = normalized[..., :rotary_dim]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos = operands.cos.to(torch.bfloat16)[None, :, None, :]
    sin = operands.sin.to(torch.bfloat16)[None, :, None, :]
    rotary = rotary * cos + rotated * sin
    return torch.cat((rotary, normalized[..., rotary_dim:]), dim=-1).contiguous()


def _materialize_value(operands: _Operands) -> torch.Tensor:
    batch, sequence_length, _input_features = operands.input_qdata.shape
    return convrot_backend.linear_prepared(
        operands.input_qdata,
        operands.input_scale,
        operands.value_weight,
        operands.value_weight_scale,
        None,
        torch.bfloat16,
    ).view(batch, sequence_length, _HEADS, _HEAD_DIM)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_quantized_sparse_piper_writes_engine_layout() -> None:
    operands = _operands()
    query, key, value = _prepare(operands)
    attention = SparsePiperAttention((0.5, 1.0))

    with torch.no_grad():
        output = _run_sparse_piper_attention_from_quantized(
            query,
            key,
            value,
            attention,
            logical_sequence_length=_SEQUENCE,
            sparse_key_blocks=_SPARSE_KEY_BLOCKS,
        )

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
    materialized_value = _materialize_value(operands)
    attention = SparsePiperAttention((1.0, 1.0))

    with torch.no_grad():
        expected = attention(
            materialized_query,
            materialized_key,
            materialized_value,
            sparse_key_blocks=_SPARSE_KEY_BLOCKS,
        )
        actual = _run_sparse_piper_attention_from_quantized(
            query,
            key,
            value,
            attention,
            logical_sequence_length=_SEQUENCE,
            sparse_key_blocks=_SPARSE_KEY_BLOCKS,
        )

    difference = actual.float() - expected.float()
    relative_l2 = difference.norm() / expected.float().norm()
    assert relative_l2 < 0.025


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("batch", "sequence_length"),
    [(1, 65), (1, 127), (2, 129), (1, 193)],
)
def test_ragged_fused_projection_matches_materialized_attention(
    batch: int,
    sequence_length: int,
) -> None:
    operands = _operands(batch=batch, sequence_length=sequence_length)
    prepared_query, prepared_key, prepared_value = _prepare(operands)
    assert not torch.count_nonzero(prepared_query[0][:, :, sequence_length:])
    assert not torch.count_nonzero(prepared_key[0][:, :, sequence_length:])
    assert not torch.count_nonzero(prepared_value[0][..., sequence_length:])
    assert all(
        bool(torch.isfinite(metadata).all())
        for metadata in (*prepared_query[1:], *prepared_key[1:], *prepared_value[1:])
    )

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
    materialized_value = _materialize_value(operands)

    sparse_key_blocks = sequence_length // 64
    attention = SparsePiperAttention((1.0, 1.0))
    with torch.no_grad():
        expected = attention(
            materialized_query,
            materialized_key,
            materialized_value,
            sparse_key_blocks=sparse_key_blocks,
        )
        actual = _run_sparse_piper_attention_from_quantized(
            prepared_query,
            prepared_key,
            prepared_value,
            attention,
            logical_sequence_length=sequence_length,
            sparse_key_blocks=sparse_key_blocks,
        )

    assert actual.shape == (batch, sequence_length, _HEADS, _HEAD_DIM)
    difference = actual.float() - expected.float()
    relative_l2 = difference.norm() / expected.float().norm()
    assert relative_l2 < 0.025


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_full_fused_sparse_piper_pipeline_compiles_as_one_graph() -> None:
    operands = _operands()
    attention = SparsePiperAttention((0.5, 1.0))

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
        return _run_sparse_piper_attention_from_quantized(
            query,
            key,
            value,
            attention,
            logical_sequence_length=_SEQUENCE,
            sparse_key_blocks=_SPARSE_KEY_BLOCKS,
        )

    compiled = torch.compile(run, fullgraph=True)
    with torch.no_grad():
        expected = run(operands.input_qdata, operands.input_scale)
        actual = compiled(operands.input_qdata, operands.input_scale)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
