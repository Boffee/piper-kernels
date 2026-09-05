"""Tests for the direct semantic chunked ConvRot INT8 SwiGLU FFN."""

from dataclasses import dataclass

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels.fusions.convrot_int8_swiglu_ffn.triton import (
    _chunked_swiglu_ffn_gated_updates_op,
    _chunked_swiglu_ffn_op,
)
from piper_kernels.linear.convrot.int8 import triton as convrot_int8_backend


@dataclass(frozen=True, slots=True)
class _Linear:
    qdata: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None
    group_size: int = 256

    def arguments(self) -> tuple[object, ...]:
        return self.qdata, self.scale, self.bias, self.group_size


@dataclass(frozen=True, slots=True)
class _Operands:
    input: torch.Tensor
    gate: _Linear
    value: _Linear
    down: _Linear

    def arguments(self, chunk_rows: int) -> tuple[object, ...]:
        return (
            self.input,
            *self.gate.arguments(),
            *self.value.arguments(),
            *self.down.arguments(),
            chunk_rows,
        )


def _linear(
    out_features: int,
    in_features: int,
    bias_dtype: torch.dtype | None,
) -> _Linear:
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    bias = (
        torch.randn(out_features, dtype=bias_dtype, device="cuda")
        if bias_dtype is not None
        else None
    )
    return _Linear(qdata, scale, bias)


def _operands(
    *,
    rows: int = 385,
    output_features: int = 384,
    bias_dtype: torch.dtype | None = torch.bfloat16,
) -> _Operands:
    input_features, intermediate_features = 256, 512
    input = torch.randn(rows, input_features, dtype=torch.bfloat16, device="cuda")  # noqa: A001
    return _Operands(
        input,
        _linear(intermediate_features, input_features, bias_dtype),
        _linear(intermediate_features, input_features, bias_dtype),
        _linear(output_features, intermediate_features, bias_dtype),
    )


def _materialized(operands: _Operands) -> torch.Tensor:
    gate = convrot_int8_backend.run_linear(operands.input, *operands.gate.arguments())
    value = convrot_int8_backend.run_linear(operands.input, *operands.value.arguments())
    return convrot_int8_backend.run_linear(
        value * F.silu(gate),
        *operands.down.arguments(),
    )


def _materialized_gated_updates(
    ffn: torch.Tensor,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
) -> torch.Tensor:
    hidden = base.float() + update_gate[gate_indices].float() * reusable_update.float()
    return (hidden + ffn_gate[gate_indices].float() * ffn.float()).to(base.dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("rows", [127, 385], ids=["short", "ragged-multi-chunk"])
@pytest.mark.parametrize("chunk_rows", [64, 128])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
def test_chunked_ffn_matches_materialized(
    rows: int,
    chunk_rows: int,
    bias_dtype: torch.dtype | None,
) -> None:
    torch.manual_seed(201 + rows)
    operands = _operands(rows=rows, bias_dtype=bias_dtype)

    expected = _materialized(operands)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(chunk_rows))

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert actual.dtype is torch.bfloat16
    assert relative_l2 < 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_chunked_ffn_accepts_adjacent_checkpoint_views() -> None:
    torch.manual_seed(209)
    operands = _operands(rows=385, bias_dtype=torch.float32)
    projection_width = operands.gate.qdata.shape[0]
    qdata = torch.cat((operands.gate.qdata, operands.value.qdata))
    scale = torch.cat((operands.gate.scale, operands.value.scale))
    assert operands.gate.bias is not None
    assert operands.value.bias is not None
    bias = torch.cat((operands.gate.bias, operands.value.bias))
    gate = _Linear(
        qdata[:projection_width],
        scale[:projection_width],
        bias[:projection_width],
    )
    value = _Linear(
        qdata[projection_width:],
        scale[projection_width:],
        bias[projection_width:],
    )
    views = _Operands(operands.input, gate, value, operands.down)

    assert gate.qdata.untyped_storage().data_ptr() == value.qdata.untyped_storage().data_ptr()
    assert value.qdata.storage_offset() > 0
    expected = _materialized(views)
    actual = _chunked_swiglu_ffn_op(*views.arguments(128))

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("invalid", ["input", "bias", "group-size"])
def test_chunked_ffn_rejects_invalid_inputs(invalid: str) -> None:
    torch.manual_seed(204)
    operands = _operands(rows=129)
    arguments = list(operands.arguments(128))
    expected_message = ""
    if invalid == "input":
        storage = torch.randn(129, 512, dtype=torch.bfloat16, device="cuda")
        arguments[0] = storage[:, ::2]
        expected_message = "contiguous"
    elif invalid == "bias":
        storage = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
        arguments[3] = storage[::2]
        expected_message = "contiguous"
    else:
        arguments[8] = 16
        expected_message = "share one group size"

    with pytest.raises(ValueError, match=expected_message):
        _chunked_swiglu_ffn_op(*arguments)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("output_features", [384, 1280], ids=["reused-workspace", "separate"])
def test_chunked_ffn_gated_updates_matches_materialized(output_features: int) -> None:
    torch.manual_seed(203)
    operands = _operands(output_features=output_features, bias_dtype=torch.float32)
    rows = operands.input.shape[0]
    base = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
    reusable_update = torch.randn_like(base)
    gate_storage = torch.randn(7, 6 * output_features, dtype=torch.bfloat16, device="cuda")
    update_gate = gate_storage[:, 2 * output_features : 3 * output_features]
    ffn_gate = gate_storage[:, 5 * output_features :]
    gate_indices = torch.randint(0, 7, (rows,), dtype=torch.int64, device="cuda")
    expected = _materialized_gated_updates(
        _materialized(operands),
        base,
        reusable_update,
        update_gate,
        ffn_gate,
        gate_indices,
    )
    actual = reusable_update.clone()

    result = _chunked_swiglu_ffn_gated_updates_op(
        *operands.arguments(128)[:-1],
        base,
        actual,
        update_gate,
        ffn_gate,
        gate_indices,
        False,
        128,
    )

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert result is None
    assert relative_l2 < 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_chunked_ffn_gated_updates_preserves_negative_python_indices() -> None:
    torch.manual_seed(205)
    operands = _operands(rows=10, bias_dtype=None)
    rows, output_features = operands.input.shape[0], operands.down.qdata.shape[0]
    base = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
    reusable_update = torch.randn_like(base)
    update_gate = torch.randn(7, output_features, dtype=torch.bfloat16, device="cuda")
    ffn_gate = torch.randn(5, output_features, dtype=torch.bfloat16, device="cuda")
    gate_indices = torch.tensor(
        [-1, -2, -3, -4, -5, 0, 1, 2, 3, 4],
        dtype=torch.int64,
        device="cuda",
    )
    expected = _materialized_gated_updates(
        _materialized(operands),
        base,
        reusable_update,
        update_gate,
        ffn_gate,
        gate_indices,
    )
    actual = reusable_update.clone()

    result = _chunked_swiglu_ffn_gated_updates_op(
        *operands.arguments(8)[:-1],
        base,
        actual,
        update_gate,
        ffn_gate,
        gate_indices,
        True,
        8,
    )

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert result is None
    assert relative_l2 < 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_chunked_ffn_runs_under_dynamic_fullgraph_compile() -> None:
    torch.manual_seed(203)
    operands = _operands(rows=257, bias_dtype=None)

    @torch.compile(fullgraph=True, dynamic=True)
    def run(activation: torch.Tensor) -> torch.Tensor:
        arguments = operands.arguments(128)
        return _chunked_swiglu_ffn_op(activation, *arguments[1:])

    for rows in (257, 385):
        activation = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
        expected = _materialized(
            _Operands(activation, operands.gate, operands.value, operands.down)
        )
        actual = run(activation)
        relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert relative_l2 < 0.01
