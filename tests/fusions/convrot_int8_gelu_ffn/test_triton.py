"""Tests for the direct chunked ConvRot INT8 GELU FFN."""

from dataclasses import dataclass

import pytest
import torch

from piper_kernels.fusions.convrot_int8_gelu_ffn.triton import (
    _chunked_gelu_ffn_gated_updates_op,
    _chunked_gelu_ffn_op,
)
from piper_kernels.linear.convrot.int8 import _ops

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm"),
]


@dataclass(frozen=True, slots=True)
class _Linear:
    qdata: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None
    group_size: int

    def arguments(self) -> tuple[object, ...]:
        return self.qdata, self.scale, self.bias, self.group_size


@dataclass(frozen=True, slots=True)
class _Operands:
    input: torch.Tensor
    up: _Linear
    down: _Linear

    def arguments(
        self,
        chunk_rows: int,
        up_input_scale: torch.Tensor | None = None,
        down_input_scale: torch.Tensor | None = None,
    ) -> tuple[object, ...]:
        return (
            self.input,
            *self.up.arguments(),
            *self.down.arguments(),
            chunk_rows,
            up_input_scale,
            down_input_scale,
        )


def _linear(
    out_features: int,
    in_features: int,
    bias_dtype: torch.dtype | None,
    group_size: int,
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
    return _Linear(qdata, scale, bias, group_size)


def _operands(
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 256,
    down_group_size: int = 256,
) -> _Operands:
    input = torch.randn(rows, input_features, dtype=dtype, device="cuda")  # noqa: A001
    return _Operands(
        input,
        _linear(intermediate_features, input_features, bias_dtype, group_size),
        _linear(output_features, intermediate_features, bias_dtype, down_group_size),
    )


def _materialized(
    operands: _Operands,
    up_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    up = _ops.linear(
        operands.input,
        *operands.up.arguments(),
        None,
        up_input_scale,
    )
    return _ops.linear(
        up,
        *operands.down.arguments(),
        "gelu_tanh",
        down_input_scale,
    )


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).norm() / expected.float().norm()


@pytest.mark.parametrize("rows", [127, 385], ids=["short", "ragged-multi-chunk"])
@pytest.mark.parametrize("chunk_rows", [64, 128])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_chunked_ffn_matches_materialized(rows, chunk_rows, bias_dtype, dtype) -> None:
    torch.manual_seed(501 + rows)
    operands = _operands(rows=rows, bias_dtype=bias_dtype, dtype=dtype)

    expected = _materialized(operands)
    actual = _chunked_gelu_ffn_op(*operands.arguments(chunk_rows))

    assert actual.dtype is dtype
    assert _relative_l2(actual, expected) < 0.01


@pytest.mark.parametrize("up_static", [False, True])
@pytest.mark.parametrize("down_static", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_static_and_dynamic_input_scales_match_materialized(up_static, down_static, dtype) -> None:
    torch.manual_seed(503)
    operands = _operands(rows=129, bias_dtype=torch.float32, dtype=dtype)
    up_scale = torch.tensor(0.02, device="cuda") if up_static else None
    down_scale = torch.tensor(0.04, device="cuda") if down_static else None

    expected = _materialized(operands, up_scale, down_scale)
    actual = _chunked_gelu_ffn_op(*operands.arguments(64, up_scale, down_scale))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("static", [False, True])
def test_custom_op_passes_opcheck(static: bool) -> None:
    operands = _operands(rows=65)
    scale = torch.tensor(0.02, device="cuda") if static else None
    with torch.inference_mode():
        results = torch.library.opcheck(
            _chunked_gelu_ffn_op,
            operands.arguments(64, scale, scale),
        )
    assert set(results.values()) == {"SUCCESS"}


@pytest.mark.parametrize("python_indexing", [False, True])
@pytest.mark.parametrize("output_features", [384, 640])
def test_gated_updates_match_materialized(python_indexing, output_features) -> None:
    torch.manual_seed(505)
    operands = _operands(rows=129, output_features=output_features, bias_dtype=torch.float32)
    rows = operands.input.shape[0]
    base = torch.randn(rows, output_features, dtype=torch.bfloat16, device="cuda")
    reusable_update = torch.randn_like(base)
    update_gate = torch.randn(7, output_features, dtype=torch.bfloat16, device="cuda")
    ffn_gate = torch.randn(7, output_features, dtype=torch.bfloat16, device="cuda")
    gate_indices = torch.randint(-7 if python_indexing else 0, 7, (rows,), device="cuda")
    hidden = base.float() + update_gate[gate_indices].float() * reusable_update.float()
    expected = (hidden + ffn_gate[gate_indices].float() * _materialized(operands).float()).to(
        base.dtype
    )
    actual = reusable_update.clone()

    result = _chunked_gelu_ffn_gated_updates_op(
        *operands.arguments(64)[:-3],
        base,
        actual,
        update_gate,
        ffn_gate,
        gate_indices,
        python_indexing,
        *operands.arguments(64)[-3:],
    )

    assert result is None
    assert _relative_l2(actual, expected) < 0.01


def test_chunked_ffn_runs_under_dynamic_fullgraph_compile() -> None:
    torch.manual_seed(507)
    operands = _operands(rows=257, bias_dtype=None)

    @torch.compile(fullgraph=True, dynamic=True)
    def run(activation: torch.Tensor) -> torch.Tensor:
        arguments = operands.arguments(128)
        return _chunked_gelu_ffn_op(activation, *arguments[1:])

    for rows in (257, 385):
        activation = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
        current = _Operands(activation, operands.up, operands.down)
        assert _relative_l2(run(activation), _materialized(current)) < 0.01
