"""Shared operands, models, and references for semantic NVFP4 SwiGLU FFN tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from .._nvfp4 import Linear, make_weight, precise_linear


@dataclass(frozen=True, slots=True)
class Operands:
    input: torch.Tensor
    gate: Linear
    value: Linear
    down: Linear

    def arguments(self, chunk_rows: int) -> tuple[object, ...]:
        return (
            self.input,
            *self.gate.arguments(),
            *self.value.arguments(),
            *self.down.arguments(),
            chunk_rows,
        )


def materialized(operands: Operands) -> torch.Tensor:
    """Run the three projections using independent portable PyTorch operations."""
    gate = precise_linear(operands.input, operands.gate)
    value = precise_linear(operands.input, operands.value)
    return precise_linear(value * F.silu(gate), operands.down)


def make_operands(
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    dynamic: bool,
    dtype: torch.dtype = torch.bfloat16,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    high_first: bool = False,
    distinct_input_scales: bool = False,
    seed: int = 951,
) -> Operands:
    torch.manual_seed(seed)
    input = torch.randn(rows, input_features, device="cuda", dtype=dtype)  # noqa: A001
    gate_dense = torch.randn(
        intermediate_features,
        input_features,
        device="cuda",
        dtype=dtype,
    )
    value_dense = torch.randn_like(gate_dense)
    down_dense = torch.randn(
        output_features,
        intermediate_features,
        device="cuda",
        dtype=dtype,
    )
    input_scale = None if dynamic else per_tensor_amax_to_scale(input.abs().amax())
    value_scale = (
        input_scale * 0.875 if distinct_input_scales and input_scale is not None else input_scale
    )

    def make_linear(dense: torch.Tensor, scale: torch.Tensor | None) -> Linear:
        bias = (
            torch.randn(dense.shape[0], device="cuda", dtype=bias_dtype)
            if bias_dtype is not None
            else None
        )
        return Linear(make_weight(dense, scale, dynamic, high_first), scale, bias, dynamic)

    gate = make_linear(gate_dense, input_scale)
    value = make_linear(value_dense, value_scale)
    down_scale = None
    if not dynamic:
        activated = precise_linear(input, value) * F.silu(precise_linear(input, gate))
        down_scale = per_tensor_amax_to_scale(activated.abs().amax())
    return Operands(input, gate, value, make_linear(down_dense, down_scale))


class SwiGluFfn(torch.nn.Module):
    def __init__(
        self,
        operands: Operands,
        *,
        promote_gate: bool = False,
        reverse_multiply: bool = False,
        expose_gate: bool = False,
    ) -> None:
        super().__init__()
        self.promote_gate = promote_gate
        self.reverse_multiply = reverse_multiply
        self.expose_gate = expose_gate
        self.gate = self._linear(operands.gate)
        self.value = self._linear(operands.value)
        self.down = self._linear(operands.down)

    @staticmethod
    def _linear(operands: Linear) -> torch.nn.Linear:
        out_features, in_features = operands.weight.shape
        linear = torch.nn.Linear(
            in_features,
            out_features,
            bias=operands.bias is not None,
            device="cuda",
            dtype=operands.weight.dtype,
        )
        linear.weight = torch.nn.Parameter(operands.weight, requires_grad=False)
        if operands.bias is not None:
            linear.bias = torch.nn.Parameter(operands.bias, requires_grad=False)
        return linear

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        value_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(input)
        value = self.value(input if value_input is None else value_input)
        activated_gate = F.silu(gate.float()).to(gate.dtype) if self.promote_gate else F.silu(gate)
        activated = activated_gate * value if self.reverse_multiply else value * activated_gate
        output = self.down(activated)
        return (output, gate) if self.expose_gate else output
