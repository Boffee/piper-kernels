"""Shared operands and references for semantic ConvRot NVFP4 FFN tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor

from .._convrot_nvfp4 import make_weight


@dataclass(frozen=True, slots=True)
class Linear:
    weight: ConvRotNVFP4Tensor
    activation_scale: torch.Tensor | None
    bias: torch.Tensor | None
    dynamic: bool

    def arguments(self) -> tuple[object, ...]:
        return (
            self.weight.qdata,
            self.weight.scale,
            self.weight.per_tensor_scale,
            self.activation_scale,
            self.bias,
            self.dynamic,
            self.weight.group_size,
            self.weight.high_first,
        )


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


def _activation_scale(input: torch.Tensor, group_size: int) -> torch.Tensor:  # noqa: A002
    return per_tensor_amax_to_scale(rotate_groups(input, group_size).abs().amax())


def precise_linear(input: torch.Tensor, linear: Linear) -> torch.Tensor:  # noqa: A002
    """Run portable rotation, quantization and FP32 affine accumulation."""
    return nvfp4_reference.linear(
        input,
        linear.weight.qdata,
        linear.weight.scale,
        linear.weight.per_tensor_scale,
        linear.activation_scale,
        linear.bias,
        linear.dynamic,
        linear.weight.high_first,
        group_size=linear.weight.group_size,
    )


def materialized(operands: Operands) -> torch.Tensor:
    """Run the equivalent three-linear graph using only portable PyTorch operations."""
    gate = precise_linear(operands.input, operands.gate)
    value = precise_linear(operands.input, operands.value)
    return precise_linear(value * F.silu(gate), operands.down)


def make_operands(  # noqa: PLR0913
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    dynamic: bool,
    dtype: torch.dtype = torch.bfloat16,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    source_group_size: int = 16,
    down_group_size: int = 64,
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
    input_scale = None if dynamic else _activation_scale(input, source_group_size)
    value_scale = (
        input_scale * 0.875 if distinct_input_scales and input_scale is not None else input_scale
    )

    def make_linear(
        dense: torch.Tensor,
        scale: torch.Tensor | None,
        group_size: int,
    ) -> Linear:
        bias = (
            torch.randn(dense.shape[0], device="cuda", dtype=bias_dtype)
            if bias_dtype is not None
            else None
        )
        return Linear(
            make_weight(dense, scale, dynamic, group_size, high_first),
            scale,
            bias,
            dynamic,
        )

    gate = make_linear(gate_dense, input_scale, source_group_size)
    value = make_linear(value_dense, value_scale, source_group_size)
    down_scale = None
    if not dynamic:
        activated = precise_linear(input, value) * F.silu(precise_linear(input, gate))
        down_scale = _activation_scale(activated, down_group_size)
    return Operands(
        input,
        gate,
        value,
        make_linear(down_dense, down_scale, down_group_size),
    )


__all__ = ["Linear", "Operands", "make_operands", "materialized"]
