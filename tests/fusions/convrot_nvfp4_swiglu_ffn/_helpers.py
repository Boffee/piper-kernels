"""Shared operands and references for semantic ConvRot NVFP4 FFN tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.linear.convrot.nvfp4 import _ops as convrot_nvfp4_ops


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


def _weight(
    dense: torch.Tensor,
    activation_scale: torch.Tensor | None,
    dynamic: bool,
    group_size: int,
    high_first: bool,
) -> ConvRotNVFP4Tensor:
    quantization = QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )
    rotated = rotate_groups(dense, group_size)
    weight = ConvRotNVFP4Tensor.from_torchao(
        TorchAONVFP4Tensor.to_nvfp4(
            rotated,
            per_tensor_scale=per_tensor_amax_to_scale(rotated.abs().amax()),
            act_per_tensor_scale=activation_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=quantization,
        ),
        group_size=group_size,
    )
    if not high_first:
        return weight
    return ConvRotNVFP4Tensor(
        ((weight.qdata & 0x0F) << 4) | (weight.qdata >> 4),
        weight.scale,
        weight.block_size,
        weight.orig_dtype,
        weight.group_size,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
        weight.is_swizzled_scales,
        weight.use_triton_kernel,
        weight.act_quant_kwargs,
        high_first=True,
    )


def _activation_scale(input: torch.Tensor, group_size: int) -> torch.Tensor:  # noqa: A002
    return per_tensor_amax_to_scale(rotate_groups(input, group_size).abs().amax())


def materialized(operands: Operands) -> torch.Tensor:
    """Run the equivalent three-linear semantic graph."""
    gate = convrot_nvfp4_ops.linear(operands.input, *operands.gate.arguments())
    value = convrot_nvfp4_ops.linear(operands.input, *operands.value.arguments())
    return convrot_nvfp4_ops.linear(value * F.silu(gate), *operands.down.arguments())


def make_operands(  # noqa: PLR0913
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    dynamic: bool,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    source_group_size: int = 16,
    down_group_size: int = 64,
    high_first: bool = False,
    distinct_input_scales: bool = False,
    seed: int = 951,
) -> Operands:
    torch.manual_seed(seed)
    input = torch.randn(rows, input_features, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    gate_dense = torch.randn(
        intermediate_features,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_dense = torch.randn_like(gate_dense)
    down_dense = torch.randn(
        output_features,
        intermediate_features,
        device="cuda",
        dtype=torch.bfloat16,
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
            _weight(dense, scale, dynamic, group_size, high_first),
            scale,
            bias,
            dynamic,
        )

    gate = make_linear(gate_dense, input_scale, source_group_size)
    value = make_linear(value_dense, value_scale, source_group_size)
    activated = convrot_nvfp4_ops.linear(input, *value.arguments()) * F.silu(
        convrot_nvfp4_ops.linear(input, *gate.arguments())
    )
    down_scale = None if dynamic else _activation_scale(activated, down_group_size)
    return Operands(
        input,
        gate,
        value,
        make_linear(down_dense, down_scale, down_group_size),
    )


__all__ = ["Linear", "Operands", "make_operands", "materialized"]
