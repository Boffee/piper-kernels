"""Operands and portable references for mixed standard/ConvRot GELU FFNs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

from ..convrot_nvfp4_swiglu_ffn._helpers import _weight as _convrot_weight
from ..nvfp4_gelu_ffn._helpers import _weight as _standard_weight


@dataclass(frozen=True, slots=True)
class Linear:
    weight: PiperNVFP4Tensor | ConvRotNVFP4Tensor
    activation_scale: torch.Tensor | None
    bias: torch.Tensor | None
    dynamic: bool

    @property
    def group_size(self) -> int | None:
        return self.weight.group_size if isinstance(self.weight, ConvRotNVFP4Tensor) else None

    def arguments(self) -> tuple[object, ...]:
        return (
            self.weight.qdata,
            self.weight.scale,
            self.weight.per_tensor_scale,
            self.activation_scale,
            self.bias,
            self.dynamic,
            self.group_size,
            self.weight.high_first,
        )


@dataclass(frozen=True, slots=True)
class Operands:
    input: torch.Tensor
    up: Linear
    down: Linear

    def arguments(self, chunk_rows: int) -> tuple[object, ...]:
        return self.input, *self.up.arguments(), *self.down.arguments(), chunk_rows


def _activation_scale(input: torch.Tensor, group_size: int | None) -> torch.Tensor:  # noqa: A002
    values = input if group_size is None else rotate_groups(input.float(), group_size)
    return per_tensor_amax_to_scale(values.abs().amax())


def _weight(
    dense: torch.Tensor,
    activation_scale: torch.Tensor | None,
    dynamic: bool,
    group_size: int | None,
    high_first: bool,
) -> PiperNVFP4Tensor | ConvRotNVFP4Tensor:
    if group_size is None:
        return _standard_weight(dense, activation_scale, dynamic, high_first)
    return _convrot_weight(dense, activation_scale, dynamic, group_size, high_first)


def precise_linear(input: torch.Tensor, linear: Linear) -> torch.Tensor:  # noqa: A002
    """Run one represented standard or ConvRot NVFP4 projection."""
    return nvfp4_reference.linear(
        input,
        linear.weight.qdata,
        linear.weight.scale,
        linear.weight.per_tensor_scale,
        linear.activation_scale,
        linear.bias,
        linear.dynamic,
        linear.weight.high_first,
        group_size=linear.group_size or 0,
    )


def activated_down(input: torch.Tensor, linear: Linear) -> torch.Tensor:  # noqa: A002
    """Apply FP32 GELU and directly prepare the down projection input."""
    prepared = nvfp4_reference.prepare_input(
        input,
        linear.activation_scale,
        linear.dynamic,
        "gelu_tanh",
        linear.weight.high_first,
        group_size=linear.group_size or 0,
    )
    return nvfp4_reference.linear_prepared(
        *prepared,
        linear.weight.qdata,
        linear.weight.scale,
        linear.weight.per_tensor_scale,
        linear.bias,
        input.dtype,
    ).reshape(*input.shape[:-1], linear.weight.shape[0])


def materialized(operands: Operands) -> torch.Tensor:
    """Run the equivalent two-projection path with direct GELU preparation."""
    return activated_down(precise_linear(operands.input, operands.up), operands.down)


def make_operands(  # noqa: PLR0913
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    up_dynamic: bool,
    down_dynamic: bool,
    up_group_size: int | None = 16,
    down_group_size: int | None = 64,
    dtype: torch.dtype = torch.bfloat16,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    up_high_first: bool = False,
    down_high_first: bool = False,
    seed: int = 981,
) -> Operands:
    torch.manual_seed(seed)
    input = torch.randn(rows, input_features, device="cuda", dtype=dtype)  # noqa: A001
    up_dense = torch.randn(
        intermediate_features,
        input_features,
        device="cuda",
        dtype=dtype,
    )
    down_dense = torch.randn(
        output_features,
        intermediate_features,
        device="cuda",
        dtype=dtype,
    )

    def make_linear(
        dense: torch.Tensor,
        scale: torch.Tensor | None,
        dynamic: bool,
        group_size: int | None,
        high_first: bool,
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

    up_scale = None if up_dynamic else _activation_scale(input, up_group_size)
    up = make_linear(up_dense, up_scale, up_dynamic, up_group_size, up_high_first)
    activated = F.gelu(precise_linear(input, up).float(), approximate="tanh")
    down_scale = None if down_dynamic else _activation_scale(activated, down_group_size)
    down = make_linear(
        down_dense,
        down_scale,
        down_dynamic,
        down_group_size,
        down_high_first,
    )
    return Operands(input, up, down)


__all__ = ["Linear", "Operands", "make_operands", "materialized"]
