"""Shared operands and references for standard NVFP4 GELU FFN tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor


@dataclass(frozen=True, slots=True)
class Linear:
    weight: PiperNVFP4Tensor
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
            self.weight.high_first,
        )


@dataclass(frozen=True, slots=True)
class Operands:
    input: torch.Tensor
    up: Linear
    down: Linear

    def arguments(self, chunk_rows: int) -> tuple[object, ...]:
        return self.input, *self.up.arguments(), *self.down.arguments(), chunk_rows


def _weight(
    dense: torch.Tensor,
    activation_scale: torch.Tensor | None,
    dynamic: bool,
    high_first: bool,
) -> PiperNVFP4Tensor:
    quantization = QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )
    quantization_input = dense.float() if dense.dtype is torch.float16 else dense
    weight = PiperNVFP4Tensor.from_torchao(
        TorchAONVFP4Tensor.to_nvfp4(
            quantization_input,
            per_tensor_scale=per_tensor_amax_to_scale(dense.abs().amax()),
            act_per_tensor_scale=activation_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=quantization,
        )
    ).to(dtype=dense.dtype)
    if not high_first:
        return weight
    return PiperNVFP4Tensor(
        ((weight.qdata & 0x0F) << 4) | (weight.qdata >> 4),
        weight.scale,
        weight.block_size,
        weight.orig_dtype,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
        weight.is_swizzled_scales,
        weight.use_triton_kernel,
        weight.act_quant_kwargs,
        high_first=True,
    )


def precise_linear(input: torch.Tensor, linear: Linear) -> torch.Tensor:  # noqa: A002
    """Run one represented standard NVFP4 affine projection."""
    return nvfp4_reference.linear(input, *linear.arguments())


def activated_down(input: torch.Tensor, linear: Linear) -> torch.Tensor:  # noqa: A002
    """Apply FP32 GELU and directly prepare the down projection input."""
    prepared = nvfp4_reference.prepare_input(
        input,
        linear.activation_scale,
        linear.dynamic,
        "gelu_tanh",
        linear.weight.high_first,
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
    dtype: torch.dtype = torch.bfloat16,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    up_high_first: bool = False,
    down_high_first: bool = False,
    seed: int = 951,
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
        activation_scale: torch.Tensor | None,
        dynamic: bool,
        high_first: bool,
    ) -> Linear:
        bias = (
            torch.randn(dense.shape[0], device="cuda", dtype=bias_dtype)
            if bias_dtype is not None
            else None
        )
        return Linear(
            _weight(dense, activation_scale, dynamic, high_first),
            activation_scale,
            bias,
            dynamic,
        )

    up_scale = None if up_dynamic else per_tensor_amax_to_scale(input.abs().amax())
    up = make_linear(up_dense, up_scale, up_dynamic, up_high_first)
    projected = precise_linear(input, up)
    activated = torch.nn.functional.gelu(projected.float(), approximate="tanh")
    down_scale = None if down_dynamic else per_tensor_amax_to_scale(activated.abs().amax())
    down = make_linear(down_dense, down_scale, down_dynamic, down_high_first)
    return Operands(input, up, down)


__all__ = ["Linear", "Operands", "make_operands", "materialized"]
