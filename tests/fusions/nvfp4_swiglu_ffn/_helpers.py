"""Shared operands and references for semantic NVFP4 SwiGLU FFN tests."""

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

from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops


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
    high_first: bool,
) -> PiperNVFP4Tensor:
    quantization = QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )
    weight = PiperNVFP4Tensor.from_torchao(
        TorchAONVFP4Tensor.to_nvfp4(
            dense,
            per_tensor_scale=per_tensor_amax_to_scale(dense.abs().amax()),
            act_per_tensor_scale=activation_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=quantization,
        )
    )
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


def materialized(operands: Operands) -> torch.Tensor:
    """Run separate projections with the fused path's affine precision semantics."""

    def project(linear: Linear) -> torch.Tensor:
        if linear.bias is None or linear.bias.dtype is operands.input.dtype:
            return precise_linear(operands.input, linear)
        return nvfp4_ops.linear(operands.input, *linear.arguments())

    gate = project(operands.gate)
    value = project(operands.value)
    return nvfp4_ops.linear(value * F.silu(gate), *operands.down.arguments())


def precise_linear(input: torch.Tensor, linear: Linear) -> torch.Tensor:  # noqa: A002
    """Reference affine accumulation in FP32 using the represented NVFP4 operands."""
    qdata, scale, global_scale = nvfp4_ops._prepare_compiled(
        input,
        linear.activation_scale,
        linear.dynamic,
        high_first=linear.weight.high_first,
    )
    prepared = PiperNVFP4Tensor(
        qdata,
        scale,
        16,
        input.dtype,
        global_scale,
        is_swizzled_scales=True,
        high_first=linear.weight.high_first,
    )
    return (
        F.linear(
            prepared.dequantize(torch.float32),
            linear.weight.dequantize(torch.float32),
            None if linear.bias is None else linear.bias.float(),
        )
        .to(input.dtype)
        .reshape(*input.shape[:-1], linear.weight.shape[0])
    )


def make_operands(
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    dynamic: bool,
    bias_dtype: torch.dtype | None = torch.bfloat16,
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
        return Linear(_weight(dense, scale, dynamic, high_first), scale, bias, dynamic)

    gate = make_linear(gate_dense, input_scale)
    value = make_linear(value_dense, value_scale)
    activated = nvfp4_ops.linear(input, *value.arguments()) * F.silu(
        nvfp4_ops.linear(input, *gate.arguments())
    )
    down_scale = None if dynamic else per_tensor_amax_to_scale(activated.abs().amax())
    return Operands(input, gate, value, make_linear(down_dense, down_scale))


__all__ = ["Linear", "Operands", "make_operands", "materialized"]
