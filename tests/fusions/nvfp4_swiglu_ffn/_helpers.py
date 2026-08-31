"""Shared operands and references for NVFP4 SwiGLU FFN tests."""

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
        )


@dataclass(frozen=True, slots=True)
class Operands:
    input: torch.Tensor
    up: Linear
    down: Linear
    dense_up: torch.Tensor
    dense_down: torch.Tensor

    def arguments(self, chunk_rows: int) -> tuple[object, ...]:
        return self.input, *self.up.arguments(), *self.down.arguments(), chunk_rows


def _weight(
    dense: torch.Tensor,
    activation_scale: torch.Tensor | None,
    dynamic: bool,
) -> PiperNVFP4Tensor:
    quantization = QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )
    weight = TorchAONVFP4Tensor.to_nvfp4(
        dense,
        per_tensor_scale=per_tensor_amax_to_scale(dense.abs().amax()),
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=quantization,
    )
    return PiperNVFP4Tensor.from_torchao(weight)


def materialized(operands: Operands) -> torch.Tensor:
    """Run the materialized activation-folded NVFP4 reference."""
    packed = nvfp4_ops.linear(operands.input, *operands.up.arguments())
    prepared = nvfp4_ops.prepare_input(
        packed,
        operands.down.activation_scale,
        operands.down.dynamic,
        "swiglu",
    )
    result = nvfp4_ops.linear_prepared(
        *prepared,
        operands.down.weight.qdata,
        operands.down.weight.scale,
        operands.down.weight.per_tensor_scale,
        operands.down.bias,
        operands.input.dtype,
    )
    return result.reshape(*operands.input.shape[:-1], operands.down.weight.shape[0])


def down_affine_reference(operands: Operands) -> torch.Tensor:
    """Run the materialized reference with the down affine in its FP32 accumulator."""
    packed = nvfp4_ops.linear(operands.input, *operands.up.arguments())
    input_qdata, input_scale, input_per_tensor_scale = nvfp4_ops.prepare_input(
        packed,
        operands.down.activation_scale,
        operands.down.dynamic,
        "swiglu",
    )
    weight_per_tensor_scale = operands.down.weight.per_tensor_scale
    assert weight_per_tensor_scale is not None
    scaling_type = F.ScalingType
    swizzle_type = F.SwizzleType
    result = F.scaled_mm(
        input_qdata.view(torch.float4_e2m1fn_x2),
        operands.down.weight.qdata.t().view(torch.float4_e2m1fn_x2),
        [input_scale.view(torch.float8_e4m3fn), input_per_tensor_scale],
        [scaling_type.BlockWise1x16, scaling_type.TensorWise],
        [operands.down.weight.scale.view(torch.float8_e4m3fn), weight_per_tensor_scale],
        [scaling_type.BlockWise1x16, scaling_type.TensorWise],
        [swizzle_type.SWIZZLE_32_4_4, swizzle_type.NO_SWIZZLE],
        [swizzle_type.SWIZZLE_32_4_4, swizzle_type.NO_SWIZZLE],
        bias=operands.down.bias,
        output_dtype=operands.input.dtype,
    )
    return result.reshape(*operands.input.shape[:-1], operands.down.weight.shape[0])


def dense_reference(operands: Operands) -> torch.Tensor:
    packed = F.linear(operands.input, operands.dense_up, operands.up.bias)
    up, gate = packed.chunk(2, dim=-1)
    return F.linear(up * F.silu(gate), operands.dense_down, operands.down.bias)


def make_operands(
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    dynamic: bool,
    with_bias: bool = True,
    seed: int = 901,
) -> Operands:
    torch.manual_seed(seed)
    input = torch.randn(rows, input_features, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    dense_up = torch.randn(
        2 * intermediate_features,
        input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dense_down = torch.randn(
        output_features,
        intermediate_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    up_bias = (
        torch.randn(2 * intermediate_features, device="cuda", dtype=torch.bfloat16)
        if with_bias
        else None
    )
    down_bias = (
        torch.randn(output_features, device="cuda", dtype=torch.bfloat16) if with_bias else None
    )
    up_activation_scale = None if dynamic else per_tensor_amax_to_scale(input.abs().amax())
    up_weight = _weight(dense_up, up_activation_scale, dynamic)
    up = Linear(up_weight, up_activation_scale, up_bias, dynamic)
    packed = nvfp4_ops.linear(input, *up.arguments())
    packed_up, packed_gate = packed.chunk(2, dim=-1)
    activated = packed_up * F.silu(packed_gate)
    down_activation_scale = None if dynamic else per_tensor_amax_to_scale(activated.abs().amax())
    down = Linear(
        _weight(dense_down, down_activation_scale, dynamic),
        down_activation_scale,
        down_bias,
        dynamic,
    )
    return Operands(input, up, down, dense_up, dense_down)


__all__ = [
    "Linear",
    "Operands",
    "dense_reference",
    "down_affine_reference",
    "make_operands",
    "materialized",
]
