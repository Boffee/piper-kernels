"""NVFP4 projection operands and portable affine reference for fusion tests."""

from dataclasses import dataclass

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor as TorchAONVFP4Tensor
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


def make_weight(
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
    # TorchAO's reference quantizer accepts BF16/FP32; retain the logical input dtype.
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
    """Reference affine accumulation in FP32 using the represented NVFP4 operands."""
    return nvfp4_reference.linear(input, *linear.arguments())
