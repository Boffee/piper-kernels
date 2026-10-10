"""ConvRot NVFP4 weight construction shared by fusion tests."""

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor as TorchAONVFP4Tensor
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor


def make_weight(
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
    # TorchAO's reference quantizer accepts BF16/FP32; retain the logical input dtype.
    quantization_input = rotated.float() if dense.dtype is torch.float16 else rotated
    storage = PiperNVFP4Tensor.from_torchao(
        TorchAONVFP4Tensor.to_nvfp4(
            quantization_input,
            per_tensor_scale=per_tensor_amax_to_scale(rotated.abs().amax()),
            act_per_tensor_scale=activation_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=quantization,
        ),
    ).to(dtype=dense.dtype)
    weight = ConvRotNVFP4Tensor.from_torchao(storage, group_size=group_size)
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
