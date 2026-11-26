"""ConvRot NVFP4 weight construction shared by fusion tests."""

import torch

from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor


def make_weight(
    dense: torch.Tensor,
    activation_scale: torch.Tensor | None,
    dynamic: bool,
    group_size: int,
    high_first: bool,
) -> ConvRotNVFP4Tensor:
    weight = ConvRotNVFP4Tensor.from_hp(
        dense,
        group_size=group_size,
        compute_per_tensor_scale=True,
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant="dynamic" if dynamic else "static",
    )
    if not high_first:
        return weight
    return ConvRotNVFP4Tensor(
        ((weight.qdata & 0x0F) << 4) | (weight.qdata >> 4),
        weight.scale,
        weight.block_size,
        weight.dtype,
        weight.group_size,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
        weight.is_swizzled_scales,
        weight.act_quant,
        high_first=True,
    )
