"""Conservative shared Triton launchers without a GPU-model allowlist."""

import torch

from piper_kernels.weights.convrot.int8.triton import quantize_input
from piper_kernels.weights.convrot.triton import rotate_input


def prepare_input(input, group_size, input_scale=None, *, out):  # noqa: A002
    """Use separate rotation and row quantization to bound fused live storage."""
    width = input.shape[-1]
    value = input.reshape(-1, width).contiguous()
    if value.shape[0] == 0:
        return out
    rotated = torch.empty(value.shape, device=value.device, dtype=torch.float32)
    rotate_input(value, rotated, group_size, num_warps=4)
    quantize_input(rotated, out[0], out[1], input_scale=input_scale, num_warps=4)
    return out
