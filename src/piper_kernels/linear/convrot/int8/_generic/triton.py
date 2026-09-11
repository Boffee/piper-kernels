"""Conservative shared Triton launchers without a GPU-model allowlist."""

# pyright: reportCallIssue=false

import torch
import triton

from piper_kernels._triton.convrot import rotate_input
from piper_kernels._triton.convrot_int8 import quantize_rows_kernel
from piper_kernels._triton.runtime import device_context


def _reciprocal_scale(value: torch.Tensor) -> bool:
    return value.device.type == "cuda" and torch.version.hip is not None


def prepare_input(input, group_size, *, out):  # noqa: A002
    """Use separate rotation and row quantization to bound fused live storage."""
    width = input.shape[-1]
    value = input.reshape(-1, width)
    rotated = torch.empty_like(value)
    rotate_input(value, rotated, group_size, num_warps=4)
    with device_context(input.device):
        quantize_rows_kernel[(value.shape[0],)](
            rotated,
            out[0],
            out[1],
            width,
            block_size=max(128, triton.next_power_of_2(width)),
            reciprocal_scale=_reciprocal_scale(value),
            accelerator_backend="hip" if _reciprocal_scale(value) else value.device.type,
            num_warps=4,
        )
        return out
