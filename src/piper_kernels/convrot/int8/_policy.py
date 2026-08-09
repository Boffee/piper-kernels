"""Backend-independent policy for optimized INT8 ConvRot preparation."""

from dataclasses import dataclass

import torch

from piper_kernels._triton.targets import AcceleratorTarget

_FUSED_GROUP_SIZE = 256
_FUSED_MIN_ROWS = 512
_FUSED_MAX_BLOCK_SIZE = 16_384
_FUSED_DTYPES = (torch.float16, torch.bfloat16)
_LARGE_SWIGLU_MIN_ROWS = 8192


@dataclass(frozen=True, slots=True)
class PreparationPlan:
    """Measured host-side choices for one ConvRot activation preparation.

    ``block_size`` and ``fused_num_warps`` describe the fused candidate launch,
    including explicit benchmark launches outside automatic eligibility.
    """

    block_size: int
    fuse_rotation_quantization: bool
    fused_num_warps: int


def select_preparation_plan(
    target: AcceleratorTarget,
    *,
    rows: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    swiglu: bool = False,
) -> PreparationPlan:
    """Select measured preparation policy from explicit target and input facts."""
    block_size = 128 if in_features <= 0 else max(128, 1 << (in_features - 1).bit_length())
    tuned_sm120 = target.is_cuda_capability(12, 0)
    fuse_rotation_quantization = (
        in_features > 0
        and group_size == _FUSED_GROUP_SIZE
        and tuned_sm120
        and dtype in _FUSED_DTYPES
        and rows >= _FUSED_MIN_ROWS
        and block_size <= _FUSED_MAX_BLOCK_SIZE
    )
    large_swiglu = tuned_sm120 and swiglu and block_size == _FUSED_MAX_BLOCK_SIZE
    fused_num_warps = (
        16 if large_swiglu and rows >= _LARGE_SWIGLU_MIN_ROWS else 8 if large_swiglu else 4
    )
    return PreparationPlan(
        block_size=block_size,
        fuse_rotation_quantization=fuse_rotation_quantization,
        fused_num_warps=fused_num_warps,
    )
