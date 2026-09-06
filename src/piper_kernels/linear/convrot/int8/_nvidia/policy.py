"""Support and measured launch policy for the NVIDIA ConvRot INT8 implementation.

These schedules are measured on SM120 and retain the existing defaults on other
supported NVIDIA targets. Hardware support does not imply per-target tuning.
"""

from dataclasses import dataclass

from piper_kernels._triton.targets import AcceleratorTarget

from .._plan import LinearExecutionPlan, fused_preparation_chunks

_FUSED_NUM_WARPS_VALUES = (2, 4, 8, 16)
_ROTATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_QUANTIZATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_MATMUL_BLOCK_M_VALUES = (16, 32, 64, 128)
_MATMUL_BLOCK_N_VALUES = (16, 32, 64, 128, 256)
_MATMUL_BLOCK_K_VALUES = (32, 64, 128)
_MATMUL_NUM_WARPS_VALUES = (2, 4, 8)
_MATMUL_NUM_STAGES_VALUES = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan(LinearExecutionPlan):
    """Launch choices accepted by the existing NVIDIA kernels and tuner."""

    def __post_init__(self) -> None:
        LinearExecutionPlan.__post_init__(self)
        if self.fused_num_warps not in _FUSED_NUM_WARPS_VALUES:
            raise ValueError("ConvRot fused preparation num_warps must be 2, 4, 8, or 16")
        if self.rotation_num_warps not in _ROTATION_NUM_WARPS_VALUES:
            raise ValueError("ConvRot split rotation num_warps must be 1, 2, 4, or 8")
        if self.quantization_num_warps not in _QUANTIZATION_NUM_WARPS_VALUES:
            raise ValueError("ConvRot split quantization num_warps must be 1, 2, 4, or 8")
        if self.matmul_block_m not in _MATMUL_BLOCK_M_VALUES:
            raise ValueError("ConvRot matmul block_m must be 16, 32, 64, or 128")
        if self.matmul_block_n not in _MATMUL_BLOCK_N_VALUES:
            raise ValueError("ConvRot matmul block_n must be 16, 32, 64, 128, or 256")
        if self.matmul_block_k not in _MATMUL_BLOCK_K_VALUES:
            raise ValueError("ConvRot matmul block_k must be 32, 64, or 128")
        if self.matmul_num_warps not in _MATMUL_NUM_WARPS_VALUES:
            raise ValueError("ConvRot matmul num_warps must be 2, 4, or 8")
        if self.matmul_num_stages not in _MATMUL_NUM_STAGES_VALUES:
            raise ValueError("ConvRot matmul num_stages must be 1, 2, 3, or 4")


_FUSED_MAX_CHUNK_SIZE = 16_384
_TWO_WARP_MAX_CHUNK_SIZE = 2_048
_DEFAULT_ROTATION_NUM_WARPS = 4
_DEFAULT_QUANTIZATION_NUM_WARPS = 8


def supports_target(target: AcceleratorTarget) -> bool:
    """Return whether the NVIDIA INT8 linear and update implementation is supported."""
    return target.cuda_capability_at_least(7, 5)


def supports_preparation_target(target: AcceleratorTarget) -> bool:
    """Return whether rotation and quantization can use the NVIDIA implementation."""
    return target.is_nvidia_cuda


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    in_features: int,
) -> NvidiaExecutionPlan:
    """Select the production preparation and GEMM schedule for one linear."""
    if not supports_target(target):
        raise ValueError(f"ConvRot INT8 execution has no optimized policy for {target}")
    fused_chunks = fused_preparation_chunks(in_features)
    fused_num_warps = 4
    if fused_chunks is not None:
        chunk_count, chunk_size = fused_chunks
        if chunk_count > 1 and chunk_size <= _TWO_WARP_MAX_CHUNK_SIZE:
            fused_num_warps = 2
        elif chunk_size == _FUSED_MAX_CHUNK_SIZE:
            fused_num_warps = 8
    return NvidiaExecutionPlan(
        # Prepared inputs may feed weights with different output widths.
        # Keep every preparation choice independent of output width.
        fuse_rotation_quantization=fused_chunks is not None,
        fused_num_warps=fused_num_warps,
        rotation_num_warps=_DEFAULT_ROTATION_NUM_WARPS,
        quantization_num_warps=_DEFAULT_QUANTIZATION_NUM_WARPS,
        matmul_block_m=128,
        matmul_block_n=256,
        matmul_block_k=128,
        matmul_num_warps=8,
    )
