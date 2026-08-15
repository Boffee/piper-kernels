"""Backend-independent execution planning for optimized INT8 ConvRot."""

from dataclasses import asdict, dataclass

import torch

from piper_kernels._triton.targets import AcceleratorTarget

_FUSED_GROUP_SIZE = 256
_FUSED_MIN_ROWS = 512
_FUSED_MAX_BLOCK_SIZE = 16_384
_FUSED_DTYPES = (torch.float16, torch.bfloat16)
_LARGE_SWIGLU_MIN_ROWS = 8192
_SM120_LARGE_MATMUL_MIN_ROWS = 512
_FUSED_NUM_WARPS_VALUES = (2, 4, 8, 16)
_ROTATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_QUANTIZATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_MATMUL_BLOCK_M_VALUES = (16, 32, 64, 128)
_MATMUL_BLOCK_N_VALUES = (16, 32, 64, 128, 256)
_MATMUL_BLOCK_K_VALUES = (32, 64, 128)
_MATMUL_NUM_WARPS_VALUES = (2, 4, 8)
_MATMUL_NUM_STAGES_VALUES = (1, 2, 3, 4)
_DEFAULT_ROTATION_NUM_WARPS = 4
_DEFAULT_QUANTIZATION_NUM_WARPS = 8


def _preparation_block_size(in_features: int) -> int:
    return 128 if in_features <= 0 else max(128, 1 << (in_features - 1).bit_length())


@dataclass(frozen=True, slots=True)
class ConvRotInt8LinearExecutionPlan:
    """Host-side preparation and GEMM choices for one ConvRot INT8 invocation."""

    fuse_rotation_quantization: bool
    fused_num_warps: int
    rotation_num_warps: int
    quantization_num_warps: int
    matmul_block_m: int
    matmul_block_n: int
    matmul_block_k: int
    matmul_num_warps: int = 4
    matmul_num_stages: int = 3

    def __post_init__(self) -> None:
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

    def as_dict(self) -> dict[str, int | bool]:
        """Return execution choices as serializable benchmark metadata."""
        return asdict(self)


def _select_fuse_rotation_quantization(
    target: AcceleratorTarget,
    *,
    rows: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
) -> bool:
    """Select whether production fuses activation rotation and quantization."""
    block_size = _preparation_block_size(in_features)
    return (
        in_features > 0
        and group_size == _FUSED_GROUP_SIZE
        and target.is_cuda_capability(12, 0)
        and dtype in _FUSED_DTYPES
        and rows >= _FUSED_MIN_ROWS
        and block_size <= _FUSED_MAX_BLOCK_SIZE
    )


def _select_fused_num_warps(
    target: AcceleratorTarget,
    *,
    rows: int,
    in_features: int,
    swiglu: bool,
) -> int:
    """Select the fused candidate's launch width, including forced tuning runs."""
    block_size = _preparation_block_size(in_features)
    large_swiglu = (
        target.is_cuda_capability(12, 0) and swiglu and block_size == _FUSED_MAX_BLOCK_SIZE
    )
    return 16 if large_swiglu and rows >= _LARGE_SWIGLU_MIN_ROWS else 8 if large_swiglu else 4


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    rows: int,
    out_features: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    swiglu: bool = False,
) -> ConvRotInt8LinearExecutionPlan:
    """Select the production preparation and GEMM schedule for one linear."""
    large_sm120 = target.is_cuda_capability(12, 0) and rows >= _SM120_LARGE_MATMUL_MIN_ROWS
    return ConvRotInt8LinearExecutionPlan(
        fuse_rotation_quantization=_select_fuse_rotation_quantization(
            target,
            rows=rows,
            in_features=in_features,
            group_size=group_size,
            dtype=dtype,
        ),
        fused_num_warps=_select_fused_num_warps(
            target,
            rows=rows,
            in_features=in_features,
            swiglu=swiglu,
        ),
        rotation_num_warps=_DEFAULT_ROTATION_NUM_WARPS,
        quantization_num_warps=_DEFAULT_QUANTIZATION_NUM_WARPS,
        matmul_block_m=128 if large_sm120 else (32 if rows < 64 else 64),
        matmul_block_n=256 if large_sm120 else (64 if out_features < 128 else 128),
        matmul_block_k=128 if large_sm120 else 32,
        matmul_num_warps=8 if large_sm120 else 4,
    )
