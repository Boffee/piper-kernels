"""Backend-independent execution planning for optimized INT8 ConvRot."""

from dataclasses import asdict, dataclass

_FUSED_MAX_BLOCK_SIZE = 16_384
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
    return max(128, 1 << (in_features - 1).bit_length())


@dataclass(frozen=True, slots=True)
class LinearExecutionPlan:
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


def _select_fuse_rotation_quantization(*, in_features: int) -> bool:
    """Select whether production fuses input rotation and quantization."""
    block_size = _preparation_block_size(in_features)
    return block_size <= _FUSED_MAX_BLOCK_SIZE


def select_execution_plan(
    *,
    in_features: int,
) -> LinearExecutionPlan:
    """Select the production preparation and GEMM schedule for one linear."""
    return LinearExecutionPlan(
        # Prepared inputs may feed weights with different output widths.
        # Keep every preparation choice independent of output width.
        fuse_rotation_quantization=_select_fuse_rotation_quantization(
            in_features=in_features,
        ),
        fused_num_warps=4,
        rotation_num_warps=_DEFAULT_ROTATION_NUM_WARPS,
        quantization_num_warps=_DEFAULT_QUANTIZATION_NUM_WARPS,
        matmul_block_m=128,
        matmul_block_n=256,
        matmul_block_k=128,
        matmul_num_warps=8,
    )
