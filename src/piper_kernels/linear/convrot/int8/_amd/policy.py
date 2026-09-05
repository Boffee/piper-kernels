"""Conservative AMD support and measured RDNA4 ConvRot INT8 launch policy."""

import sys
from dataclasses import dataclass

from piper_kernels._triton.targets import AcceleratorTarget

from .._plan import LinearExecutionPlan

# gfx1201 has hardware correctness/performance coverage; the remaining targets
# have offline compilation coverage. Unknown architectures use the reference.
_SUPPORTED_ARCHITECTURES = ("gfx942", "gfx1100", "gfx1151", "gfx1200", "gfx1201")


@dataclass(frozen=True, slots=True)
class AmdExecutionPlan(LinearExecutionPlan):
    """AMD-owned launch constraints, including wave32 preparation with 32 warps."""

    def __post_init__(self) -> None:
        LinearExecutionPlan.__post_init__(self)
        for name in ("fused_num_warps", "rotation_num_warps", "quantization_num_warps"):
            if getattr(self, name) not in (1, 2, 4, 8, 16, 32):
                raise ValueError(f"AMD {name} must be a power of two from 1 through 32")
        if self.matmul_num_warps not in (4, 8):
            raise ValueError("AMD matmul_num_warps must be 4 or 8")
        for name in ("matmul_block_m", "matmul_block_n", "matmul_block_k"):
            if getattr(self, name) not in (16, 32, 64, 128, 256):
                raise ValueError(f"AMD {name} must be a power of two from 16 through 256")
        if self.matmul_num_stages not in (1, 2, 3, 4):
            raise ValueError("AMD matmul_num_stages must be 1, 2, 3, or 4")


def supports_target(target: AcceleratorTarget) -> bool:
    """Select only Linux HIP architectures covered by the AMD compile tests."""
    return (
        sys.platform == "linux"
        and target.is_amd_hip
        and target.is_architecture(*_SUPPORTED_ARCHITECTURES)
    )


def preparation_blocks(row_width: int) -> tuple[int, int, int]:
    """Choose up to three group-aligned power-of-two row segments."""
    if row_width <= 0:
        raise ValueError("ConvRot preparation row width must be positive")
    if (row_width & (row_width - 1)) == 0:
        return row_width, 0, 0

    leading_block = 1 << (row_width.bit_length() - 1)
    tail_width = row_width - leading_block
    tail_block = 1 << (tail_width - 1).bit_length()
    leading_tail = (leading_block, tail_block, 0)

    equal_block_width = (row_width + 2) // 3
    equal_block = 1 << (equal_block_width - 1).bit_length()
    equal_count = (row_width + equal_block - 1) // equal_block
    if equal_count in (2, 3) and equal_count * equal_block <= leading_block + tail_block:
        return (equal_block, equal_block, equal_block if equal_count == 3 else 0)
    return leading_tail


def select_execution_plan(target: AcceleratorTarget, *, in_features: int) -> AmdExecutionPlan:
    """Keep input preparation independent of the weights and output width."""
    if not supports_target(target):
        raise ValueError(f"ConvRot INT8 execution has no optimized policy for {target}")
    if in_features <= 0:
        raise ValueError("ConvRot preparation row width must be positive")
    block_size = max(128, 1 << (in_features - 1).bit_length())
    rdna4 = target.is_architecture("gfx1200", "gfx1201")
    fused_num_warps = 4
    if rdna4:
        if in_features & (in_features - 1):
            fused_num_warps = 16 if block_size > 8192 else 4
        else:
            fused_num_warps = 32 if block_size > 8192 else 8
    return AmdExecutionPlan(
        fuse_rotation_quantization=block_size <= 16384,
        fused_num_warps=fused_num_warps,
        rotation_num_warps=4,
        quantization_num_warps=4,
        matmul_block_m=128 if rdna4 else 64,
        matmul_block_n=256 if rdna4 else 64,
        matmul_block_k=64,
        matmul_num_warps=8 if rdna4 else 4,
        matmul_num_stages=2 if rdna4 else 1,
    )
