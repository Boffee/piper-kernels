"""AMD ConvRot INT8 execution choices and implementation constraints."""

from dataclasses import dataclass

from .._plan import LinearExecutionPlan


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
