"""Shared launch-plan values for ConvRot INT8 implementations."""

from dataclasses import asdict, dataclass, fields


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
        if not isinstance(self.fuse_rotation_quantization, bool):
            raise ValueError("ConvRot preparation fusion choice must be boolean")
        for field in fields(LinearExecutionPlan):
            name, value = field.name, getattr(self, field.name)
            if name == "fuse_rotation_quantization":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"ConvRot launch dimension {name} must be a positive integer")

    def as_dict(self) -> dict[str, int | bool]:
        """Return execution choices as serializable benchmark metadata."""
        return asdict(self)
