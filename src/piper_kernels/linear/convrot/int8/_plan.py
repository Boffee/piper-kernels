"""Shared launch-plan values for ConvRot INT8 implementations."""

from dataclasses import asdict, dataclass, fields


def fused_preparation_chunks(in_features: int) -> tuple[int, int] | None:
    """Return ``(chunk_count, chunk_size)`` for fused input preparation.

    Preserve the inexpensive single-chunk path for small rows. Above it, choose
    the supported layout with the least padded work, preferring fewer chunks on
    ties.
    """
    block_size = max(128, 1 << (in_features - 1).bit_length())
    if block_size <= 4096:
        return 1, block_size

    candidates: list[tuple[int, int]] = []
    for chunk_count in (1, 2, 3):
        chunk_width = (in_features + chunk_count - 1) // chunk_count
        chunk_size = max(128, 1 << (chunk_width - 1).bit_length())
        if chunk_size > 16384:
            continue
        if chunk_count == 1 and chunk_size > 8192:
            continue
        if chunk_size * (chunk_count - 1) >= in_features:
            continue
        candidates.append((chunk_count, chunk_size))

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (candidate[0] * candidate[1], candidate[0]),
    )


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
