"""Execution plan for fused ConvRot INT8 matrix-product updates."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AddmmExecutionPlan:
    """One shared schedule; only column width varies with rotation groups."""

    block_m: int = field(default=16, init=False)
    block_n: int = 128
    block_k: int = field(default=32, init=False)
    num_warps: int = field(default=4, init=False)
    num_stages: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.block_n not in (128, 256):
            raise ValueError("INT8 update feature tiles must be 128 or 256")
