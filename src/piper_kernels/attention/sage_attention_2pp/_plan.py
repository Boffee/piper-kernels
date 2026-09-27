"""SageAttention2++ execution choices, validation, and reporting."""

from dataclasses import asdict, dataclass

from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)


@dataclass(frozen=True, slots=True)
class SageAttention2ppExecutionPlan:
    """Host-side specialization choices for one SageAttention2++ invocation."""

    block_m: int
    grouped_qk: bool
    use_tensor_descriptors: bool
    use_packed_probability_conversion: bool = True
    num_warps: int = 4
    num_stages: int = 3
    reverse_causal_blocks: bool = False
    loop_num_stages: int | None = None
    loop_licm: bool = False

    def __post_init__(self) -> None:
        if self.block_m not in BLOCK_M_VALUES:
            raise ValueError("SageAttention2++ block_m must be 64 or 128")
        if self.num_warps not in NUM_WARPS_VALUES:
            raise ValueError("SageAttention2++ num_warps must be 2, 4, or 8")
        if self.num_stages not in NUM_STAGES_VALUES:
            raise ValueError("SageAttention2++ num_stages must be 1, 2, 3, or 4")
        if self.loop_num_stages not in LOOP_NUM_STAGES_VALUES:
            raise ValueError("SageAttention2++ loop_num_stages must be None, 1, 2, 3, or 4")

    def as_dict(self) -> dict[str, object]:
        """Return the execution-plan fields as plain metadata."""
        return asdict(self)
