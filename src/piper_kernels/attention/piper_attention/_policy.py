"""Backend-independent execution planning for Piper Attention."""

from dataclasses import asdict, dataclass

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)


@dataclass(frozen=True, slots=True)
class PiperAttentionExecutionPlan:
    """Host-side specialization and launch choices for one Piper invocation."""

    block_m: int
    grouped_qk: bool
    native_uint8: bool
    split_pv_head_dim: bool
    scaled_fp16_numerator: bool
    use_tensor_descriptors: bool
    num_warps: int = 4
    num_stages: int = 3
    reverse_causal_blocks: bool = False
    loop_num_stages: int | None = None
    loop_licm: bool = False
    use_packed_probability_conversion: bool = False

    def __post_init__(self) -> None:
        if self.block_m not in BLOCK_M_VALUES:
            raise ValueError("Piper Attention block_m must be 32, 64, or 128")
        if self.num_warps not in NUM_WARPS_VALUES:
            raise ValueError("Piper Attention num_warps must be 2, 4, or 8")
        if self.num_stages not in NUM_STAGES_VALUES:
            raise ValueError("Piper Attention num_stages must be 1, 2, 3, or 4")
        if self.loop_num_stages not in LOOP_NUM_STAGES_VALUES:
            raise ValueError("Piper Attention loop_num_stages must be None, 1, 2, 3, or 4")
        if self.scaled_fp16_numerator and not self.split_pv_head_dim:
            raise ValueError("scaled FP16 numerator recurrence requires split PV")
        if self.use_packed_probability_conversion and not self.native_uint8:
            raise ValueError("packed probability conversion requires native UINT8 MMA")

    def as_dict(self) -> dict[str, object]:
        """Return execution choices as serializable benchmark metadata."""
        return asdict(self)


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    candidate_block_m: int,
    query_length: int,
    key_length: int,
    head_dim: int,
    is_causal: bool,
) -> PiperAttentionExecutionPlan:
    """Select established policy without borrowing schedules from other kernels."""
    grouped_qk = target.is_cuda_capability(12)
    native_uint8 = target.supports_uint8_int8_mma
    split_pv_head_dim = (
        target.is_cuda_capability(12)
        and not is_causal
        and head_dim == 128
        and query_length >= 1024
        and key_length >= 1024
    )
    scaled_fp16_numerator = split_pv_head_dim and key_length <= 131072
    # Paired SM120 measurements favor packed conversion for D64 and
    # non-causal D128, while the D128 causal specialization is neutral to
    # slightly slower and retains stock Triton lowering.
    use_packed_probability_conversion = target.is_cuda_capability(12, 0) and not (
        is_causal and head_dim == 128
    )

    block_m = (
        64
        if is_causal
        else 128
        if scaled_fp16_numerator and query_length >= 8192 and key_length >= 8192
        else 64
        if split_pv_head_dim
        else candidate_block_m
    )
    use_tensor_descriptors = target.is_cuda_capability(12) and block_m == 128 and head_dim == 128
    return PiperAttentionExecutionPlan(
        block_m=block_m,
        grouped_qk=grouped_qk,
        native_uint8=native_uint8,
        split_pv_head_dim=split_pv_head_dim,
        scaled_fp16_numerator=scaled_fp16_numerator,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=2 if use_tensor_descriptors else 3,
        use_packed_probability_conversion=use_packed_probability_conversion,
    )
