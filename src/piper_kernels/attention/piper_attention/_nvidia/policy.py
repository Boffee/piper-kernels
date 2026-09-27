"""NVIDIA execution planning for Piper Attention."""

from dataclasses import asdict, dataclass

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)

_SM120_CAUSAL_DESCRIPTOR_MIN_QUERY_LENGTH = 1024
# D64 needs enough K traversal to amortize descriptor setup in the public
# floating-input path. Sixty-four full K64 tiles clear the measured crossover
# even for a single query tile/head; the rule also covers rectangular attention.
_SM120_D64_DESCRIPTOR_MIN_KEY_TILES = 64


def supports_target(target: AcceleratorTarget) -> bool:
    """Require the NVIDIA MMAv2 lowering handled by the mixed-sign extension."""
    return target.supports_uint8_int8_mma


@dataclass(frozen=True, slots=True)
class PiperAttentionExecutionPlan:
    """Host-side specialization and launch choices for one Piper invocation."""

    block_m: int
    grouped_qk: bool
    split_pv_head_dim: bool
    use_tensor_descriptors: bool
    derive_value_log_bound: bool = False
    optimize_causal_traversal: bool = False
    num_warps: int = 4
    num_stages: int = 3
    loop_num_stages: int | None = None
    loop_licm: bool = False
    use_packed_probability_conversion: bool = False
    retain_query_tail_for_strided_output: bool = False
    output_ctas_per_sm: int = 0
    strided_output_query_group: int = 0
    ragged_strided_output_maxnreg: int | None = None

    def __post_init__(self) -> None:
        if self.block_m not in BLOCK_M_VALUES:
            raise ValueError("Piper Attention block_m must be 64 or 128")
        if self.num_warps not in NUM_WARPS_VALUES:
            raise ValueError("Piper Attention num_warps must be 2, 4, or 8")
        if self.num_stages not in NUM_STAGES_VALUES:
            raise ValueError("Piper Attention num_stages must be 1, 2, 3, or 4")
        if self.loop_num_stages not in LOOP_NUM_STAGES_VALUES:
            raise ValueError("Piper Attention loop_num_stages must be None, 1, 2, 3, or 4")

    def as_dict(self) -> dict[str, object]:
        """Return execution choices as serializable benchmark metadata."""
        return asdict(self)


def _generic_execution_plan(
    target: AcceleratorTarget,
    *,
    head_dim: int,
    is_causal: bool,
) -> PiperAttentionExecutionPlan:
    """Build capability-based defaults before exact-target tuning is applied."""
    grouped_qk = target.is_cuda_capability(12)
    split_pv_head_dim = target.is_cuda_capability(12) and not is_causal and head_dim == 128
    block_m = 64 if is_causal or split_pv_head_dim else 128
    use_tensor_descriptors = target.is_cuda_capability(12) and block_m == 128 and head_dim == 128
    return PiperAttentionExecutionPlan(
        block_m=block_m,
        grouped_qk=grouped_qk,
        split_pv_head_dim=split_pv_head_dim,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=2 if use_tensor_descriptors else 3,
    )


def _sm89_execution_plan(
    *,
    head_dim: int,
    is_causal: bool,
) -> PiperAttentionExecutionPlan:
    """Build the exact-SM89 plan, including its measured D128 schedule."""
    noncausal_d128 = not is_causal and head_dim == 128
    return PiperAttentionExecutionPlan(
        block_m=64 if is_causal else 128,
        grouped_qk=False,
        split_pv_head_dim=noncausal_d128,
        use_tensor_descriptors=False,
        num_stages=1 if noncausal_d128 else 3,
        loop_num_stages=3 if noncausal_d128 else None,
        loop_licm=noncausal_d128,
        use_packed_probability_conversion=noncausal_d128,
    )


def _sm120_execution_plan(
    *,
    head_dim: int,
    is_causal: bool,
    query_length: int,
    key_length: int,
) -> PiperAttentionExecutionPlan:
    """Build the loop, probability, and value-metadata plan measured on exact SM120."""
    split_pv_head_dim = head_dim == 128
    # Short calls retain pointer loads where descriptor setup dominates.
    use_tensor_descriptors = (
        key_length // 64 >= _SM120_D64_DESCRIPTOR_MIN_KEY_TILES
        if head_dim == 64
        else not is_causal or query_length >= _SM120_CAUSAL_DESCRIPTOR_MIN_QUERY_LENGTH
    )
    block_m = 64 if is_causal else 128
    strided_causal_d64 = is_causal and head_dim == 64 and use_tensor_descriptors
    # Packed conversion wins for D64 and non-causal D128. The D128 causal
    # specialization is neutral to slightly slower and retains stock lowering.
    return PiperAttentionExecutionPlan(
        block_m=block_m,
        grouped_qk=True,
        split_pv_head_dim=split_pv_head_dim,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=2 if use_tensor_descriptors and not is_causal else 3,
        derive_value_log_bound=not is_causal,
        optimize_causal_traversal=is_causal,
        # The combined full/tail kernel benefits from loop-invariant motion
        # for causal D64. Aligned grids retain their measured original loop.
        loop_licm=is_causal and head_dim == 64 and query_length % block_m != 0,
        use_packed_probability_conversion=not (is_causal and head_dim == 128),
        # On SM120, retaining the CTA-uniform query-tail branch improves the
        # strided D128 kernel with masked K tails. Aligned K keeps its own winner.
        retain_query_tail_for_strided_output=not is_causal and head_dim == 128,
        # Measured Q128/four-warp noncausal kernels fit two CTAs per SM.
        # Zero leaves output chunk planning independent of device occupancy.
        output_ctas_per_sm=0 if is_causal else 2,
        # Interleave heads within small groups of causal query tiles to spread
        # their different prefix lengths across the GPU. Contiguous output
        # retains its measured original ordering.
        strided_output_query_group=8 if strided_causal_d64 else 0,
        # Three four-warp CTAs fit in 64K registers at 168 registers/thread.
        # Constrain only the ragged strided variant that otherwise drops to
        # two CTAs; even a nonbinding cap can change the aligned kernel's code.
        ragged_strided_output_maxnreg=168 if strided_causal_d64 else None,
    )


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    head_dim: int,
    is_causal: bool,
    query_length: int,
    key_length: int | None = None,
) -> PiperAttentionExecutionPlan:
    """Combine capability defaults with measured policy; omitted K length means self-attention."""
    if target.is_cuda_capability(8, 9):
        return _sm89_execution_plan(
            head_dim=head_dim,
            is_causal=is_causal,
        )
    if target.is_cuda_capability(12, 0):
        return _sm120_execution_plan(
            head_dim=head_dim,
            is_causal=is_causal,
            query_length=query_length,
            key_length=query_length if key_length is None else key_length,
        )
    return _generic_execution_plan(
        target,
        head_dim=head_dim,
        is_causal=is_causal,
    )
