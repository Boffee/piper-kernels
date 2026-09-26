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


def supports_target(target: AcceleratorTarget) -> bool:
    """Require the NVIDIA MMAv2 lowering handled by the mixed-sign extension."""
    return target.supports_uint8_int8_mma


@dataclass(frozen=True, slots=True)
class PiperAttentionExecutionPlan:
    """Host-side specialization and launch choices for one Piper invocation.

    ``use_gluon_kernel`` replaces the Triton recurrence with the ``cp.async`` Gluon
    kernel; ``max_registers`` and ``fuse_query_quantization`` apply only to it.
    ``unspecialized_value_stride`` quantizes V without specializing on its
    key-length-dependent row stride.
    """

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
    unspecialized_value_stride: bool = False
    use_gluon_kernel: bool = False
    max_registers: int | None = None
    fuse_query_quantization: bool = False

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


def _sm89_execution_plan(*, head_dim: int) -> PiperAttentionExecutionPlan:
    """Build the exact-SM89 plan measured on an RTX 4070 Ti SUPER.

    Every mode runs the ``cp.async`` Gluon kernel with per-thread Q/K scales:
    SM120's grouped scales raise the error against exact attention by 7-12% here.
    D64 gives each warp 32 query rows and quantizes Q in the kernel prologue. D128
    gives each warp 16 rows under a register cap that fits two CTAs per SM; there
    the prologue would cost more than the Q preparation pass it replaces. V
    quantization leaves the V row stride unspecialized, which keeps SM89 within
    SM120's compile count and is also 3-5x faster here.
    """
    wide = head_dim == 128
    return PiperAttentionExecutionPlan(
        block_m=64 if wide else 128,
        grouped_qk=False,
        split_pv_head_dim=False,
        use_tensor_descriptors=False,
        derive_value_log_bound=True,
        num_stages=1,
        use_packed_probability_conversion=True,
        unspecialized_value_stride=True,
        use_gluon_kernel=True,
        max_registers=232 if wide else None,
        fuse_query_quantization=not wide,
    )


def _sm120_execution_plan(
    *,
    head_dim: int,
    is_causal: bool,
    query_length: int,
) -> PiperAttentionExecutionPlan:
    """Build the loop, probability, and value-metadata plan measured on exact SM120."""
    split_pv_head_dim = head_dim == 128
    # Short causal calls keep pointer loads to avoid descriptor setup overhead.
    use_tensor_descriptors = head_dim == 128 and (
        not is_causal or query_length >= _SM120_CAUSAL_DESCRIPTOR_MIN_QUERY_LENGTH
    )
    block_m = 64 if is_causal else 128
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
    )


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    head_dim: int,
    is_causal: bool,
    query_length: int,
) -> PiperAttentionExecutionPlan:
    """Combine portable capability defaults with exact-target measured policy."""
    if target.is_cuda_capability(8, 9):
        return _sm89_execution_plan(head_dim=head_dim)
    if target.is_cuda_capability(12, 0):
        return _sm120_execution_plan(
            head_dim=head_dim,
            is_causal=is_causal,
            query_length=query_length,
        )
    return _generic_execution_plan(
        target,
        head_dim=head_dim,
        is_causal=is_causal,
    )
