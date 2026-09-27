"""NVIDIA Piper Attention execution choices, validation, and reporting."""

from dataclasses import asdict, dataclass
from typing import Literal

from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)

ATTENTION_KERNELS = ("triton", "gluon_async_copy")


@dataclass(frozen=True, slots=True)
class PiperAttentionExecutionPlan:
    """Host-side specialization and launch choices for one Piper invocation.

    ``attention_kernel`` names the implementation independently of the target.
    ``max_registers`` and ``fuse_query_quantization`` apply only to ``gluon_async_copy``.
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
    retain_query_tail_for_strided_output: bool = False
    output_ctas_per_sm: int = 0
    strided_output_query_group: int = 0
    ragged_strided_output_maxnreg: int | None = None
    unspecialized_value_stride: bool = False
    attention_kernel: Literal["triton", "gluon_async_copy"] = "triton"
    max_registers: int | None = None
    fuse_query_quantization: bool = False

    def __post_init__(self) -> None:
        if self.attention_kernel not in ATTENTION_KERNELS:
            raise ValueError("Piper Attention kernel must be triton or gluon_async_copy")
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

    def validate_kernel(self) -> None:
        """Check implementation capabilities before preparation or tuner execution.

        Tuning may construct unsupported combinations; validate them here so
        the tuner can report those candidates without aborting the whole search.
        """
        if self.attention_kernel == "triton":
            if self.fuse_query_quantization:
                raise ValueError("fused query quantization requires the Gluon kernel")
            if self.max_registers is not None:
                raise ValueError("a register cap requires the Gluon kernel")
            return
        if (
            self.num_warps != 4
            or self.num_stages != 1
            or self.grouped_qk
            or self.split_pv_head_dim
            or self.use_tensor_descriptors
            or not self.derive_value_log_bound
            or not self.use_packed_probability_conversion
            or self.optimize_causal_traversal
            or self.loop_num_stages is not None
            or self.loop_licm
            or self.retain_query_tail_for_strided_output
            or self.strided_output_query_group
            or self.ragged_strided_output_maxnreg is not None
        ):
            raise ValueError(
                "the Gluon kernel requires Q64 or Q128 tiles on four warps, per-thread Q/K scales, "
                "an unsplit PV product, pointer loads, a derived V log bound, and packed "
                "probability codes, with no Triton loop or strided-output controls"
            )
