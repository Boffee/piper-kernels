"""NVIDIA execution planning for dense, chunked prefill, and Split-KV decode Piper Attention.

Scheduling and backend boundaries:
- Full query tiles and a ragged final tile share one launch. Full tiles use
  unmasked access; the tail uses masked Q pointer loads and output stores, even
  when full tiles use descriptors. Optimized causal traversal starts with the
  longest prefixes. Quantization and statistics use separate preparation launches.
- Exact SM120 uses Triton (``attention_kernel="triton"``). Its execution plan
  chooses descriptor loads, tiling, pipeline stages, and causal ordering from
  host metadata. Descriptor setup must be amortized by traversal work; rectangular
  attention uses actual K length. Output strides matter for fused output buffers:
  the causal strided D64 schedule interleaves heads across query-tile groups and
  bounds ragged-kernel registers to preserve occupancy, while D128 splits PV into
  two FP32 D64 accumulators.
- Exact SM89 uses the Gluon kernel (``attention_kernel="gluon_async_copy"``),
  staging Q/K/V and per-key V multipliers through ``cp.async`` commit groups. It
  retains per-thread Q/K scales, derives the V log-scale bound, computes per-chunk
  ``k_bias`` in FP32 directly from SMEM Q and ``key_center``, and masks the final K
  tile plus the causal diagonal when needed. D64 quantizes Q in the prologue; D128
  uses a smaller query tile and a register cap to retain occupancy. Ragged lengths
  reuse one compiled kernel. Q quantized by a fused producer skips the prologue;
  Q128 tiles zero-fill rows past Q64-padded storage, and producer K scales and V
  multipliers span K64-padded storage.
- Capability checks belong to ``PiperAttentionExecutionPlan``; offline tuning
  (``benchmarks/tune_piper_attention.py``) can compare implementations without
  changing production dispatch.
"""

from dataclasses import asdict, dataclass
from typing import Literal

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)

from .._decode_schedule import (
    CENTER_CHUNK,
    FAST_MAX_DECODE_SPLITS,
    DecodeSchedule,
    default_decode_schedule,
)

ATTENTION_KERNELS = ("triton", "gluon_async_copy")


@dataclass(frozen=True, slots=True)
class PiperAttentionExecutionPlan:
    """Host-side specialization and launch choices for one Piper invocation.

    ``attention_kernel`` names the implementation independently of the target.
    ``max_registers`` and ``fuse_query_quantization`` apply only to ``gluon_async_copy``.
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


_SM120_CAUSAL_DESCRIPTOR_MIN_QUERY_LENGTH = 1024
# D64 needs enough K traversal to amortize descriptor setup in the public
# floating-input path. Sixty-four full K64 tiles clear the measured crossover
# even for a single query tile/head; the rule also covers rectangular attention.
_SM120_D64_DESCRIPTOR_MIN_KEY_TILES = 64


def supports_target(target: AcceleratorTarget) -> bool:
    """Require the NVIDIA MMAv2 lowering handled by the mixed-sign extension."""
    return target.supports_uint8_int8_mma


def supports_quantized_query(target: AcceleratorTarget) -> bool:
    """Return whether NVIDIA ``target`` supports native quantized-query dense Piper."""
    return target.is_cuda_capability(12) or target.is_cuda_capability(8, 9)


def groups_qk_scales(target: AcceleratorTarget) -> bool:
    """Return whether every plan for target reads Q32/K64 group scales.

    Otherwise Q/K scales are per thread, stored per row and key. The choice
    depends on the target alone, so external Q/K producers can match it.
    """
    return target.is_cuda_capability(12)


def _generic_execution_plan(
    target: AcceleratorTarget,
    *,
    head_dim: int,
    is_causal: bool = False,
    window_mode: Literal["none", "causal", "windowed"] | None = None,
) -> PiperAttentionExecutionPlan:
    """Build capability-based defaults before exact-target tuning is applied."""
    resolved_mode = ("causal" if is_causal else "none") if window_mode is None else window_mode
    grouped_qk = groups_qk_scales(target)
    unbounded = resolved_mode == "none"
    split_pv_head_dim = target.is_cuda_capability(12) and unbounded and head_dim == 128
    block_m = 64 if not unbounded or split_pv_head_dim else 128
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
    window_mode: Literal["none", "causal", "windowed"],
    quantized_query: bool,
) -> PiperAttentionExecutionPlan:
    """Build the exact-SM89 plan measured on an RTX 4070 Ti SUPER.

    Every mode runs the ``cp.async`` Gluon kernel with per-thread Q/K scales:
    SM120's grouped scales raise the error against exact attention by 7-12% here.
    D64 gives each warp 32 query rows and quantizes floating-point Q in the kernel
    prologue. Already quantized causal D64 Q runs 16 rows per warp instead, which
    trims diagonal work by 0.5-8%. D128 gives each warp 16 rows under a register
    cap that fits two CTAs per SM; there the prologue would cost more than the Q
    preparation pass it replaces.
    """
    wide = head_dim == 128
    causal_mode = window_mode == "causal"
    return PiperAttentionExecutionPlan(
        block_m=64 if wide or (causal_mode and quantized_query) else 128,
        grouped_qk=False,
        split_pv_head_dim=False,
        use_tensor_descriptors=False,
        derive_value_log_bound=True,
        num_stages=1,
        use_packed_probability_conversion=True,
        attention_kernel="gluon_async_copy",
        max_registers=232 if wide else None,
        fuse_query_quantization=not wide and not quantized_query,
    )


def _sm120_execution_plan(
    *,
    head_dim: int,
    window_mode: Literal["none", "causal", "windowed"],
    query_length: int,
    key_length: int,
) -> PiperAttentionExecutionPlan:
    """Build the loop, probability, and value-metadata plan measured on exact SM120."""
    causal_mode = window_mode == "causal"
    unbounded = window_mode == "none"
    split_pv_head_dim = head_dim == 128
    # Short calls retain pointer loads where descriptor setup dominates.
    use_tensor_descriptors = (
        key_length // 64 >= _SM120_D64_DESCRIPTOR_MIN_KEY_TILES
        if head_dim == 64
        else unbounded or query_length >= _SM120_CAUSAL_DESCRIPTOR_MIN_QUERY_LENGTH
    )
    block_m = 64 if not unbounded else 128
    strided_causal_d64 = causal_mode and head_dim == 64 and use_tensor_descriptors
    # Packed conversion wins for D64 and non-causal D128. The D128 causal
    # specialization is neutral to slightly slower and retains stock lowering.
    return PiperAttentionExecutionPlan(
        block_m=block_m,
        grouped_qk=True,
        split_pv_head_dim=split_pv_head_dim,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=2 if use_tensor_descriptors and unbounded else 3,
        derive_value_log_bound=unbounded,
        optimize_causal_traversal=causal_mode,
        # The combined full/tail kernel benefits from loop-invariant motion
        # for causal D64. Aligned grids retain their measured original loop.
        loop_licm=causal_mode and head_dim == 64 and query_length % block_m != 0,
        use_packed_probability_conversion=not (causal_mode and head_dim == 128),
        # On SM120, retaining the CTA-uniform query-tail branch improves the
        # strided D128 kernel with masked K tails. Aligned K keeps its own winner.
        retain_query_tail_for_strided_output=unbounded and head_dim == 128,
        # Measured Q128/four-warp noncausal kernels fit two CTAs per SM.
        # Zero leaves output chunk planning independent of device occupancy.
        output_ctas_per_sm=0 if not unbounded else 2,
        # Interleave heads within small groups of causal query tiles to spread
        # their different prefix lengths across the GPU. Contiguous output
        # retains its measured original ordering.
        strided_output_query_group=8 if strided_causal_d64 else 0,
        # Three four-warp CTAs fit in 64K registers at 168 registers/thread.
        # Constrain only the ragged strided variant that otherwise drops to
        # two CTAs; even a nonbinding cap can change the aligned kernel's code.
        ragged_strided_output_maxnreg=168 if strided_causal_d64 else None,
    )


def _sm120_decode_schedule() -> DecodeSchedule:
    return DecodeSchedule(
        block_n=64,
        num_warps=4,
        num_stages=2,
        loop_num_stages=2,
        frozen_qk_mode="tf32x2",
        tail_qk_precision="tf32x3",
        pv_precision="tf32",
        max_splits=FAST_MAX_DECODE_SPLITS,
        target_ctas_per_sm=3,
        tiles_per_chunk_cap=CENTER_CHUNK // 64,
        max_tail_splits=4,
    )


def select_decode_schedule(target: AcceleratorTarget, head_dim: int) -> DecodeSchedule:
    """Return the NVIDIA Split-KV decode schedule for ``target`` and ``head_dim``."""
    if target.is_cuda_capability(12):
        return _sm120_decode_schedule()
    return default_decode_schedule(head_dim)


def select_frozen_prefill_splits(
    base_blocks: int,
    hint_chunks: int,
    sm_count: int,
    bytes_per_split: int,
    *,
    allow_subchunk_single: bool = False,
) -> int:
    """Select NVIDIA frozen chunk-prefill split count capped by occupancy and 64 MiB partials."""
    if hint_chunks <= 0:
        return 1
    total_tiles = hint_chunks * 16
    mem_cap = max(1, (64 * 1024 * 1024) // max(1, bytes_per_split))
    if allow_subchunk_single:
        if base_blocks < sm_count * 3 and total_tiles >= 8:
            return min(32, total_tiles // 4, max(1, (sm_count * 3) // base_blocks), mem_cap)
        return 1
    if base_blocks < sm_count and hint_chunks >= 2:
        return min(16, total_tiles // 4, max(1, (sm_count * 2) // base_blocks), mem_cap)
    return 1


def select_prefill_config(
    target: AcceleratorTarget,
    *,
    head_dim: int,
    is_tail: bool,
    has_chunk_map: bool = True,
    has_chunk_splits: bool = False,
    window_mode: Literal["none", "causal", "windowed"] = "none",
) -> tuple[int, int, bool, int, int]:
    """Return ``(block_m, num_warps, split_pv_head_dim, num_stages, loop_num_stages)``."""
    if target.is_cuda_capability(12):
        if head_dim == 64:
            # Free one pipeline buffer register when unmapped FP32 split partials spill at loop=2.
            loop_stages = 1 if (has_chunk_splits and not has_chunk_map) else 3
            return 64, 4, False, 3, loop_stages
        if not is_tail:
            return 64, 8, True, 2, 2
        # Windowed D=128 tail needs 8 warps on SM12x to keep zero SASS spills.
        return 64, 8 if window_mode == "windowed" else 4, True, 2, 1
    if not is_tail:
        num_warps = 8 if head_dim == 128 else 4
        # Unify the PV accumulator for SM80 windowed D=128 to keep zero SASS spills.
        split_pv = head_dim == 128 and not (
            window_mode == "windowed" and target.is_cuda_capability(8, 0)
        )
        return 64, num_warps, split_pv, 1, 1
    if head_dim == 128 and not target.is_cuda_capability(8, 0):
        return 32, 4, False, 1, 1
    split_pv = head_dim == 128 or target.is_cuda_capability(8, 9)
    return 64, 8, split_pv, 1, 1


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    head_dim: int,
    is_causal: bool,
    query_length: int,
    key_length: int | None = None,
    quantized_query: bool = False,
    window_mode: Literal["none", "causal", "windowed"] | None = None,
) -> PiperAttentionExecutionPlan:
    """Combine capability defaults with measured policy; omitted K length means self-attention.

    ``quantized_query`` selects a plan for Q that a producer already quantized.
    """
    resolved_mode: Literal["none", "causal", "windowed"] = (
        ("causal" if is_causal else "none") if window_mode is None else window_mode
    )
    if target.is_cuda_capability(8, 9):
        return _sm89_execution_plan(
            head_dim=head_dim, window_mode=resolved_mode, quantized_query=quantized_query
        )
    if target.is_cuda_capability(12, 0):
        return _sm120_execution_plan(
            head_dim=head_dim,
            window_mode=resolved_mode,
            query_length=query_length,
            key_length=query_length if key_length is None else key_length,
        )
    return _generic_execution_plan(
        target,
        head_dim=head_dim,
        window_mode=resolved_mode,
    )
