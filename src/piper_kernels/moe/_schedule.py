"""Shared Triton-free launch schedules, tiling constants, and Split-K sizing for MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from piper_kernels.linear._splitk import select_split_k

from ._ops import _INT8_MAX_REDUCTION_WIDTH

NVFP4_GEMV_BLOCK_N = 16
NVFP4_GEMV_BLOCK_K = 128
NVFP4_GEMV_NUM_WARPS = 4
NVFP4_SM120_GEMV_BLOCK_K = 512
NVFP4_SM120_M1_SPLITK_CTA_MULT = 3
NVFP4_SM75_SCALAR_BLOCK_K = 64
NVFP4_SM120_DOT_NUM_WARPS = 4
NVFP4_DOT_NUM_WARPS_CUDA = 8
NVFP4_DOT_BLOCK_N = 64
NVFP4_DOT_BLOCK_K = 128
NVFP4_DOT_BLOCK_K_GFX120X = 256
NVFP4_REDUCE_BLOCK_N = 128
NVFP4_MAX_SPLIT_K = 16
NVFP4_MIN_SPLITK_WEIGHT_BYTES = 4 * 1024 * 1024

INT8_GEMM_BLOCK_M = 16
INT8_GEMM_BLOCK_N = 64
INT8_REDUCE_BLOCK_N = 128
INT8_MAX_SPLIT_K = 16
INT8_MIN_SPLITK_WEIGHT_BYTES = 8 * 1024 * 1024
INT8_MIN_K_TILES = 8
INT8_MIN_TILES_PER_SPLIT = 4
INT8_PARTIAL_RATIO_DIVISOR = 64

TRELLIS_GEMV_BATCH_BLOCK_M = 4
TRELLIS_GEMV_GFX120X_SPARSE_BLOCK_M = 2
TRELLIS_GEMV_NUM_WARPS = 1
TRELLIS_MAX_SPLIT_K = 8
TRELLIS_MIN_SPLITK_WEIGHT_BYTES = 1024 * 1024
TRELLIS_SCRATCH_BYTES = 32 * 1024 * 1024

BATCH_BLOCK_M = 16
BATCH_MAX_ROWS = 16
BUCKET_MAX_PAIRS = 1024

PREFILL_WORKSPACE_BYTES = 256 * 1024 * 1024
PREFILL_MAX_PAIRS = 65_535
PREFILL_BUCKET_BLOCK_P = 256
PREFILL_GATHER_BLOCK_K = 256
PREFILL_GROUP_M = 8
PREFILL_NVFP4_SPARSE_BLOCK_M = 16
PREFILL_NVFP4_DENSE_BLOCK_M = 32
PREFILL_NVFP4_SPARSE_BLOCK_N = 64
PREFILL_NVFP4_DENSE_BLOCK_N = 128
PREFILL_NVFP4_BLOCK_K = 64
NVFP4_SM120_PREFILL_BLOCK_K = 128
PREFILL_INT8_SPARSE_BLOCK_M = 16
PREFILL_INT8_DENSE_BLOCK_M = 64
PREFILL_INT8_SPARSE_BLOCK_N = 64
PREFILL_INT8_DENSE_BLOCK_N = 128
PREFILL_INT8_DENSE_BLOCK_K = 64

WEIGHTED_REDUCE_BLOCK_N = 128
ACTIVATE_BLOCK_SIZE = 256


@dataclass(frozen=True, slots=True)
class Nvfp4M1Schedule:
    """Single-token NVFP4 MoE projection tiling and Split-K sizing schedule."""

    block_n: int
    block_k: int
    num_warps: int
    splitk_cta_multiplier: int
    use_sm120_e2m1x2: bool = False


@dataclass(frozen=True, slots=True)
class Nvfp4BatchedSchedule:
    """Small-batch NVFP4 MoE projection schedule."""

    use_dot: bool
    cast_fp32_to_bf16: bool
    post_scale_only: bool
    block_n: int
    block_k: int
    num_warps: int
    single_dot: bool
    use_sm120_e2m1x2: bool = False


@dataclass(frozen=True, slots=True)
class Nvfp4PrefillSchedule:
    """Grouped prefill NVFP4 MoE projection schedule."""

    cast_fp32_to_bf16: bool
    force_fp32_dot: bool
    half_scales: bool
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    single_dot: bool
    use_sm120_e2m1x2: bool = False


@dataclass(frozen=True, slots=True)
class Int8LaunchSchedule:
    """ConvRot INT8 MoE GEMV/batched K-tiling and barrier-fence schedule."""

    block_k: int
    num_stages: int
    lds_only_fences: bool = False


@dataclass(frozen=True, slots=True)
class Int8PrefillSchedule:
    """ConvRot INT8 MoE grouped prefill GEMM schedule."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    lds_only_fences: bool = False


@dataclass(frozen=True, slots=True)
class TrellisMoeSchedule:
    """Target-dependent kernel choices for ``trellis_moe_gemv_kernel``."""

    block_m: int
    block_n: int
    num_warps: int
    unroll_k: int
    use_dp4a: bool
    use_sad_u8: bool = False
    split_x_loads: bool = False
    use_subtile_splitk: bool = False


def cdiv(value: int, divisor: int) -> int:
    """Return ceiling integer division ``ceil(value / divisor)``."""
    return (value + divisor - 1) // divisor


def max_bucket_tiles(num_pairs: int, num_slots: int, block_m: int = BATCH_BLOCK_M) -> int:
    """Upper-bound compacted slot-tile count across ``num_pairs`` and ``num_slots``."""
    max_unique = min(num_pairs, num_slots)
    return max_unique + (num_pairs - max_unique) // block_m


def expected_bucket_tiles(
    num_pairs: int,
    num_slots: int,
    active_slots_hint: int,
    *,
    block_m: int = BATCH_BLOCK_M,
    batched: bool,
) -> int:
    """Estimate active slot tiles for Split-K and accumulator-seed heuristics."""
    if not batched:
        return max(1, min(num_pairs, active_slots_hint) if active_slots_hint > 0 else num_pairs)
    eff_slots = min(num_slots, active_slots_hint) if active_slots_hint > 0 else num_slots
    return max(1, max_bucket_tiles(num_pairs, eff_slots, block_m))


def prefill_bytes_per_pair(
    *,
    is_int8: bool,
    hidden_size: int,
    intermediate_size: int,
    up_gate_out: int,
    logical_elem_bytes: int,
    group_size: int,
) -> int:
    """Estimate temporary workspace bytes required per routed `(token, k)` pair."""
    if is_int8:
        return hidden_size + up_gate_out * logical_elem_bytes + intermediate_size + hidden_size * 4
    active_elem_bytes = 4 if group_size > 0 else logical_elem_bytes
    return (
        hidden_size * active_elem_bytes
        + up_gate_out * 4
        + intermediate_size * active_elem_bytes
        + hidden_size * 4
    )


def select_prefill_chunk_rows(rows: int, top_k: int, bytes_per_pair: int) -> int:
    """Bound rows per prefill chunk to fit temporary workspace and pair budgets."""
    max_pairs_mem = max(1, PREFILL_WORKSPACE_BYTES // max(1, bytes_per_pair))
    max_rows = min(rows, PREFILL_MAX_PAIRS // top_k, max_pairs_mem // top_k)
    return max(1, max_rows)


def select_moe_execution_regime(
    rows: int,
    top_k: int,
    bytes_per_pair: int,
) -> tuple[Literal["m1", "batched", "prefill"], int]:
    """Choose between ``m1``, ``batched``, and chunked ``prefill`` execution."""
    if rows == 1:
        return "m1", 1
    if rows <= BATCH_MAX_ROWS and rows * top_k <= BUCKET_MAX_PAIRS:
        return "batched", rows
    chunk_rows = select_prefill_chunk_rows(rows, top_k, bytes_per_pair)
    if chunk_rows == 1 and top_k > PREFILL_MAX_PAIRS:
        return "m1", 1
    return "prefill", chunk_rows


def select_prefill_block_m(
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    *,
    sparse_m: int,
    dense_m: int,
) -> int:
    """Choose sparse or dense M-tile height from expected pairs per active expert."""
    eff_slots = min(num_slots, active_slots_hint) if active_slots_hint > 0 else num_slots
    avg_m = cdiv(chunk_rows * top_k, max(1, eff_slots))
    return sparse_m if avg_m <= sparse_m else dense_m


def select_nvfp4_split_k(
    num_sms: int,
    active_tiles: int,
    out_features: int,
    in_features: int,
    *,
    block_n: int = NVFP4_GEMV_BLOCK_N,
    block_k: int = NVFP4_GEMV_BLOCK_K,
    cta_multiplier: int = 8,
) -> int:
    """Select Split-K count for NVFP4 MoE projections."""
    return select_split_k(
        num_sms,
        output_tiles=active_tiles * cdiv(out_features, block_n),
        k_tiles=cdiv(in_features, block_k),
        weight_bytes=active_tiles * out_features * (in_features // 2),
        min_weight_bytes=NVFP4_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=NVFP4_MAX_SPLIT_K,
        cta_multiplier=cta_multiplier,
        min_k_tiles=8,
        min_tiles_per_split=4,
        round_up=True,
    )


def select_int8_split_k(
    num_sms: int,
    active_slots: int,
    out_features: int,
    in_features: int,
    block_k: int,
) -> int:
    """Select Split-K count for ConvRot INT8 MoE projections."""
    if in_features > _INT8_MAX_REDUCTION_WIDTH:
        return 1
    max_for_partial = max(1, in_features // INT8_PARTIAL_RATIO_DIVISOR)
    return select_split_k(
        num_sms,
        output_tiles=active_slots * cdiv(out_features, INT8_GEMM_BLOCK_N),
        k_tiles=cdiv(in_features, block_k),
        weight_bytes=active_slots * out_features * in_features,
        min_weight_bytes=INT8_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=min(INT8_MAX_SPLIT_K, max_for_partial),
        cta_multiplier=1,
        min_k_tiles=INT8_MIN_K_TILES,
        min_tiles_per_split=INT8_MIN_TILES_PER_SPLIT,
        round_up=False,
    )


def select_trellis_split_k(
    num_sms: int,
    active_tiles: int,
    out_features: int,
    blocks_per_row: int,
    weight_bytes: int,
) -> int:
    """Select Split-K count for ConvRot Trellis MoE projections."""
    return select_split_k(
        num_sms,
        output_tiles=active_tiles * out_features,
        k_tiles=blocks_per_row,
        weight_bytes=weight_bytes,
        min_weight_bytes=TRELLIS_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=TRELLIS_MAX_SPLIT_K,
        cta_multiplier=4,
        min_k_tiles=2,
        min_tiles_per_split=2,
        round_up=False,
    )


def build_int8_prefill_schedule(
    launch: Int8LaunchSchedule,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
) -> Int8PrefillSchedule:
    """Resolve grouped prefill INT8 tiling from a target's base launch schedule."""
    block_m = select_prefill_block_m(
        chunk_rows,
        top_k,
        num_slots,
        active_slots_hint,
        sparse_m=PREFILL_INT8_SPARSE_BLOCK_M,
        dense_m=PREFILL_INT8_DENSE_BLOCK_M,
    )
    if block_m == PREFILL_INT8_SPARSE_BLOCK_M:
        return Int8PrefillSchedule(
            block_m=block_m,
            block_n=PREFILL_INT8_SPARSE_BLOCK_N,
            block_k=launch.block_k,
            num_warps=4,
            num_stages=launch.num_stages,
            lds_only_fences=launch.lds_only_fences,
        )
    return Int8PrefillSchedule(
        block_m=block_m,
        block_n=PREFILL_INT8_DENSE_BLOCK_N,
        block_k=PREFILL_INT8_DENSE_BLOCK_K,
        num_warps=8,
        num_stages=launch.num_stages,
        lds_only_fences=launch.lds_only_fences,
    )


def default_nvfp4_m1_schedule() -> Nvfp4M1Schedule:
    """Return the portable baseline single-token NVFP4 schedule."""
    return Nvfp4M1Schedule(
        block_n=NVFP4_GEMV_BLOCK_N,
        block_k=NVFP4_GEMV_BLOCK_K,
        num_warps=NVFP4_GEMV_NUM_WARPS,
        splitk_cta_multiplier=8,
        use_sm120_e2m1x2=False,
    )


def default_nvfp4_batched_schedule(
    *,
    rows: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
    allow_dot: bool = True,
) -> Nvfp4BatchedSchedule:
    """Return the portable baseline batched NVFP4 schedule."""
    use_dot = (
        allow_dot
        and rows > 4
        and (
            active_dtype in (torch.float16, torch.bfloat16)
            or (active_dtype is torch.float32 and logical_dtype is torch.bfloat16)
        )
    )
    cast_fp32_to_bf16 = use_dot and active_dtype is torch.float32
    post_scale_only = not (use_dot and active_dtype is torch.float16)
    if use_dot:
        return Nvfp4BatchedSchedule(
            use_dot=True,
            cast_fp32_to_bf16=cast_fp32_to_bf16,
            post_scale_only=post_scale_only,
            block_n=NVFP4_DOT_BLOCK_N,
            block_k=NVFP4_DOT_BLOCK_K,
            num_warps=NVFP4_GEMV_NUM_WARPS,
            single_dot=False,
            use_sm120_e2m1x2=False,
        )
    return Nvfp4BatchedSchedule(
        use_dot=False,
        cast_fp32_to_bf16=False,
        post_scale_only=post_scale_only,
        block_n=NVFP4_GEMV_BLOCK_N,
        block_k=NVFP4_GEMV_BLOCK_K,
        num_warps=NVFP4_GEMV_NUM_WARPS,
        single_dot=False,
        use_sm120_e2m1x2=False,
    )


def default_nvfp4_prefill_schedule(
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
    force_fp32_dot_override: bool = False,
) -> Nvfp4PrefillSchedule:
    """Return the portable baseline prefill NVFP4 schedule."""
    force_fp32_dot = force_fp32_dot_override or (
        active_dtype is torch.float32 and logical_dtype is not torch.bfloat16
    )
    cast_fp32_to_bf16 = (not force_fp32_dot) and (active_dtype is torch.float32)
    half_scales = not force_fp32_dot
    block_m = select_prefill_block_m(
        chunk_rows,
        top_k,
        num_slots,
        active_slots_hint,
        sparse_m=PREFILL_NVFP4_SPARSE_BLOCK_M,
        dense_m=PREFILL_NVFP4_DENSE_BLOCK_M,
    )
    if block_m == PREFILL_NVFP4_SPARSE_BLOCK_M:
        block_n = PREFILL_NVFP4_SPARSE_BLOCK_N
        num_warps = NVFP4_DOT_NUM_WARPS_CUDA if force_fp32_dot else NVFP4_GEMV_NUM_WARPS
    else:
        block_n = PREFILL_NVFP4_SPARSE_BLOCK_N if force_fp32_dot else PREFILL_NVFP4_DENSE_BLOCK_N
        num_warps = NVFP4_DOT_NUM_WARPS_CUDA
    return Nvfp4PrefillSchedule(
        cast_fp32_to_bf16=cast_fp32_to_bf16,
        force_fp32_dot=force_fp32_dot,
        half_scales=half_scales,
        block_m=block_m,
        block_n=block_n,
        block_k=PREFILL_NVFP4_BLOCK_K,
        num_warps=num_warps,
        single_dot=False,
        use_sm120_e2m1x2=False,
    )


def default_int8_launch_schedule() -> Int8LaunchSchedule:
    """Return the portable baseline ConvRot INT8 MoE launch schedule."""
    return Int8LaunchSchedule(block_k=64, num_stages=1, lds_only_fences=False)


def default_trellis_schedule(*, batched: bool) -> TrellisMoeSchedule:
    """Return the portable baseline ConvRot Trellis MoE GEMV schedule."""
    return TrellisMoeSchedule(
        block_m=TRELLIS_GEMV_BATCH_BLOCK_M if batched else 1,
        block_n=1,
        num_warps=TRELLIS_GEMV_NUM_WARPS,
        unroll_k=1,
        use_dp4a=False,
        use_sad_u8=False,
        split_x_loads=False,
        use_subtile_splitk=False,
    )
