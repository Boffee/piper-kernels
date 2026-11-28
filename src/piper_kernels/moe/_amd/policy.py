"""AMD HIP MoE launch schedules for NVFP4, ConvRot INT8, and ConvRot Trellis."""

from __future__ import annotations

from dataclasses import replace

import torch

from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from .._schedule import (
    NVFP4_DOT_BLOCK_K_GFX120X,
    NVFP4_GEMV_NUM_WARPS,
    TRELLIS_GEMV_BATCH_BLOCK_M,
    TRELLIS_GEMV_GFX120X_SPARSE_BLOCK_M,
    TRELLIS_GEMV_NUM_WARPS,
    Int8LaunchSchedule,
    Nvfp4BatchedSchedule,
    Nvfp4M1Schedule,
    Nvfp4PrefillSchedule,
    TrellisMoeSchedule,
    default_nvfp4_batched_schedule,
    default_nvfp4_m1_schedule,
    default_nvfp4_prefill_schedule,
)


def select_nvfp4_m1_schedule(_target: AcceleratorTarget) -> Nvfp4M1Schedule:
    """Resolve single-token NVFP4 MoE projection schedule for AMD HIP targets."""
    return default_nvfp4_m1_schedule()


def select_nvfp4_batched_schedule(
    target: AcceleratorTarget,
    *,
    rows: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> Nvfp4BatchedSchedule:
    """Resolve small-batch NVFP4 MoE schedule for AMD gfx120x and fallback HIP targets."""
    base = default_nvfp4_batched_schedule(
        rows=rows, active_dtype=active_dtype, logical_dtype=logical_dtype
    )
    if base.use_dot and target.is_hip_architecture(*GFX120X_ARCHITECTURES):
        return replace(base, block_k=NVFP4_DOT_BLOCK_K_GFX120X)
    return base


def select_nvfp4_prefill_schedule(
    target: AcceleratorTarget,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> Nvfp4PrefillSchedule:
    """Resolve grouped prefill NVFP4 MoE schedule for AMD wave32/wave64 targets."""
    base = default_nvfp4_prefill_schedule(
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        active_dtype=active_dtype,
        logical_dtype=logical_dtype,
    )
    if target.is_hip_architecture("gfx942"):
        return replace(base, num_warps=NVFP4_GEMV_NUM_WARPS)
    return base


def select_int8_launch_schedule(target: AcceleratorTarget) -> Int8LaunchSchedule:
    """Resolve AMD ConvRot INT8 MoE K-tiling and LDS-only fence options."""
    is_gfx120x = target.is_hip_architecture(*GFX120X_ARCHITECTURES)
    return Int8LaunchSchedule(
        block_k=256 if is_gfx120x else 64,
        num_stages=2 if is_gfx120x else 1,
        lds_only_fences=is_gfx120x,
    )


def select_trellis_schedule(
    target: AcceleratorTarget,
    *,
    batched: bool,
    num_pairs: int,
    expected_slots: int,
) -> TrellisMoeSchedule:
    """Resolve AMD ConvRot Trellis MoE GEMV schedule for gfx120x and fallback targets."""
    is_gfx120x = target.is_hip_architecture(*GFX120X_ARCHITECTURES)
    if not batched:
        block_m, block_n = 1, 1
    elif is_gfx120x:
        block_m = (
            TRELLIS_GEMV_GFX120X_SPARSE_BLOCK_M
            if num_pairs * 2 <= 3 * max(1, expected_slots)
            else TRELLIS_GEMV_BATCH_BLOCK_M
        )
        block_n = 2
    else:
        block_m, block_n = TRELLIS_GEMV_BATCH_BLOCK_M, 1
    return TrellisMoeSchedule(
        block_m=block_m,
        block_n=block_n,
        num_warps=TRELLIS_GEMV_NUM_WARPS,
        unroll_k=1,
        use_dp4a=False,
        use_sad_u8=is_gfx120x,
        split_x_loads=is_gfx120x,
        use_subtile_splitk=is_gfx120x,
    )
