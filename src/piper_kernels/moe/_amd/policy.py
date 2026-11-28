"""AMD HIP MoE launch schedules for ConvRot INT8 and ConvRot Trellis."""

from __future__ import annotations

from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from .._schedule import (
    TRELLIS_GEMV_BATCH_BLOCK_M,
    TRELLIS_GEMV_GFX120X_SPARSE_BLOCK_M,
    TRELLIS_GEMV_NUM_WARPS,
    Int8LaunchSchedule,
    TrellisMoeSchedule,
)


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
