"""AMD HIP MoE launch schedules for NVFP4, ConvRot INT8, and ConvRot Trellis."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from .._schedule import (
    NVFP4_DOT_BLOCK_K,
    NVFP4_DOT_BLOCK_K_GFX120X,
    NVFP4_DOT_BLOCK_N,
    NVFP4_DOT_NUM_WARPS_CUDA,
    NVFP4_GEMV_BLOCK_K,
    NVFP4_GEMV_BLOCK_N,
    NVFP4_GEMV_NUM_WARPS,
    PREFILL_NVFP4_BLOCK_K,
    PREFILL_NVFP4_DENSE_BLOCK_M,
    PREFILL_NVFP4_DENSE_BLOCK_N,
    PREFILL_NVFP4_SPARSE_BLOCK_M,
    PREFILL_NVFP4_SPARSE_BLOCK_N,
    TRELLIS_GEMV_BATCH_BLOCK_M,
    TRELLIS_GEMV_GFX120X_SPARSE_BLOCK_M,
    TRELLIS_GEMV_NUM_WARPS,
    Int8LaunchSchedule,
    Nvfp4BatchedSchedule,
    Nvfp4M1Schedule,
    Nvfp4PrefillSchedule,
    TrellisMoeSchedule,
    default_nvfp4_m1_schedule,
    select_prefill_block_m,
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
    use_dot = rows > 4 and (
        active_dtype in (torch.float16, torch.bfloat16)
        or (active_dtype is torch.float32 and logical_dtype is torch.bfloat16)
    )
    cast_fp32_to_bf16 = use_dot and active_dtype is torch.float32
    post_scale_only = not (use_dot and active_dtype is torch.float16)
    if use_dot:
        block_k = (
            NVFP4_DOT_BLOCK_K_GFX120X
            if target.is_hip_architecture(*GFX120X_ARCHITECTURES)
            else NVFP4_DOT_BLOCK_K
        )
        return Nvfp4BatchedSchedule(
            use_dot=True,
            cast_fp32_to_bf16=cast_fp32_to_bf16,
            post_scale_only=post_scale_only,
            block_n=NVFP4_DOT_BLOCK_N,
            block_k=block_k,
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
    force_fp32_dot = active_dtype is torch.float32 and logical_dtype is not torch.bfloat16
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
    is_wave64 = target.is_hip_architecture("gfx942")
    if block_m == PREFILL_NVFP4_SPARSE_BLOCK_M:
        block_n = PREFILL_NVFP4_SPARSE_BLOCK_N
        num_warps = (
            NVFP4_DOT_NUM_WARPS_CUDA if (force_fp32_dot and not is_wave64) else NVFP4_GEMV_NUM_WARPS
        )
    else:
        block_n = PREFILL_NVFP4_SPARSE_BLOCK_N if force_fp32_dot else PREFILL_NVFP4_DENSE_BLOCK_N
        num_warps = NVFP4_GEMV_NUM_WARPS if is_wave64 else NVFP4_DOT_NUM_WARPS_CUDA
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
