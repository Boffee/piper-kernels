"""NVIDIA MoE launch schedules for NVFP4, ConvRot INT8, and ConvRot Trellis."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from .._schedule import (
    NVFP4_DOT_BLOCK_K,
    NVFP4_DOT_BLOCK_N,
    NVFP4_DOT_NUM_WARPS_CUDA,
    NVFP4_GEMV_BLOCK_K,
    NVFP4_GEMV_BLOCK_N,
    NVFP4_GEMV_NUM_WARPS,
    NVFP4_SM75_SCALAR_BLOCK_K,
    NVFP4_SM120_DOT_NUM_WARPS,
    NVFP4_SM120_GEMV_BLOCK_K,
    NVFP4_SM120_M1_SPLITK_CTA_MULT,
    NVFP4_SM120_PREFILL_BLOCK_K,
    PREFILL_NVFP4_BLOCK_K,
    PREFILL_NVFP4_DENSE_BLOCK_M,
    PREFILL_NVFP4_DENSE_BLOCK_N,
    PREFILL_NVFP4_SPARSE_BLOCK_M,
    PREFILL_NVFP4_SPARSE_BLOCK_N,
    TRELLIS_GEMV_BATCH_BLOCK_M,
    TRELLIS_GEMV_NUM_WARPS,
    Int8LaunchSchedule,
    Nvfp4BatchedSchedule,
    Nvfp4M1Schedule,
    Nvfp4PrefillSchedule,
    TrellisMoeSchedule,
    default_nvfp4_m1_schedule,
    select_prefill_block_m,
)


def select_nvfp4_m1_schedule(target: AcceleratorTarget) -> Nvfp4M1Schedule:
    """Resolve single-token NVFP4 MoE projection schedule for ``target``."""
    if target.is_cuda_capability(12):
        return Nvfp4M1Schedule(
            block_n=NVFP4_GEMV_BLOCK_N,
            block_k=NVFP4_SM120_GEMV_BLOCK_K,
            num_warps=NVFP4_GEMV_NUM_WARPS,
            splitk_cta_multiplier=NVFP4_SM120_M1_SPLITK_CTA_MULT,
            use_sm120_e2m1x2=True,
        )
    return default_nvfp4_m1_schedule()


def select_nvfp4_batched_schedule(
    target: AcceleratorTarget,
    *,
    rows: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> Nvfp4BatchedSchedule:
    """Resolve small-batch NVFP4 MoE schedule for NVIDIA SM75/SM8x/SM12x."""
    is_sm75 = not target.cuda_capability_at_least(8)
    sm75_bf16 = is_sm75 and logical_dtype is torch.bfloat16
    use_dot = (
        rows > 4
        and not sm75_bf16
        and (
            active_dtype in (torch.float16, torch.bfloat16)
            or (active_dtype is torch.float32 and logical_dtype is torch.bfloat16)
        )
    )
    cast_fp32_to_bf16 = use_dot and active_dtype is torch.float32
    post_scale_only = not (use_dot and active_dtype is torch.float16)
    is_sm120 = target.is_cuda_capability(12)
    if use_dot:
        return Nvfp4BatchedSchedule(
            use_dot=True,
            cast_fp32_to_bf16=cast_fp32_to_bf16,
            post_scale_only=post_scale_only,
            block_n=NVFP4_DOT_BLOCK_N,
            block_k=NVFP4_DOT_BLOCK_K,
            num_warps=NVFP4_SM120_DOT_NUM_WARPS if is_sm120 else NVFP4_DOT_NUM_WARPS_CUDA,
            single_dot=is_sm120,
            use_sm120_e2m1x2=is_sm120,
        )
    block_k = (
        NVFP4_SM120_GEMV_BLOCK_K
        if is_sm120
        else (NVFP4_SM75_SCALAR_BLOCK_K if is_sm75 else NVFP4_GEMV_BLOCK_K)
    )
    return Nvfp4BatchedSchedule(
        use_dot=False,
        cast_fp32_to_bf16=False,
        post_scale_only=post_scale_only,
        block_n=NVFP4_GEMV_BLOCK_N,
        block_k=block_k,
        num_warps=NVFP4_GEMV_NUM_WARPS,
        single_dot=False,
        use_sm120_e2m1x2=is_sm120,
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
    """Resolve grouped prefill NVFP4 MoE schedule for NVIDIA SM75/SM8x/SM12x."""
    is_sm120 = target.is_cuda_capability(12)
    sm75_bf16 = not target.cuda_capability_at_least(8) and logical_dtype is torch.bfloat16
    force_fp32_dot = (
        active_dtype is torch.float32 and logical_dtype is not torch.bfloat16
    ) or sm75_bf16
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
    if is_sm120 and not force_fp32_dot:
        return Nvfp4PrefillSchedule(
            cast_fp32_to_bf16=cast_fp32_to_bf16,
            force_fp32_dot=False,
            half_scales=half_scales,
            block_m=block_m,
            block_n=NVFP4_DOT_BLOCK_N,
            block_k=NVFP4_SM120_PREFILL_BLOCK_K,
            num_warps=NVFP4_SM120_DOT_NUM_WARPS,
            single_dot=True,
            use_sm120_e2m1x2=True,
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
        use_sm120_e2m1x2=is_sm120,
    )


def select_int8_launch_schedule(_target: AcceleratorTarget) -> Int8LaunchSchedule:
    """Resolve NVIDIA ConvRot INT8 MoE K-tiling and pipeline-stage schedule."""
    return Int8LaunchSchedule(block_k=64, num_stages=2, lds_only_fences=False)


def select_trellis_schedule(
    target: AcceleratorTarget,
    *,
    batched: bool,
    num_pairs: int,
    expected_slots: int,
) -> TrellisMoeSchedule:
    """Resolve NVIDIA ConvRot Trellis MoE GEMV schedule for ``target``."""
    del num_pairs, expected_slots
    block_m = TRELLIS_GEMV_BATCH_BLOCK_M if batched else 1
    return TrellisMoeSchedule(
        block_m=block_m,
        block_n=1,
        num_warps=TRELLIS_GEMV_NUM_WARPS,
        unroll_k=2 if (target.is_cuda_capability(12) and block_m == 1) else 1,
        use_dp4a=True,
        use_sad_u8=False,
        split_x_loads=False,
        use_subtile_splitk=False,
    )


def supports_trellis_int8_prefill(target: AcceleratorTarget) -> bool:
    """Return whether NVIDIA ``target`` supports Trellis INT8 W8A8 prefill scratch."""
    return target.cuda_capability_at_least(8)
