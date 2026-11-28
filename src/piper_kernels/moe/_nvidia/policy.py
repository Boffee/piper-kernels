"""NVIDIA MoE launch schedules for NVFP4, ConvRot INT8, and ConvRot Trellis."""

from __future__ import annotations

from dataclasses import replace

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from .._schedule import (
    NVFP4_DOT_BLOCK_N,
    NVFP4_DOT_NUM_WARPS_CUDA,
    NVFP4_SM75_SCALAR_BLOCK_K,
    NVFP4_SM120_DOT_NUM_WARPS,
    NVFP4_SM120_GEMV_BLOCK_K,
    NVFP4_SM120_M1_SPLITK_CTA_MULT,
    NVFP4_SM120_PREFILL_BLOCK_K,
    TRELLIS_GEMV_BATCH_BLOCK_M,
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


def select_nvfp4_m1_schedule(target: AcceleratorTarget) -> Nvfp4M1Schedule:
    """Resolve single-token NVFP4 MoE projection schedule for ``target``."""
    base = default_nvfp4_m1_schedule()
    if target.is_cuda_capability(12):
        return replace(
            base,
            block_k=NVFP4_SM120_GEMV_BLOCK_K,
            splitk_cta_multiplier=NVFP4_SM120_M1_SPLITK_CTA_MULT,
            use_sm120_e2m1x2=True,
        )
    return base


def select_nvfp4_batched_schedule(
    target: AcceleratorTarget,
    *,
    rows: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> Nvfp4BatchedSchedule:
    """Resolve small-batch NVFP4 MoE schedule for NVIDIA SM75/SM8x/SM12x."""
    is_sm75 = not target.cuda_capability_at_least(8)
    is_sm120 = target.is_cuda_capability(12)
    base = default_nvfp4_batched_schedule(
        rows=rows,
        active_dtype=active_dtype,
        logical_dtype=logical_dtype,
        allow_dot=not (is_sm75 and logical_dtype is torch.bfloat16),
    )
    if base.use_dot:
        return replace(
            base,
            num_warps=NVFP4_SM120_DOT_NUM_WARPS if is_sm120 else NVFP4_DOT_NUM_WARPS_CUDA,
            single_dot=is_sm120,
            use_sm120_e2m1x2=is_sm120,
        )
    if is_sm120:
        return replace(base, block_k=NVFP4_SM120_GEMV_BLOCK_K, use_sm120_e2m1x2=True)
    if is_sm75:
        return replace(base, block_k=NVFP4_SM75_SCALAR_BLOCK_K)
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
    """Resolve grouped prefill NVFP4 MoE schedule for NVIDIA SM75/SM8x/SM12x."""
    is_sm120 = target.is_cuda_capability(12)
    sm75_bf16 = not target.cuda_capability_at_least(8) and logical_dtype is torch.bfloat16
    base = default_nvfp4_prefill_schedule(
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        active_dtype=active_dtype,
        logical_dtype=logical_dtype,
        force_fp32_dot_override=sm75_bf16,
    )
    if not is_sm120:
        return base
    if not base.force_fp32_dot:
        return replace(
            base,
            block_n=NVFP4_DOT_BLOCK_N,
            block_k=NVFP4_SM120_PREFILL_BLOCK_K,
            num_warps=NVFP4_SM120_DOT_NUM_WARPS,
            single_dot=True,
            use_sm120_e2m1x2=True,
        )
    return replace(base, use_sm120_e2m1x2=True)


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
