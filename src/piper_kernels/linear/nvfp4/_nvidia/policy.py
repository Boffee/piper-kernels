"""NVIDIA SM12x NVFP4 W4A16 launch schedules."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from .._w4a16_schedule import (
    W4A16_DOT_BLOCK_K,
    W4A16_DOT_BLOCK_N,
    W4A16_GEMV_BLOCK_K,
    W4A16_GEMV_BLOCK_N,
    W4A16_GEMV_MAX_ROWS,
    W4A16_NARROW_BLOCK_N,
    W4A16Schedule,
    cdiv,
    select_w4a16_split_k,
)

SM120_DEQUANT_SCHEDULE = W4A16Schedule(
    mode="dequant",
    block_n=64,
    block_k=256,
    num_warps=8,
)


def _sm120_w4a16_fp32_schedule(
    num_sms: int,
    rows: int,
    out_features: int,
    input_features: int,
) -> W4A16Schedule:
    if rows > W4A16_GEMV_MAX_ROWS:
        return SM120_DEQUANT_SCHEDULE
    block_m = 4 if rows <= 4 else (8 if rows <= 8 else W4A16_GEMV_MAX_ROWS)
    split_k = select_w4a16_split_k(
        num_sms,
        out_features,
        input_features,
        rows=rows,
        block_m=block_m,
        block_n=W4A16_GEMV_BLOCK_N,
        block_k=W4A16_GEMV_BLOCK_K,
        cta_multiplier=4,
        power_of_two=True,
        cap_workspace=True,
    )
    return W4A16Schedule(
        mode="gemv",
        block_m=block_m,
        block_n=W4A16_GEMV_BLOCK_N,
        block_k=W4A16_GEMV_BLOCK_K,
        num_warps=4,
        num_stages=2,
        use_dot=False,
        split_k=split_k,
    )


def _sm120_w4a16_large_m_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> W4A16Schedule:
    # Float16 above M=256 is compute-bound: cuBLAS on pre-dequantized weights beats Triton.
    if input_dtype is torch.float16 and rows > 256:
        return SM120_DEQUANT_SCHEDULE
    tiles_64x128 = cdiv(rows, 64) * cdiv(out_features, 128)
    group_m = 8 if rows > 64 else 0
    # Use 64x128 split_k=1 above 60% SM occupancy (5/3 ratio); hand off tall deep-K wave
    # tails below 80% full-wave fill (5/4 ratio) to dequantize + cuBLAS.
    if tiles_64x128 * 5 >= max(1, num_sms) * 3:
        waves = cdiv(tiles_64x128, max(1, num_sms))
        if (
            num_sms > 0
            and rows >= 1024
            and input_features >= 8192
            and tiles_64x128 * 5 < waves * num_sms * 4
        ):
            return SM120_DEQUANT_SCHEDULE
        return W4A16Schedule(
            mode="gemv",
            block_m=64,
            block_n=128,
            block_k=64,
            num_warps=8,
            num_stages=3,
            use_dot=True,
            single_dot=False,
            split_k=1,
            group_m=group_m,
        )
    split_k = select_w4a16_split_k(
        num_sms,
        out_features,
        input_features,
        rows=rows,
        block_m=64,
        block_n=W4A16_DOT_BLOCK_N,
        block_k=W4A16_DOT_BLOCK_K,
        cta_multiplier=1,
        max_split_k=8,
        round_up=False,
        power_of_two=True,
        cap_workspace=True,
    )
    return W4A16Schedule(
        mode="gemv",
        block_m=64,
        block_n=W4A16_DOT_BLOCK_N,
        block_k=W4A16_DOT_BLOCK_K,
        num_warps=8,
        num_stages=3,
        use_dot=True,
        single_dot=True,
        split_k=split_k,
        group_m=group_m,
    )


def _sm120_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> W4A16Schedule:
    if rows == 1:
        output_tiles = cdiv(out_features, W4A16_NARROW_BLOCK_N)
        underfilled = output_tiles < num_sms
        split_k = (
            select_w4a16_split_k(
                num_sms,
                out_features,
                input_features,
                block_m=1,
                block_n=W4A16_NARROW_BLOCK_N,
                block_k=512,
                cta_multiplier=6 if underfilled else 5,
                min_k_tiles=4 if underfilled else 20,
                min_tiles_per_split=2,
                power_of_two=True,
                cap_workspace=True,
            )
            if (num_sms > 0 and output_tiles < 3 * num_sms)
            else 1
        )
        return W4A16Schedule(
            mode="gemv",
            block_m=1,
            block_n=W4A16_NARROW_BLOCK_N,
            block_k=512,
            num_warps=4,
            num_stages=2,
            use_dot=False,
            split_k=split_k,
        )
    if input_dtype is torch.float32:
        return _sm120_w4a16_fp32_schedule(num_sms, rows, out_features, input_features)
    if rows <= 32:
        # 16x128 uses 6x CTA oversubscription and exact split_k for low-M latency;
        # 32x128 backs off to 2x / power-of-two so larger batches avoid reduce overhead.
        small_batch = rows <= W4A16_GEMV_MAX_ROWS
        block_m = W4A16_GEMV_MAX_ROWS if small_batch else 32
        split_k = select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=block_m,
            block_n=W4A16_DOT_BLOCK_N,
            block_k=W4A16_DOT_BLOCK_K,
            cta_multiplier=6 if small_batch else 2,
            max_split_k=16 if small_batch else 8,
            round_up=False,
            power_of_two=not small_batch,
            cap_workspace=True,
        )
        return W4A16Schedule(
            mode="gemv",
            block_m=block_m,
            block_n=W4A16_DOT_BLOCK_N,
            block_k=W4A16_DOT_BLOCK_K,
            num_warps=4,
            num_stages=3,
            use_dot=True,
            single_dot=False,
            split_k=split_k,
        )
    return _sm120_w4a16_large_m_schedule(num_sms, rows, input_dtype, out_features, input_features)


def select_w4a16_schedule(
    target: AcceleratorTarget,
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> W4A16Schedule:
    """Resolve the NVIDIA SM12x W4A16 schedule for ``target``."""
    if not target.supports_nvfp4_w4a16:
        raise ValueError(
            "NVFP4 W4A16 requires NVIDIA SM12x (use ConvRot INT8 or ConvRot Trellis on other GPUs)"
        )
    return _sm120_w4a16_schedule(num_sms, rows, input_dtype, out_features, input_features)
