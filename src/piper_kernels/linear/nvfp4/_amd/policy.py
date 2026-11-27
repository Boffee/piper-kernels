"""AMD ROCm NVFP4 W4A16 launch schedules (RDNA4 and CDNA/RDNA3 fallback)."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from .._w4a16_schedule import (
    W4A16_DOT_BLOCK_K,
    W4A16_DOT_BLOCK_N,
    W4A16_GEMV_MAX_ROWS,
    W4A16_NARROW_BLOCK_N,
    W4A16Schedule,
    cdiv,
    default_w4a16_schedule,
    select_w4a16_split_k,
)

RDNA4_DEQUANT_SCHEDULE = W4A16Schedule(
    mode="dequant",
    block_n=64,
    block_k=256,
    num_warps=8,
)


def _rdna4_w4a16_large_m_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> W4A16Schedule:
    if rows <= 32:
        split_k = select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=32,
            block_n=W4A16_DOT_BLOCK_N,
            block_k=W4A16_DOT_BLOCK_K,
            cta_multiplier=8,
            max_split_k=4,
            round_up=False,
            power_of_two=True,
            cap_workspace=True,
        )
        return W4A16Schedule(
            mode="gemv",
            block_m=32,
            block_n=W4A16_DOT_BLOCK_N,
            block_k=W4A16_DOT_BLOCK_K,
            num_warps=8,
            use_dot=True,
            split_k=split_k,
        )
    if rows <= 64:
        split_k = select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=64,
            block_n=W4A16_DOT_BLOCK_N,
            block_k=W4A16_DOT_BLOCK_K,
            cta_multiplier=3,
            max_split_k=2,
            min_tiles_per_split=16,
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
            use_dot=True,
            split_k=split_k,
            group_m=8,
        )
    dequant_crossover = 512 if input_dtype is torch.bfloat16 else 1024
    if rows > dequant_crossover:
        return RDNA4_DEQUANT_SCHEDULE
    tiles_128x128 = cdiv(rows, 128) * cdiv(out_features, 128)
    if (
        input_dtype is torch.bfloat16
        and rows > 128
        and num_sms > 0
        and tiles_128x128 >= 2 * num_sms
    ):
        return W4A16Schedule(
            mode="gemv",
            block_m=128,
            block_n=128,
            block_k=64,
            num_warps=8,
            use_dot=True,
            single_dot=False,
            split_k=1,
            group_m=8,
        )
    tiles_128x64 = cdiv(rows, 128) * cdiv(out_features, 64)
    if num_sms > 0 and tiles_128x64 >= 2 * num_sms:
        return W4A16Schedule(
            mode="gemv",
            block_m=128,
            block_n=64,
            block_k=64,
            num_warps=8,
            use_dot=True,
            single_dot=True,
            split_k=1,
            group_m=8,
        )
    split_k = select_w4a16_split_k(
        num_sms,
        out_features,
        input_features,
        rows=rows,
        block_m=64,
        block_n=W4A16_DOT_BLOCK_N,
        block_k=W4A16_DOT_BLOCK_K,
        cta_multiplier=2,
        max_split_k=2,
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
        use_dot=True,
        split_k=split_k,
        group_m=8,
    )


def _rdna4_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> W4A16Schedule:
    if input_dtype is torch.float32 and rows > 1:
        if rows > W4A16_GEMV_MAX_ROWS:
            return RDNA4_DEQUANT_SCHEDULE
        block_m = 4 if rows <= 4 else (8 if rows <= 8 else W4A16_GEMV_MAX_ROWS)
        split_k = select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=block_m,
            block_n=W4A16_NARROW_BLOCK_N,
            block_k=512,
            cta_multiplier=16,
            min_k_tiles=4,
            min_tiles_per_split=2,
            max_split_k=8,
            power_of_two=True,
            cap_workspace=True,
        )
        return W4A16Schedule(
            mode="gemv",
            block_m=block_m,
            block_n=W4A16_NARROW_BLOCK_N,
            block_k=512,
            num_warps=4,
            use_dot=False,
            split_k=split_k,
        )
    if rows <= 4:
        block_m, block_n, block_k = (
            (1, 1, 512 if input_features < 1024 else 1024) if rows == 1 else (4, 2, 512)
        )
        split_k = (
            select_w4a16_split_k(
                num_sms,
                out_features,
                input_features,
                rows=rows,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                cta_multiplier=4,
                min_k_tiles=4,
                min_tiles_per_split=2,
                power_of_two=True,
                cap_workspace=True,
            )
            if (num_sms > 0 and cdiv(out_features, block_n) < 4 * num_sms)
            else 1
        )
        return W4A16Schedule(
            mode="gemv",
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=1,
            use_dot=False,
            split_k=split_k,
        )
    if rows <= W4A16_GEMV_MAX_ROWS:
        split_k = select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            rows=rows,
            block_m=W4A16_GEMV_MAX_ROWS,
            block_n=32,
            block_k=W4A16_DOT_BLOCK_K,
            cta_multiplier=4,
            max_split_k=4,
            round_up=False,
            power_of_two=True,
            cap_workspace=True,
        )
        return W4A16Schedule(
            mode="gemv",
            block_m=W4A16_GEMV_MAX_ROWS,
            block_n=32,
            block_k=W4A16_DOT_BLOCK_K,
            num_warps=4,
            use_dot=True,
            split_k=split_k,
        )
    return _rdna4_w4a16_large_m_schedule(num_sms, rows, input_dtype, out_features, input_features)


def select_w4a16_schedule(
    target: AcceleratorTarget,
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
) -> W4A16Schedule:
    """Resolve the AMD ROCm W4A16 schedule for ``target``."""
    if target.is_architecture("gfx1200", "gfx1201"):
        return _rdna4_w4a16_schedule(num_sms, rows, input_dtype, out_features, input_features)
    return default_w4a16_schedule(
        num_sms,
        rows,
        input_dtype,
        out_features,
        input_features,
        allow_dot=True,
        dot_num_warps=4,
    )
