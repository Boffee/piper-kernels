"""Shared Triton-free W4A16 launch schedules and Split-K sizing for NVFP4 linear."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from piper_kernels.linear._splitk import select_split_k

W4A16_GEMV_MAX_ROWS = 16
W4A16_GEMV_BLOCK_N = 32
W4A16_NARROW_BLOCK_N = 16
W4A16_GEMV_BLOCK_K = 256
W4A16_DOT_BLOCK_N = 64
W4A16_DOT_BLOCK_K = 128
W4A16_DEQUANT_BLOCK_N = 32
W4A16_DEQUANT_BLOCK_K = 128
W4A16_REDUCE_BLOCK_N = 64
W4A16_FP32_WORKSPACE_BYTES = 32 * 1024 * 1024
W4A16_MAX_REDUCE_ROWS = 65_535
W4A16_MAX_SPLIT_K = 32
W4A16_MIN_SPLITK_WEIGHT_BYTES = 2 * 1024 * 1024
W4A16_NARROW_NO_SPLITK_WEIGHT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class W4A16Schedule:
    """Resolved NVFP4 W4A16 execution schedule for one target and workload."""

    mode: Literal["gemv", "dequant"]
    block_m: int = 1
    block_n: int = W4A16_GEMV_BLOCK_N
    block_k: int = W4A16_GEMV_BLOCK_K
    num_warps: int = 4
    num_stages: int | None = None
    use_dot: bool = False
    single_dot: bool = False
    use_sm120_e2m1x2: bool = False
    dequant_as_fp32: bool = False
    split_k: int = 1
    group_m: int = 0


DEFAULT_DEQUANT_SCHEDULE = W4A16Schedule(
    mode="dequant",
    block_n=W4A16_DEQUANT_BLOCK_N,
    block_k=W4A16_DEQUANT_BLOCK_K,
    num_warps=4,
)


def cdiv(value: int, divisor: int) -> int:
    """Return ceiling integer division ``ceil(value / divisor)``."""
    return (value + divisor - 1) // divisor


def select_w4a16_split_k(  # noqa: PLR0913
    num_sms: int,
    out_features: int,
    input_features: int,
    *,
    block_n: int,
    block_k: int,
    rows: int = 1,
    block_m: int = 1,
    cta_multiplier: int = 8,
    min_k_tiles: int = 8,
    min_tiles_per_split: int = 4,
    max_split_k: int = W4A16_MAX_SPLIT_K,
    round_up: bool = True,
    power_of_two: bool = False,
    cap_workspace: bool = False,
) -> int:
    """Select bounded Split-K count for a W4A16 workload."""
    if cap_workspace:
        if rows > W4A16_MAX_REDUCE_ROWS:
            return 1
        max_split_k = min(
            max_split_k,
            max(1, W4A16_FP32_WORKSPACE_BYTES // max(1, rows * out_features * 4)),
        )
        if max_split_k <= 1:
            return 1
    output_tiles = cdiv(rows, block_m) * cdiv(out_features, block_n)
    split_k = select_split_k(
        num_sms,
        output_tiles=output_tiles,
        k_tiles=cdiv(input_features, block_k),
        weight_bytes=out_features * (input_features // 2),
        min_weight_bytes=W4A16_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=max_split_k,
        cta_multiplier=cta_multiplier,
        min_k_tiles=min_k_tiles,
        min_tiles_per_split=min_tiles_per_split,
        round_up=round_up,
    )
    if power_of_two and split_k > 1:
        return 1 << (split_k.bit_length() - 1)
    return split_k


def default_w4a16_schedule(
    num_sms: int,
    rows: int,
    input_dtype: torch.dtype,
    out_features: int,
    input_features: int,
    *,
    allow_dot: bool = True,
    dot_num_warps: int = 4,
    dequant_as_fp32: bool = False,
) -> W4A16Schedule:
    """Resolve the baseline portable W4A16 schedule for generic GPU targets."""
    if rows > W4A16_GEMV_MAX_ROWS:
        if dequant_as_fp32:
            return W4A16Schedule(
                mode="dequant",
                block_n=W4A16_DEQUANT_BLOCK_N,
                block_k=W4A16_DEQUANT_BLOCK_K,
                num_warps=4,
                dequant_as_fp32=True,
            )
        return DEFAULT_DEQUANT_SCHEDULE
    weight_bytes = out_features * (input_features // 2)
    if rows == 1:
        if (
            cdiv(out_features, W4A16_GEMV_BLOCK_N) <= num_sms
            and weight_bytes < W4A16_NARROW_NO_SPLITK_WEIGHT_BYTES
        ):
            return W4A16Schedule(
                mode="gemv",
                block_m=1,
                block_n=W4A16_NARROW_BLOCK_N,
                block_k=W4A16_GEMV_BLOCK_K,
                num_warps=8,
                use_dot=False,
                split_k=1,
            )
        split_k = select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            block_n=W4A16_GEMV_BLOCK_N,
            block_k=W4A16_GEMV_BLOCK_K,
        )
        return W4A16Schedule(
            mode="gemv",
            block_m=1,
            block_n=W4A16_GEMV_BLOCK_N,
            block_k=W4A16_GEMV_BLOCK_K,
            num_warps=4,
            use_dot=False,
            split_k=split_k,
        )
    if rows <= 4 or input_dtype is torch.float32 or not allow_dot:
        block_m = 4 if rows <= 4 else (8 if rows <= 8 else W4A16_GEMV_MAX_ROWS)
        split_k = select_w4a16_split_k(
            num_sms,
            out_features,
            input_features,
            block_n=W4A16_GEMV_BLOCK_N,
            block_k=W4A16_GEMV_BLOCK_K,
        )
        return W4A16Schedule(
            mode="gemv",
            block_m=block_m,
            block_n=W4A16_GEMV_BLOCK_N,
            block_k=W4A16_GEMV_BLOCK_K,
            num_warps=4,
            use_dot=False,
            split_k=split_k,
        )
    split_k = select_w4a16_split_k(
        num_sms,
        out_features,
        input_features,
        block_n=W4A16_DOT_BLOCK_N,
        block_k=W4A16_DOT_BLOCK_K,
    )
    return W4A16Schedule(
        mode="gemv",
        block_m=W4A16_GEMV_MAX_ROWS,
        block_n=W4A16_DOT_BLOCK_N,
        block_k=W4A16_DOT_BLOCK_K,
        num_warps=dot_num_warps,
        use_dot=True,
        split_k=split_k,
    )
