"""Triton kernels and custom operations for in-place NVFP4 weight updates."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.convrot import rotate_hadamard_groups
from piper_kernels._triton.nvfp4 import (
    _amax_partial_kernel,
    _decode_fp4_code,
    _normalize_nvfp4_blocks,
    pack_e2m1_pairs,
    swizzled_scale_offsets,
)
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.stochastic_quantization import _random, seed_argument
from piper_kernels.weights.nvfp4 import _layout

_NVFP4_BLOCK_SIZE = _layout.BLOCK_SIZE
_NVFP4_QDATA_BLOCK_SIZE = _layout.QDATA_BLOCK_SIZE


@triton.jit
def _stochastic_e2m1(values, seed, offsets):
    """Sample adjacent E2M1 magnitudes, retaining exact values and saturation."""
    magnitude = tl.abs(values)
    lower = tl.where(
        magnitude < 2.0,
        tl.floor(magnitude * 2.0) * 0.5,
        tl.where(magnitude < 4.0, tl.floor(magnitude), 4.0),
    )
    width = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
    probability = (magnitude - lower) / width
    sampled = lower + tl.where(_random(seed, offsets) < probability, width, 0.0)
    sampled = tl.where(values < 0.0, -sampled, sampled)
    return tl.where((magnitude > 0.0) & (magnitude < 6.0), sampled, values)


@triton.jit
def _scale_offsets(rows, columns, features: tl.constexpr, swizzled: tl.constexpr):
    if swizzled:
        return swizzled_scale_offsets(rows, columns, tl.cdiv(features, 64))
    return rows * (features // 16) + columns


@triton.jit
def _update_kernel(  # noqa: PLR0915
    qdata_ptr,
    scale_ptr,
    old_global_ptr,
    new_global_ptr,
    mat1_ptr,
    mat2_ptr,
    partial_ptr,
    rows_count,
    features: tl.constexpr,
    rank: tl.constexpr,
    stride_a_row: tl.constexpr,
    stride_a_col: tl.constexpr,
    stride_b_row: tl.constexpr,
    stride_b_col: tl.constexpr,
    beta,
    alpha,
    seed,
    group_size: tl.constexpr,
    has_base: tl.constexpr,
    has_update: tl.constexpr,
    matmul: tl.constexpr,
    two_level: tl.constexpr,
    swizzled: tl.constexpr,
    high_first: tl.constexpr,
    stochastic: tl.constexpr,
    amax_only: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    row_tile = tl.program_id(0)
    column_tile = tl.program_id(1)
    rows = (row_tile * block_m + tl.arange(0, block_m)).to(tl.int64)
    columns = (column_tile * block_n + tl.arange(0, block_n)).to(tl.int64)
    valid = (rows[:, None] < rows_count) & (columns[None, :] < features)
    logical_dtype: tl.constexpr = mat1_ptr.dtype.element_ty

    update = tl.full((block_m, block_n), 0.0, tl.float32)
    if has_update:
        if matmul:
            reduction = tl.arange(0, block_k)
            for start in range(tl.cdiv(rank, block_k)):
                k = (start * block_k + reduction).to(tl.int64)
                left = tl.load(
                    mat1_ptr + rows[:, None] * stride_a_row + k[None, :] * stride_a_col,
                    mask=(rows[:, None] < rows_count) & (k[None, :] < rank),
                    other=0.0,
                )
                right = tl.load(
                    mat2_ptr + k[:, None] * stride_b_row + columns[None, :] * stride_b_col,
                    mask=(k[:, None] < rank) & (columns[None, :] < features),
                    other=0.0,
                )
                if group_size:
                    rotated = rotate_hadamard_groups(
                        tl.reshape(right.to(tl.float32), (block_k * block_n,)),
                        block_k * block_n,
                        group_size,
                    )
                    right = tl.reshape(rotated * (group_size**-0.5), (block_k, block_n)).to(
                        logical_dtype
                    )
                update = tl.dot(left, right, update, input_precision="tf32x3")
        else:
            update = tl.load(
                mat1_ptr + rows[:, None] * stride_a_row + columns[None, :] * stride_a_col,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            if group_size:
                rotated = rotate_hadamard_groups(
                    tl.reshape(update, (block_m * block_n,)),
                    block_m * block_n,
                    group_size,
                )
                update = tl.reshape(rotated * (group_size**-0.5), (block_m, block_n))

    base = tl.full((block_m, block_n), 0.0, tl.float32)
    if has_base:
        packed = tl.load(
            qdata_ptr + rows[:, None] * (features // 2) + columns[None, :] // 2,
            mask=valid,
            other=0,
        )
        low_lane = columns[None, :] % 2 == (1 if high_first else 0)
        code = tl.where(low_lane, packed & 15, packed >> 4)
        scale_offsets = _scale_offsets(rows[:, None], columns[None, :] // 16, features, swizzled)
        scales = tl.load(scale_ptr + scale_offsets, mask=valid, other=0.0).to(tl.float32)
        if two_level:
            scales *= tl.load(old_global_ptr).to(tl.float32)
        base = _decode_fp4_code(code) * scales

    merged = beta * base + alpha * update
    merged = tl.where(valid, merged, 0.0)
    if amax_only:
        amax = tl.max(tl.max(tl.abs(merged), axis=1), axis=0)
        tl.store(partial_ptr + row_tile * tl.cdiv(features, block_n) + column_tile, amax)
    else:
        global_scale = 1.0
        if two_level:
            global_scale = tl.load(new_global_ptr).to(tl.float32)
        block_count: tl.constexpr = block_m * block_n // 16
        blocks = tl.reshape(merged, (block_count, 16))
        normalized, encoded_scale = _normalize_nvfp4_blocks(  # pyright: ignore[reportGeneralTypeIssues]
            blocks, global_scale
        )
        if stochastic:
            element_scale = encoded_scale.to(tl.float32) * global_scale
            valid_scale = (element_scale > 0.0) & (element_scale < float("inf"))
            stochastic_values = tl.div_rn(blocks, element_scale[:, None])
            offsets = tl.reshape(rows[:, None] * features + columns[None, :], (block_count, 16))
            sampled = _stochastic_e2m1(stochastic_values, seed, offsets)
            normalized = tl.where(valid_scale[:, None], sampled, normalized)
        low, high = tl.split(tl.reshape(normalized, (block_m, block_n // 2, 2)))
        packed = pack_e2m1_pairs(high, low) if high_first else pack_e2m1_pairs(low, high)
        packed_columns = column_tile * (block_n // 2) + tl.arange(0, block_n // 2)
        tl.store(
            qdata_ptr + rows[:, None] * (features // 2) + packed_columns[None, :],
            packed,
            mask=(rows[:, None] < rows_count) & (packed_columns[None, :] < features // 2),
        )
        scale_columns = column_tile * (block_n // 16) + tl.arange(0, block_n // 16)
        scale_offsets = _scale_offsets(rows[:, None], scale_columns[None, :], features, swizzled)
        tl.store(
            scale_ptr + scale_offsets,
            tl.reshape(encoded_scale, (block_m, block_n // 16)),
            mask=(rows[:, None] < rows_count) & (scale_columns[None, :] < features // 16),
        )


@triton.jit
def _global_scale_kernel(partial_ptr, output_ptr, count, block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    values = tl.load(partial_ptr + offsets, mask=offsets < count, other=0.0)
    scale = tl.max(values, axis=0) * (1.0 / (448.0 * 6.0))  # pyright: ignore[reportOperatorIssue]
    # Keep the global reciprocal / smallest normal FP8 scale finite for zero weights.
    tl.store(output_ptr, tl.maximum(scale, 2.0**-120))


def _update_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    mat1: torch.Tensor,
    mat2: torch.Tensor | None,
    group_size: int,
    beta: float,
    alpha: float,
    swizzled: bool,
    high_first: bool,
    rounding_seed: int | None,
) -> None:
    """Merge, rotate, and pack updates with only tile maxima as workspace.

    Two-level scaling uses a read-only amax pass followed by recomputation and
    in-place packing, without materializing dense weights or a matmul result.
    """
    rows, packed_features = qdata.shape
    features = packed_features * 2
    block_m = 16 if mat2 is not None else 4
    block_n = max(128 if mat2 is not None else 256, group_size)
    grid = ((rows + block_m - 1) // block_m, (features + block_n - 1) // block_n)
    two_level = per_tensor_scale is not None
    partial = (
        torch.empty(grid[0] * grid[1], device=qdata.device, dtype=torch.float32)
        if two_level
        else None
    )
    new_global = torch.empty_like(per_tensor_scale) if per_tensor_scale is not None else None
    arguments = (
        qdata,
        scale,
        per_tensor_scale,
        new_global,
        mat1,
        mat2,
        partial,
        rows,
        features,
        mat1.shape[1] if mat2 is not None else 0,
        *mat1.stride(),
        *(mat2.stride() if mat2 is not None else (0, 0)),
        beta,
        alpha,
        seed_argument(rounding_seed),
    )
    options = {
        "group_size": group_size,
        "has_base": beta != 0,
        "has_update": alpha != 0 and (mat2 is None or mat1.shape[1] != 0),
        "matmul": mat2 is not None,
        "two_level": two_level,
        "swizzled": swizzled,
        "high_first": high_first,
        "stochastic": rounding_seed is not None,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": 32,
        "num_warps": 4,
        "enable_fp_fusion": False,
    }
    with device_context(qdata.device):
        if two_level:
            _update_kernel[grid](*arguments, amax_only=True, **options)  # pyright: ignore[reportArgumentType]
            assert partial is not None
            while partial.numel() > 1024:
                count = (partial.numel() + 1023) // 1024
                reduced = torch.empty(count, device=qdata.device, dtype=torch.float32)
                _amax_partial_kernel[(count,)](
                    partial,
                    reduced,
                    partial.numel(),
                    block_size=1024,
                    num_warps=4,
                )
                partial = reduced
            _global_scale_kernel[(1,)](
                partial,
                new_global,
                partial.numel(),
                block_size=triton.next_power_of_2(partial.numel()),
                num_warps=4,
            )
        _update_kernel[grid](*arguments, amax_only=False, **options)  # pyright: ignore[reportArgumentType]
        if per_tensor_scale is not None:
            assert new_global is not None
            # All packing CTAs must finish reading the old global scale first.
            per_tensor_scale.copy_(new_global)


@torch.library.custom_op(
    "piper_kernels::nvfp4_addmm_",
    mutates_args=("qdata", "scale", "per_tensor_scale"),
)
def addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
    swizzled: bool,
    high_first: bool,
    rounding_seed: int | None = None,
) -> None:
    """Fuse an addmm update and optional rotation into in-place NVFP4 packing."""
    _update_(
        qdata,
        scale,
        per_tensor_scale,
        mat1,
        mat2,
        group_size,
        beta,
        alpha,
        swizzled,
        high_first,
        rounding_seed,
    )


@addmm_.register_fake
def _addmm_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _per_tensor_scale: torch.Tensor | None,
    _mat1: torch.Tensor,
    _mat2: torch.Tensor,
    _group_size: int,
    _beta: float,
    _alpha: float,
    _swizzled: bool,
    _high_first: bool,
    _rounding_seed: int | None = None,
) -> None:
    return None


@torch.library.custom_op(
    "piper_kernels::nvfp4_add_",
    mutates_args=("qdata", "scale", "per_tensor_scale"),
)
def add_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    update: torch.Tensor,
    group_size: int,
    alpha: float,
    swizzled: bool,
    high_first: bool,
    rounding_seed: int | None = None,
) -> None:
    """Fuse a dense update and optional rotation into in-place NVFP4 packing."""
    _update_(
        qdata,
        scale,
        per_tensor_scale,
        update,
        None,
        group_size,
        1.0,
        alpha,
        swizzled,
        high_first,
        rounding_seed,
    )


@add_.register_fake
def _add_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _per_tensor_scale: torch.Tensor | None,
    _update: torch.Tensor,
    _group_size: int,
    _alpha: float,
    _swizzled: bool,
    _high_first: bool,
    _rounding_seed: int | None = None,
) -> None:
    return None
