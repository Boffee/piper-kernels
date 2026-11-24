"""Triton kernels and GPU dispatcher for single-token (``M = 1``) quantized MoE decode."""

# Triton's JIT launcher and constexpr arguments are not fully represented by its
# Python typing surface.
# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._input_activations import parse_input_activation
from piper_kernels._input_activations.triton import gelu_tanh, swiglu
from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._splitk import device_sm_count, select_split_k
from piper_kernels.linear.convrot.int8.triton import _accumulate_int8_tile, splitk_accumulator_seed
from piper_kernels.linear.nvfp4.triton import (
    _accumulate_scaled_row,
    _load_nvfp4_tile,
    _split_global_scale,
)
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.convrot.triton import rotate_hadamard_groups, rotate_input
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

if TYPE_CHECKING:
    from piper_kernels.linear.convrot.int8._backend import LinearBackend, LinearExecutionPlan

_NVFP4_BLOCK_SIZE_TL = tl.constexpr(nvfp4_layout.BLOCK_SIZE)
_NVFP4_GEMV_BLOCK_N = 16
_NVFP4_GEMV_BLOCK_K = 128
_NVFP4_GEMV_NUM_WARPS = 4
_NVFP4_REDUCE_BLOCK_N = 128
_NVFP4_MAX_SPLIT_K = 16
_NVFP4_MIN_SPLITK_WEIGHT_BYTES = 4 * 1024 * 1024

_INT8_GEMM_BLOCK_M = 16
_INT8_GEMM_BLOCK_N = 64
_INT8_REDUCE_BLOCK_N = 128
_INT8_MAX_SPLIT_K = 16
_INT8_MIN_SPLITK_WEIGHT_BYTES = 8 * 1024 * 1024
_INT8_MIN_K_TILES = 8
_INT8_MIN_TILES_PER_SPLIT = 4
_INT8_PARTIAL_RATIO_DIVISOR = 64
_INT8_MAX_SAFE_REDUCTION = (2**31 - 1) // (128 * 128)

_WEIGHTED_REDUCE_BLOCK_N = 128
_ACTIVATE_BLOCK_SIZE = 256


@triton.jit
def _resolve_slot(
    topk_ids_ptr,
    expert_to_slot_ptr,
    k_idx,
    num_experts,
    num_slots,
    has_slot_map: tl.constexpr,
):
    """Resolve logical expert ID for ``k_idx`` to a slot index, or ``-1`` on miss."""
    expert_id = tl.load(topk_ids_ptr + k_idx).to(tl.int64)
    if has_slot_map:
        in_range = (expert_id >= 0) & (expert_id < num_experts)
        safe_expert = tl.where(in_range, expert_id, 0)
        slot = tl.load(expert_to_slot_ptr + safe_expert).to(tl.int64)
        slot = tl.where(in_range, slot, -1)
    else:
        slot = expert_id
    return tl.where((slot >= 0) & (slot < num_slots), slot, -1)


@triton.jit(do_not_specialize=["split_k"])
def nvfp4_moe_m1_gemv_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    output_ptr,
    out_features,
    input_features,
    scale_column_blocks,
    expert_scale_stride,
    num_experts,
    num_slots,
    top_k,
    split_k,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    has_slot_map: tl.constexpr,
    has_global_scale: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
):
    """Single-token slot-indirect NVFP4 W4A16 GEMV across ``(N, top_k, split_k)``."""
    pid_n = tl.program_id(0)
    k_idx = tl.program_id(1)
    pid_k = tl.program_id(2)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        return

    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    packed_k = input_features // 2
    scale_k = input_features // _NVFP4_BLOCK_SIZE_TL
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * packed_k
    slot_scale_ptr = weight_scale_ptr + slot * expert_scale_stride
    slot_input_ptr = input_ptr + (k_idx.to(tl.int64) * input_features if per_expert_input else 0)
    global_scale = (
        tl.load(weight_global_scale_ptr + slot).to(tl.float32) if has_global_scale else 1.0
    )
    pre_scale, post_scale = _split_global_scale(global_scale)

    total_k_tiles = tl.cdiv(input_features, block_k)
    base_tiles = total_k_tiles // split_k
    remainder_tiles = total_k_tiles % split_k
    k_tile_start = pid_k * base_tiles + tl.minimum(pid_k, remainder_tiles)
    k_tile_end = k_tile_start + base_tiles + tl.where(pid_k < remainder_tiles, 1, 0)

    acc = tl.zeros((block_n,), dtype=tl.float32)
    for k_tile in range(k_tile_start, k_tile_end):
        k_base = k_tile * block_k
        w_even, w_odd, scales = _load_nvfp4_tile(
            slot_weight_ptr,
            slot_scale_ptr,
            n_offs_i64,
            n_valid,
            k_base,
            packed_k,
            scale_k,
            scale_column_blocks,
            block_k,
            high_first,
        )
        k_elem_offs = k_base + tl.arange(0, block_k)
        k_elem_valid = k_elem_offs < input_features
        x_vec = tl.load(slot_input_ptr + k_elem_offs, mask=k_elem_valid, other=0.0).to(tl.float32)
        if has_global_scale:
            x_vec = x_vec * pre_scale
        acc += _accumulate_scaled_row(x_vec, w_even, w_odd, scales, block_n, block_k)

    if has_global_scale and not is_splitk:
        acc *= post_scale
    row_idx = pid_k.to(tl.int64) * top_k + k_idx if is_splitk else k_idx.to(tl.int64)
    tl.store(output_ptr + row_idx * out_features + n_offs_i64, acc, mask=n_valid)


@triton.jit(do_not_specialize=["split_k"])
def nvfp4_moe_m1_reduce_kernel(
    partial_ptr,
    weight_global_scale_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    output_ptr,
    out_features,
    num_experts,
    num_slots,
    top_k,
    split_k,
    block_n: tl.constexpr,
    has_slot_map: tl.constexpr,
    has_global_scale: tl.constexpr = False,
):
    """Deterministically reduce NVFP4 up/gate Split-K partials across active slots."""
    pid_n = tl.program_id(0)
    k_idx = tl.program_id(1)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        return

    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    acc = tl.zeros((block_n,), dtype=tl.float32)
    split_stride = tl.full((), top_k, tl.int64) * out_features
    ptr = partial_ptr + tl.full((), k_idx, tl.int64) * out_features + n_offs_i64
    for _ in range(split_k):
        acc += tl.load(ptr, mask=n_valid, other=0.0)
        ptr += split_stride

    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        _, post_scale = _split_global_scale(global_scale)
        acc *= post_scale
    tl.store(output_ptr + k_idx.to(tl.int64) * out_features + n_offs_i64, acc, mask=n_valid)


@triton.jit
def moe_m1_activate_rotate_kernel(
    mid_ptr,
    act_ptr,
    logical_ref_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    intermediate_size,
    num_experts,
    num_slots,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    activation_fn: tl.constexpr,
    gate_first: tl.constexpr,
    accelerator_backend: tl.constexpr,
    has_slot_map: tl.constexpr,
):
    """Apply SwiGLU or GeLU-tanh and optional ConvRot transform for active ``M = 1`` slots."""
    tile_idx = tl.program_id(0)
    k_idx = tl.program_id(1)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        return

    offs = tile_idx * block_size + tl.arange(0, block_size)
    valid = offs < intermediate_size
    up_gate_stride = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    k_base = k_idx.to(tl.int64) * up_gate_stride

    if activation_fn == "swiglu":
        first = tl.load(mid_ptr + k_base + offs, mask=valid, other=0.0).to(tl.float32)
        second = tl.load(mid_ptr + k_base + intermediate_size + offs, mask=valid, other=0.0).to(
            tl.float32
        )
        act = swiglu(second, first) if gate_first else swiglu(first, second)
    else:
        vals = tl.load(mid_ptr + k_base + offs, mask=valid, other=0.0).to(tl.float32)
        act = gelu_tanh(vals, accelerator_backend)

    # Round activation outputs to logical_dtype before optional FP32 ConvRot rotation.
    act = act.to(logical_ref_ptr.dtype.element_ty).to(tl.float32)
    if group_size > 0:
        act = rotate_hadamard_groups(act, block_size, group_size, normalize=True)
    tl.store(
        act_ptr + k_idx.to(tl.int64) * intermediate_size + offs,
        act.to(act_ptr.dtype.element_ty),
        mask=valid,
    )


@triton.jit(do_not_specialize=["split_k"])
def int8_moe_m1_matmul_kernel(
    input_ptr,
    weight_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    output_ptr,
    out_features,
    in_features,
    num_experts,
    num_slots,
    top_k,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_slot_map: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    dequant_output: tl.constexpr,
    swap_halves: tl.constexpr,
    aligned_nk: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    """Single-token slot-indirect ConvRot INT8 projection across ``(N, top_k, split_k)``."""
    pid_n = tl.program_id(0)
    k_idx = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        if not is_splitk:
            zeros = tl.zeros((block_n,), dtype=tl.float32).to(output_ptr.dtype.element_ty)
            tl.store(
                output_ptr + k_idx.to(tl.int64) * out_features + offs_n_i64,
                zeros,
                mask=offs_n < out_features,
            )
        return

    offs_m = tl.arange(0, block_m)

    total_k_tiles = tl.cdiv(in_features, block_k)
    base_tiles = total_k_tiles // split_k
    remainder_tiles = total_k_tiles % split_k
    k_start = pid_k * base_tiles + tl.minimum(pid_k, remainder_tiles)
    k_end = k_start + base_tiles + tl.where(pid_k < remainder_tiles, 1, 0)

    slot_input_ptr = input_ptr + (k_idx.to(tl.int64) * in_features if per_expert_input else 0)
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * in_features
    acc = _accumulate_int8_tile(
        slot_input_ptr,
        slot_weight_ptr,
        offs_m,
        offs_n,
        1,
        out_features,
        in_features,
        k_start,
        k_end,
        block_m,
        block_n,
        block_k,
        False,
        aligned_nk,
        accumulator_seed,
    )
    centered = acc - accumulator_seed
    offs_n_i64 = offs_n.to(tl.int64)

    if dequant_output:
        in_scale = tl.load(input_scale_ptr + (k_idx if per_expert_input else 0))
        w_scale = tl.load(
            weight_scale_ptr + slot * out_features + offs_n_i64,
            mask=offs_n < out_features,
            other=0.0,
        )
        out_vals = (centered.to(tl.float32) * in_scale * w_scale[None, :]).to(
            output_ptr.dtype.element_ty
        )
    else:
        out_vals = centered

    if swap_halves:
        half_n = out_features // 2
        col_out = tl.where(offs_n_i64 < half_n, offs_n_i64 + half_n, offs_n_i64 - half_n)
    else:
        col_out = offs_n_i64

    row_idx = pid_k.to(tl.int64) * top_k + k_idx if is_splitk else k_idx.to(tl.int64)
    out_ptrs = (
        output_ptr + row_idx * out_features + tl.broadcast_to(col_out[None, :], (block_m, block_n))
    )
    row0_mask = (offs_m[:, None] == 0) & (offs_n[None, :] < out_features)
    tl.store(out_ptrs, out_vals, mask=row0_mask)


@triton.jit(do_not_specialize=["split_k"])
def int8_moe_m1_reduce_kernel(
    partial_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    output_ptr,
    out_features,
    num_experts,
    num_slots,
    top_k,
    split_k,
    block_n: tl.constexpr,
    has_slot_map: tl.constexpr,
    per_expert_input: tl.constexpr,
    swap_halves: tl.constexpr,
):
    """Deterministically reduce INT8 up/gate Split-K partials in INT32."""
    pid_n = tl.program_id(0)
    k_idx = tl.program_id(1)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    n_valid = offs_n < out_features
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        zeros = tl.zeros((block_n,), dtype=tl.float32).to(output_ptr.dtype.element_ty)
        tl.store(
            output_ptr + k_idx.to(tl.int64) * out_features + offs_n_i64,
            zeros,
            mask=n_valid,
        )
        return

    acc = tl.zeros((block_n,), dtype=tl.int32)
    split_stride = tl.full((), top_k, tl.int64) * out_features
    ptr = partial_ptr + tl.full((), k_idx, tl.int64) * out_features + offs_n_i64
    for _ in range(split_k):
        acc += tl.load(ptr, mask=n_valid, other=0)
        ptr += split_stride

    in_scale = tl.load(input_scale_ptr + (k_idx if per_expert_input else 0))
    w_scale = tl.load(weight_scale_ptr + slot * out_features + offs_n_i64, mask=n_valid, other=0.0)
    scaled = (acc.to(tl.float32) * in_scale * w_scale).to(output_ptr.dtype.element_ty)

    if swap_halves:
        half_n = out_features // 2
        col_out = tl.where(offs_n_i64 < half_n, offs_n_i64 + half_n, offs_n_i64 - half_n)
    else:
        col_out = offs_n_i64

    tl.store(output_ptr + k_idx.to(tl.int64) * out_features + col_out, scaled, mask=n_valid)


@triton.jit(do_not_specialize=["split_k"])
def moe_m1_weighted_reduce_kernel(
    partial_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    expert_to_slot_ptr,
    output_ptr,
    out_features,
    num_experts,
    num_slots,
    top_k,
    split_k,
    block_n: tl.constexpr,
    is_int8: tl.constexpr,
    has_slot_map: tl.constexpr,
    accumulate: tl.constexpr,
    has_global_scale: tl.constexpr = False,
):
    """Deterministically combine active expert ``down`` partials across ``top_k``."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    n_valid = offs_n < out_features
    split_stride = tl.full((), top_k, tl.int64) * out_features

    total = tl.zeros((block_n,), dtype=tl.float32)
    had_hit = tl.full((), 0, tl.int32)
    for k_idx in range(top_k):
        slot = _resolve_slot(
            topk_ids_ptr,
            expert_to_slot_ptr,
            k_idx,
            num_experts,
            num_slots,
            has_slot_map,
        )
        if slot >= 0:
            had_hit = tl.full((), 1, tl.int32)
            base_ptr = partial_ptr + tl.full((), k_idx, tl.int64) * out_features + offs_n_i64
            if is_int8:
                acc_i32 = tl.zeros((block_n,), dtype=tl.int32)
                ptr = base_ptr
                for _ in range(split_k):
                    acc_i32 += tl.load(ptr, mask=n_valid, other=0)
                    ptr += split_stride
                in_scale = tl.load(input_scale_ptr + k_idx)
                w_scale = tl.load(
                    weight_scale_ptr + slot * out_features + offs_n_i64,
                    mask=n_valid,
                    other=0.0,
                )
                expert_val = acc_i32.to(tl.float32) * in_scale * w_scale
            else:
                expert_val = tl.zeros((block_n,), dtype=tl.float32)
                ptr = base_ptr
                for _ in range(split_k):
                    expert_val += tl.load(ptr, mask=n_valid, other=0.0)
                    ptr += split_stride
                if has_global_scale:
                    global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
                    _, post_scale = _split_global_scale(global_scale)
                    expert_val *= post_scale
            router_weight = tl.load(topk_weights_ptr + k_idx).to(tl.float32)
            total += expert_val * router_weight

    if accumulate:
        if had_hit != 0:
            existing = tl.load(output_ptr + offs_n_i64, mask=n_valid, other=0.0).to(tl.float32)
            tl.store(
                output_ptr + offs_n_i64,
                (existing + total).to(output_ptr.dtype.element_ty),
                mask=n_valid,
            )
    else:
        tl.store(output_ptr + offs_n_i64, total.to(output_ptr.dtype.element_ty), mask=n_valid)


def _int8_launch_config(target: AcceleratorTarget) -> tuple[int, int, dict[str, object]]:
    is_gfx1201 = target.is_amd_hip and target.is_architecture("gfx1201")
    is_rdna4 = target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201")
    block_k = 256 if is_gfx1201 else 64
    num_stages = 2 if (target.is_nvidia_cuda or is_rdna4) else 1
    compiler_options: dict[str, object] = (
        dict(amd_compiler_workarounds.lds_only_fence_options(is_gfx1201))
        if target.is_amd_hip
        else {}
    )
    return block_k, num_stages, compiler_options


def _select_nvfp4_split_k(
    device: torch.device,
    active_slots: int,
    out_features: int,
    in_features: int,
) -> int:
    return select_split_k(
        device_sm_count(device),
        output_tiles=active_slots * triton.cdiv(out_features, _NVFP4_GEMV_BLOCK_N),
        k_tiles=triton.cdiv(in_features, _NVFP4_GEMV_BLOCK_K),
        weight_bytes=active_slots * out_features * (in_features // 2),
        min_weight_bytes=_NVFP4_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=_NVFP4_MAX_SPLIT_K,
        cta_multiplier=8,
        min_k_tiles=8,
        min_tiles_per_split=4,
        round_up=True,
    )


def _select_int8_split_k(
    device: torch.device,
    active_slots: int,
    out_features: int,
    in_features: int,
    block_k: int,
) -> int:
    if in_features > _INT8_MAX_SAFE_REDUCTION:
        return 1
    max_for_partial = max(1, in_features // _INT8_PARTIAL_RATIO_DIVISOR)
    return select_split_k(
        device_sm_count(device),
        output_tiles=active_slots * triton.cdiv(out_features, _INT8_GEMM_BLOCK_N),
        k_tiles=triton.cdiv(in_features, block_k),
        weight_bytes=active_slots * out_features * in_features,
        min_weight_bytes=_INT8_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=min(_INT8_MAX_SPLIT_K, max_for_partial),
        cta_multiplier=1,
        min_k_tiles=_INT8_MIN_K_TILES,
        min_tiles_per_split=_INT8_MIN_TILES_PER_SPLIT,
        round_up=False,
    )


def _moe_m1_nvfp4(
    active_x_row: torch.Tensor,
    topk_ids_1d: torch.Tensor,
    topk_weights_1d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    out_row: torch.Tensor,
    *,
    target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    split_k_up: int,
    split_k_down: int,
    up_scale_cols: int,
    up_scale_stride: int,
    down_scale_cols: int,
    down_scale_stride: int,
) -> None:
    device = active_x_row.device
    num_slots, hidden_size, packed_inter = down_qdata.shape
    intermediate_size = 2 * packed_inter
    up_gate_out = up_gate_qdata.shape[1]
    top_k = topk_ids_1d.shape[0]
    has_slot_map = expert_to_slot is not None
    slot_map_ptr = expert_to_slot if expert_to_slot is not None else topk_ids_1d
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots

    mid = torch.empty((top_k, up_gate_out), device=device, dtype=torch.float32)
    up_target = (
        torch.empty((split_k_up, top_k, up_gate_out), device=device, dtype=torch.float32)
        if split_k_up > 1
        else mid
    )
    up_gs_ptr = up_gate_global_scale if up_gate_global_scale is not None else mid

    act_dtype = torch.float32 if group_size > 0 else out_row.dtype
    act = torch.empty((top_k, intermediate_size), device=device, dtype=act_dtype)

    down_partial = torch.empty(
        (split_k_down, top_k, hidden_size), device=device, dtype=torch.float32
    )
    down_gs_ptr = down_global_scale if down_global_scale is not None else down_partial

    up_gemv_grid = (triton.cdiv(up_gate_out, _NVFP4_GEMV_BLOCK_N), top_k, split_k_up)
    up_reduce_grid = (triton.cdiv(up_gate_out, _NVFP4_REDUCE_BLOCK_N), top_k)
    activate_grid = (triton.cdiv(intermediate_size, _ACTIVATE_BLOCK_SIZE), top_k)
    down_gemv_grid = (triton.cdiv(hidden_size, _NVFP4_GEMV_BLOCK_N), top_k, split_k_down)

    with device_context(device):
        nvfp4_moe_m1_gemv_kernel[up_gemv_grid](
            active_x_row,
            up_gate_qdata,
            up_gate_scale.view(torch.uint8),
            up_gs_ptr,
            topk_ids_1d,
            slot_map_ptr,
            up_target,
            up_gate_out,
            hidden_size,
            up_scale_cols,
            up_scale_stride,
            num_experts,
            num_slots,
            top_k,
            split_k_up,
            block_n=_NVFP4_GEMV_BLOCK_N,
            block_k=_NVFP4_GEMV_BLOCK_K,
            high_first=high_first,
            has_slot_map=has_slot_map,
            has_global_scale=up_gate_global_scale is not None,
            per_expert_input=False,
            is_splitk=split_k_up > 1,
            num_warps=_NVFP4_GEMV_NUM_WARPS,
        )
        if split_k_up > 1:
            nvfp4_moe_m1_reduce_kernel[up_reduce_grid](
                up_target,
                up_gs_ptr,
                topk_ids_1d,
                slot_map_ptr,
                mid,
                up_gate_out,
                num_experts,
                num_slots,
                top_k,
                split_k_up,
                block_n=_NVFP4_REDUCE_BLOCK_N,
                has_slot_map=has_slot_map,
                has_global_scale=up_gate_global_scale is not None,
                num_warps=4,
            )
        moe_m1_activate_rotate_kernel[activate_grid](
            mid,
            act,
            out_row,
            topk_ids_1d,
            slot_map_ptr,
            intermediate_size,
            num_experts,
            num_slots,
            block_size=_ACTIVATE_BLOCK_SIZE,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            accelerator_backend=target.backend,
            has_slot_map=has_slot_map,
            num_warps=4,
        )
        nvfp4_moe_m1_gemv_kernel[down_gemv_grid](
            act,
            down_qdata,
            down_scale.view(torch.uint8),
            down_gs_ptr,
            topk_ids_1d,
            slot_map_ptr,
            down_partial,
            hidden_size,
            intermediate_size,
            down_scale_cols,
            down_scale_stride,
            num_experts,
            num_slots,
            top_k,
            split_k_down,
            block_n=_NVFP4_GEMV_BLOCK_N,
            block_k=_NVFP4_GEMV_BLOCK_K,
            high_first=high_first,
            has_slot_map=has_slot_map,
            has_global_scale=down_global_scale is not None,
            per_expert_input=True,
            is_splitk=True,
            num_warps=_NVFP4_GEMV_NUM_WARPS,
        )
        moe_m1_weighted_reduce_kernel[(triton.cdiv(hidden_size, _WEIGHTED_REDUCE_BLOCK_N),)](
            down_partial,
            down_partial,
            down_partial,
            down_gs_ptr,
            topk_ids_1d,
            topk_weights_1d,
            slot_map_ptr,
            out_row,
            hidden_size,
            num_experts,
            num_slots,
            top_k,
            split_k_down,
            block_n=_WEIGHTED_REDUCE_BLOCK_N,
            is_int8=False,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            has_global_scale=down_global_scale is not None,
            num_warps=4,
        )


def _moe_m1_int8(
    input_q_row: torch.Tensor,
    input_s_row: torch.Tensor,
    topk_ids_1d: torch.Tensor,
    topk_weights_1d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    out_row: torch.Tensor,
    *,
    target: AcceleratorTarget,
    int8_backend: LinearBackend,
    down_prep_plan: LinearExecutionPlan,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    swap_halves: bool,
    block_k: int,
    num_stages: int,
    compiler_options: dict[str, object],
    split_k_up: int,
    split_k_down: int,
    up_seed: int,
    down_seed: int,
    up_aligned_nk: bool,
    down_aligned_nk: bool,
    up_tiles: int,
    down_tiles: int,
) -> None:
    device = input_q_row.device
    num_slots, hidden_size, intermediate_size = down_qdata.shape
    up_gate_out = up_gate_qdata.shape[1]
    top_k = topk_ids_1d.shape[0]
    has_slot_map = expert_to_slot is not None
    slot_map_ptr = expert_to_slot if expert_to_slot is not None else topk_ids_1d
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots

    mid = torch.empty((top_k, up_gate_out), device=device, dtype=out_row.dtype)
    up_target = (
        torch.empty((split_k_up, top_k, up_gate_out), device=device, dtype=torch.int32)
        if split_k_up > 1
        else mid
    )

    with device_context(device):
        int8_moe_m1_matmul_kernel[(up_tiles, top_k, split_k_up)](
            input_q_row,
            up_gate_qdata,
            topk_ids_1d,
            slot_map_ptr,
            input_s_row,
            up_gate_scale,
            up_target,
            up_gate_out,
            hidden_size,
            num_experts,
            num_slots,
            top_k,
            split_k_up,
            block_m=_INT8_GEMM_BLOCK_M,
            block_n=_INT8_GEMM_BLOCK_N,
            block_k=block_k,
            has_slot_map=has_slot_map,
            per_expert_input=False,
            is_splitk=split_k_up > 1,
            dequant_output=split_k_up == 1,
            swap_halves=(swap_halves and split_k_up == 1),
            aligned_nk=up_aligned_nk,
            accumulator_seed=up_seed,
            **compiler_options,
            num_stages=num_stages,
            num_warps=4,
        )
        if split_k_up > 1:
            int8_moe_m1_reduce_kernel[(triton.cdiv(up_gate_out, _INT8_REDUCE_BLOCK_N), top_k)](
                up_target,
                input_s_row,
                up_gate_scale,
                topk_ids_1d,
                slot_map_ptr,
                mid,
                up_gate_out,
                num_experts,
                num_slots,
                top_k,
                split_k_up,
                block_n=_INT8_REDUCE_BLOCK_N,
                has_slot_map=has_slot_map,
                per_expert_input=False,
                swap_halves=swap_halves,
                num_warps=4,
            )

    act_q, act_s = int8_backend.prepare_input_with_plan(
        mid,
        intermediate_size,
        group_size,
        activation_fn=activation_fn,
        execution_plan=down_prep_plan,
        target=target,
    )
    down_partial = torch.empty((split_k_down, top_k, hidden_size), device=device, dtype=torch.int32)

    with device_context(device):
        int8_moe_m1_matmul_kernel[(down_tiles, top_k, split_k_down)](
            act_q,
            down_qdata,
            topk_ids_1d,
            slot_map_ptr,
            act_s,
            down_scale,
            down_partial,
            hidden_size,
            intermediate_size,
            num_experts,
            num_slots,
            top_k,
            split_k_down,
            block_m=_INT8_GEMM_BLOCK_M,
            block_n=_INT8_GEMM_BLOCK_N,
            block_k=block_k,
            has_slot_map=has_slot_map,
            per_expert_input=True,
            is_splitk=True,
            dequant_output=False,
            swap_halves=False,
            aligned_nk=down_aligned_nk,
            accumulator_seed=down_seed,
            **compiler_options,
            num_stages=num_stages,
            num_warps=4,
        )
        moe_m1_weighted_reduce_kernel[(triton.cdiv(hidden_size, _WEIGHTED_REDUCE_BLOCK_N),)](
            down_partial,
            act_s,
            down_scale,
            down_scale,
            topk_ids_1d,
            topk_weights_1d,
            slot_map_ptr,
            out_row,
            hidden_size,
            num_experts,
            num_slots,
            top_k,
            split_k_down,
            block_n=_WEIGHTED_REDUCE_BLOCK_N,
            is_int8=True,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            has_global_scale=False,
            num_warps=4,
        )


def _prepare_nvfp4_input_gpu(
    input_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    if prepared_data is not None:
        return prepared_data.reshape(input_2d.shape[0], input_2d.shape[1]).contiguous()
    input_c = input_2d.contiguous()
    if group_size == 0:
        return input_c
    rows, hidden_size = input_c.shape
    rotated = torch.empty((rows, hidden_size), device=input_c.device, dtype=torch.float32)
    rotate_input(input_c, rotated, group_size, num_warps=4)
    return rotated


def _prepare_int8_input_gpu(
    input_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    group_size: int,
    *,
    target: AcceleratorTarget,
    int8_backend: LinearBackend,
    up_prep_plan: LinearExecutionPlan | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prepared_data is not None:
        assert prepared_scale is not None
        rows, hidden_size = input_2d.shape
        return prepared_data.reshape(rows, hidden_size).contiguous(), prepared_scale.reshape(
            rows
        ).contiguous()
    assert up_prep_plan is not None
    return int8_backend.prepare_input_with_plan(
        input_2d.contiguous(),
        input_2d.shape[1],
        group_size,
        activation_fn=None,
        execution_plan=up_prep_plan,
        target=target,
    )


def moe_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None = None,
    down_global_scale: torch.Tensor | None = None,
    expert_to_slot: torch.Tensor | None = None,
    prepared_data: torch.Tensor | None = None,
    prepared_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    target: AcceleratorTarget | None = None,
    accumulate: bool = False,
    group_size: int = 0,
    activation_fn: str = "swiglu",
    gate_first: bool = True,
    high_first: bool = False,
    active_slots_hint: int = 0,
) -> torch.Tensor:
    """Execute the single-token MoE decode kernels on GPU (row-looping when ``M > 1``)."""
    device = input.device
    rows = math.prod(input.shape[:-1])
    hidden_size = input.shape[-1]
    top_k = topk_ids.shape[-1]
    result = torch.empty_like(input, memory_format=torch.contiguous_format) if out is None else out
    out_2d = result.view(rows, hidden_size)
    input_2d = input.reshape(rows, hidden_size)
    ids_2d = topk_ids.reshape(rows, top_k).contiguous()
    weights_2d = topk_weights.reshape(rows, top_k).contiguous()
    resolved_target = AcceleratorTarget.from_device(device) if target is None else target
    active_slots = max(1, min(top_k, active_slots_hint) if active_slots_hint > 0 else top_k)

    is_int8 = up_gate_qdata.dtype is torch.int8
    if is_int8:
        from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

        intermediate_size = down_qdata.shape[2]
        up_gate_out = up_gate_qdata.shape[1]
        int8_backend = int8_backend_mod.select_linear_backend_for_target(resolved_target)
        assert int8_backend is not None
        up_prep_plan = (
            None
            if prepared_data is not None
            else int8_backend.policy.select_execution_plan(resolved_target, in_features=hidden_size)
        )
        down_prep_plan = int8_backend.policy.select_execution_plan(
            resolved_target,
            in_features=intermediate_size,
            rows=top_k,
            out_features=None,
        )
        block_k, num_stages, compiler_options = _int8_launch_config(resolved_target)
        split_k_up = _select_int8_split_k(device, active_slots, up_gate_out, hidden_size, block_k)
        split_k_down = _select_int8_split_k(
            device, active_slots, hidden_size, intermediate_size, block_k
        )
        up_seed = splitk_accumulator_seed(
            resolved_target,
            rows=active_slots * _INT8_GEMM_BLOCK_M,
            out_features=up_gate_out,
            in_features=hidden_size,
            block_m=_INT8_GEMM_BLOCK_M,
            block_n=_INT8_GEMM_BLOCK_N,
            block_k=block_k,
            paired=False,
            split_k=split_k_up,
        )
        down_seed = splitk_accumulator_seed(
            resolved_target,
            rows=active_slots * _INT8_GEMM_BLOCK_M,
            out_features=hidden_size,
            in_features=intermediate_size,
            block_m=_INT8_GEMM_BLOCK_M,
            block_n=_INT8_GEMM_BLOCK_N,
            block_k=block_k,
            paired=False,
            split_k=split_k_down,
        )
        swap_halves = activation_fn == "swiglu" and gate_first
        up_aligned_nk = up_gate_out % _INT8_GEMM_BLOCK_N == 0 and hidden_size % block_k == 0
        down_aligned_nk = hidden_size % _INT8_GEMM_BLOCK_N == 0 and intermediate_size % block_k == 0
        up_tiles = triton.cdiv(up_gate_out, _INT8_GEMM_BLOCK_N)
        down_tiles = triton.cdiv(hidden_size, _INT8_GEMM_BLOCK_N)

        inp_q_2d, inp_s_1d = _prepare_int8_input_gpu(
            input_2d,
            prepared_data,
            prepared_scale,
            group_size,
            target=resolved_target,
            int8_backend=int8_backend,
            up_prep_plan=up_prep_plan,
        )
        for row_idx in range(rows):
            _moe_m1_int8(
                inp_q_2d[row_idx],
                inp_s_1d[row_idx : row_idx + 1],
                ids_2d[row_idx],
                weights_2d[row_idx],
                up_gate_qdata,
                up_gate_scale,
                down_qdata,
                down_scale,
                expert_to_slot,
                out_2d[row_idx],
                target=resolved_target,
                int8_backend=int8_backend,
                down_prep_plan=down_prep_plan,
                accumulate=accumulate,
                group_size=group_size,
                activation_fn=activation_fn,
                swap_halves=swap_halves,
                block_k=block_k,
                num_stages=num_stages,
                compiler_options=compiler_options,
                split_k_up=split_k_up,
                split_k_down=split_k_down,
                up_seed=up_seed,
                down_seed=down_seed,
                up_aligned_nk=up_aligned_nk,
                down_aligned_nk=down_aligned_nk,
                up_tiles=up_tiles,
                down_tiles=down_tiles,
            )
    else:
        intermediate_size = 2 * down_qdata.shape[2]
        up_gate_out = up_gate_qdata.shape[1]
        split_k_up = _select_nvfp4_split_k(device, active_slots, up_gate_out, hidden_size)
        split_k_down = _select_nvfp4_split_k(device, active_slots, hidden_size, intermediate_size)
        up_scale_cols = int(triton.cdiv(hidden_size, nvfp4_layout.SCALE_COLUMN_TILE))
        up_scale_stride = up_gate_scale.shape[1] * up_gate_scale.shape[2]
        down_scale_cols = int(triton.cdiv(intermediate_size, nvfp4_layout.SCALE_COLUMN_TILE))
        down_scale_stride = down_scale.shape[1] * down_scale.shape[2]

        active_2d = _prepare_nvfp4_input_gpu(input_2d, prepared_data, group_size)
        for row_idx in range(rows):
            _moe_m1_nvfp4(
                active_2d[row_idx],
                ids_2d[row_idx],
                weights_2d[row_idx],
                up_gate_qdata,
                up_gate_scale,
                down_qdata,
                down_scale,
                up_gate_global_scale,
                down_global_scale,
                expert_to_slot,
                out_2d[row_idx],
                target=resolved_target,
                accumulate=accumulate,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                high_first=high_first,
                split_k_up=split_k_up,
                split_k_down=split_k_down,
                up_scale_cols=up_scale_cols,
                up_scale_stride=up_scale_stride,
                down_scale_cols=down_scale_cols,
                down_scale_stride=down_scale_stride,
            )
    return result


_TRITON_FLOAT_POINTER_TYPES = {
    torch.float16: "*fp16",
    torch.bfloat16: "*bf16",
    torch.float32: "*fp32",
}


def _compile_int8_m1_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    has_slot_map: bool,
    split_k: int,
    accumulate: bool,
    compile_fn: Callable[[ASTSource, GPUTarget, dict[str, object]], triton.compiler.CompiledKernel],
) -> dict[str, triton.compiler.CompiledKernel]:
    if high_first:
        raise ValueError("ConvRot INT8 MoE weights do not support high_first")
    if accel.is_nvidia_cuda and not accel.cuda_capability_at_least(8):
        raise ValueError(
            f"ConvRot INT8 MoE requires NVIDIA SM80+ or supported AMD GPU, got {accel}"
        )
    validate_group_size(group_size)
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError(
            f"ConvRot INT8 MoE dimensions must be divisible by group_size {group_size}"
        )
    if hidden_size > _INT8_MAX_SAFE_REDUCTION or intermediate_size > _INT8_MAX_SAFE_REDUCTION:
        raise ValueError(
            f"ConvRot INT8 MoE widths must not exceed {_INT8_MAX_SAFE_REDUCTION} "
            f"for INT32 accumulation, got hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}"
        )
    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    block_k, num_stages, compiler_options = _int8_launch_config(accel)
    is_splitk = split_k > 1
    swap_halves = activation_fn == "swiglu" and gate_first
    up_seed = splitk_accumulator_seed(
        accel,
        rows=_INT8_GEMM_BLOCK_M,
        out_features=up_gate_out,
        in_features=hidden_size,
        block_m=_INT8_GEMM_BLOCK_M,
        block_n=_INT8_GEMM_BLOCK_N,
        block_k=block_k,
        paired=False,
        split_k=split_k,
    )
    down_seed = splitk_accumulator_seed(
        accel,
        rows=_INT8_GEMM_BLOCK_M,
        out_features=hidden_size,
        in_features=intermediate_size,
        block_m=_INT8_GEMM_BLOCK_M,
        block_n=_INT8_GEMM_BLOCK_N,
        block_k=block_k,
        paired=False,
        split_k=split_k,
    )
    int8_signature = {
        "input_ptr": "*i8",
        "weight_ptr": "*i8",
        "topk_ids_ptr": "*i64",
        "expert_to_slot_ptr": "*i64",
        "input_scale_ptr": "*fp32",
        "weight_scale_ptr": "*fp32",
        "output_ptr": "*i32" if is_splitk else out_ty,
        "out_features": "i32",
        "in_features": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "top_k": "i32",
        "split_k": "i32",
    }
    compiled_int8: dict[str, triton.compiler.CompiledKernel] = {
        "int8_matmul_up": compile_fn(
            ASTSource(
                int8_moe_m1_matmul_kernel,
                int8_signature,
                constexprs={
                    "block_m": _INT8_GEMM_BLOCK_M,
                    "block_n": _INT8_GEMM_BLOCK_N,
                    "block_k": block_k,
                    "has_slot_map": has_slot_map,
                    "per_expert_input": False,
                    "is_splitk": is_splitk,
                    "dequant_output": not is_splitk,
                    "swap_halves": (swap_halves and not is_splitk),
                    "aligned_nk": (
                        up_gate_out % _INT8_GEMM_BLOCK_N == 0 and hidden_size % block_k == 0
                    ),
                    "accumulator_seed": up_seed,
                },
            ),
            target,
            {"num_stages": num_stages, "num_warps": 4, **compiler_options},
        ),
        "int8_matmul_down": compile_fn(
            ASTSource(
                int8_moe_m1_matmul_kernel,
                {**int8_signature, "output_ptr": "*i32"},
                constexprs={
                    "block_m": _INT8_GEMM_BLOCK_M,
                    "block_n": _INT8_GEMM_BLOCK_N,
                    "block_k": block_k,
                    "has_slot_map": has_slot_map,
                    "per_expert_input": True,
                    "is_splitk": True,
                    "dequant_output": False,
                    "swap_halves": False,
                    "aligned_nk": (
                        hidden_size % _INT8_GEMM_BLOCK_N == 0 and intermediate_size % block_k == 0
                    ),
                    "accumulator_seed": down_seed,
                },
            ),
            target,
            {"num_stages": num_stages, "num_warps": 4, **compiler_options},
        ),
    }
    if is_splitk:
        compiled_int8["int8_reduce"] = compile_fn(
            ASTSource(
                int8_moe_m1_reduce_kernel,
                {
                    "partial_ptr": "*i32",
                    "input_scale_ptr": "*fp32",
                    "weight_scale_ptr": "*fp32",
                    "topk_ids_ptr": "*i64",
                    "expert_to_slot_ptr": "*i64",
                    "output_ptr": out_ty,
                    "out_features": "i32",
                    "num_experts": "i32",
                    "num_slots": "i32",
                    "top_k": "i32",
                    "split_k": "i32",
                },
                constexprs={
                    "block_n": _INT8_REDUCE_BLOCK_N,
                    "has_slot_map": has_slot_map,
                    "per_expert_input": False,
                    "swap_halves": swap_halves,
                },
            ),
            target,
            {"num_warps": 4},
        )
    compiled_int8["weighted_reduce"] = compile_fn(
        ASTSource(
            moe_m1_weighted_reduce_kernel,
            {
                "partial_ptr": "*i32",
                "input_scale_ptr": "*fp32",
                "weight_scale_ptr": "*fp32",
                "weight_global_scale_ptr": "*fp32",
                "topk_ids_ptr": "*i64",
                "topk_weights_ptr": out_ty,
                "expert_to_slot_ptr": "*i64",
                "output_ptr": out_ty,
                "out_features": "i32",
                "num_experts": "i32",
                "num_slots": "i32",
                "top_k": "i32",
                "split_k": "i32",
            },
            constexprs={
                "block_n": _WEIGHTED_REDUCE_BLOCK_N,
                "is_int8": True,
                "has_slot_map": has_slot_map,
                "accumulate": accumulate,
                "has_global_scale": False,
            },
        ),
        target,
        {"num_warps": 4},
    )
    return compiled_int8


def _compile_nvfp4_m1_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    weight_format: str,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    split_k: int,
    accumulate: bool,
    compile_fn: Callable[[ASTSource, GPUTarget, dict[str, object]], triton.compiler.CompiledKernel],
) -> dict[str, triton.compiler.CompiledKernel]:
    if hidden_size % nvfp4_layout.BLOCK_SIZE or intermediate_size % nvfp4_layout.BLOCK_SIZE:
        raise ValueError("NVFP4 MoE hidden_size and intermediate_size must be multiples of 16")
    if weight_format == "nvfp4_w4a16":
        if group_size != 0:
            raise ValueError("nvfp4_w4a16 requires group_size=0; use convrot_nvfp4_w4a16")
    elif weight_format == "convrot_nvfp4_w4a16":
        if group_size == 0:
            raise ValueError("convrot_nvfp4_w4a16 requires group_size in (16, 64, 256)")
        validate_group_size(group_size)
        if hidden_size % group_size or intermediate_size % group_size:
            raise ValueError(
                f"ConvRot NVFP4 MoE dimensions must be divisible by group_size {group_size}"
            )
    else:
        raise ValueError(f"Unsupported MoE weight_format {weight_format!r}")

    active_ty = _TRITON_FLOAT_POINTER_TYPES[torch.float32] if group_size > 0 else out_ty
    is_splitk = split_k > 1
    nvfp4_signature = {
        "input_ptr": active_ty,
        "weight_ptr": "*u8",
        "weight_scale_ptr": "*u8",
        "weight_global_scale_ptr": "*fp32",
        "topk_ids_ptr": "*i64",
        "expert_to_slot_ptr": "*i64",
        "output_ptr": "*fp32",
        "out_features": "i32",
        "input_features": "i32",
        "scale_column_blocks": "i32",
        "expert_scale_stride": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "top_k": "i32",
        "split_k": "i32",
    }
    compiled_nvfp4: dict[str, triton.compiler.CompiledKernel] = {
        "nvfp4_gemv_up": compile_fn(
            ASTSource(
                nvfp4_moe_m1_gemv_kernel,
                nvfp4_signature,
                constexprs={
                    "block_n": _NVFP4_GEMV_BLOCK_N,
                    "block_k": _NVFP4_GEMV_BLOCK_K,
                    "high_first": high_first,
                    "has_slot_map": has_slot_map,
                    "has_global_scale": has_global_scale,
                    "per_expert_input": False,
                    "is_splitk": is_splitk,
                },
            ),
            target,
            {"num_warps": _NVFP4_GEMV_NUM_WARPS},
        ),
        "nvfp4_gemv_down": compile_fn(
            ASTSource(
                nvfp4_moe_m1_gemv_kernel,
                nvfp4_signature,
                constexprs={
                    "block_n": _NVFP4_GEMV_BLOCK_N,
                    "block_k": _NVFP4_GEMV_BLOCK_K,
                    "high_first": high_first,
                    "has_slot_map": has_slot_map,
                    "has_global_scale": has_global_scale,
                    "per_expert_input": True,
                    "is_splitk": True,
                },
            ),
            target,
            {"num_warps": _NVFP4_GEMV_NUM_WARPS},
        ),
    }
    if is_splitk:
        compiled_nvfp4["nvfp4_reduce"] = compile_fn(
            ASTSource(
                nvfp4_moe_m1_reduce_kernel,
                {
                    "partial_ptr": "*fp32",
                    "weight_global_scale_ptr": "*fp32",
                    "topk_ids_ptr": "*i64",
                    "expert_to_slot_ptr": "*i64",
                    "output_ptr": "*fp32",
                    "out_features": "i32",
                    "num_experts": "i32",
                    "num_slots": "i32",
                    "top_k": "i32",
                    "split_k": "i32",
                },
                constexprs={
                    "block_n": _NVFP4_REDUCE_BLOCK_N,
                    "has_slot_map": has_slot_map,
                    "has_global_scale": has_global_scale,
                },
            ),
            target,
            {"num_warps": 4},
        )
    compiled_nvfp4["activate_rotate"] = compile_fn(
        ASTSource(
            moe_m1_activate_rotate_kernel,
            {
                "mid_ptr": "*fp32",
                "act_ptr": active_ty,
                "logical_ref_ptr": out_ty,
                "topk_ids_ptr": "*i64",
                "expert_to_slot_ptr": "*i64",
                "intermediate_size": "i32",
                "num_experts": "i32",
                "num_slots": "i32",
            },
            constexprs={
                "block_size": _ACTIVATE_BLOCK_SIZE,
                "group_size": group_size,
                "activation_fn": activation_fn,
                "gate_first": gate_first,
                "accelerator_backend": accel.backend,
                "has_slot_map": has_slot_map,
            },
        ),
        target,
        {"num_warps": 4},
    )
    compiled_nvfp4["weighted_reduce"] = compile_fn(
        ASTSource(
            moe_m1_weighted_reduce_kernel,
            {
                "partial_ptr": "*fp32",
                "input_scale_ptr": "*fp32",
                "weight_scale_ptr": "*fp32",
                "weight_global_scale_ptr": "*fp32",
                "topk_ids_ptr": "*i64",
                "topk_weights_ptr": out_ty,
                "expert_to_slot_ptr": "*i64",
                "output_ptr": out_ty,
                "out_features": "i32",
                "num_experts": "i32",
                "num_slots": "i32",
                "top_k": "i32",
                "split_k": "i32",
            },
            constexprs={
                "block_n": _WEIGHTED_REDUCE_BLOCK_N,
                "is_int8": False,
                "has_slot_map": has_slot_map,
                "accumulate": accumulate,
                "has_global_scale": has_global_scale,
            },
        ),
        target,
        {"num_warps": 4},
    )
    return compiled_nvfp4


def compile_m1_kernels(
    target: GPUTarget,
    *,
    weight_format: str = "nvfp4_w4a16",
    logical_dtype: torch.dtype = torch.bfloat16,
    hidden_size: int = 256,
    intermediate_size: int = 256,
    group_size: int = 0,
    activation_fn: str = "swiglu",
    gate_first: bool = True,
    high_first: bool = False,
    has_slot_map: bool = True,
    has_global_scale: bool = True,
    split_k: int = 1,
    accumulate: bool = False,
    compiler: Callable[[ASTSource, GPUTarget, dict[str, object]], triton.compiler.CompiledKernel]
    | None = None,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile the single-token MoE Triton kernels for an offline accelerator target."""
    if logical_dtype not in _TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported MoE logical dtype: {logical_dtype}")
    if hidden_size <= 0 or intermediate_size <= 0 or split_k <= 0:
        raise ValueError("MoE dimensions and split_k must be positive")
    parse_input_activation(activation_fn, name="MoE")

    compile_fn = compiler or (lambda src, tgt, opts: triton.compile(src, target=tgt, options=opts))
    out_ty = _TRITON_FLOAT_POINTER_TYPES[logical_dtype]
    accel = AcceleratorTarget.from_compiler_target(target)
    if weight_format == "convrot_int8":
        return _compile_int8_m1_kernels(
            target,
            accel,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            has_slot_map=has_slot_map,
            split_k=split_k,
            accumulate=accumulate,
            compile_fn=compile_fn,
        )
    return _compile_nvfp4_m1_kernels(
        target,
        accel,
        weight_format=weight_format,
        out_ty=out_ty,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        has_slot_map=has_slot_map,
        has_global_scale=has_global_scale,
        split_k=split_k,
        accumulate=accumulate,
        compile_fn=compile_fn,
    )


__all__ = [
    "compile_m1_kernels",
    "int8_moe_m1_matmul_kernel",
    "int8_moe_m1_reduce_kernel",
    "moe_ffn",
    "moe_m1_activate_rotate_kernel",
    "moe_m1_weighted_reduce_kernel",
    "nvfp4_moe_m1_gemv_kernel",
    "nvfp4_moe_m1_reduce_kernel",
]
