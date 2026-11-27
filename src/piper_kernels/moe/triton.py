"""Triton kernels and GPU dispatcher for quantized MoE decode and prefill."""

# Triton's JIT launcher and constexpr arguments are not fully represented by its
# Python typing surface.
# pyright: reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._input_activations import parse_input_activation
from piper_kernels._input_activations.triton import gelu_tanh, swiglu
from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.primitives import (
    TRITON_FLOAT_POINTER_TYPES,
    TritonCompileFn,
    default_triton_compile,
    splitk_tile_bounds,
)
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget, rdna4_int8_accumulator_seed
from piper_kernels.linear._splitk import device_sm_count, select_split_k
from piper_kernels.linear.convrot.int8.trellis_triton import _gemv_accumulate_k_blocks
from piper_kernels.linear.convrot.int8.triton import _accumulate_int8_tile, splitk_accumulator_seed
from piper_kernels.linear.nvfp4._w4a16_triton import (
    _accumulate_scaled_row,
    _accumulate_scaled_row_blocks,
    _load_nvfp4_tile,
    _load_scaled_nvfp4_pairs,
    _split_global_scale,
)
from piper_kernels.trellis._layout import (
    WRAP_GUARD_BYTES,
    TrellisLayout,
    default_layout_for_block_size,
    validate_block_size,
)
from piper_kernels.trellis.triton import (
    _decode_stream_flags,
    _le_step_bits_4,
    compute_row_offsets_2d,
    decode_slots_int8_kernel,
    decode_slots_int8_out,
)
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.convrot.int8._quantization import dynamic_quantize_rows
from piper_kernels.weights.convrot.triton import rotate_hadamard_groups, rotate_input
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

from ._ops import _INT8_MAX_REDUCTION_WIDTH

if TYPE_CHECKING:
    from piper_kernels.linear.convrot.int8._backend import LinearBackend, LinearExecutionPlan

_NVFP4_BLOCK_SIZE_TL = tl.constexpr(nvfp4_layout.BLOCK_SIZE)
_NVFP4_QDATA_BLOCK_SIZE_TL = tl.constexpr(nvfp4_layout.QDATA_BLOCK_SIZE)
_NVFP4_GEMV_BLOCK_N = 16
_NVFP4_GEMV_BLOCK_K = 128
_NVFP4_GEMV_NUM_WARPS = 4
_NVFP4_SM120_GEMV_BLOCK_K = 512
_NVFP4_SM120_M1_SPLITK_CTA_MULT = 3
_NVFP4_SM75_SCALAR_BLOCK_K = 64
_NVFP4_SM120_DOT_NUM_WARPS = 4
_NVFP4_DOT_NUM_WARPS_CUDA = 8
_NVFP4_DOT_BLOCK_N = 64
_NVFP4_DOT_BLOCK_K = 128
_NVFP4_DOT_BLOCK_K_RDNA4 = 256
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

_TRELLIS_GEMV_BATCH_BLOCK_M = 4
_TRELLIS_GEMV_RDNA4_SPARSE_BLOCK_M = 2
_TRELLIS_GEMV_NUM_WARPS = 1
_TRELLIS_MAX_SPLIT_K = 8
_TRELLIS_MIN_SPLITK_WEIGHT_BYTES = 1024 * 1024
_TRELLIS_SCRATCH_BYTES = 32 * 1024 * 1024
_WRAP_GUARD_BYTES = tl.constexpr(WRAP_GUARD_BYTES)

_BATCH_BLOCK_M = 16
_BATCH_BLOCK_M_TL = tl.constexpr(_BATCH_BLOCK_M)
_BATCH_MAX_ROWS = 16
_BUCKET_MAX_PAIRS = 1024

_PREFILL_WORKSPACE_BYTES = 256 * 1024 * 1024
_PREFILL_MAX_PAIRS = 65_535
_PREFILL_BUCKET_BLOCK_P = 256
_PREFILL_GATHER_BLOCK_K = 256
_PREFILL_GROUP_M = 8
_PREFILL_NVFP4_SPARSE_BLOCK_M = 16
_PREFILL_NVFP4_DENSE_BLOCK_M = 32
_PREFILL_NVFP4_SPARSE_BLOCK_N = 64
_PREFILL_NVFP4_DENSE_BLOCK_N = 128
_PREFILL_NVFP4_BLOCK_K = 64
_NVFP4_SM120_PREFILL_BLOCK_K = 128
_PREFILL_INT8_SPARSE_BLOCK_M = 16
_PREFILL_INT8_DENSE_BLOCK_M = 64
_PREFILL_INT8_SPARSE_BLOCK_N = 64
_PREFILL_INT8_DENSE_BLOCK_N = 128
_PREFILL_INT8_DENSE_BLOCK_K = 64

_WEIGHTED_REDUCE_BLOCK_N = 128
_ACTIVATE_BLOCK_SIZE = 256


@triton.jit
def _swap_halves_col(offs_n_i64, out_features, swap_halves: tl.constexpr):
    """Map column offsets to swap upper/lower SwiGLU halves when gate precedes up."""
    if not swap_halves:
        return offs_n_i64
    half_n = out_features // 2
    return tl.where(offs_n_i64 < half_n, offs_n_i64 + half_n, offs_n_i64 - half_n)


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
    packed_k,
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
    use_sm120_e2m1x2: tl.constexpr = False,
):
    """Single-token slot-indirect NVFP4 W4A16 GEMV across ``(N * top_k, split_k)``."""
    n_tiles = tl.cdiv(out_features, block_n)
    pid_0 = tl.program_id(0)
    pid_n = pid_0 % n_tiles
    k_idx = pid_0 // n_tiles
    pid_k = tl.program_id(1)
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

    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    input_features = packed_k * 2
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * packed_k
    slot_scale_ptr = weight_scale_ptr + slot * expert_scale_stride
    slot_input_ptr = input_ptr + (k_idx.to(tl.int64) * input_features if per_expert_input else 0)
    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        pre_scale, post_scale = _split_global_scale(global_scale)
    else:
        pre_scale = 1.0
        post_scale = 1.0

    k_tile_start, k_tile_end = splitk_tile_bounds(tl.cdiv(packed_k, block_k // 2), pid_k, split_k)

    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    acc = tl.zeros((block_n, block_scale_k), dtype=tl.float32)
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
            use_sm120_e2m1x2,
        )
        k_elem_offs = k_base + tl.arange(0, block_k)
        k_elem_valid = k_elem_offs < input_features
        x_vec = tl.load(slot_input_ptr + k_elem_offs, mask=k_elem_valid, other=0.0).to(tl.float32)
        if has_global_scale:
            x_vec = x_vec * pre_scale
        acc += _accumulate_scaled_row_blocks(x_vec, w_even, w_odd, scales, block_n, block_k)

    acc = tl.sum(acc, axis=1)
    if has_global_scale and not is_splitk:
        acc *= post_scale
    row_idx = pid_k.to(tl.int64) * top_k + k_idx if is_splitk else k_idx.to(tl.int64)
    tl.store(output_ptr + row_idx * out_features + n_offs_i64, acc, mask=n_valid)


@triton.jit
def _segmented_count_combine(slot_a, cnt_a, slot_b, cnt_b):
    """Segmented inclusive-scan combiner for counting positions within each sorted slot run."""
    same = slot_a == slot_b
    return slot_b, tl.where(same, cnt_a + cnt_b, cnt_b)


@triton.jit
def moe_bucket_tokens_kernel(
    topk_ids_ptr,
    expert_to_slot_ptr,
    tile_slots_ptr,
    tile_pair_ids_ptr,
    num_experts,
    num_slots,
    num_pairs,
    max_tiles,
    block_p: tl.constexpr,
    block_m: tl.constexpr,
    has_slot_map: tl.constexpr,
):
    """Group ``[0, num_pairs)`` hit assignments into fixed-width ``block_m`` slot tiles."""
    p_offs = tl.arange(0, block_p)
    valid = p_offs < num_pairs
    safe_p = tl.where(valid, p_offs, 0)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        safe_p,
        num_experts,
        num_slots,
        has_slot_map,
    )
    sort_slot = tl.where(valid & (slot >= 0), slot, num_slots)

    keys = (sort_slot << 32) | p_offs.to(tl.int64)
    sorted_keys = tl.sort(keys)
    # Triton's sort stub returns a tensor-or-tuple union.
    sorted_slot = (sorted_keys >> 32).to(tl.int32)  # pyright: ignore[reportOperatorIssue]
    sorted_pair = (sorted_keys & 0xFFFFFFFF).to(tl.int32)  # pyright: ignore[reportOperatorIssue]
    is_hit = sorted_slot < num_slots

    _, run_pos_1based = tl.associative_scan(
        (sorted_slot, tl.where(is_hit, 1, 0).to(tl.int32)),
        axis=0,
        combine_fn=_segmented_count_combine,
    )
    pos_in_run = run_pos_1based - 1
    lane_in_tile = pos_in_run % block_m
    is_tile_start = is_hit & (lane_in_tile == 0)
    tile_id = tl.cumsum(is_tile_start.to(tl.int32), axis=0) - 1
    num_active_tiles = tl.sum(is_tile_start.to(tl.int32))

    # Active tiles fill [0, num_active_tiles); inactive tail fills [num_active_tiles, max_tiles).
    tl.store(tile_slots_ptr + tile_id.to(tl.int64), sorted_slot, mask=is_tile_start)
    tl.store(
        tile_slots_ptr + p_offs.to(tl.int64),
        tl.full((block_p,), -1, tl.int32),
        mask=(p_offs >= num_active_tiles) & (p_offs < max_tiles),
    )
    tl.store(
        tile_pair_ids_ptr + tile_id.to(tl.int64) * block_m + lane_in_tile.to(tl.int64),
        sorted_pair,
        mask=is_hit,
    )


@triton.jit
def _accumulate_nvfp4_batched_nodot(
    acc,
    input_ptr,
    tile_pair_base,
    k_elem_offs,
    k_elem_valid,
    k_stride,
    pre_scale,
    w_even,
    w_odd,
    scales,
    m_offs,
    tile_len,
    top_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    pre_scale_in_gemv: tl.constexpr,
    per_expert_input: tl.constexpr,
):
    # Triton's typing does not declare static_range iterable.
    for row_idx in tl.static_range(block_m):  # pyright: ignore[reportGeneralTypeIssues]
        if row_idx < tile_len:
            pair_id = tl.load(tile_pair_base + row_idx)
            src_row = (pair_id if per_expert_input else pair_id // top_k).to(tl.int64)
            x_vec = tl.load(
                input_ptr + src_row * k_stride + k_elem_offs,
                mask=k_elem_valid,
                other=0.0,
            ).to(tl.float32)
            if pre_scale_in_gemv:
                x_vec = x_vec * pre_scale
            row_acc = _accumulate_scaled_row(x_vec, w_even, w_odd, scales, block_n, block_k)
            acc += tl.where((m_offs == row_idx)[:, None], row_acc[None, :], 0.0)
    return acc


@triton.jit
def _accumulate_nvfp4_batched_dot(
    acc,
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    n_offs_i64,
    n_valid,
    k_base,
    packed_k,
    scale_k,
    scale_column_blocks,
    src_row_offsets,
    init_valid,
    pre_scale,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    pre_scale_in_gemv: tl.constexpr,
    cast_fp32_to_bf16: tl.constexpr,
    force_fp32_dot: tl.constexpr = False,
    half_scales: tl.constexpr = False,
    use_sm120_e2m1x2: tl.constexpr = False,
    single_dot: tl.constexpr = False,
):
    dot_dtype = (
        tl.float32
        if force_fp32_dot
        else (tl.bfloat16 if cast_fp32_to_bf16 else input_ptr.dtype.element_ty)
    )
    # Every finite (E2M1, E4M3FN) product fits in FP16, halving live weight registers on SM120.
    pair_dtype = tl.float16 if (use_sm120_e2m1x2 and force_fp32_dot) else dot_dtype
    w_e, w_o = _load_scaled_nvfp4_pairs(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_n,
        block_k,
        high_first,
        tl.constexpr(pair_dtype),
        use_sm120_e2m1x2,
        half_scales and not force_fp32_dot,
    )
    input_features = packed_k * 2
    k_elem_offs = k_base + tl.arange(0, block_k)
    k_elem_valid = k_elem_offs < input_features
    x_tile = tl.load(
        input_ptr + src_row_offsets[:, None] + k_elem_offs[None, :],
        mask=init_valid[:, None] & k_elem_valid[None, :],
        other=0.0,
    )
    if pre_scale_in_gemv:
        x_tile = (x_tile.to(tl.float32) * pre_scale).to(dot_dtype)
    elif cast_fp32_to_bf16 or force_fp32_dot:
        x_tile = x_tile.to(dot_dtype)
    if single_dot:
        w_full = tl.reshape(tl.join(w_e, w_o), (block_n, block_k))
        if force_fp32_dot:
            return tl.dot(x_tile, tl.trans(w_full.to(tl.float32)), acc, input_precision="ieee")
        else:
            return tl.dot(x_tile, tl.trans(w_full), acc)
    else:
        x_even, x_odd = tl.split(tl.reshape(x_tile, (block_m, block_k // 2, 2)))
        if force_fp32_dot:
            acc = tl.dot(x_even, tl.trans(w_e.to(tl.float32)), acc, input_precision="ieee")
            return tl.dot(x_odd, tl.trans(w_o.to(tl.float32)), acc, input_precision="ieee")
        else:
            acc = tl.dot(x_even, tl.trans(w_e), acc)
            return tl.dot(x_odd, tl.trans(w_o), acc)


@triton.jit(do_not_specialize=["split_k"])
def nvfp4_moe_batched_gemv_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    tile_slots_ptr,
    tile_pair_ids_ptr,
    output_ptr,
    out_features,
    packed_k,
    scale_column_blocks,
    expert_scale_stride,
    num_pairs,
    top_k,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    has_global_scale: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    use_dot: tl.constexpr,
    cast_fp32_to_bf16: tl.constexpr = False,
    use_sm120_e2m1x2: tl.constexpr = False,
    single_dot: tl.constexpr = False,
):
    """Slot-batched NVFP4 W4A16 projection across ``(N, bucket_tile, split_k)``."""
    pid_n = tl.program_id(0)
    tile_idx = tl.program_id(1)
    pid_k = tl.program_id(2)
    slot = tl.load(tile_slots_ptr + tile_idx).to(tl.int64)
    if slot < 0:
        return

    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features
    m_offs = tl.arange(0, block_m)
    tile_pair_base = tile_pair_ids_ptr + tile_idx.to(tl.int64) * block_m
    init_pair_ids = tl.load(tile_pair_base + m_offs)
    init_valid = init_pair_ids >= 0
    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    input_features = packed_k * 2
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    k_stride = tl.full((), input_features, tl.int64)
    if use_dot:
        src_pairs = tl.where(
            init_valid, init_pair_ids if per_expert_input else init_pair_ids // top_k, 0
        ).to(tl.int64)
        src_row_offsets = src_pairs * k_stride
    else:
        tile_len = tl.sum(init_valid.to(tl.int32))

    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * packed_k
    slot_scale_ptr = weight_scale_ptr + slot * expert_scale_stride

    # Keep FP16 tl.dot unscaled (cannot overflow FP32, would underflow FP16);
    # BF16 pre-scales against FP32 overflow.
    pre_scale_in_gemv: tl.constexpr = tl.constexpr(
        has_global_scale and not (use_dot and input_ptr.dtype.element_ty == tl.float16)
    )
    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        pre_scale, post_scale = _split_global_scale(global_scale)
        epilogue_scale = post_scale if pre_scale_in_gemv else global_scale
    else:
        pre_scale = 1.0
        epilogue_scale = 1.0

    k_tile_start, k_tile_end = splitk_tile_bounds(tl.cdiv(packed_k, block_k // 2), pid_k, split_k)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_tile in range(k_tile_start, k_tile_end):
        k_base = k_tile * block_k
        if use_dot:
            acc = _accumulate_nvfp4_batched_dot(
                acc,
                input_ptr,
                slot_weight_ptr,
                slot_scale_ptr,
                n_offs_i64,
                n_valid,
                k_base,
                packed_k,
                scale_k,
                scale_column_blocks,
                src_row_offsets,  # pyright: ignore[reportPossiblyUnboundVariable]
                init_valid,
                pre_scale,
                block_m,
                block_n,
                block_k,
                high_first,
                pre_scale_in_gemv,
                cast_fp32_to_bf16,
                use_sm120_e2m1x2=use_sm120_e2m1x2,
                single_dot=single_dot,
            )
        else:
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
                use_sm120_e2m1x2,
            )
            k_elem_offs = k_base + tl.arange(0, block_k)
            k_elem_valid = k_elem_offs < input_features
            acc = _accumulate_nvfp4_batched_nodot(
                acc,
                input_ptr,
                tile_pair_base,
                k_elem_offs,
                k_elem_valid,
                k_stride,
                pre_scale,
                w_even,
                w_odd,
                scales,
                m_offs,
                tile_len,  # pyright: ignore[reportPossiblyUnboundVariable]
                top_k,
                block_m,
                block_n,
                block_k,
                pre_scale_in_gemv,
                per_expert_input,
            )

    if has_global_scale and not is_splitk:
        acc *= epilogue_scale
    pair_ids = tl.load(tile_pair_base + m_offs).to(tl.int64)
    row_valid = pair_ids >= 0
    dst_rows = tl.where(
        row_valid, pid_k.to(tl.int64) * num_pairs + pair_ids if is_splitk else pair_ids, 0
    )
    tl.store(
        output_ptr + dst_rows[:, None] * out_features + n_offs_i64[None, :],
        acc,
        mask=row_valid[:, None] & n_valid[None, :],
    )


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
    """Apply SwiGLU or GeLU-tanh and optional ConvRot transform for active slots."""
    n_tiles = tl.cdiv(intermediate_size, block_size)
    pid_0 = tl.program_id(0)
    tile_idx = pid_0 % n_tiles
    k_idx = pid_0 // n_tiles
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
    """Single-token slot-indirect ConvRot INT8 projection across ``(N * top_k, split_k)``."""
    n_tiles = tl.cdiv(out_features, block_n)
    pid_0 = tl.program_id(0)
    pid_n = pid_0 % n_tiles
    k_idx = pid_0 // n_tiles
    pid_k = tl.program_id(1)
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

    k_start, k_end = splitk_tile_bounds(tl.cdiv(in_features, block_k), pid_k, split_k)

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

    col_out = _swap_halves_col(offs_n_i64, out_features, swap_halves)

    row_idx = pid_k.to(tl.int64) * top_k + k_idx if is_splitk else k_idx.to(tl.int64)
    out_ptrs = (
        output_ptr + row_idx * out_features + tl.broadcast_to(col_out[None, :], (block_m, block_n))
    )
    row0_mask = (offs_m[:, None] == 0) & (offs_n[None, :] < out_features)
    tl.store(out_ptrs, out_vals, mask=row0_mask)


@triton.jit(do_not_specialize=["split_k"])
def int8_moe_batched_matmul_kernel(
    input_ptr,
    weight_ptr,
    tile_slots_ptr,
    tile_pair_ids_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    output_ptr,
    out_features,
    in_features,
    num_pairs,
    top_k,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    dequant_output: tl.constexpr,
    swap_halves: tl.constexpr,
    aligned_nk: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
):
    """Slot-batched ConvRot INT8 projection across ``(N, bucket_tile, split_k)``."""
    pid_n = tl.program_id(0)
    tile_idx = tl.program_id(1)
    pid_k = tl.program_id(2)
    slot = tl.load(tile_slots_ptr + tile_idx).to(tl.int64)
    if slot < 0:
        return

    offs_m = tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    pair_ids = tl.load(tile_pair_ids_ptr + tile_idx.to(tl.int64) * block_m + offs_m).to(tl.int64)
    row_valid = pair_ids >= 0
    src_rows = tl.where(row_valid, pair_ids if per_expert_input else pair_ids // top_k, 0)

    k_start, k_end = splitk_tile_bounds(tl.cdiv(in_features, block_k), pid_k, split_k)

    offsets_k = tl.arange(0, block_k)
    offsets_k_i64 = offsets_k.to(tl.int64)
    k_stride = tl.full((), in_features, tl.int64)
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * in_features
    weight_pointers = (
        slot_weight_ptr
        + offs_n_i64[None, :] * k_stride
        + (tl.full((), k_start, tl.int64) * block_k + offsets_k_i64)[:, None]
    )
    src_row_offsets = src_rows * k_stride
    accumulator = tl.full((block_m, block_n), accumulator_seed, tl.int32)

    for k_offset in range(k_start, k_end):
        k_base = tl.full((), k_offset, tl.int64) * block_k
        in_ptrs = input_ptr + src_row_offsets[:, None] + (k_base + offsets_k_i64)[None, :]
        if aligned_nk:
            input_values = tl.load(in_ptrs, mask=row_valid[:, None], other=0)
            weight = tl.load(weight_pointers)
        else:
            remaining_k = in_features - k_offset * block_k
            input_values = tl.load(
                in_ptrs,
                mask=row_valid[:, None] & (offsets_k[None, :] < remaining_k),
                other=0,
            )
            weight = tl.load(
                weight_pointers,
                mask=(offs_n[None, :] < out_features) & (offsets_k[:, None] < remaining_k),
                other=0,
            )
        if accumulator_seed:
            accumulator = tl.dot(input_values, weight, accumulator)
        else:
            accumulator += tl.dot(input_values, weight)
        weight_pointers += block_k

    centered = accumulator - accumulator_seed
    if dequant_output:
        in_scales = tl.load(input_scale_ptr + src_rows, mask=row_valid, other=0.0)
        w_scale = tl.load(
            weight_scale_ptr + slot * out_features + offs_n_i64,
            mask=offs_n < out_features,
            other=0.0,
        )
        out_vals = (centered.to(tl.float32) * in_scales[:, None] * w_scale[None, :]).to(
            output_ptr.dtype.element_ty
        )
    else:
        out_vals = centered

    col_out = _swap_halves_col(offs_n_i64, out_features, swap_halves)

    dst_rows = tl.where(
        row_valid, pid_k.to(tl.int64) * num_pairs + pair_ids if is_splitk else pair_ids, 0
    )
    tl.store(
        output_ptr + dst_rows[:, None] * out_features + col_out[None, :],
        out_vals,
        mask=row_valid[:, None] & (offs_n[None, :] < out_features),
    )


@triton.jit
def _reduce_splitk_slot_value(
    partial_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    pair_idx,
    slot,
    offs_n_i64,
    n_valid,
    out_features,
    split_stride,
    split_k,
    block_n: tl.constexpr,
    is_int8: tl.constexpr,
    has_global_scale: tl.constexpr,
    post_scale_only: tl.constexpr,
    has_row_scale: tl.constexpr = False,
    top_k=1,
    scale_by_token: tl.constexpr = False,
):
    base_ptr = partial_ptr + pair_idx * out_features + offs_n_i64
    if is_int8:
        acc_i32 = tl.zeros((block_n,), dtype=tl.int32)
        ptr = base_ptr
        for _ in range(split_k):
            acc_i32 += tl.load(ptr, mask=n_valid, other=0)
            ptr += split_stride
        in_scale = tl.load(input_scale_ptr + (pair_idx // top_k if scale_by_token else pair_idx))
        w_scale = tl.load(
            weight_scale_ptr + slot * out_features + offs_n_i64,
            mask=n_valid,
            other=0.0,
        )
        return acc_i32.to(tl.float32) * in_scale * w_scale
    expert_val = tl.zeros((block_n,), dtype=tl.float32)
    ptr = base_ptr
    for _ in range(split_k):
        expert_val += tl.load(ptr, mask=n_valid, other=0.0)
        ptr += split_stride
    if has_row_scale:
        w_scale = tl.load(
            weight_scale_ptr + slot * out_features + offs_n_i64,
            mask=n_valid,
            other=0.0,
        ).to(tl.float32)
        return expert_val * w_scale
    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        if post_scale_only:
            _, post_scale = _split_global_scale(global_scale)
            return expert_val * post_scale
        return expert_val * global_scale
    return expert_val


@triton.jit(do_not_specialize=["split_k"])
def moe_splitk_reduce_kernel(
    partial_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    output_ptr,
    out_features,
    num_experts,
    num_slots,
    num_pairs,
    top_k,
    split_k,
    block_n: tl.constexpr,
    is_int8: tl.constexpr,
    has_slot_map: tl.constexpr,
    has_global_scale: tl.constexpr = False,
    post_scale_only: tl.constexpr = True,
    has_row_scale: tl.constexpr = False,
    swap_halves: tl.constexpr = False,
):
    """Deterministically reduce Split-K up/gate partials across active slots."""
    n_tiles = tl.cdiv(out_features, block_n)
    pid_0 = tl.program_id(0)
    pid_n = pid_0 % n_tiles
    k_idx = pid_0 // n_tiles
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
        if is_int8:
            zeros = tl.zeros((block_n,), dtype=tl.float32).to(output_ptr.dtype.element_ty)
            tl.store(
                output_ptr + k_idx.to(tl.int64) * out_features + offs_n_i64,
                zeros,
                mask=n_valid,
            )
        return

    split_stride = tl.full((), num_pairs, tl.int64) * out_features
    reduced = _reduce_splitk_slot_value(
        partial_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        weight_global_scale_ptr,
        k_idx.to(tl.int64),
        slot,
        offs_n_i64,
        n_valid,
        out_features,
        split_stride,
        split_k,
        block_n,
        is_int8,
        has_global_scale,
        post_scale_only,
        has_row_scale,
        top_k=top_k,
        scale_by_token=is_int8,
    )
    out_val = reduced.to(output_ptr.dtype.element_ty) if is_int8 else reduced
    col_out = _swap_halves_col(offs_n_i64, out_features, swap_halves)
    tl.store(output_ptr + k_idx.to(tl.int64) * out_features + col_out, out_val, mask=n_valid)


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
    out_row_stride,
    num_experts,
    num_slots,
    num_pairs,
    top_k,
    split_k,
    block_n: tl.constexpr,
    is_int8: tl.constexpr,
    has_slot_map: tl.constexpr,
    accumulate: tl.constexpr,
    has_global_scale: tl.constexpr = False,
    post_scale_only: tl.constexpr = True,
    use_compact_map: tl.constexpr = False,
    has_row_scale: tl.constexpr = False,
):
    """Deterministically combine active expert ``down`` partials across ``(N, rows)``."""
    pid_n = tl.program_id(0)
    row_idx = tl.program_id(1).to(tl.int64)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_n_i64 = offs_n.to(tl.int64)
    n_valid = offs_n < out_features
    split_stride = tl.full((), num_pairs, tl.int64) * out_features

    total = tl.zeros((block_n,), dtype=tl.float32)
    had_hit = tl.full((), 0, tl.int32)
    pair_base = row_idx * top_k
    for k_idx in range(top_k):
        pair_idx = pair_base + k_idx
        if use_compact_map:
            compact_idx = tl.load(topk_ids_ptr + pair_idx).to(tl.int64)
            if compact_idx >= 0:
                had_hit = tl.full((), 1, tl.int32)
                expert_val = tl.load(
                    partial_ptr + compact_idx * out_features + offs_n_i64,
                    mask=n_valid,
                    other=0.0,
                )
                router_weight = tl.load(topk_weights_ptr + pair_idx).to(tl.float32)
                total += expert_val * router_weight
        else:
            slot = _resolve_slot(
                topk_ids_ptr,
                expert_to_slot_ptr,
                pair_idx,
                num_experts,
                num_slots,
                has_slot_map,
            )
            if slot >= 0:
                had_hit = tl.full((), 1, tl.int32)
                expert_val = _reduce_splitk_slot_value(
                    partial_ptr,
                    input_scale_ptr,
                    weight_scale_ptr,
                    weight_global_scale_ptr,
                    pair_idx,
                    slot,
                    offs_n_i64,
                    n_valid,
                    out_features,
                    split_stride,
                    split_k,
                    block_n,
                    is_int8,
                    has_global_scale,
                    post_scale_only,
                    has_row_scale,
                )
                router_weight = tl.load(topk_weights_ptr + pair_idx).to(tl.float32)
                total += expert_val * router_weight

    out_row_ptr = output_ptr + row_idx * out_row_stride.to(tl.int64) + offs_n_i64
    if accumulate:
        if had_hit != 0:
            existing = tl.load(out_row_ptr, mask=n_valid, other=0.0).to(tl.float32)
            tl.store(
                out_row_ptr,
                (existing + total).to(output_ptr.dtype.element_ty),
                mask=n_valid,
            )
    else:
        tl.store(out_row_ptr, total.to(output_ptr.dtype.element_ty), mask=n_valid)


@triton.jit
def _swizzle_grouped_pid(pid, max_tiles, num_pid_n, group_m: tl.constexpr):
    if group_m > 0:
        num_pid_in_group = group_m * num_pid_n
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * group_m
        actual_group_m = tl.minimum(max_tiles - first_pid_m, group_m)
        tile_idx = first_pid_m + pid_in_group % actual_group_m
        pid_n = pid_in_group // actual_group_m
    else:
        tile_idx = pid // num_pid_n
        pid_n = pid % num_pid_n
    return tile_idx, pid_n


@triton.jit
def moe_prefill_bucket_count_kernel(
    topk_ids_ptr,
    expert_to_slot_ptr,
    slot_counts_ptr,
    pair_slots_ptr,
    pair_pos_ptr,
    num_experts,
    num_slots,
    num_pairs,
    block_p: tl.constexpr,
    has_slot_map: tl.constexpr,
):
    """Resolve slot IDs and atomically count per-slot hit positions without CPU syncs."""
    p_offs = tl.program_id(0).to(tl.int64) * block_p + tl.arange(0, block_p)
    valid = p_offs < num_pairs
    safe_p = tl.where(valid, p_offs, 0)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        safe_p,
        num_experts,
        num_slots,
        has_slot_map,
    )
    slot = tl.where(valid, slot, -1)
    tl.store(pair_slots_ptr + p_offs, slot.to(tl.int32), mask=valid)
    is_hit = valid & (slot >= 0)
    safe_slot = tl.where(is_hit, slot, 0)
    pos = tl.atomic_add(slot_counts_ptr + safe_slot, 1, mask=is_hit)
    tl.store(pair_pos_ptr + p_offs, pos, mask=is_hit)


@triton.jit
def moe_prefill_bucket_scatter_kernel(
    pair_slots_ptr,
    pair_pos_ptr,
    slot_counts_ptr,
    slot_offsets_ptr,
    slot_tile_base_ptr,
    compact_to_pair_ptr,
    pair_to_compact_ptr,
    tile_slots_ptr,
    tile_m_start_ptr,
    tile_m_len_ptr,
    num_pairs,
    block_p: tl.constexpr,
    block_m: tl.constexpr,
):
    """Scatter active ``(row, k)`` pairs into compacted rows and ``block_m`` tiles."""
    p_offs = tl.program_id(0).to(tl.int64) * block_p + tl.arange(0, block_p)
    valid = p_offs < num_pairs
    slot = tl.load(pair_slots_ptr + p_offs, mask=valid, other=-1).to(tl.int64)
    is_hit = valid & (slot >= 0)
    safe_slot = tl.where(is_hit, slot, 0)
    pos = tl.load(pair_pos_ptr + p_offs, mask=is_hit, other=0).to(tl.int64)
    slot_off = tl.load(slot_offsets_ptr + safe_slot, mask=is_hit, other=0).to(tl.int64)
    compact_idx = slot_off + pos
    tl.store(compact_to_pair_ptr + compact_idx, p_offs.to(tl.int32), mask=is_hit)
    tl.store(pair_to_compact_ptr + p_offs, compact_idx.to(tl.int32), mask=is_hit)

    is_tile_start = is_hit & ((pos % block_m) == 0)
    tile_base = tl.load(slot_tile_base_ptr + safe_slot, mask=is_tile_start, other=0).to(tl.int64)
    slot_cnt = tl.load(slot_counts_ptr + safe_slot, mask=is_tile_start, other=0).to(tl.int64)
    tile_id = tile_base + (pos // block_m)
    tile_len = tl.minimum(slot_cnt - pos, block_m).to(tl.int32)
    tl.store(tile_slots_ptr + tile_id, slot.to(tl.int32), mask=is_tile_start)
    tl.store(tile_m_start_ptr + tile_id, compact_idx.to(tl.int32), mask=is_tile_start)
    tl.store(tile_m_len_ptr + tile_id, tile_len, mask=is_tile_start)


@triton.jit
def moe_prefill_gather_kernel(
    input_ptr,
    input_scale_ptr,
    compact_to_pair_ptr,
    out_ptr,
    out_scale_ptr,
    hidden_size,
    top_k,
    block_k: tl.constexpr,
    has_scale: tl.constexpr,
):
    """Gather routed input rows (and optional INT8 scales) into contiguous slot-compacted order."""
    pid_k = tl.program_id(0)
    compact_idx = tl.program_id(1).to(tl.int64)
    pair_id = tl.load(compact_to_pair_ptr + compact_idx).to(tl.int64)
    if pair_id < 0:
        return
    src_row = pair_id // top_k
    k_offs = pid_k * block_k + tl.arange(0, block_k)
    k_valid = k_offs < hidden_size
    stride = hidden_size.to(tl.int64)
    vals = tl.load(input_ptr + src_row * stride + k_offs, mask=k_valid)
    tl.store(out_ptr + compact_idx * stride + k_offs, vals, mask=k_valid)
    if has_scale and pid_k == 0:
        in_scale = tl.load(input_scale_ptr + src_row)
        tl.store(out_scale_ptr + compact_idx, in_scale)


@triton.jit
def nvfp4_moe_grouped_gemm_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    tile_slots_ptr,
    tile_m_start_ptr,
    tile_m_len_ptr,
    output_ptr,
    max_tiles,
    out_features,
    packed_k,
    scale_column_blocks,
    expert_scale_stride,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    high_first: tl.constexpr,
    has_global_scale: tl.constexpr,
    cast_fp32_to_bf16: tl.constexpr,
    force_fp32_dot: tl.constexpr,
    half_scales: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr = False,
    single_dot: tl.constexpr = False,
):
    """Grouped NVFP4 W4A16 GEMM over contiguous slot-compacted tiles."""
    num_pid_n = tl.cdiv(out_features, block_n)
    tile_idx, pid_n = _swizzle_grouped_pid(tl.program_id(0), max_tiles, num_pid_n, group_m)
    slot = tl.load(tile_slots_ptr + tile_idx).to(tl.int64)
    if slot < 0:
        return
    m_start = tl.load(tile_m_start_ptr + tile_idx).to(tl.int64)
    tile_len = tl.load(tile_m_len_ptr + tile_idx)

    m_offs = tl.arange(0, block_m)
    m_offs_i64 = m_offs.to(tl.int64)
    m_valid = m_offs < tile_len
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    in_features = packed_k * 2
    k_stride = in_features.to(tl.int64)
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    slot_input_ptr = input_ptr + m_start * k_stride
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * packed_k
    slot_scale_ptr = weight_scale_ptr + slot * expert_scale_stride
    src_row_offsets = m_offs_i64 * k_stride

    # Keep native FP16 tl.dot unscaled to avoid underflow; pre-scale BF16/FP32 against overflow.
    is_fp16_dot: tl.constexpr = tl.constexpr(
        (not force_fp32_dot)
        and (not cast_fp32_to_bf16)
        and (input_ptr.dtype.element_ty == tl.float16)
    )
    pre_scale_in_gemm: tl.constexpr = tl.constexpr(has_global_scale and not is_fp16_dot)
    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        pre_scale, post_scale = _split_global_scale(global_scale)
        epilogue_scale = post_scale if pre_scale_in_gemm else global_scale
    else:
        pre_scale = 1.0
        epilogue_scale = 1.0

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    num_k_tiles = tl.cdiv(packed_k, block_k // 2)
    for k_tile in range(num_k_tiles):
        k_base = k_tile * block_k
        acc = _accumulate_nvfp4_batched_dot(
            acc,
            slot_input_ptr,
            slot_weight_ptr,
            slot_scale_ptr,
            n_offs_i64,
            n_valid,
            k_base,
            packed_k,
            scale_k,
            scale_column_blocks,
            src_row_offsets,
            m_valid,
            pre_scale,
            block_m,
            block_n,
            block_k,
            high_first,
            pre_scale_in_gemm,
            cast_fp32_to_bf16,
            force_fp32_dot,
            half_scales,
            use_sm120_e2m1x2,
            single_dot,
        )

    if has_global_scale:
        acc *= epilogue_scale
    tl.store(
        output_ptr + (m_start + m_offs_i64)[:, None] * out_features + n_offs_i64[None, :],
        acc,
        mask=m_valid[:, None] & n_valid[None, :],
    )


@triton.jit(do_not_specialize=["slot_start", "slot_end", "n_start", "n_rows"])
def int8_moe_grouped_gemm_kernel(
    input_ptr,
    weight_ptr,
    tile_slots_ptr,
    tile_m_start_ptr,
    tile_m_len_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    output_ptr,
    max_tiles,
    out_features,
    in_features,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    swap_halves: tl.constexpr,
    aligned_nk: tl.constexpr,
    accumulator_seed: tl.constexpr = 0,
    slot_start=0,
    slot_end=0,
    n_start=0,
    n_rows=0,
    slot_window: tl.constexpr = False,
):
    """Grouped ConvRot INT8 W8A8 GEMM over contiguous slot-compacted tiles."""
    active_n_rows = n_rows if slot_window else out_features
    n_base = n_start if slot_window else 0
    num_pid_n = tl.cdiv(active_n_rows, block_n)
    tile_idx, pid_n = _swizzle_grouped_pid(tl.program_id(0), max_tiles, num_pid_n, group_m)
    slot = tl.load(tile_slots_ptr + tile_idx).to(tl.int64)
    if slot < 0:
        return
    if slot_window:  # noqa: SIM102 - keep constexpr guard outer so LLVM strips bounds checks
        if slot < slot_start or slot >= slot_end:
            return
    m_start = tl.load(tile_m_start_ptr + tile_idx).to(tl.int64)
    tile_len = tl.load(tile_m_len_ptr + tile_idx)

    offs_m = tl.arange(0, block_m)
    offs_m_i64 = offs_m.to(tl.int64)
    offs_n_local = pid_n * block_n + tl.arange(0, block_n)
    offs_n_global_i64 = (n_base + offs_n_local).to(tl.int64)
    m_valid = offs_m < tile_len
    n_valid = offs_n_local < active_n_rows
    k_stride = in_features.to(tl.int64)
    slot_input_ptr = input_ptr + m_start * k_stride
    weight_slot = (slot - slot_start) if slot_window else slot
    slot_weight_ptr = weight_ptr + weight_slot * tl.full((), active_n_rows, tl.int64) * k_stride

    acc = _accumulate_int8_tile(
        slot_input_ptr,
        slot_weight_ptr,
        offs_m,
        offs_n_local,
        tile_len,
        active_n_rows,
        in_features,
        0,
        tl.cdiv(in_features, block_k),
        block_m,
        block_n,
        block_k,
        False,
        aligned_nk,
        accumulator_seed,
    )
    centered = acc - accumulator_seed
    in_scales = tl.load(input_scale_ptr + m_start + offs_m_i64, mask=m_valid, other=0.0)
    w_scale = tl.load(
        weight_scale_ptr + slot * out_features + offs_n_global_i64,
        mask=n_valid,
        other=0.0,
    )
    out_vals = (centered.to(tl.float32) * in_scales[:, None] * w_scale[None, :]).to(
        output_ptr.dtype.element_ty
    )

    col_out = _swap_halves_col(offs_n_global_i64, out_features, swap_halves)

    tl.store(
        output_ptr + (m_start + offs_m_i64)[:, None] * out_features + col_out[None, :],
        out_vals,
        mask=m_valid[:, None] & n_valid[None, :],
    )


@triton.jit
def _trellis_slot_row_params(
    row_widths_ptr,
    row_offsets_ptr,
    slot_offsets_ptr,
    slot,
    width_slot_idx,
    pid_n,
    out_features,
    blocks_per_row,
    bitstreams_bytes,
    denom,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    slot_bytes,
    variable_rows: tl.constexpr,
    has_slot_offsets: tl.constexpr,
):
    """Resolve slot/row bitstream offsets, byte bounds, and LE step widths for MoE GEMV."""
    if variable_rows:
        width_idx = width_slot_idx * out_features + pid_n
        width = tl.load(row_widths_ptr + width_idx).to(tl.int32)
        row_offset = tl.load(row_offsets_ptr + width_idx).to(tl.int64)
        width64 = width.to(tl.int64)
        eff_code_bits = tl.maximum(width - _WRAP_GUARD_BYTES, 0) * 8
        eff_step_bits = eff_code_bits // denom
        eff_wide_steps = eff_code_bits - eff_step_bits * denom
        if has_slot_offsets:
            raw_slot_base = tl.load(slot_offsets_ptr + slot).to(tl.int64)
            valid_base = raw_slot_base >= 0
            slot_base = tl.where(valid_base, raw_slot_base, 0)
            row_base_bytes = slot_base + row_offset * blocks_per_row
            limit_bytes = tl.where(valid_base, bitstreams_bytes.to(tl.int64), 0)
            non_empty = valid_base & (width > _WRAP_GUARD_BYTES)
        else:
            slot_base = slot * slot_bytes.to(tl.int64)
            row_base_bytes = slot_base + row_offset * blocks_per_row
            limit_bytes = tl.minimum(
                slot_base + slot_bytes.to(tl.int64), bitstreams_bytes.to(tl.int64)
            )
            non_empty = width > _WRAP_GUARD_BYTES
    else:
        width64 = bytes_per_block.to(tl.int64)
        row_base_bytes = (slot * out_features.to(tl.int64) + pid_n) * blocks_per_row * width64
        limit_bytes = bitstreams_bytes.to(tl.int64)
        eff_step_bits = step_bits
        eff_wide_steps = num_wide_steps
        non_empty = True
    return row_base_bytes, width64, limit_bytes, eff_step_bits, eff_wide_steps, non_empty


@triton.jit
def _trellis_slot_row_step(
    row_widths_ptr,
    row_offsets_ptr,
    slot_offsets_ptr,
    slot,
    width_slot_idx,
    pid_n,
    out_features,
    blocks_per_row,
    bitstreams_bytes,
    denom,
    block_size,
    first_step,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    slot_bytes,
    variable_rows: tl.constexpr,
    has_slot_offsets: tl.constexpr,
    exact_pow2: tl.constexpr,
):
    """Resolve one MoE trellis output row's storage offsets and 4-step bit indices."""
    row_base, width64, limit_bytes, esb, ews, non_empty = _trellis_slot_row_params(
        row_widths_ptr, row_offsets_ptr, slot_offsets_ptr, slot, width_slot_idx, pid_n,
        out_features, blocks_per_row, bitstreams_bytes, denom, step_bits, num_wide_steps,
        bytes_per_block, slot_bytes, variable_rows=variable_rows, has_slot_offsets=has_slot_offsets,
    )  # fmt: skip
    sb0, sb1, sb2, sb3 = _le_step_bits_4(first_step, esb, ews, denom, block_size, exact_pow2)
    row_blk = (slot * out_features.to(tl.int64) + pid_n) * blocks_per_row
    return row_blk, row_base, width64, limit_bytes, sb0, sb1, sb2, sb3, sb0 >> 3, non_empty


@triton.jit
def _trellis_slot_second_row(
    row_widths_ptr,
    row_offsets_ptr,
    slot_offsets_ptr,
    slot,
    width_slot_idx,
    n1,
    n1_in_bounds,
    out_features,
    blocks_per_row,
    bitstreams_bytes,
    denom,
    block_size,
    first_step,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    slot_bytes,
    width64_0,
    sb0_0,
    sb0_1,
    sb0_2,
    sb0_3,
    fb0,
    variable_rows: tl.constexpr,
    has_slot_offsets: tl.constexpr,
    exact_pow2: tl.constexpr,
):
    """Resolve the paired second output row for ``block_n == 2`` MoE Trellis GEMV."""
    if variable_rows:
        row_blk1, row_base1, width64_1, _, sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1_raw = (
            _trellis_slot_row_step(
                row_widths_ptr, row_offsets_ptr, slot_offsets_ptr, slot, width_slot_idx, n1,
                out_features, blocks_per_row, bitstreams_bytes, denom, block_size, first_step,
                step_bits, num_wide_steps, bytes_per_block, slot_bytes, variable_rows=True,
                has_slot_offsets=has_slot_offsets, exact_pow2=exact_pow2,
            )
        )  # fmt: skip
        ne1 = ne1_raw & n1_in_bounds
    else:
        row_blk1 = (slot * out_features.to(tl.int64) + n1) * blocks_per_row
        width64_1 = width64_0
        row_base1 = row_blk1 * width64_1
        sb1_0, sb1_1, sb1_2, sb1_3, fb1 = sb0_0, sb0_1, sb0_2, sb0_3, fb0
        ne1 = n1_in_bounds
    return row_blk1, row_base1, width64_1, sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1


@triton.jit
def _store_moe_trellis_row(
    acc,
    row_scale_ptr,
    output_ptr,
    slot,
    safe_pairs,
    m_mask,
    pid_k,
    pid_n,
    n_in_bounds,
    num_pairs,
    out_features,
    is_splitk: tl.constexpr,
):
    """Reduce K-group accumulators for one MoE output row and store partials or outputs."""
    sums = tl.sum(acc, axis=1)
    n_valid = m_mask & n_in_bounds
    if is_splitk:
        partial_offsets = (
            pid_k.to(tl.int64) * (num_pairs.to(tl.int64) * out_features)
            + safe_pairs * out_features
            + pid_n
        )
        tl.store(output_ptr + partial_offsets, sums, mask=n_valid)
    else:
        row_scale = tl.load(
            row_scale_ptr + slot * out_features + pid_n,
            mask=n_in_bounds,
            other=0.0,
        ).to(tl.float32)
        tl.store(output_ptr + safe_pairs * out_features + pid_n, sums * row_scale, mask=n_valid)


@triton.jit(
    do_not_specialize=[
        "out_features",
        "in_features",
        "blocks_per_row",
        "bitstreams_bytes",
        "block_size",
        "step_bits",
        "num_wide_steps",
        "bytes_per_block",
        "slot_bytes",
        "num_experts",
        "num_slots",
        "num_pairs",
        "top_k",
        "split_k",
    ]
)
def trellis_moe_gemv_kernel(
    x_rot_ptr,
    bitstreams_ptr,
    block_scales_ptr,
    row_scale_ptr,
    row_widths_ptr,
    row_offsets_ptr,
    slot_offsets_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    tile_slots_ptr,
    tile_pair_ids_ptr,
    output_ptr,
    out_features,
    in_features,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    slot_bytes,
    num_experts,
    num_slots,
    num_pairs,
    top_k,
    split_k,
    tile_steps: tl.constexpr,
    block_m: tl.constexpr,
    batched: tl.constexpr,
    has_slot_map: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    variable_rows: tl.constexpr,
    has_slot_offsets: tl.constexpr,
    exact_pow2: tl.constexpr,
    needs_w2: tl.constexpr,
    use_dp4a: tl.constexpr,
    has_tail_bytes: tl.constexpr,
    ptr_aligned: tl.constexpr,
    x_mul4: tl.constexpr,
    unroll_k: tl.constexpr = 1,
    block_n: tl.constexpr = 1,
    use_sad_u8: tl.constexpr = False,
    split_x_loads: tl.constexpr = False,
):
    """Fused in-register LE Trellis -> A16 Split-K GEMV for ``m1`` and bucketed ``batched`` MoE."""
    tl.static_assert((block_n == 1) | (block_n == 2))
    n_tiles = tl.cdiv(out_features, block_n)
    pid_0 = tl.program_id(0)
    pid_n_tile = pid_0 % n_tiles
    pid_tile = pid_0 // n_tiles
    pid_k = tl.program_id(1)
    n0 = pid_n_tile * block_n

    if batched:
        sub_tiles = _BATCH_BLOCK_M_TL // block_m
        bucket_idx = pid_tile // sub_tiles
        sub_idx = pid_tile - bucket_idx * sub_tiles
        slot = tl.load(tile_slots_ptr + bucket_idx).to(tl.int64)
        if slot < 0:
            return
        pair_offs = (
            bucket_idx.to(tl.int64) * _BATCH_BLOCK_M_TL + sub_idx * block_m + tl.arange(0, block_m)
        )
        pair_ids = tl.load(tile_pair_ids_ptr + pair_offs)
        m_mask = pair_ids >= 0
        if tl.sum(m_mask.to(tl.int32)) == 0:
            return
        safe_pairs = tl.where(m_mask, pair_ids, 0).to(tl.int64)
        ms = safe_pairs if per_expert_input else (safe_pairs // top_k)
        m0 = tl.full((), 0, tl.int64)
        width_slot_idx = bucket_idx.to(tl.int64)
    else:
        slot = _resolve_slot(
            topk_ids_ptr,
            expert_to_slot_ptr,
            pid_tile,
            num_experts,
            num_slots,
            has_slot_map,
        )
        if slot < 0:
            return
        m_mask = tl.arange(0, 1) >= 0
        safe_pairs = tl.full((1,), pid_tile, tl.int64)
        m0 = pid_tile.to(tl.int64) if per_expert_input else tl.full((), 0, tl.int64)
        ms = tl.full((1,), pid_tile if per_expert_input else 0, tl.int64)
        width_slot_idx = pid_tile.to(tl.int64)

    denom = tile_steps if exact_pow2 else block_size
    first_step = tl.arange(0, tile_steps // 4) * 4

    row_blk0, row_base0, width64_0, limit_bytes, sb0_0, sb0_1, sb0_2, sb0_3, fb0, ne0 = (
        _trellis_slot_row_step(
            row_widths_ptr, row_offsets_ptr, slot_offsets_ptr, slot, width_slot_idx, n0,
            out_features, blocks_per_row, bitstreams_bytes, denom, block_size, first_step,
            step_bits, num_wide_steps, bytes_per_block, slot_bytes, variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets, exact_pow2=exact_pow2,
        )
    )  # fmt: skip

    if block_n == 2:
        n1_in_bounds = (n0 + 1) < out_features
        n1 = tl.minimum(n0 + 1, out_features - 1)
        row_blk1, row_base1, width64_1, sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1 = (
            _trellis_slot_second_row(
                row_widths_ptr, row_offsets_ptr, slot_offsets_ptr, slot, width_slot_idx, n1,
                n1_in_bounds, out_features, blocks_per_row, bitstreams_bytes, denom, block_size,
                first_step, step_bits, num_wide_steps, bytes_per_block, slot_bytes, width64_0,
                sb0_0, sb0_1, sb0_2, sb0_3, fb0, variable_rows=variable_rows,
                has_slot_offsets=has_slot_offsets, exact_pow2=exact_pow2,
            )
        )  # fmt: skip
    else:
        n1_in_bounds = False
        n1 = n0
        row_blk1, row_base1, width64_1, sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1 = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    acc0, acc1 = _gemv_accumulate_k_blocks(
        x_rot_ptr, bitstreams_ptr, block_scales_ptr, row_blk0, row_base0, width64_0,
        sb0_0, sb0_1, sb0_2, sb0_3, fb0, m0, ms, m_mask, pid_k, blocks_per_row,
        limit_bytes, block_size, in_features, split_k, ne0, row_blk1, row_base1, width64_1,
        sb1_0, sb1_1, sb1_2, sb1_3, fb1, ne1, tile_steps=tile_steps, block_m=block_m,
        exact_pow2=exact_pow2, needs_w2=needs_w2, use_dp4a=use_dp4a,
        has_tail_bytes=has_tail_bytes, ptr_aligned=ptr_aligned, x_mul4=x_mul4,
        unroll_k=unroll_k, block_n=block_n, use_sad_u8=use_sad_u8,
        split_x_loads=split_x_loads,
    )  # fmt: skip
    _store_moe_trellis_row(
        acc0, row_scale_ptr, output_ptr, slot, safe_pairs, m_mask, pid_k, n0,
        n0 < out_features, num_pairs, out_features, is_splitk=is_splitk,
    )  # fmt: skip
    if block_n == 2:
        _store_moe_trellis_row(
            acc1, row_scale_ptr, output_ptr, slot, safe_pairs, m_mask, pid_k, n1,
            n1_in_bounds, num_pairs, out_features, is_splitk=is_splitk,
        )  # fmt: skip


def _int8_launch_config(target: AcceleratorTarget) -> tuple[int, int, dict[str, object]]:
    is_rdna4 = target.is_rdna4
    block_k = 256 if is_rdna4 else 64
    num_stages = 2 if (target.is_nvidia_cuda or is_rdna4) else 1
    compiler_options: dict[str, object] = (
        dict(amd_compiler_workarounds.lds_only_fence_options(is_rdna4)) if target.is_amd_hip else {}
    )
    return block_k, num_stages, compiler_options


def _select_nvfp4_split_k(
    device: torch.device,
    active_tiles: int,
    out_features: int,
    in_features: int,
    *,
    block_n: int = _NVFP4_GEMV_BLOCK_N,
    block_k: int = _NVFP4_GEMV_BLOCK_K,
    cta_multiplier: int = 8,
) -> int:
    return select_split_k(
        device_sm_count(device),
        output_tiles=active_tiles * triton.cdiv(out_features, block_n),
        k_tiles=triton.cdiv(in_features, block_k),
        weight_bytes=active_tiles * out_features * (in_features // 2),
        min_weight_bytes=_NVFP4_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=_NVFP4_MAX_SPLIT_K,
        cta_multiplier=cta_multiplier,
        min_k_tiles=8,
        min_tiles_per_split=4,
        round_up=True,
    )


def _nvfp4_m1_config(target: AcceleratorTarget) -> tuple[int, int, int, int]:
    """Return ``(block_n, block_k, num_warps, splitk_cta_multiplier)``."""
    if target.is_cuda_capability(12):
        return (
            _NVFP4_GEMV_BLOCK_N,
            _NVFP4_SM120_GEMV_BLOCK_K,
            _NVFP4_GEMV_NUM_WARPS,
            _NVFP4_SM120_M1_SPLITK_CTA_MULT,
        )
    return _NVFP4_GEMV_BLOCK_N, _NVFP4_GEMV_BLOCK_K, _NVFP4_GEMV_NUM_WARPS, 8


def _max_bucket_tiles(num_pairs: int, num_slots: int, block_m: int = _BATCH_BLOCK_M) -> int:
    max_unique = min(num_pairs, num_slots)
    return max_unique + (num_pairs - max_unique) // block_m


def _expected_bucket_tiles(
    num_pairs: int,
    num_slots: int,
    active_slots_hint: int,
    *,
    block_m: int = _BATCH_BLOCK_M,
    batched: bool,
) -> int:
    if not batched:
        return max(1, min(num_pairs, active_slots_hint) if active_slots_hint > 0 else num_pairs)
    eff_slots = min(num_slots, active_slots_hint) if active_slots_hint > 0 else num_slots
    return max(1, _max_bucket_tiles(num_pairs, eff_slots, block_m))


def _prefill_bytes_per_pair(
    *,
    is_int8: bool,
    hidden_size: int,
    intermediate_size: int,
    up_gate_out: int,
    logical_elem_bytes: int,
    group_size: int,
) -> int:
    if is_int8:
        return hidden_size + up_gate_out * logical_elem_bytes + intermediate_size + hidden_size * 4
    active_elem_bytes = 4 if group_size > 0 else logical_elem_bytes
    return (
        hidden_size * active_elem_bytes
        + up_gate_out * 4
        + intermediate_size * active_elem_bytes
        + hidden_size * 4
    )


def _select_prefill_chunk_rows(rows: int, top_k: int, bytes_per_pair: int) -> int:
    max_pairs_mem = max(1, _PREFILL_WORKSPACE_BYTES // max(1, bytes_per_pair))
    max_rows = min(rows, _PREFILL_MAX_PAIRS // top_k, max_pairs_mem // top_k)
    return max(1, max_rows)


def _select_moe_execution_regime(
    rows: int,
    top_k: int,
    bytes_per_pair: int,
) -> tuple[Literal["m1", "batched", "prefill"], int]:
    if rows == 1:
        return "m1", 1
    if rows <= _BATCH_MAX_ROWS and rows * top_k <= _BUCKET_MAX_PAIRS:
        return "batched", rows
    chunk_rows = _select_prefill_chunk_rows(rows, top_k, bytes_per_pair)
    if chunk_rows == 1 and top_k > _PREFILL_MAX_PAIRS:
        return "m1", 1
    return "prefill", chunk_rows


def _select_prefill_block_m(
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    *,
    sparse_m: int,
    dense_m: int,
) -> int:
    eff_slots = min(num_slots, active_slots_hint) if active_slots_hint > 0 else num_slots
    avg_m = triton.cdiv(chunk_rows * top_k, max(1, eff_slots))
    return sparse_m if avg_m <= sparse_m else dense_m


def _nvfp4_prefill_config(
    target: AcceleratorTarget,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> tuple[bool, bool, bool, int, int, int, int, bool]:
    """Return ``(cast_bf16, force_fp32_dot, half_scales, bm, bn, bk, warps, single_dot)``."""
    sm75_bf16 = (
        target.is_nvidia_cuda
        and not target.cuda_capability_at_least(8)
        and logical_dtype is torch.bfloat16
    )
    force_fp32_dot = (
        active_dtype is torch.float32 and logical_dtype is not torch.bfloat16
    ) or sm75_bf16
    cast_fp32_to_bf16 = (not force_fp32_dot) and (active_dtype is torch.float32)
    half_scales = not force_fp32_dot
    block_m = _select_prefill_block_m(
        chunk_rows,
        top_k,
        num_slots,
        active_slots_hint,
        sparse_m=_PREFILL_NVFP4_SPARSE_BLOCK_M,
        dense_m=_PREFILL_NVFP4_DENSE_BLOCK_M,
    )
    if target.is_cuda_capability(12) and not force_fp32_dot:
        return (
            cast_fp32_to_bf16,
            False,
            half_scales,
            block_m,
            _NVFP4_DOT_BLOCK_N,
            _NVFP4_SM120_PREFILL_BLOCK_K,
            _NVFP4_SM120_DOT_NUM_WARPS,
            True,
        )
    is_wave64 = target.is_amd_hip and target.is_architecture("gfx942")
    if block_m == _PREFILL_NVFP4_SPARSE_BLOCK_M:
        block_n = _PREFILL_NVFP4_SPARSE_BLOCK_N
        num_warps = (
            _NVFP4_DOT_NUM_WARPS_CUDA
            if (force_fp32_dot and not is_wave64)
            else _NVFP4_GEMV_NUM_WARPS
        )
    else:
        block_n = _PREFILL_NVFP4_SPARSE_BLOCK_N if force_fp32_dot else _PREFILL_NVFP4_DENSE_BLOCK_N
        num_warps = _NVFP4_GEMV_NUM_WARPS if is_wave64 else _NVFP4_DOT_NUM_WARPS_CUDA
    return (
        cast_fp32_to_bf16,
        force_fp32_dot,
        half_scales,
        block_m,
        block_n,
        _PREFILL_NVFP4_BLOCK_K,
        num_warps,
        False,
    )


def _int8_prefill_config(
    target: AcceleratorTarget,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
) -> tuple[int, int, int, int, int, dict[str, object]]:
    """Return ``(block_m, block_n, block_k, num_warps, num_stages, compiler_options)``."""
    base_block_k, num_stages, compiler_options = _int8_launch_config(target)
    block_m = _select_prefill_block_m(
        chunk_rows,
        top_k,
        num_slots,
        active_slots_hint,
        sparse_m=_PREFILL_INT8_SPARSE_BLOCK_M,
        dense_m=_PREFILL_INT8_DENSE_BLOCK_M,
    )
    if block_m == _PREFILL_INT8_SPARSE_BLOCK_M:
        return block_m, _PREFILL_INT8_SPARSE_BLOCK_N, base_block_k, 4, num_stages, compiler_options
    return (
        block_m,
        _PREFILL_INT8_DENSE_BLOCK_N,
        _PREFILL_INT8_DENSE_BLOCK_K,
        8,
        num_stages,
        compiler_options,
    )


def _launch_moe_prefill_bucket(
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
    block_m: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
]:
    """Group arbitrary prefill pairs into compacted slot order and ``block_m`` tile descriptors."""
    device = topk_ids_flat.device
    num_pairs = topk_ids_flat.shape[0]
    max_tiles = _max_bucket_tiles(num_pairs, num_slots, block_m)
    has_slot_map = expert_to_slot is not None
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots

    slot_counts = torch.zeros((num_slots,), device=device, dtype=torch.int32)
    pair_slots = torch.empty((num_pairs,), device=device, dtype=torch.int32)
    pair_pos = torch.empty((num_pairs,), device=device, dtype=torch.int32)
    tile_slots = torch.full((max_tiles,), -1, device=device, dtype=torch.int32)
    tile_m_start = torch.zeros((max_tiles,), device=device, dtype=torch.int32)
    tile_m_len = torch.zeros((max_tiles,), device=device, dtype=torch.int32)
    compact_to_pair = torch.full((num_pairs,), -1, device=device, dtype=torch.int32)
    pair_to_compact = torch.full((num_pairs,), -1, device=device, dtype=torch.int32)
    grid = (triton.cdiv(num_pairs, _PREFILL_BUCKET_BLOCK_P),)

    with device_context(device):
        moe_prefill_bucket_count_kernel[grid](
            topk_ids_flat,
            expert_to_slot,
            slot_counts,
            pair_slots,
            pair_pos,
            num_experts,
            num_slots,
            num_pairs,
            block_p=_PREFILL_BUCKET_BLOCK_P,
            has_slot_map=has_slot_map,
            num_warps=4,
        )
        slot_offsets = torch.cumsum(slot_counts, dim=0, dtype=torch.int32) - slot_counts
        tiles_per_slot = (slot_counts + (block_m - 1)) // block_m
        slot_tile_base = torch.cumsum(tiles_per_slot, dim=0, dtype=torch.int32) - tiles_per_slot
        moe_prefill_bucket_scatter_kernel[grid](
            pair_slots,
            pair_pos,
            slot_counts,
            slot_offsets,
            slot_tile_base,
            compact_to_pair,
            pair_to_compact,
            tile_slots,
            tile_m_start,
            tile_m_len,
            num_pairs,
            block_p=_PREFILL_BUCKET_BLOCK_P,
            block_m=block_m,
            num_warps=4,
        )
    return (
        tile_slots,
        tile_m_start,
        tile_m_len,
        compact_to_pair,
        pair_to_compact,
        max_tiles,
        slot_counts,
    )


def _launch_moe_prefill_gather(
    input_chunk: torch.Tensor,
    input_scale_chunk: torch.Tensor | None,
    compact_to_pair: torch.Tensor,
    out_compact: torch.Tensor,
    out_scale_compact: torch.Tensor | None,
    hidden_size: int,
    top_k: int,
    *,
    has_scale: bool,
) -> None:
    num_pairs = compact_to_pair.shape[0]
    with device_context(input_chunk.device):
        moe_prefill_gather_kernel[(triton.cdiv(hidden_size, _PREFILL_GATHER_BLOCK_K), num_pairs)](
            input_chunk,
            input_scale_chunk,
            compact_to_pair,
            out_compact,
            out_scale_compact,
            hidden_size,
            top_k,
            block_k=_PREFILL_GATHER_BLOCK_K,
            has_scale=has_scale,
            num_warps=4,
        )


def _launch_moe_prefill_weighted_reduce(
    down_out: torch.Tensor,
    pair_to_compact: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    out_chunk: torch.Tensor,
    hidden_size: int,
    top_k: int,
    *,
    accumulate: bool,
) -> None:
    chunk_rows = out_chunk.shape[0]
    with device_context(down_out.device):
        moe_m1_weighted_reduce_kernel[
            (triton.cdiv(hidden_size, _WEIGHTED_REDUCE_BLOCK_N), chunk_rows)
        ](
            down_out,
            None,
            None,
            None,
            pair_to_compact,
            topk_weights_flat,
            None,
            out_chunk,
            hidden_size,
            out_chunk.stride(0),
            1,
            1,
            pair_to_compact.shape[0],
            top_k,
            1,
            block_n=_WEIGHTED_REDUCE_BLOCK_N,
            is_int8=False,
            has_slot_map=False,
            accumulate=accumulate,
            has_global_scale=False,
            post_scale_only=True,
            use_compact_map=True,
            num_warps=4,
        )


def _nvfp4_batched_config(
    target: AcceleratorTarget,
    *,
    rows: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> tuple[bool, bool, bool, int, int, int, bool]:
    """Return ``(use_dot, cast_bf16, post_scale_only, block_n, block_k, num_warps, single_dot)``."""
    is_sm75 = target.is_nvidia_cuda and not target.cuda_capability_at_least(8)
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
        block_k = _NVFP4_DOT_BLOCK_K_RDNA4 if target.is_rdna4 else _NVFP4_DOT_BLOCK_K
        num_warps = (
            _NVFP4_SM120_DOT_NUM_WARPS
            if is_sm120
            else (_NVFP4_DOT_NUM_WARPS_CUDA if target.is_nvidia_cuda else _NVFP4_GEMV_NUM_WARPS)
        )
        return (
            True,
            cast_fp32_to_bf16,
            post_scale_only,
            _NVFP4_DOT_BLOCK_N,
            block_k,
            num_warps,
            is_sm120,
        )
    block_k = (
        _NVFP4_SM120_GEMV_BLOCK_K
        if is_sm120
        else (_NVFP4_SM75_SCALAR_BLOCK_K if is_sm75 else _NVFP4_GEMV_BLOCK_K)
    )
    return (
        False,
        False,
        post_scale_only,
        _NVFP4_GEMV_BLOCK_N,
        block_k,
        _NVFP4_GEMV_NUM_WARPS,
        False,
    )


def _launch_moe_bucket(
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Group ``topk_ids_flat`` hits into fixed ``_BATCH_BLOCK_M`` slot tiles on GPU."""
    device = topk_ids_flat.device
    num_pairs = topk_ids_flat.shape[0]
    max_tiles = _max_bucket_tiles(num_pairs, num_slots, _BATCH_BLOCK_M)
    block_p = triton.next_power_of_2(max(16, num_pairs))
    has_slot_map = expert_to_slot is not None
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots
    tile_slots = torch.empty((max_tiles,), device=device, dtype=torch.int32)
    tile_pair_ids = torch.full((max_tiles, _BATCH_BLOCK_M), -1, device=device, dtype=torch.int32)
    with device_context(device):
        moe_bucket_tokens_kernel[(1,)](
            topk_ids_flat,
            expert_to_slot,
            tile_slots,
            tile_pair_ids,
            num_experts,
            num_slots,
            num_pairs,
            max_tiles,
            block_p=block_p,
            block_m=_BATCH_BLOCK_M,
            has_slot_map=has_slot_map,
            num_warps=4,
        )
    return tile_slots, tile_pair_ids, max_tiles


def _select_int8_split_k(
    device: torch.device,
    active_slots: int,
    out_features: int,
    in_features: int,
    block_k: int,
) -> int:
    if in_features > _INT8_MAX_REDUCTION_WIDTH:
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


def _launch_nvfp4_proj(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None,
    routing: _A16ChunkRouting,
    output: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    scale_cols: int,
    scale_stride: int,
    num_slots: int,
    split_k: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    high_first: bool,
    has_global_scale: bool,
    per_expert_input: bool,
    is_splitk: bool,
    use_dot: bool,
    cast_fp32_to_bf16: bool,
    use_sm120_e2m1x2: bool = False,
    single_dot: bool = False,
) -> None:
    scale_u8 = scale.view(torch.uint8)
    packed_k = in_features // 2
    n_tiles = triton.cdiv(out_features, block_n)
    with device_context(x.device):
        if routing.tile_slots is not None:
            assert routing.tile_pair_ids is not None
            nvfp4_moe_batched_gemv_kernel[(n_tiles, routing.tile_slots.shape[0], split_k)](
                x,
                qdata,
                scale_u8,
                global_scale,
                routing.tile_slots,
                routing.tile_pair_ids,
                output,
                out_features,
                packed_k,
                scale_cols,
                scale_stride,
                routing.num_pairs,
                routing.top_k,
                split_k,
                block_m=_BATCH_BLOCK_M,
                block_n=block_n,
                block_k=block_k,
                high_first=high_first,
                has_global_scale=has_global_scale,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                use_dot=use_dot,
                cast_fp32_to_bf16=cast_fp32_to_bf16,
                use_sm120_e2m1x2=use_sm120_e2m1x2,
                single_dot=single_dot,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            nvfp4_moe_m1_gemv_kernel[(n_tiles * routing.top_k, split_k)](
                x,
                qdata,
                scale_u8,
                global_scale,
                routing.topk_ids_flat,
                routing.expert_to_slot,
                output,
                out_features,
                packed_k,
                scale_cols,
                scale_stride,
                routing.num_experts,
                num_slots,
                routing.top_k,
                split_k,
                block_n=block_n,
                block_k=block_k,
                high_first=high_first,
                has_slot_map=routing.has_slot_map,
                has_global_scale=has_global_scale,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                use_sm120_e2m1x2=use_sm120_e2m1x2,
                num_warps=num_warps,
            )


@dataclass(frozen=True, slots=True)
class _A16ChunkRouting:
    topk_ids_flat: torch.Tensor
    expert_to_slot: torch.Tensor | None
    tile_slots: torch.Tensor | None
    tile_pair_ids: torch.Tensor | None
    num_experts: int
    num_pairs: int
    top_k: int
    has_slot_map: bool


def _run_a16_chunk(
    active_x_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    out_chunk: torch.Tensor,
    *,
    num_slots: int,
    hidden_size: int,
    intermediate_size: int,
    up_gate_out: int,
    expert_to_slot: torch.Tensor | None,
    target: AcceleratorTarget,
    batched: bool,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    split_k_up: int,
    split_k_down: int,
    up_row_scale: torch.Tensor | None,
    down_row_scale: torch.Tensor | None,
    up_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    post_scale_only: bool,
    launch_proj: Callable[[bool, torch.Tensor, _A16ChunkRouting, torch.Tensor, int], None],
) -> None:
    device = active_x_chunk.device
    chunk_rows = active_x_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    topk_ids_flat = topk_ids_chunk.reshape(-1)
    topk_weights_flat = topk_weights_chunk.reshape(-1)
    has_slot_map = expert_to_slot is not None
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots
    tile_slots, tile_pair_ids = (
        _launch_moe_bucket(topk_ids_flat, expert_to_slot, num_slots)[:2]
        if batched
        else (None, None)
    )
    routing = _A16ChunkRouting(
        topk_ids_flat=topk_ids_flat,
        expert_to_slot=expert_to_slot,
        tile_slots=tile_slots,
        tile_pair_ids=tile_pair_ids,
        num_experts=num_experts,
        num_pairs=num_pairs,
        top_k=top_k,
        has_slot_map=has_slot_map,
    )
    has_row_scale = up_row_scale is not None

    mid = torch.empty((num_pairs, up_gate_out), device=device, dtype=torch.float32)
    up_target = (
        torch.empty((split_k_up, num_pairs, up_gate_out), device=device, dtype=torch.float32)
        if split_k_up > 1
        else mid
    )
    act_dtype = torch.float32 if group_size > 0 else out_chunk.dtype
    act = torch.empty((num_pairs, intermediate_size), device=device, dtype=act_dtype)
    down_partial = torch.empty(
        (split_k_down, num_pairs, hidden_size), device=device, dtype=torch.float32
    )

    with device_context(device):
        launch_proj(False, active_x_chunk, routing, up_target, split_k_up)
        if split_k_up > 1:
            moe_splitk_reduce_kernel[
                (triton.cdiv(up_gate_out, _NVFP4_REDUCE_BLOCK_N) * num_pairs,)
            ](
                up_target,
                None,
                up_row_scale,
                up_global_scale,
                topk_ids_flat,
                expert_to_slot,
                mid,
                up_gate_out,
                num_experts,
                num_slots,
                num_pairs,
                top_k,
                split_k_up,
                block_n=_NVFP4_REDUCE_BLOCK_N,
                is_int8=False,
                has_slot_map=has_slot_map,
                has_global_scale=up_global_scale is not None,
                post_scale_only=post_scale_only,
                has_row_scale=has_row_scale,
                swap_halves=False,
                num_warps=4,
            )
        moe_m1_activate_rotate_kernel[
            (triton.cdiv(intermediate_size, _ACTIVATE_BLOCK_SIZE) * num_pairs,)
        ](
            mid,
            act,
            out_chunk,
            topk_ids_flat,
            expert_to_slot,
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
        launch_proj(True, act, routing, down_partial, split_k_down)
        moe_m1_weighted_reduce_kernel[
            (triton.cdiv(hidden_size, _WEIGHTED_REDUCE_BLOCK_N), chunk_rows)
        ](
            down_partial,
            None,
            down_row_scale,
            down_global_scale,
            topk_ids_flat,
            topk_weights_flat,
            expert_to_slot,
            out_chunk,
            hidden_size,
            out_chunk.stride(0),
            num_experts,
            num_slots,
            num_pairs,
            top_k,
            split_k_down,
            block_n=_WEIGHTED_REDUCE_BLOCK_N,
            is_int8=False,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            has_global_scale=down_global_scale is not None,
            post_scale_only=post_scale_only,
            has_row_scale=has_row_scale,
            num_warps=4,
        )


def _launch_int8_proj(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    in_scale: torch.Tensor,
    w_scale: torch.Tensor,
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    tile_slots: torch.Tensor | None,
    tile_pair_ids: torch.Tensor | None,
    output: torch.Tensor,
    *,
    out_tiles: int,
    out_features: int,
    in_features: int,
    num_experts: int,
    num_slots: int,
    num_pairs: int,
    top_k: int,
    split_k: int,
    block_m: int,
    block_k: int,
    num_stages: int,
    compiler_options: dict[str, object],
    has_slot_map: bool,
    per_expert_input: bool,
    is_splitk: bool,
    swap_halves: bool,
    aligned_nk: bool,
    accumulator_seed: int,
) -> None:
    with device_context(x_q.device):
        if tile_slots is not None:
            assert tile_pair_ids is not None
            int8_moe_batched_matmul_kernel[(out_tiles, tile_slots.shape[0], split_k)](
                x_q,
                w_q,
                tile_slots,
                tile_pair_ids,
                in_scale,
                w_scale,
                output,
                out_features,
                in_features,
                num_pairs,
                top_k,
                split_k,
                block_m=block_m,
                block_n=_INT8_GEMM_BLOCK_N,
                block_k=block_k,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                dequant_output=not is_splitk,
                swap_halves=swap_halves,
                aligned_nk=aligned_nk,
                accumulator_seed=accumulator_seed,
                **compiler_options,
                num_stages=num_stages,
                num_warps=4,
            )
        else:
            int8_moe_m1_matmul_kernel[(out_tiles * top_k, split_k)](
                x_q,
                w_q,
                topk_ids_flat,
                expert_to_slot,
                in_scale,
                w_scale,
                output,
                out_features,
                in_features,
                num_experts,
                num_slots,
                top_k,
                split_k,
                block_m=block_m,
                block_n=_INT8_GEMM_BLOCK_N,
                block_k=block_k,
                has_slot_map=has_slot_map,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                dequant_output=not is_splitk,
                swap_halves=swap_halves,
                aligned_nk=aligned_nk,
                accumulator_seed=accumulator_seed,
                **compiler_options,
                num_stages=num_stages,
                num_warps=4,
            )


def _run_int8_chunk(
    input_q_chunk: torch.Tensor,
    input_s_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    out_chunk: torch.Tensor,
    *,
    target: AcceleratorTarget,
    int8_backend: LinearBackend,
    down_prep_plan: LinearExecutionPlan,
    batched: bool,
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
    device = input_q_chunk.device
    chunk_rows = input_q_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    num_slots, hidden_size, intermediate_size = down_qdata.shape
    up_gate_out = up_gate_qdata.shape[1]
    topk_ids_flat = topk_ids_chunk.reshape(-1)
    topk_weights_flat = topk_weights_chunk.reshape(-1)
    has_slot_map = expert_to_slot is not None
    num_experts = expert_to_slot.shape[0] if expert_to_slot is not None else num_slots

    if batched:
        tile_slots, tile_pair_ids, _ = _launch_moe_bucket(topk_ids_flat, expert_to_slot, num_slots)
        block_m = _BATCH_BLOCK_M
    else:
        tile_slots = tile_pair_ids = None
        block_m = _INT8_GEMM_BLOCK_M

    # Batched INT8 skips missed rows when split_k_up == 1; zero mid before preparing down.
    mid = (
        torch.zeros((num_pairs, up_gate_out), device=device, dtype=out_chunk.dtype)
        if batched and split_k_up == 1
        else torch.empty((num_pairs, up_gate_out), device=device, dtype=out_chunk.dtype)
    )
    up_target = (
        torch.empty((split_k_up, num_pairs, up_gate_out), device=device, dtype=torch.int32)
        if split_k_up > 1
        else mid
    )

    with device_context(device):
        _launch_int8_proj(
            input_q_chunk,
            up_gate_qdata,
            input_s_chunk,
            up_gate_scale,
            topk_ids_flat,
            expert_to_slot,
            tile_slots,
            tile_pair_ids,
            up_target,
            out_tiles=up_tiles,
            out_features=up_gate_out,
            in_features=hidden_size,
            num_experts=num_experts,
            num_slots=num_slots,
            num_pairs=num_pairs,
            top_k=top_k,
            split_k=split_k_up,
            block_m=block_m,
            block_k=block_k,
            num_stages=num_stages,
            compiler_options=compiler_options,
            has_slot_map=has_slot_map,
            per_expert_input=False,
            is_splitk=split_k_up > 1,
            swap_halves=(swap_halves and split_k_up == 1),
            aligned_nk=up_aligned_nk,
            accumulator_seed=up_seed,
        )
        if split_k_up > 1:
            moe_splitk_reduce_kernel[(triton.cdiv(up_gate_out, _INT8_REDUCE_BLOCK_N) * num_pairs,)](
                up_target,
                input_s_chunk,
                up_gate_scale,
                None,
                topk_ids_flat,
                expert_to_slot,
                mid,
                up_gate_out,
                num_experts,
                num_slots,
                num_pairs,
                top_k,
                split_k_up,
                block_n=_INT8_REDUCE_BLOCK_N,
                is_int8=True,
                has_slot_map=has_slot_map,
                has_global_scale=False,
                post_scale_only=True,
                has_row_scale=False,
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
    down_partial = torch.empty(
        (split_k_down, num_pairs, hidden_size), device=device, dtype=torch.int32
    )

    with device_context(device):
        _launch_int8_proj(
            act_q,
            down_qdata,
            act_s,
            down_scale,
            topk_ids_flat,
            expert_to_slot,
            tile_slots,
            tile_pair_ids,
            down_partial,
            out_tiles=down_tiles,
            out_features=hidden_size,
            in_features=intermediate_size,
            num_experts=num_experts,
            num_slots=num_slots,
            num_pairs=num_pairs,
            top_k=top_k,
            split_k=split_k_down,
            block_m=block_m,
            block_k=block_k,
            num_stages=num_stages,
            compiler_options=compiler_options,
            has_slot_map=has_slot_map,
            per_expert_input=True,
            is_splitk=True,
            swap_halves=False,
            aligned_nk=down_aligned_nk,
            accumulator_seed=down_seed,
        )
        moe_m1_weighted_reduce_kernel[
            (triton.cdiv(hidden_size, _WEIGHTED_REDUCE_BLOCK_N), chunk_rows)
        ](
            down_partial,
            act_s,
            down_scale,
            None,
            topk_ids_flat,
            topk_weights_flat,
            expert_to_slot,
            out_chunk,
            hidden_size,
            out_chunk.stride(0),
            num_experts,
            num_slots,
            num_pairs,
            top_k,
            split_k_down,
            block_n=_WEIGHTED_REDUCE_BLOCK_N,
            is_int8=True,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            has_global_scale=False,
            post_scale_only=True,
            num_warps=4,
        )


def _launch_nvfp4_grouped_gemm(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None,
    tile_slots: torch.Tensor,
    tile_m_start: torch.Tensor,
    tile_m_len: torch.Tensor,
    output: torch.Tensor,
    *,
    max_tiles: int,
    out_features: int,
    in_features: int,
    scale_cols: int,
    scale_stride: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    high_first: bool,
    has_global_scale: bool,
    cast_fp32_to_bf16: bool,
    force_fp32_dot: bool,
    half_scales: bool,
    use_sm120_e2m1x2: bool = False,
    single_dot: bool = False,
) -> None:
    grid = (max_tiles * triton.cdiv(out_features, block_n),)
    with device_context(x.device):
        nvfp4_moe_grouped_gemm_kernel[grid](
            x,
            qdata,
            scale.view(torch.uint8),
            global_scale,
            tile_slots,
            tile_m_start,
            tile_m_len,
            output,
            max_tiles,
            out_features,
            in_features // 2,
            scale_cols,
            scale_stride,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=_PREFILL_GROUP_M,
            high_first=high_first,
            has_global_scale=has_global_scale,
            cast_fp32_to_bf16=cast_fp32_to_bf16,
            force_fp32_dot=force_fp32_dot,
            half_scales=half_scales,
            use_sm120_e2m1x2=use_sm120_e2m1x2,
            single_dot=single_dot,
            num_warps=num_warps,
            num_stages=1,
        )


def _run_nvfp4_prefill_chunk(
    active_x_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    out_chunk: torch.Tensor,
    *,
    target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
    up_scale_cols: int,
    up_scale_stride: int,
    down_scale_cols: int,
    down_scale_stride: int,
) -> None:
    device = active_x_chunk.device
    chunk_rows = active_x_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    num_slots, hidden_size, packed_inter = down_qdata.shape
    intermediate_size = 2 * packed_inter
    up_gate_out = up_gate_qdata.shape[1]
    cast_bf16, force_fp32_dot, half_scales, block_m, block_n, block_k, num_warps, single_dot = (
        _nvfp4_prefill_config(
            target,
            chunk_rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            active_dtype=active_x_chunk.dtype,
            logical_dtype=out_chunk.dtype,
        )
    )
    tile_slots, tile_m_start, tile_m_len, compact_to_pair, pair_to_compact, max_tiles, _ = (
        _launch_moe_prefill_bucket(topk_ids_chunk.reshape(-1), expert_to_slot, num_slots, block_m)
    )
    x_compact = torch.empty((num_pairs, hidden_size), device=device, dtype=active_x_chunk.dtype)
    mid = torch.empty((num_pairs, up_gate_out), device=device, dtype=torch.float32)

    _launch_moe_prefill_gather(
        active_x_chunk,
        None,
        compact_to_pair,
        x_compact,
        None,
        hidden_size,
        top_k,
        has_scale=False,
    )
    use_sm120_e2m1x2 = target.is_cuda_capability(12)
    _launch_nvfp4_grouped_gemm(
        x_compact,
        up_gate_qdata,
        up_gate_scale,
        up_gate_global_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        mid,
        max_tiles=max_tiles,
        out_features=up_gate_out,
        in_features=hidden_size,
        scale_cols=up_scale_cols,
        scale_stride=up_scale_stride,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        high_first=high_first,
        has_global_scale=up_gate_global_scale is not None,
        cast_fp32_to_bf16=cast_bf16,
        force_fp32_dot=force_fp32_dot,
        half_scales=half_scales,
        use_sm120_e2m1x2=use_sm120_e2m1x2,
        single_dot=single_dot,
    )
    del x_compact
    act_dtype = torch.float32 if group_size > 0 else out_chunk.dtype
    act = torch.empty((num_pairs, intermediate_size), device=device, dtype=act_dtype)
    with device_context(device):
        moe_m1_activate_rotate_kernel[
            (triton.cdiv(intermediate_size, _ACTIVATE_BLOCK_SIZE) * num_pairs,)
        ](
            mid,
            act,
            out_chunk,
            compact_to_pair,
            None,
            intermediate_size,
            num_pairs,
            num_pairs,
            block_size=_ACTIVATE_BLOCK_SIZE,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            accelerator_backend=target.backend,
            has_slot_map=False,
            num_warps=4,
        )
    del mid
    down_out = torch.empty((num_pairs, hidden_size), device=device, dtype=torch.float32)
    _launch_nvfp4_grouped_gemm(
        act,
        down_qdata,
        down_scale,
        down_global_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        down_out,
        max_tiles=max_tiles,
        out_features=hidden_size,
        in_features=intermediate_size,
        scale_cols=down_scale_cols,
        scale_stride=down_scale_stride,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        high_first=high_first,
        has_global_scale=down_global_scale is not None,
        cast_fp32_to_bf16=cast_bf16,
        force_fp32_dot=force_fp32_dot,
        half_scales=half_scales,
        use_sm120_e2m1x2=use_sm120_e2m1x2,
        single_dot=single_dot,
    )
    del act
    _launch_moe_prefill_weighted_reduce(
        down_out,
        pair_to_compact,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        hidden_size,
        top_k,
        accumulate=accumulate,
    )


def _launch_int8_grouped_gemm(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    tile_slots: torch.Tensor,
    tile_m_start: torch.Tensor,
    tile_m_len: torch.Tensor,
    in_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    *,
    max_tiles: int,
    out_features: int,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    compiler_options: dict[str, object],
    swap_halves: bool,
    aligned_nk: bool,
    accumulator_seed: int,
    slot_start: int = 0,
    slot_end: int = 0,
    n_start: int = 0,
    n_rows: int = 0,
    slot_window: bool = False,
) -> None:
    active_n_rows = n_rows if slot_window else out_features
    grid = (max_tiles * triton.cdiv(active_n_rows, block_n),)
    with device_context(x_q.device):
        int8_moe_grouped_gemm_kernel[grid](
            x_q,
            w_q,
            tile_slots,
            tile_m_start,
            tile_m_len,
            in_scale,
            w_scale,
            output,
            max_tiles,
            out_features,
            in_features,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=_PREFILL_GROUP_M,
            swap_halves=swap_halves,
            aligned_nk=aligned_nk,
            accumulator_seed=accumulator_seed,
            slot_start=slot_start,
            slot_end=slot_end,
            n_start=n_start,
            n_rows=n_rows,
            slot_window=slot_window,
            **compiler_options,
            num_stages=num_stages,
            num_warps=num_warps,
        )


def _run_prefill_int8_proj(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    tile_slots: torch.Tensor,
    tile_m_start: torch.Tensor,
    tile_m_len: torch.Tensor,
    output: torch.Tensor,
    *,
    block_scales: torch.Tensor | None,
    row_widths: torch.Tensor | None,
    slot_offsets: torch.Tensor | None,
    block_size: int,
    slot_counts: torch.Tensor,
    num_slots: int,
    max_tiles: int,
    out_features: int,
    in_features: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    compiler_options: dict[str, object],
    swap_halves: bool,
    accumulator_seed: int,
) -> None:
    if block_scales is None:
        _launch_int8_grouped_gemm(
            x_q,
            qdata,
            tile_slots,
            tile_m_start,
            tile_m_len,
            x_scale,
            scale,
            output,
            max_tiles=max_tiles,
            out_features=out_features,
            in_features=in_features,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            compiler_options=compiler_options,
            swap_halves=swap_halves,
            aligned_nk=(out_features % block_n == 0 and in_features % block_k == 0),
            accumulator_seed=accumulator_seed,
        )
        return

    slot_decoded_bytes = out_features * in_features
    if slot_decoded_bytes <= _TRELLIS_SCRATCH_BYTES:
        max_chunk_slots = min(num_slots, _TRELLIS_SCRATCH_BYTES // max(1, slot_decoded_bytes))
        n_chunk = out_features
    else:
        max_chunk_slots = 1
        raw_chunk_rows = max(1, _TRELLIS_SCRATCH_BYTES // max(1, in_features))
        n_chunk = (
            (raw_chunk_rows // block_n) * block_n if raw_chunk_rows >= block_n else raw_chunk_rows
        )

    scratch = torch.empty(
        (max_chunk_slots, n_chunk, in_features), device=x_q.device, dtype=torch.int8
    )
    block_scales_u8 = block_scales.view(torch.uint8)
    for slot_start in range(0, num_slots, max_chunk_slots):
        slot_end = min(slot_start + max_chunk_slots, num_slots)
        window_slots = slot_end - slot_start
        window_counts = slot_counts[slot_start:slot_end]
        window_slot_offsets = None if slot_offsets is None else slot_offsets[slot_start:slot_end]
        window_widths = None if row_widths is None else row_widths[slot_start:slot_end]
        window_offsets = (
            None
            if window_widths is None
            else compute_row_offsets_2d(window_widths, dtype=torch.int32)
        )
        for n_start in range(0, out_features, n_chunk):
            n_end = min(n_start + n_chunk, out_features)
            n_rows = n_end - n_start
            q_view = scratch[:window_slots, :n_rows]
            if row_widths is None:
                chunk_streams = (
                    qdata[slot_start:slot_end, n_start:n_end].contiguous()
                    if n_rows < out_features
                    else qdata[slot_start:slot_end]
                )
            elif slot_offsets is None:
                chunk_streams = qdata[slot_start:slot_end]
            else:
                chunk_streams = qdata
            decode_slots_int8_out(
                chunk_streams,
                block_scales_u8[slot_start:slot_end, n_start:n_end],
                q_view,
                block_size=block_size,
                row_widths=None if window_widths is None else window_widths[:, n_start:n_end],
                slot_offsets=window_slot_offsets,
                active_counts=window_counts,
                row_offsets=None if window_offsets is None else window_offsets[:, n_start:n_end],
            )
            _launch_int8_grouped_gemm(
                x_q,
                q_view,
                tile_slots,
                tile_m_start,
                tile_m_len,
                x_scale,
                scale,
                output,
                max_tiles=max_tiles,
                out_features=out_features,
                in_features=in_features,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                compiler_options=compiler_options,
                swap_halves=swap_halves,
                aligned_nk=(n_rows % block_n == 0 and in_features % block_k == 0),
                accumulator_seed=accumulator_seed,
                slot_start=slot_start,
                slot_end=slot_end,
                n_start=n_start,
                n_rows=n_rows,
                slot_window=True,
            )


def _run_int8_prefill_chunk(
    input_q_chunk: torch.Tensor,
    input_s_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    out_chunk: torch.Tensor,
    *,
    target: AcceleratorTarget,
    int8_backend: LinearBackend,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    swap_halves: bool,
    active_slots_hint: int,
    up_gate_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_scales: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    down_block_size: int = 0,
) -> None:
    device = input_q_chunk.device
    chunk_rows = input_q_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    if down_block_scales is not None:
        assert up_gate_block_scales is not None
        num_slots, hidden_size, down_blocks = down_block_scales.shape
        intermediate_size = down_blocks * down_block_size
        up_gate_out = up_gate_block_scales.shape[1]
    else:
        num_slots, hidden_size, intermediate_size = down_qdata.shape
        up_gate_out = up_gate_qdata.shape[1]
    block_m, block_n, block_k, num_warps, num_stages, compiler_options = _int8_prefill_config(
        target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
    )
    (
        tile_slots,
        tile_m_start,
        tile_m_len,
        compact_to_pair,
        pair_to_compact,
        max_tiles,
        slot_counts,
    ) = _launch_moe_prefill_bucket(topk_ids_chunk.reshape(-1), expert_to_slot, num_slots, block_m)
    expected_tiles = _expected_bucket_tiles(
        num_pairs, num_slots, active_slots_hint, block_m=block_m, batched=True
    )
    up_seed = rdna4_int8_accumulator_seed(
        target, hidden_size, expected_tiles * triton.cdiv(up_gate_out, block_n)
    )
    down_seed = rdna4_int8_accumulator_seed(
        target, intermediate_size, expected_tiles * triton.cdiv(hidden_size, block_n)
    )
    x_compact = torch.empty((num_pairs, hidden_size), device=device, dtype=torch.int8)
    x_scale_compact = torch.empty((num_pairs,), device=device, dtype=torch.float32)
    _launch_moe_prefill_gather(
        input_q_chunk,
        input_s_chunk,
        compact_to_pair,
        x_compact,
        x_scale_compact,
        hidden_size,
        top_k,
        has_scale=True,
    )
    # Zero mid so inactive compacted tail rows produce finite zeroed activations.
    mid = torch.zeros((num_pairs, up_gate_out), device=device, dtype=out_chunk.dtype)
    _run_prefill_int8_proj(
        x_compact,
        x_scale_compact,
        up_gate_qdata,
        up_gate_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        mid,
        block_scales=up_gate_block_scales,
        row_widths=up_gate_row_widths,
        slot_offsets=up_gate_slot_offsets,
        block_size=up_gate_block_size,
        slot_counts=slot_counts,
        num_slots=num_slots,
        max_tiles=max_tiles,
        out_features=up_gate_out,
        in_features=hidden_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        compiler_options=compiler_options,
        swap_halves=swap_halves,
        accumulator_seed=up_seed,
    )
    del x_compact, x_scale_compact
    down_prep_plan = int8_backend.policy.select_execution_plan(
        target, in_features=intermediate_size, rows=num_pairs, out_features=None
    )
    act_q, act_s = int8_backend.prepare_input_with_plan(
        mid,
        intermediate_size,
        group_size,
        activation_fn=activation_fn,
        execution_plan=down_prep_plan,
        target=target,
    )
    del mid
    down_out = torch.empty((num_pairs, hidden_size), device=device, dtype=torch.float32)
    _run_prefill_int8_proj(
        act_q,
        act_s,
        down_qdata,
        down_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        down_out,
        block_scales=down_block_scales,
        row_widths=down_row_widths,
        slot_offsets=down_slot_offsets,
        block_size=down_block_size,
        slot_counts=slot_counts,
        num_slots=num_slots,
        max_tiles=max_tiles,
        out_features=hidden_size,
        in_features=intermediate_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        compiler_options=compiler_options,
        swap_halves=False,
        accumulator_seed=down_seed,
    )
    del act_q, act_s
    _launch_moe_prefill_weighted_reduce(
        down_out,
        pair_to_compact,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        hidden_size,
        top_k,
        accumulate=accumulate,
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
        rows, hidden_size = input_2d.shape
        prep_2d = prepared_data.reshape(rows, hidden_size).contiguous()
        if prepared_scale is not None:
            return prep_2d, prepared_scale.reshape(rows).contiguous()
        inp_q, inp_s = dynamic_quantize_rows(prep_2d)
        return inp_q, inp_s.squeeze(-1).contiguous()
    assert up_prep_plan is not None
    return int8_backend.prepare_input_with_plan(
        input_2d.contiguous(),
        input_2d.shape[1],
        group_size,
        activation_fn=None,
        execution_plan=up_prep_plan,
        target=target,
    )


def _dispatch_int8_prefill_chunks(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    resolved_target: AcceleratorTarget,
    int8_backend: LinearBackend,
    chunk_rows: int,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    active_slots_hint: int,
    up_gate_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_scales: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    down_block_size: int = 0,
) -> None:
    rows, hidden_size = input_2d.shape
    swap_halves = activation_fn == "swiglu" and gate_first
    prep_q_full = None if prepared_data is None else prepared_data.reshape(rows, hidden_size)
    prep_s_full = None if prepared_scale is None else prepared_scale.reshape(rows)
    up_prep_plan = (
        None
        if prepared_data is not None
        else int8_backend.policy.select_execution_plan(resolved_target, in_features=hidden_size)
    )
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        inp_q_chunk, inp_s_chunk = _prepare_int8_input_gpu(
            input_2d[start:end],
            None if prep_q_full is None else prep_q_full[start:end],
            None if prep_s_full is None else prep_s_full[start:end],
            group_size,
            target=resolved_target,
            int8_backend=int8_backend,
            up_prep_plan=up_prep_plan,
        )
        _run_int8_prefill_chunk(
            inp_q_chunk,
            inp_s_chunk,
            ids_2d[start:end],
            weights_2d[start:end],
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            out_2d[start:end],
            target=resolved_target,
            int8_backend=int8_backend,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            swap_halves=swap_halves,
            active_slots_hint=active_slots_hint,
            up_gate_block_scales=up_gate_block_scales,
            up_gate_row_widths=up_gate_row_widths,
            up_gate_slot_offsets=up_gate_slot_offsets,
            up_gate_block_size=up_gate_block_size,
            down_block_scales=down_block_scales,
            down_row_widths=down_row_widths,
            down_slot_offsets=down_slot_offsets,
            down_block_size=down_block_size,
        )


def _iter_a16_chunks(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    out_2d: torch.Tensor,
    prepared_data: torch.Tensor | None,
    group_size: int,
    chunk_rows: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    rows, hidden_size = input_2d.shape
    prep_full = None if prepared_data is None else prepared_data.reshape(rows, hidden_size)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        active_chunk = _prepare_nvfp4_input_gpu(
            input_2d[start:end],
            None if prep_full is None else prep_full[start:end],
            group_size,
        )
        yield active_chunk, ids_2d[start:end], weights_2d[start:end], out_2d[start:end]


def _dispatch_int8_moe(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    prepared_scale: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    resolved_target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    active_slots_hint: int,
) -> None:
    from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

    device = input_2d.device
    rows, hidden_size = input_2d.shape
    num_slots, _, intermediate_size = down_qdata.shape
    up_gate_out = up_gate_qdata.shape[1]
    top_k = ids_2d.shape[1]
    int8_backend = int8_backend_mod.select_linear_backend_for_target(resolved_target)
    assert int8_backend is not None
    bytes_per_pair = _prefill_bytes_per_pair(
        is_int8=True,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=out_2d.element_size(),
        group_size=group_size,
    )
    regime, chunk_rows = _select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill":
        _dispatch_int8_prefill_chunks(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            out_2d,
            resolved_target=resolved_target,
            int8_backend=int8_backend,
            chunk_rows=chunk_rows,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            active_slots_hint=active_slots_hint,
        )
        return

    swap_halves = activation_fn == "swiglu" and gate_first
    up_prep_plan = (
        None
        if prepared_data is not None
        else int8_backend.policy.select_execution_plan(resolved_target, in_features=hidden_size)
    )
    batched = regime == "batched"
    expected_tiles = _expected_bucket_tiles(
        chunk_rows * top_k, num_slots, active_slots_hint, batched=batched
    )
    down_prep_plan = int8_backend.policy.select_execution_plan(
        resolved_target,
        in_features=intermediate_size,
        rows=chunk_rows * top_k,
        out_features=None,
    )
    block_k, num_stages, compiler_options = _int8_launch_config(resolved_target)
    split_k_up = _select_int8_split_k(device, expected_tiles, up_gate_out, hidden_size, block_k)
    split_k_down = _select_int8_split_k(
        device, expected_tiles, hidden_size, intermediate_size, block_k
    )
    up_seed = splitk_accumulator_seed(
        resolved_target,
        rows=expected_tiles * _INT8_GEMM_BLOCK_M,
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
        rows=expected_tiles * _INT8_GEMM_BLOCK_M,
        out_features=hidden_size,
        in_features=intermediate_size,
        block_m=_INT8_GEMM_BLOCK_M,
        block_n=_INT8_GEMM_BLOCK_N,
        block_k=block_k,
        paired=False,
        split_k=split_k_down,
    )
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
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        _run_int8_chunk(
            inp_q_2d[start:end],
            inp_s_1d[start:end],
            ids_2d[start:end],
            weights_2d[start:end],
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            out_2d[start:end],
            target=resolved_target,
            int8_backend=int8_backend,
            down_prep_plan=down_prep_plan,
            batched=batched,
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


def _dispatch_nvfp4_moe(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    resolved_target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
) -> None:
    device = input_2d.device
    rows, hidden_size = input_2d.shape
    num_slots, _, packed_inter = down_qdata.shape
    intermediate_size = 2 * packed_inter
    up_gate_out = up_gate_qdata.shape[1]
    top_k = ids_2d.shape[1]
    up_scale_cols = int(triton.cdiv(hidden_size, nvfp4_layout.SCALE_COLUMN_TILE))
    up_scale_stride = up_gate_scale.shape[1] * up_gate_scale.shape[2]
    down_scale_cols = int(triton.cdiv(intermediate_size, nvfp4_layout.SCALE_COLUMN_TILE))
    down_scale_stride = down_scale.shape[1] * down_scale.shape[2]

    bytes_per_pair = _prefill_bytes_per_pair(
        is_int8=False,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=input_2d.element_size(),
        group_size=group_size,
    )
    regime, chunk_rows = _select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill":
        for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
            input_2d, ids_2d, weights_2d, out_2d, prepared_data, group_size, chunk_rows
        ):
            _run_nvfp4_prefill_chunk(
                active_chunk,
                ids_chunk,
                weights_chunk,
                up_gate_qdata,
                up_gate_scale,
                down_qdata,
                down_scale,
                up_gate_global_scale,
                down_global_scale,
                expert_to_slot,
                out_chunk,
                target=resolved_target,
                accumulate=accumulate,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                high_first=high_first,
                active_slots_hint=active_slots_hint,
                up_scale_cols=up_scale_cols,
                up_scale_stride=up_scale_stride,
                down_scale_cols=down_scale_cols,
                down_scale_stride=down_scale_stride,
            )
        return

    batched = regime == "batched"
    use_sm120_e2m1x2 = resolved_target.is_cuda_capability(12)
    split_k_up = split_k_down = 1
    use_dot = cast_fp32_to_bf16 = single_dot = False
    post_scale_only = True
    block_n, block_k, num_warps, m1_cta_mult = _nvfp4_m1_config(resolved_target)
    if not batched:
        active_slots = _expected_bucket_tiles(top_k, num_slots, active_slots_hint, batched=False)
        split_k_up = _select_nvfp4_split_k(
            device,
            active_slots,
            up_gate_out,
            hidden_size,
            block_n=block_n,
            block_k=block_k,
            cta_multiplier=m1_cta_mult,
        )
        split_k_down = _select_nvfp4_split_k(
            device,
            active_slots,
            hidden_size,
            intermediate_size,
            block_n=block_n,
            block_k=block_k,
            cta_multiplier=m1_cta_mult,
        )

    for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
        input_2d, ids_2d, weights_2d, out_2d, prepared_data, group_size, chunk_rows
    ):
        if batched:
            chunk_n = active_chunk.shape[0]
            use_dot, cast_fp32_to_bf16, post_scale_only, block_n, block_k, num_warps, single_dot = (
                _nvfp4_batched_config(
                    resolved_target,
                    rows=chunk_n,
                    active_dtype=active_chunk.dtype,
                    logical_dtype=out_chunk.dtype,
                )
            )
            tiles = _expected_bucket_tiles(
                chunk_n * top_k, num_slots, active_slots_hint, batched=True
            )
            split_k_up = _select_nvfp4_split_k(
                device, tiles, up_gate_out, hidden_size, block_n=block_n, block_k=block_k
            )
            split_k_down = _select_nvfp4_split_k(
                device, tiles, hidden_size, intermediate_size, block_n=block_n, block_k=block_k
            )

        def _launch_nvfp4_step(
            per_expert_input: bool,
            x: torch.Tensor,
            routing: _A16ChunkRouting,
            output: torch.Tensor,
            split_k: int,
            *,
            _bn: int = block_n,
            _bk: int = block_k,
            _nw: int = num_warps,
            _ud: bool = use_dot,
            _bf16: bool = cast_fp32_to_bf16,
            _sm120_e2m1x2: bool = use_sm120_e2m1x2,
            _sd: bool = single_dot,
        ) -> None:
            if not per_expert_input:
                q, s, gs = up_gate_qdata, up_gate_scale, up_gate_global_scale
                n_out, n_in, sc, ss = up_gate_out, hidden_size, up_scale_cols, up_scale_stride
            else:
                q, s, gs = down_qdata, down_scale, down_global_scale
                n_out, n_in, sc, ss = (
                    hidden_size,
                    intermediate_size,
                    down_scale_cols,
                    down_scale_stride,
                )
            _launch_nvfp4_proj(
                x,
                q,
                s,
                gs,
                routing,
                output,
                out_features=n_out,
                in_features=n_in,
                scale_cols=sc,
                scale_stride=ss,
                num_slots=num_slots,
                split_k=split_k,
                block_n=_bn,
                block_k=_bk,
                num_warps=_nw,
                high_first=high_first,
                has_global_scale=gs is not None,
                per_expert_input=per_expert_input,
                is_splitk=per_expert_input or split_k > 1,
                use_dot=_ud,
                cast_fp32_to_bf16=_bf16,
                use_sm120_e2m1x2=_sm120_e2m1x2,
                single_dot=_sd,
            )

        _run_a16_chunk(
            active_chunk,
            ids_chunk,
            weights_chunk,
            out_chunk,
            num_slots=num_slots,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            up_gate_out=up_gate_out,
            expert_to_slot=expert_to_slot,
            target=resolved_target,
            batched=batched,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            split_k_up=split_k_up,
            split_k_down=split_k_down,
            up_row_scale=None,
            down_row_scale=None,
            up_global_scale=up_gate_global_scale,
            down_global_scale=down_global_scale,
            post_scale_only=post_scale_only,
            launch_proj=_launch_nvfp4_step,
        )


def _resolve_m1_active_slots(
    topk_ids_flat: torch.Tensor,
    expert_to_slot: torch.Tensor | None,
    num_slots: int,
) -> torch.Tensor:
    if expert_to_slot is None:
        return topk_ids_flat.to(torch.int64).clamp(0, num_slots - 1)
    num_experts = expert_to_slot.shape[0]
    in_range = (topk_ids_flat >= 0) & (topk_ids_flat < num_experts)
    safe_experts = torch.where(in_range, topk_ids_flat.to(torch.int64), 0)
    slots = expert_to_slot.index_select(0, safe_experts).to(torch.int64)
    valid = in_range & (slots >= 0) & (slots < num_slots)
    return torch.where(valid, slots, 0)


@dataclass(frozen=True, slots=True)
class _TrellisMoeSchedule:
    """Resolved ConvRot Trellis MoE A16 GEMV execution schedule."""

    block_m: int
    block_n: int
    num_warps: int
    unroll_k: int
    use_dp4a: bool
    use_sad_u8: bool
    split_x_loads: bool


def _select_trellis_moe_schedule(
    target: AcceleratorTarget,
    *,
    batched: bool,
    num_pairs: int,
    expected_slots: int,
) -> _TrellisMoeSchedule:
    """Resolve all target-dependent kernel choices for ``trellis_moe_gemv_kernel``."""
    if not batched:
        block_m, block_n = (1, 1)
    elif target.is_rdna4:
        block_m = (
            _TRELLIS_GEMV_RDNA4_SPARSE_BLOCK_M
            if num_pairs * 2 <= 3 * max(1, expected_slots)
            else _TRELLIS_GEMV_BATCH_BLOCK_M
        )
        block_n = 2
    else:
        block_m, block_n = (_TRELLIS_GEMV_BATCH_BLOCK_M, 1)
    return _TrellisMoeSchedule(
        block_m=block_m,
        block_n=block_n,
        num_warps=_TRELLIS_GEMV_NUM_WARPS,
        unroll_k=2 if (target.is_cuda_capability(12) and block_m == 1) else 1,
        use_dp4a=target.is_nvidia_cuda,
        use_sad_u8=target.is_rdna4,
        split_x_loads=target.is_rdna4,
    )


def _select_trellis_split_k(
    device: torch.device,
    active_tiles: int,
    out_features: int,
    blocks_per_row: int,
    weight_bytes: int,
) -> int:
    return select_split_k(
        device_sm_count(device),
        output_tiles=active_tiles * out_features,
        k_tiles=blocks_per_row,
        weight_bytes=weight_bytes,
        min_weight_bytes=_TRELLIS_MIN_SPLITK_WEIGHT_BYTES,
        max_split_k=_TRELLIS_MAX_SPLIT_K,
        cta_multiplier=4,
        min_k_tiles=2,
        min_tiles_per_split=2,
        round_up=False,
    )


def _launch_trellis_proj(
    x_rot: torch.Tensor,
    qdata: torch.Tensor,
    block_scales: torch.Tensor,
    row_scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    slot_offsets: torch.Tensor | None,
    block_size: int,
    routing: _A16ChunkRouting,
    expert_to_slot: torch.Tensor | None,
    output: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    num_slots: int,
    split_k: int,
    batched: bool,
    per_expert_input: bool,
    is_splitk: bool,
    schedule: _TrellisMoeSchedule,
) -> None:
    blocks_per_row = block_scales.shape[2]
    tile_steps = triton.next_power_of_2(block_size)
    exact_pow2 = block_size == tile_steps
    variable_rows = row_widths is not None
    has_slot_offsets = slot_offsets is not None

    if batched:
        assert routing.tile_slots is not None
        assert routing.tile_pair_ids is not None
        sub_tiles = _BATCH_BLOCK_M // schedule.block_m
        grid_tiles = routing.tile_slots.shape[0] * sub_tiles
        active_slots = (
            routing.tile_slots.clamp(0, num_slots - 1).to(torch.int64) if variable_rows else None
        )
    else:
        grid_tiles = routing.top_k
        active_slots = (
            _resolve_m1_active_slots(routing.topk_ids_flat, expert_to_slot, num_slots)
            if variable_rows
            else None
        )

    per_slot_2d = variable_rows and not has_slot_offsets
    if variable_rows:
        assert row_widths is not None
        assert active_slots is not None
        active_widths = row_widths.index_select(0, active_slots).contiguous()
        active_offsets = compute_row_offsets_2d(active_widths, dtype=torch.int32)
        slot_bytes = 0 if has_slot_offsets else qdata.shape[1]
        step_bits = num_wide_steps = bytes_per_block = 0
        needs_w2 = True
    else:
        layout = TrellisLayout.from_storage(block_size, qdata.shape[3])
        active_widths = active_offsets = None
        slot_bytes = 0
        step_bits = layout.step_bits
        num_wide_steps = layout.num_wide_steps
        bytes_per_block = layout.bytes_per_block
        needs_w2 = layout.block_bits > 5 * layout.block_size

    flat_qdata = qdata.view(-1)
    ptr_aligned = qdata.is_contiguous() and qdata.data_ptr() % 4 == 0
    has_tail_bytes = per_slot_2d or (qdata.numel() % 4 != 0)
    x_mul4 = block_size % 4 == 0 and in_features % 4 == 0

    grid = (triton.cdiv(out_features, schedule.block_n) * grid_tiles, split_k)
    with device_context(x_rot.device):
        trellis_moe_gemv_kernel[grid](
            x_rot,
            flat_qdata,
            block_scales.view(torch.uint8),
            row_scale,
            active_widths,
            active_offsets,
            slot_offsets,
            None if batched else routing.topk_ids_flat,
            None if batched else routing.expert_to_slot,
            routing.tile_slots,
            routing.tile_pair_ids,
            output,
            out_features,
            in_features,
            blocks_per_row,
            qdata.numel(),
            block_size,
            step_bits,
            num_wide_steps,
            bytes_per_block,
            slot_bytes,
            routing.num_experts,
            num_slots,
            routing.num_pairs,
            routing.top_k,
            split_k,
            tile_steps=tile_steps,
            block_m=schedule.block_m,
            batched=batched,
            has_slot_map=False if batched else routing.has_slot_map,
            per_expert_input=per_expert_input,
            is_splitk=is_splitk,
            variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets,
            exact_pow2=exact_pow2,
            needs_w2=needs_w2,
            use_dp4a=schedule.use_dp4a,
            has_tail_bytes=has_tail_bytes,
            ptr_aligned=ptr_aligned,
            x_mul4=x_mul4,
            unroll_k=schedule.unroll_k,
            block_n=schedule.block_n,
            use_sad_u8=schedule.use_sad_u8,
            split_x_loads=schedule.split_x_loads,
            num_warps=schedule.num_warps,
        )


def _dispatch_trellis_moe(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_block_scales: torch.Tensor,
    up_gate_scale: torch.Tensor,
    up_gate_row_widths: torch.Tensor | None,
    up_gate_slot_offsets: torch.Tensor | None,
    up_gate_block_size: int,
    down_qdata: torch.Tensor,
    down_block_scales: torch.Tensor,
    down_scale: torch.Tensor,
    down_row_widths: torch.Tensor | None,
    down_slot_offsets: torch.Tensor | None,
    down_block_size: int,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    resolved_target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    active_slots_hint: int,
) -> None:
    from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

    device = input_2d.device
    rows, hidden_size = input_2d.shape
    num_slots, _, down_blocks = down_block_scales.shape
    intermediate_size = down_blocks * down_block_size
    up_gate_out, up_blocks = up_gate_block_scales.shape[1], up_gate_block_scales.shape[2]
    top_k = ids_2d.shape[1]

    if resolved_target.is_nvidia_cuda and not resolved_target.cuda_capability_at_least(8):
        int8_backend = None
    else:
        int8_backend = int8_backend_mod.select_linear_backend_for_target(resolved_target)

    bytes_per_pair = _prefill_bytes_per_pair(
        is_int8=int8_backend is not None,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=out_2d.element_size(),
        group_size=group_size,
    )
    regime, chunk_rows = _select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill" and int8_backend is not None:
        _dispatch_int8_prefill_chunks(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            prepared_data,
            None,
            out_2d,
            resolved_target=resolved_target,
            int8_backend=int8_backend,
            chunk_rows=chunk_rows,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            active_slots_hint=active_slots_hint,
            up_gate_block_scales=up_gate_block_scales,
            up_gate_row_widths=up_gate_row_widths,
            up_gate_slot_offsets=up_gate_slot_offsets,
            up_gate_block_size=up_gate_block_size,
            down_block_scales=down_block_scales,
            down_row_widths=down_row_widths,
            down_slot_offsets=down_slot_offsets,
            down_block_size=down_block_size,
        )
        return

    if regime == "prefill":
        # Non-INT8 accelerators (e.g. SM75) fall back to chunked batched A16 GEMV.
        a16_chunk_rows = min(rows, _BATCH_MAX_ROWS, max(1, _BUCKET_MAX_PAIRS // max(1, top_k)))
        batched = a16_chunk_rows > 1
    else:
        a16_chunk_rows = chunk_rows
        batched = regime == "batched"

    up_slot_bytes = up_gate_qdata.numel() // max(1, num_slots)
    down_slot_bytes = down_qdata.numel() // max(1, num_slots)
    expected_slots = min(num_slots, active_slots_hint) if active_slots_hint > 0 else num_slots
    split_k_up = split_k_down = 1
    schedule = _select_trellis_moe_schedule(
        resolved_target,
        batched=batched,
        num_pairs=top_k,
        expected_slots=expected_slots,
    )
    if not batched:
        active_slots = _expected_bucket_tiles(top_k, num_slots, active_slots_hint, batched=False)
        split_k_up = _select_trellis_split_k(
            device, active_slots, up_gate_out, up_blocks, max(1, up_slot_bytes * active_slots)
        )
        split_k_down = _select_trellis_split_k(
            device, active_slots, hidden_size, down_blocks, max(1, down_slot_bytes * active_slots)
        )

    def _launch_trellis_step(
        per_expert_input: bool,
        x: torch.Tensor,
        routing: _A16ChunkRouting,
        output: torch.Tensor,
        split_k: int,
    ) -> None:
        if not per_expert_input:
            q, bs, rs, rw, so, bsz, n_out, n_in = (
                up_gate_qdata,
                up_gate_block_scales,
                up_gate_scale,
                up_gate_row_widths,
                up_gate_slot_offsets,
                up_gate_block_size,
                up_gate_out,
                hidden_size,
            )
        else:
            q, bs, rs, rw, so, bsz, n_out, n_in = (
                down_qdata,
                down_block_scales,
                down_scale,
                down_row_widths,
                down_slot_offsets,
                down_block_size,
                hidden_size,
                intermediate_size,
            )
        _launch_trellis_proj(
            x,
            q,
            bs,
            rs,
            rw,
            so,
            bsz,
            routing,
            expert_to_slot,
            output,
            out_features=n_out,
            in_features=n_in,
            num_slots=num_slots,
            split_k=split_k,
            batched=batched,
            per_expert_input=per_expert_input,
            is_splitk=per_expert_input or split_k > 1,
            schedule=schedule,
        )

    for active_chunk, ids_chunk, weights_chunk, out_chunk in _iter_a16_chunks(
        input_2d, ids_2d, weights_2d, out_2d, prepared_data, group_size, a16_chunk_rows
    ):
        if batched:
            num_pairs = active_chunk.shape[0] * top_k
            schedule = _select_trellis_moe_schedule(
                resolved_target,
                batched=True,
                num_pairs=num_pairs,
                expected_slots=expected_slots,
            )
            weight_buckets = min(
                _max_bucket_tiles(num_pairs, num_slots, _BATCH_BLOCK_M),
                _expected_bucket_tiles(num_pairs, num_slots, active_slots_hint, batched=True),
            )
            active_proj_tiles = (
                min(
                    _max_bucket_tiles(num_pairs, num_slots, schedule.block_m),
                    _expected_bucket_tiles(
                        num_pairs,
                        num_slots,
                        active_slots_hint,
                        block_m=schedule.block_m,
                        batched=True,
                    ),
                )
                if resolved_target.is_rdna4
                else weight_buckets
            )
            split_k_up = _select_trellis_split_k(
                device,
                active_proj_tiles,
                triton.cdiv(up_gate_out, schedule.block_n),
                up_blocks,
                max(1, up_slot_bytes * weight_buckets),
            )
            split_k_down = _select_trellis_split_k(
                device,
                active_proj_tiles,
                triton.cdiv(hidden_size, schedule.block_n),
                down_blocks,
                max(1, down_slot_bytes * weight_buckets),
            )
        _run_a16_chunk(
            active_chunk,
            ids_chunk,
            weights_chunk,
            out_chunk,
            num_slots=num_slots,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            up_gate_out=up_gate_out,
            expert_to_slot=expert_to_slot,
            target=resolved_target,
            batched=batched,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            split_k_up=split_k_up,
            split_k_down=split_k_down,
            up_row_scale=up_gate_scale,
            down_row_scale=down_scale,
            up_global_scale=None,
            down_global_scale=None,
            post_scale_only=True,
            launch_proj=_launch_trellis_step,
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
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_size: int = 0,
) -> torch.Tensor:
    """Execute single-token, small-batch, or prefill quantized MoE kernels on GPU."""
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

    if up_gate_block_scales is not None:
        assert down_block_scales is not None
        _dispatch_trellis_moe(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_block_scales,
            up_gate_scale,
            up_gate_row_widths,
            up_gate_slot_offsets,
            up_gate_block_size,
            down_qdata,
            down_block_scales,
            down_scale,
            down_row_widths,
            down_slot_offsets,
            down_block_size,
            expert_to_slot,
            prepared_data,
            out_2d,
            resolved_target=resolved_target,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            active_slots_hint=active_slots_hint,
        )
    elif up_gate_qdata.dtype is torch.int8:
        _dispatch_int8_moe(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            out_2d,
            resolved_target=resolved_target,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            active_slots_hint=active_slots_hint,
        )
    else:
        _dispatch_nvfp4_moe(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            up_gate_global_scale,
            down_global_scale,
            expert_to_slot,
            prepared_data,
            out_2d,
            resolved_target=resolved_target,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            active_slots_hint=active_slots_hint,
        )
    return result


def _bind_opt_ptrs(
    sig: dict[str, str],
    constexprs: dict[str, object],
    **opt_ptrs: str | None,
) -> None:
    for name, ptr_type in opt_ptrs.items():
        if ptr_type is None:
            constexprs[name] = None
        else:
            sig[name] = ptr_type


def _nvfp4_aligned_attrs(
    arg_names: Sequence[str],
    signature: dict[str, str],
    *,
    out_features: int,
    in_features: int,
) -> dict[tuple[int, ...], list[list[object]]]:
    """Attach runtime pointer and packed-stride divisibility hints for offline compile."""
    aligned = [name for name, ty in signature.items() if ty.startswith("*")]
    aligned.append("expert_scale_stride")
    if out_features % 16 == 0:
        aligned.append("out_features")
    if (in_features // 2) % 16 == 0:
        aligned.append("packed_k")
    return {(arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned}


def _compile_bucket_kernel(
    target: GPUTarget,
    *,
    num_pairs: int,
    has_slot_map: bool,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    block_p = triton.next_power_of_2(max(16, num_pairs))
    sig: dict[str, str] = {
        "topk_ids_ptr": "*i64",
        "tile_slots_ptr": "*i32",
        "tile_pair_ids_ptr": "*i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "max_tiles": "i32",
    }
    constexprs: dict[str, object] = {
        "block_p": block_p,
        "block_m": _BATCH_BLOCK_M,
        "has_slot_map": has_slot_map,
    }
    _bind_opt_ptrs(sig, constexprs, expert_to_slot_ptr="*i64" if has_slot_map else None)
    return compile_fn(
        ASTSource(moe_bucket_tokens_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": 4},
    )


def _compile_weighted_reduce(
    target: GPUTarget,
    *,
    out_ty: str,
    is_int8: bool,
    has_slot_map: bool,
    accumulate: bool,
    has_global_scale: bool,
    post_scale_only: bool,
    use_compact_map: bool = False,
    has_row_scale: bool = False,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    eff_int8 = False if use_compact_map else is_int8
    eff_slot_map = False if use_compact_map else has_slot_map
    eff_global_scale = False if use_compact_map else has_global_scale
    eff_post_scale_only = True if use_compact_map else post_scale_only
    eff_row_scale = False if use_compact_map else has_row_scale
    idx_ty = "*i32" if use_compact_map else "*i64"
    sig: dict[str, str] = {
        "partial_ptr": "*i32" if eff_int8 else "*fp32",
        "topk_ids_ptr": idx_ty,
        "topk_weights_ptr": out_ty,
        "output_ptr": out_ty,
        "out_features": "i32",
        "out_row_stride": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "top_k": "i32",
        "split_k": "i32",
    }
    constexprs: dict[str, object] = {
        "block_n": _WEIGHTED_REDUCE_BLOCK_N,
        "is_int8": eff_int8,
        "has_slot_map": eff_slot_map,
        "accumulate": accumulate,
        "has_global_scale": eff_global_scale,
        "post_scale_only": eff_post_scale_only,
        "use_compact_map": use_compact_map,
        "has_row_scale": eff_row_scale,
    }
    _bind_opt_ptrs(
        sig,
        constexprs,
        input_scale_ptr="*fp32" if eff_int8 else None,
        weight_scale_ptr="*fp32" if (eff_int8 or eff_row_scale) else None,
        weight_global_scale_ptr="*fp32" if (not eff_int8 and eff_global_scale) else None,
        expert_to_slot_ptr="*i64" if eff_slot_map else None,
    )
    return compile_fn(
        ASTSource(moe_m1_weighted_reduce_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": 4},
    )


def _compile_splitk_reduce(
    target: GPUTarget,
    *,
    out_ty: str,
    is_int8: bool,
    has_slot_map: bool,
    has_global_scale: bool = False,
    post_scale_only: bool = True,
    has_row_scale: bool = False,
    swap_halves: bool = False,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    eff_global_scale = False if is_int8 else has_global_scale
    eff_row_scale = False if is_int8 else has_row_scale
    sig: dict[str, str] = {
        "partial_ptr": "*i32" if is_int8 else "*fp32",
        "topk_ids_ptr": "*i64",
        "output_ptr": out_ty if is_int8 else "*fp32",
        "out_features": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "top_k": "i32",
        "split_k": "i32",
    }
    constexprs: dict[str, object] = {
        "block_n": _INT8_REDUCE_BLOCK_N if is_int8 else _NVFP4_REDUCE_BLOCK_N,
        "is_int8": is_int8,
        "has_slot_map": has_slot_map,
        "has_global_scale": eff_global_scale,
        "post_scale_only": True if is_int8 else post_scale_only,
        "has_row_scale": eff_row_scale,
        "swap_halves": swap_halves if is_int8 else False,
    }
    _bind_opt_ptrs(
        sig,
        constexprs,
        input_scale_ptr="*fp32" if is_int8 else None,
        weight_scale_ptr="*fp32" if (is_int8 or eff_row_scale) else None,
        weight_global_scale_ptr="*fp32" if eff_global_scale else None,
        expert_to_slot_ptr="*i64" if has_slot_map else None,
    )
    return compile_fn(
        ASTSource(moe_splitk_reduce_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": 4},
    )


def _compile_activate_rotate(
    target: GPUTarget,
    *,
    active_ty: str,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    accelerator_backend: str,
    has_slot_map: bool,
    use_compact_map: bool = False,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    eff_slot_map = False if use_compact_map else has_slot_map
    idx_ty = "*i32" if use_compact_map else "*i64"
    sig: dict[str, str] = {
        "mid_ptr": "*fp32",
        "act_ptr": active_ty,
        "logical_ref_ptr": out_ty,
        "topk_ids_ptr": idx_ty,
        "intermediate_size": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
    }
    constexprs: dict[str, object] = {
        "block_size": _ACTIVATE_BLOCK_SIZE,
        "group_size": group_size,
        "activation_fn": activation_fn,
        "gate_first": gate_first,
        "accelerator_backend": accelerator_backend,
        "has_slot_map": eff_slot_map,
    }
    _bind_opt_ptrs(sig, constexprs, expert_to_slot_ptr="*i64" if eff_slot_map else None)
    return compile_fn(
        ASTSource(moe_m1_activate_rotate_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": 4},
    )


def _validate_nvfp4_compile_format(
    *,
    weight_format: str,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    if hidden_size % nvfp4_layout.BLOCK_SIZE or intermediate_size % nvfp4_layout.BLOCK_SIZE:
        raise ValueError("NVFP4 MoE hidden_size and intermediate_size must be multiples of 16")
    if weight_format == "nvfp4_w4a16":
        if group_size != 0:
            raise ValueError("nvfp4_w4a16 requires group_size=0; use convrot_nvfp4_w4a16")
        return
    if group_size == 0:
        raise ValueError("convrot_nvfp4_w4a16 requires group_size in (16, 64, 256)")
    validate_group_size(group_size)
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError(
            f"ConvRot NVFP4 MoE dimensions must be divisible by group_size {group_size}"
        )


def _validate_convrot_compile_format(
    accel: AcceleratorTarget,
    *,
    weight_format: str,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
    high_first: bool,
    block_size: int,
    variable_rows: bool,
    has_slot_offsets: bool,
    bytes_per_block: int | None,
) -> None:
    label = "ConvRot INT8 MoE" if weight_format == "convrot_int8" else "ConvRot Trellis MoE"
    if high_first:
        raise ValueError(f"{label} weights do not support high_first")
    if (
        weight_format == "convrot_int8"
        and accel.is_nvidia_cuda
        and not accel.cuda_capability_at_least(8)
    ):
        raise ValueError(
            f"ConvRot INT8 MoE requires NVIDIA SM80+ or supported AMD GPU, got {accel}"
        )
    validate_group_size(group_size)
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError(f"{label} dimensions must be divisible by group_size {group_size}")
    if hidden_size > _INT8_MAX_REDUCTION_WIDTH or intermediate_size > _INT8_MAX_REDUCTION_WIDTH:
        raise ValueError(
            f"{label} widths must not exceed {_INT8_MAX_REDUCTION_WIDTH} "
            f"for INT32 accumulation, got hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}"
        )
    if weight_format != "convrot_trellis":
        return
    if has_slot_offsets and not variable_rows:
        raise ValueError("has_slot_offsets requires variable_rows=True")
    validate_block_size(block_size)
    if hidden_size % block_size or intermediate_size % block_size:
        raise ValueError(
            f"ConvRot Trellis MoE dimensions must be divisible by block_size {block_size}"
        )
    if bytes_per_block is not None:
        if variable_rows:
            raise ValueError(
                "bytes_per_block applies only to uniform (variable_rows=False) layouts"
            )
        TrellisLayout.from_storage(block_size, bytes_per_block)


def _validate_compile_format(
    accel: AcceleratorTarget,
    *,
    weight_format: str,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
    high_first: bool,
    block_size: int = 256,
    variable_rows: bool = False,
    has_slot_offsets: bool = False,
    bytes_per_block: int | None = None,
) -> None:
    if weight_format in ("nvfp4_w4a16", "convrot_nvfp4_w4a16"):
        _validate_nvfp4_compile_format(
            weight_format=weight_format,
            group_size=group_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        return
    if weight_format not in ("convrot_int8", "convrot_trellis"):
        raise ValueError(f"Unsupported MoE weight_format {weight_format!r}")
    _validate_convrot_compile_format(
        accel,
        weight_format=weight_format,
        group_size=group_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        high_first=high_first,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
        bytes_per_block=bytes_per_block,
    )


def _compile_prefill_utility_kernels(
    target: GPUTarget,
    *,
    gather_ty: str,
    out_ty: str,
    is_int8: bool,
    block_m: int,
    has_slot_map: bool,
    accumulate: bool,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    count_sig: dict[str, str] = {
        "topk_ids_ptr": "*i64",
        "slot_counts_ptr": "*i32",
        "pair_slots_ptr": "*i32",
        "pair_pos_ptr": "*i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
    }
    count_constexprs: dict[str, object] = {
        "block_p": _PREFILL_BUCKET_BLOCK_P,
        "has_slot_map": has_slot_map,
    }
    _bind_opt_ptrs(count_sig, count_constexprs, expert_to_slot_ptr="*i64" if has_slot_map else None)
    gather_sig: dict[str, str] = {
        "input_ptr": gather_ty,
        "compact_to_pair_ptr": "*i32",
        "out_ptr": gather_ty,
        "hidden_size": "i32",
        "top_k": "i32",
    }
    gather_constexprs: dict[str, object] = {
        "block_k": _PREFILL_GATHER_BLOCK_K,
        "has_scale": is_int8,
    }
    _bind_opt_ptrs(
        gather_sig,
        gather_constexprs,
        input_scale_ptr="*fp32" if is_int8 else None,
        out_scale_ptr="*fp32" if is_int8 else None,
    )
    return {
        "bucket_count": compile_fn(
            ASTSource(moe_prefill_bucket_count_kernel, count_sig, constexprs=count_constexprs),
            target,
            {"num_warps": 4},
        ),
        "bucket_scatter": compile_fn(
            ASTSource(
                moe_prefill_bucket_scatter_kernel,
                {
                    "pair_slots_ptr": "*i32",
                    "pair_pos_ptr": "*i32",
                    "slot_counts_ptr": "*i32",
                    "slot_offsets_ptr": "*i32",
                    "slot_tile_base_ptr": "*i32",
                    "compact_to_pair_ptr": "*i32",
                    "pair_to_compact_ptr": "*i32",
                    "tile_slots_ptr": "*i32",
                    "tile_m_start_ptr": "*i32",
                    "tile_m_len_ptr": "*i32",
                    "num_pairs": "i32",
                },
                constexprs={"block_p": _PREFILL_BUCKET_BLOCK_P, "block_m": block_m},
            ),
            target,
            {"num_warps": 4},
        ),
        "gather": compile_fn(
            ASTSource(moe_prefill_gather_kernel, gather_sig, constexprs=gather_constexprs),
            target,
            {"num_warps": 4},
        ),
        "weighted_reduce": _compile_weighted_reduce(
            target,
            out_ty=out_ty,
            is_int8=False,
            has_slot_map=False,
            accumulate=accumulate,
            has_global_scale=False,
            post_scale_only=True,
            use_compact_map=True,
            compile_fn=compile_fn,
        ),
    }


def _compile_int8_prefill_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    accumulate: bool,
    slot_window: bool = False,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    block_m, block_n, block_k, num_warps, num_stages, compiler_options = _int8_prefill_config(
        accel,
        chunk_rows=rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
    )
    expected_tiles = _expected_bucket_tiles(
        rows * top_k, num_slots, active_slots_hint, block_m=block_m, batched=True
    )
    up_seed = rdna4_int8_accumulator_seed(
        accel, hidden_size, expected_tiles * triton.cdiv(up_gate_out, block_n)
    )
    down_seed = rdna4_int8_accumulator_seed(
        accel, intermediate_size, expected_tiles * triton.cdiv(hidden_size, block_n)
    )
    swap_halves = activation_fn == "swiglu" and gate_first
    sig = {
        "input_ptr": "*i8",
        "weight_ptr": "*i8",
        "tile_slots_ptr": "*i32",
        "tile_m_start_ptr": "*i32",
        "tile_m_len_ptr": "*i32",
        "input_scale_ptr": "*fp32",
        "weight_scale_ptr": "*fp32",
        "output_ptr": out_ty,
        "max_tiles": "i32",
        "out_features": "i32",
        "in_features": "i32",
        "slot_start": "i32",
        "slot_end": "i32",
        "n_start": "i32",
        "n_rows": "i32",
    }
    opts = {"num_stages": num_stages, "num_warps": num_warps, **compiler_options}
    compiled = _compile_prefill_utility_kernels(
        target,
        gather_ty="*i8",
        out_ty=out_ty,
        is_int8=True,
        block_m=block_m,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        compile_fn=compile_fn,
    )
    compiled["int8_grouped_up"] = compile_fn(
        ASTSource(
            int8_moe_grouped_gemm_kernel,
            sig,
            constexprs={
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "group_m": _PREFILL_GROUP_M,
                "swap_halves": swap_halves,
                "aligned_nk": (up_gate_out % block_n == 0 and hidden_size % block_k == 0),
                "accumulator_seed": up_seed,
                "slot_window": slot_window,
            },
        ),
        target,
        opts,
    )
    compiled["int8_grouped_down"] = compile_fn(
        ASTSource(
            int8_moe_grouped_gemm_kernel,
            {**sig, "output_ptr": "*fp32"},
            constexprs={
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "group_m": _PREFILL_GROUP_M,
                "swap_halves": False,
                "aligned_nk": (hidden_size % block_n == 0 and intermediate_size % block_k == 0),
                "accumulator_seed": down_seed,
                "slot_window": slot_window,
            },
        ),
        target,
        opts,
    )
    return compiled


def _compile_nvfp4_prefill_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    hidden_size: int,
    intermediate_size: int,
    logical_dtype: torch.dtype,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    accumulate: bool,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    active_dtype = torch.float32 if group_size > 0 else logical_dtype
    active_ty = TRITON_FLOAT_POINTER_TYPES[active_dtype]
    cast_bf16, force_fp32_dot, half_scales, block_m, block_n, block_k, num_warps, single_dot = (
        _nvfp4_prefill_config(
            accel,
            chunk_rows=rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            active_dtype=active_dtype,
            logical_dtype=logical_dtype,
        )
    )
    use_sm120_e2m1x2 = accel.is_cuda_capability(12)
    sig: dict[str, str] = {
        "input_ptr": active_ty,
        "weight_ptr": "*u8",
        "weight_scale_ptr": "*u8",
        "tile_slots_ptr": "*i32",
        "tile_m_start_ptr": "*i32",
        "tile_m_len_ptr": "*i32",
        "output_ptr": "*fp32",
        "max_tiles": "i32",
        "out_features": "i32",
        "packed_k": "i32",
        "scale_column_blocks": "i32",
        "expert_scale_stride": "i32",
    }
    constexprs: dict[str, object] = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "group_m": _PREFILL_GROUP_M,
        "high_first": high_first,
        "has_global_scale": has_global_scale,
        "cast_fp32_to_bf16": cast_bf16,
        "force_fp32_dot": force_fp32_dot,
        "half_scales": half_scales,
        "use_sm120_e2m1x2": use_sm120_e2m1x2,
        "single_dot": single_dot,
    }
    _bind_opt_ptrs(
        sig,
        constexprs,
        weight_global_scale_ptr="*fp32" if has_global_scale else None,
    )
    opts: dict[str, object] = {"num_warps": num_warps, "num_stages": 1}
    compiled = _compile_prefill_utility_kernels(
        target,
        gather_ty=active_ty,
        out_ty=out_ty,
        is_int8=False,
        block_m=block_m,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        compile_fn=compile_fn,
    )
    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    grouped_attrs = _nvfp4_aligned_attrs(
        nvfp4_moe_grouped_gemm_kernel.arg_names,
        sig,
        out_features=math.gcd(up_gate_out, hidden_size),
        in_features=math.gcd(hidden_size, intermediate_size),
    )
    compiled["nvfp4_grouped"] = compile_fn(
        ASTSource(nvfp4_moe_grouped_gemm_kernel, sig, constexprs, attrs=grouped_attrs),
        target,
        opts,
    )
    compiled["activate_rotate"] = _compile_activate_rotate(
        target,
        active_ty=active_ty,
        out_ty=out_ty,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        accelerator_backend=accel.backend,
        has_slot_map=has_slot_map,
        use_compact_map=True,
        compile_fn=compile_fn,
    )
    return compiled


def _compile_int8_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    batched: bool,
    num_pairs: int = 1,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    split_k: int,
    accumulate: bool,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    block_k, num_stages, compiler_options = _int8_launch_config(accel)
    is_splitk = split_k > 1
    swap_halves = activation_fn == "swiglu" and gate_first
    seed_rows = (num_pairs if batched else 1) * _INT8_GEMM_BLOCK_M
    up_seed = splitk_accumulator_seed(
        accel,
        rows=seed_rows,
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
        rows=seed_rows,
        out_features=hidden_size,
        in_features=intermediate_size,
        block_m=_INT8_GEMM_BLOCK_M,
        block_n=_INT8_GEMM_BLOCK_N,
        block_k=block_k,
        paired=False,
        split_k=split_k,
    )
    kernel_fn = int8_moe_batched_matmul_kernel if batched else int8_moe_m1_matmul_kernel
    int8_signature: dict[str, str] = (
        {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "tile_slots_ptr": "*i32",
            "tile_pair_ids_ptr": "*i32",
            "input_scale_ptr": "*fp32",
            "weight_scale_ptr": "*fp32",
            "output_ptr": "*i32" if is_splitk else out_ty,
            "out_features": "i32",
            "in_features": "i32",
            "num_pairs": "i32",
            "top_k": "i32",
            "split_k": "i32",
        }
        if batched
        else {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "topk_ids_ptr": "*i64",
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
    )
    slot_map_cfg: dict[str, object] = {}
    if not batched:
        slot_map_cfg["has_slot_map"] = has_slot_map
        _bind_opt_ptrs(
            int8_signature,
            slot_map_cfg,
            expert_to_slot_ptr="*i64" if has_slot_map else None,
        )
    up_block_m = _BATCH_BLOCK_M if batched else _INT8_GEMM_BLOCK_M
    compiled_int8: dict[str, triton.compiler.CompiledKernel] = {
        "int8_matmul_up": compile_fn(
            ASTSource(
                kernel_fn,
                int8_signature,
                constexprs={
                    "block_m": up_block_m,
                    "block_n": _INT8_GEMM_BLOCK_N,
                    "block_k": block_k,
                    **slot_map_cfg,
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
                kernel_fn,
                {**int8_signature, "output_ptr": "*i32"},
                constexprs={
                    "block_m": up_block_m,
                    "block_n": _INT8_GEMM_BLOCK_N,
                    "block_k": block_k,
                    **slot_map_cfg,
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
        compiled_int8["int8_reduce"] = _compile_splitk_reduce(
            target,
            out_ty=out_ty,
            is_int8=True,
            has_slot_map=has_slot_map,
            swap_halves=swap_halves,
            compile_fn=compile_fn,
        )
    compiled_int8["weighted_reduce"] = _compile_weighted_reduce(
        target,
        out_ty=out_ty,
        is_int8=True,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        has_global_scale=False,
        post_scale_only=True,
        compile_fn=compile_fn,
    )
    return compiled_int8


def _compile_a16_epilogue(
    compiled: dict[str, triton.compiler.CompiledKernel],
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    reduce_key: str,
    active_ty: str,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    post_scale_only: bool,
    has_row_scale: bool,
    is_splitk: bool,
    accumulate: bool,
    compile_fn: TritonCompileFn,
) -> None:
    if is_splitk:
        compiled[reduce_key] = _compile_splitk_reduce(
            target,
            out_ty=out_ty,
            is_int8=False,
            has_slot_map=has_slot_map,
            has_global_scale=has_global_scale,
            post_scale_only=post_scale_only,
            has_row_scale=has_row_scale,
            compile_fn=compile_fn,
        )
    compiled["activate_rotate"] = _compile_activate_rotate(
        target,
        active_ty=active_ty,
        out_ty=out_ty,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        accelerator_backend=accel.backend,
        has_slot_map=has_slot_map,
        use_compact_map=False,
        compile_fn=compile_fn,
    )
    compiled["weighted_reduce"] = _compile_weighted_reduce(
        target,
        out_ty=out_ty,
        is_int8=False,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        has_global_scale=has_global_scale,
        post_scale_only=post_scale_only,
        has_row_scale=has_row_scale,
        compile_fn=compile_fn,
    )


def _compile_nvfp4_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    batched: bool,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    logical_dtype: torch.dtype,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    split_k: int,
    accumulate: bool,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    active_dtype = torch.float32 if group_size > 0 else logical_dtype
    active_ty = TRITON_FLOAT_POINTER_TYPES[active_dtype]
    use_sm120_e2m1x2 = accel.is_cuda_capability(12)
    is_splitk = split_k > 1
    if batched:
        use_dot, cast_fp32_to_bf16, post_scale_only, block_n, block_k, num_warps, single_dot = (
            _nvfp4_batched_config(
                accel,
                rows=rows,
                active_dtype=active_dtype,
                logical_dtype=logical_dtype,
            )
        )
        nvfp4_signature: dict[str, str] = {
            "input_ptr": active_ty,
            "weight_ptr": "*u8",
            "weight_scale_ptr": "*u8",
            "tile_slots_ptr": "*i32",
            "tile_pair_ids_ptr": "*i32",
            "output_ptr": "*fp32",
            "out_features": "i32",
            "packed_k": "i32",
            "scale_column_blocks": "i32",
            "expert_scale_stride": "i32",
            "num_pairs": "i32",
            "top_k": "i32",
            "split_k": "i32",
        }
        kernel_fn = nvfp4_moe_batched_gemv_kernel
        mode_constexprs: dict[str, object] = {
            "block_m": _BATCH_BLOCK_M,
            "block_n": block_n,
            "block_k": block_k,
            "use_dot": use_dot,
            "cast_fp32_to_bf16": cast_fp32_to_bf16,
            "single_dot": single_dot,
        }
        gemv_opts: dict[str, object] = {"num_warps": num_warps, "num_stages": 1}
    else:
        block_n, block_k, num_warps, _ = _nvfp4_m1_config(accel)
        post_scale_only = True
        nvfp4_signature = {
            "input_ptr": active_ty,
            "weight_ptr": "*u8",
            "weight_scale_ptr": "*u8",
            "topk_ids_ptr": "*i64",
            "output_ptr": "*fp32",
            "out_features": "i32",
            "packed_k": "i32",
            "scale_column_blocks": "i32",
            "expert_scale_stride": "i32",
            "num_experts": "i32",
            "num_slots": "i32",
            "top_k": "i32",
            "split_k": "i32",
        }
        kernel_fn = nvfp4_moe_m1_gemv_kernel
        mode_constexprs = {
            "block_n": block_n,
            "block_k": block_k,
            "has_slot_map": has_slot_map,
        }
        _bind_opt_ptrs(
            nvfp4_signature,
            mode_constexprs,
            expert_to_slot_ptr="*i64" if has_slot_map else None,
        )
        gemv_opts = {"num_warps": num_warps}
    _bind_opt_ptrs(
        nvfp4_signature,
        mode_constexprs,
        weight_global_scale_ptr="*fp32" if has_global_scale else None,
    )

    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    up_attrs = _nvfp4_aligned_attrs(
        kernel_fn.arg_names,
        nvfp4_signature,
        out_features=up_gate_out,
        in_features=hidden_size,
    )
    down_attrs = _nvfp4_aligned_attrs(
        kernel_fn.arg_names,
        nvfp4_signature,
        out_features=hidden_size,
        in_features=intermediate_size,
    )
    compiled_nvfp4: dict[str, triton.compiler.CompiledKernel] = {
        "nvfp4_gemv_up": compile_fn(
            ASTSource(
                kernel_fn,
                nvfp4_signature,
                constexprs={
                    **mode_constexprs,
                    "high_first": high_first,
                    "has_global_scale": has_global_scale,
                    "per_expert_input": False,
                    "is_splitk": is_splitk,
                    "use_sm120_e2m1x2": use_sm120_e2m1x2,
                },
                attrs=up_attrs,
            ),
            target,
            gemv_opts,
        ),
        "nvfp4_gemv_down": compile_fn(
            ASTSource(
                kernel_fn,
                nvfp4_signature,
                constexprs={
                    **mode_constexprs,
                    "high_first": high_first,
                    "has_global_scale": has_global_scale,
                    "per_expert_input": True,
                    "is_splitk": True,
                    "use_sm120_e2m1x2": use_sm120_e2m1x2,
                },
                attrs=down_attrs,
            ),
            target,
            gemv_opts,
        ),
    }
    _compile_a16_epilogue(
        compiled_nvfp4,
        target,
        accel,
        reduce_key="nvfp4_reduce",
        active_ty=active_ty,
        out_ty=out_ty,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        has_slot_map=has_slot_map,
        has_global_scale=has_global_scale,
        post_scale_only=post_scale_only,
        has_row_scale=False,
        is_splitk=is_splitk,
        accumulate=accumulate,
        compile_fn=compile_fn,
    )
    return compiled_nvfp4


def _compile_trellis_prefill_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    block_size: int,
    variable_rows: bool,
    has_slot_offsets: bool,
    bytes_per_block: int | None,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    accumulate: bool,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    compiled = _compile_int8_prefill_kernels(
        target,
        accel,
        rows=rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        out_ty=out_ty,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        slot_window=True,
        compile_fn=compile_fn,
    )

    tile_steps = triton.next_power_of_2(block_size)
    per_slot_2d = variable_rows and not has_slot_offsets
    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    if variable_rows:
        needs_w2 = True
        stream_bytes = 1
    else:
        layout = default_layout_for_block_size(block_size, bytes_per_block)
        needs_w2 = layout.block_bits > 5 * layout.block_size
        up_slot_bytes = up_gate_out * (hidden_size // block_size) * layout.bytes_per_block
        down_slot_bytes = hidden_size * (intermediate_size // block_size) * layout.bytes_per_block
        stream_bytes = up_slot_bytes if (down_slot_bytes % 4 == 0) else down_slot_bytes
    use_dp4a, use_sad_u8, has_tail_bytes, clamped_load, num_warps = _decode_stream_flags(
        accel,
        stream_bytes,
        variable_rows=variable_rows,
        ptr_aligned=True,
        per_slot_2d=per_slot_2d,
    )
    dec_sig: dict[str, str] = {
        "bitstreams": "*u8",
        "block_scales": "*u8",
        "qdata": "*i8",
        "active_counts_ptr": "*i32",
        "num_blocks": "i32",
        "blocks_per_row": "i32",
        "bitstreams_bytes": "i64",
        "block_size": "i32",
        "step_bits": "i32",
        "num_wide_steps": "i32",
        "bytes_per_block": "i32",
        "rows_per_slot": "i32",
        "slot_bytes": "i32",
    }
    dec_constexprs: dict[str, object] = {
        "tile_steps": tile_steps,
        "tile_blocks": 4,
        "variable_rows": variable_rows,
        "exact_pow2": block_size == tile_steps,
        "filter_active": True,
        "has_slot_offsets": has_slot_offsets,
        "per_slot_2d": per_slot_2d,
        "needs_w2": needs_w2,
        "use_dp4a": use_dp4a,
        "use_sad_u8": use_sad_u8,
        "has_tail_bytes": has_tail_bytes,
        "ptr_aligned": True,
        "clamped_load": clamped_load,
        "q_u32_aligned": (block_size % 4 == 0),
    }
    _bind_opt_ptrs(
        dec_sig,
        dec_constexprs,
        row_widths="*u8" if variable_rows else None,
        row_offsets="*i32" if variable_rows else None,
        slot_offsets_ptr="*i64" if has_slot_offsets else None,
    )

    compiled["trellis_decode"] = compile_fn(
        ASTSource(decode_slots_int8_kernel, dec_sig, constexprs=dec_constexprs),
        target,
        {"num_warps": num_warps},
    )
    return compiled


def _compile_trellis_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    batched: bool,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    block_size: int,
    variable_rows: bool,
    has_slot_offsets: bool,
    bytes_per_block: int | None,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    split_k: int,
    accumulate: bool,
    compile_fn: TritonCompileFn,
    rows: int = 1,
    top_k: int = 1,
    num_slots: int = 64,
    active_slots_hint: int = 0,
) -> dict[str, triton.compiler.CompiledKernel]:
    tile_steps = triton.next_power_of_2(block_size)
    exact_pow2 = block_size == tile_steps
    expected_slots = min(num_slots, active_slots_hint) if active_slots_hint > 0 else num_slots
    schedule = _select_trellis_moe_schedule(
        accel,
        batched=batched,
        num_pairs=rows * top_k,
        expected_slots=expected_slots,
    )
    is_splitk = split_k > 1
    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    per_slot_2d = variable_rows and not has_slot_offsets
    if variable_rows:
        needs_w2 = True
        up_has_tail = True
        down_has_tail = True
    else:
        layout = default_layout_for_block_size(block_size, bytes_per_block)
        needs_w2 = layout.block_bits > 5 * layout.block_size
        up_bytes = num_slots * up_gate_out * (hidden_size // block_size) * layout.bytes_per_block
        down_bytes = (
            num_slots * hidden_size * (intermediate_size // block_size) * layout.bytes_per_block
        )
        up_has_tail = _decode_stream_flags(
            accel, up_bytes, variable_rows=False, per_slot_2d=per_slot_2d
        )[2]
        down_has_tail = _decode_stream_flags(
            accel, down_bytes, variable_rows=False, per_slot_2d=per_slot_2d
        )[2]

    gemv_sig: dict[str, str] = {
        "x_rot_ptr": "*fp32",
        "bitstreams_ptr": "*u8",
        "block_scales_ptr": "*u8",
        "row_scale_ptr": "*fp32",
        "output_ptr": "*fp32",
        "out_features": "i32",
        "in_features": "i32",
        "blocks_per_row": "i32",
        "bitstreams_bytes": "i64",
        "block_size": "i32",
        "step_bits": "i32",
        "num_wide_steps": "i32",
        "bytes_per_block": "i32",
        "slot_bytes": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "top_k": "i32",
        "split_k": "i32",
    }
    gemv_constexprs: dict[str, object] = {
        "tile_steps": tile_steps,
        "block_m": schedule.block_m,
        "block_n": schedule.block_n,
        "batched": batched,
        "has_slot_map": False if batched else has_slot_map,
        "is_splitk": is_splitk,
        "variable_rows": variable_rows,
        "has_slot_offsets": has_slot_offsets,
        "exact_pow2": exact_pow2,
        "needs_w2": needs_w2,
        "use_dp4a": schedule.use_dp4a,
        "use_sad_u8": schedule.use_sad_u8,
        "split_x_loads": schedule.split_x_loads,
        "ptr_aligned": True,
        "x_mul4": (block_size % 4 == 0),
        "unroll_k": schedule.unroll_k,
    }
    _bind_opt_ptrs(
        gemv_sig,
        gemv_constexprs,
        row_widths_ptr="*u8" if variable_rows else None,
        row_offsets_ptr="*i32" if variable_rows else None,
        slot_offsets_ptr="*i64" if has_slot_offsets else None,
        tile_slots_ptr="*i32" if batched else None,
        tile_pair_ids_ptr="*i32" if batched else None,
        topk_ids_ptr=None if batched else "*i64",
        expert_to_slot_ptr="*i64" if (not batched and has_slot_map) else None,
    )

    compiled: dict[str, triton.compiler.CompiledKernel] = {
        "trellis_gemv_up": compile_fn(
            ASTSource(
                trellis_moe_gemv_kernel,
                gemv_sig,
                constexprs={
                    **gemv_constexprs,
                    "has_tail_bytes": up_has_tail,
                    "per_expert_input": False,
                },
            ),
            target,
            {"num_warps": schedule.num_warps},
        ),
        "trellis_gemv_down": compile_fn(
            ASTSource(
                trellis_moe_gemv_kernel,
                gemv_sig,
                constexprs={
                    **gemv_constexprs,
                    "has_tail_bytes": down_has_tail,
                    "per_expert_input": True,
                    "is_splitk": True,
                },
            ),
            target,
            {"num_warps": schedule.num_warps},
        ),
    }
    _compile_a16_epilogue(
        compiled,
        target,
        accel,
        reduce_key="trellis_reduce",
        active_ty="*fp32",
        out_ty=out_ty,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        has_slot_map=has_slot_map,
        has_global_scale=False,
        post_scale_only=True,
        has_row_scale=True,
        is_splitk=is_splitk,
        accumulate=accumulate,
        compile_fn=compile_fn,
    )
    return compiled


def compile_moe_kernels(
    target: GPUTarget,
    *,
    weight_format: str = "nvfp4_w4a16",
    logical_dtype: torch.dtype = torch.bfloat16,
    rows: int = 1,
    top_k: int = 1,
    num_slots: int = 64,
    active_slots_hint: int = 0,
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
    block_size: int = 256,
    variable_rows: bool = False,
    has_slot_offsets: bool = False,
    bytes_per_block: int | None = None,
    compiler: TritonCompileFn | None = None,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile quantized MoE Triton kernels matching runtime regime and chunking selection."""
    if logical_dtype not in TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported MoE logical dtype: {logical_dtype}")
    if (
        hidden_size <= 0
        or intermediate_size <= 0
        or split_k <= 0
        or rows <= 0
        or top_k <= 0
        or num_slots <= 0
        or active_slots_hint < 0
    ):
        raise ValueError("MoE dimensions, rows, top_k, num_slots, and split_k must be positive")
    parse_input_activation(activation_fn, name="MoE")
    accel = AcceleratorTarget.from_compiler_target(target)
    _validate_compile_format(
        accel,
        weight_format=weight_format,
        group_size=group_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        high_first=high_first,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
        bytes_per_block=bytes_per_block,
    )

    up_gate_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    is_int8 = weight_format == "convrot_int8"
    is_trellis = weight_format == "convrot_trellis"
    compile_fn = compiler or default_triton_compile
    out_ty = TRITON_FLOAT_POINTER_TYPES[logical_dtype]

    has_int8_backend = is_int8
    if is_trellis:
        from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

        has_int8_backend = (
            not (accel.is_nvidia_cuda and not accel.cuda_capability_at_least(8))
            and int8_backend_mod.select_linear_backend_for_target(accel) is not None
        )

    bytes_per_pair = _prefill_bytes_per_pair(
        is_int8=has_int8_backend,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=logical_dtype.itemsize,
        group_size=group_size,
    )
    regime, chunk_rows = _select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill" and has_int8_backend:
        if is_trellis:
            return _compile_trellis_prefill_kernels(
                target,
                accel,
                rows=chunk_rows,
                top_k=top_k,
                num_slots=num_slots,
                active_slots_hint=active_slots_hint,
                out_ty=out_ty,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                block_size=block_size,
                variable_rows=variable_rows,
                has_slot_offsets=has_slot_offsets,
                bytes_per_block=bytes_per_block,
                activation_fn=activation_fn,
                gate_first=gate_first,
                has_slot_map=has_slot_map,
                accumulate=accumulate,
                compile_fn=compile_fn,
            )
        return _compile_int8_prefill_kernels(
            target,
            accel,
            rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            compile_fn=compile_fn,
        )
    if regime == "prefill" and not is_trellis:
        return _compile_nvfp4_prefill_kernels(
            target,
            accel,
            rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            logical_dtype=logical_dtype,
            out_ty=out_ty,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            has_slot_map=has_slot_map,
            has_global_scale=has_global_scale,
            accumulate=accumulate,
            compile_fn=compile_fn,
        )

    if regime == "prefill":
        a16_rows = min(rows, _BATCH_MAX_ROWS, max(1, _BUCKET_MAX_PAIRS // max(1, top_k)))
        batched = a16_rows > 1
    else:
        a16_rows = chunk_rows
        batched = regime == "batched"
    num_pairs = a16_rows * top_k

    if is_trellis:
        compiled = _compile_trellis_kernels(
            target,
            accel,
            batched=batched,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_size=block_size,
            variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets,
            bytes_per_block=bytes_per_block,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            has_slot_map=has_slot_map,
            split_k=split_k,
            accumulate=accumulate,
            compile_fn=compile_fn,
            rows=a16_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
        )
    elif is_int8:
        compiled = _compile_int8_kernels(
            target,
            accel,
            batched=batched,
            num_pairs=num_pairs,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            has_slot_map=has_slot_map,
            split_k=split_k,
            accumulate=accumulate,
            compile_fn=compile_fn,
        )
    else:
        compiled = _compile_nvfp4_kernels(
            target,
            accel,
            batched=batched,
            rows=chunk_rows,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            logical_dtype=logical_dtype,
            out_ty=out_ty,
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
    if batched:
        compiled["bucket"] = _compile_bucket_kernel(
            target,
            num_pairs=num_pairs,
            has_slot_map=has_slot_map,
            compile_fn=compile_fn,
        )
    return compiled


__all__ = [
    "compile_moe_kernels",
    "int8_moe_batched_matmul_kernel",
    "int8_moe_grouped_gemm_kernel",
    "int8_moe_m1_matmul_kernel",
    "moe_bucket_tokens_kernel",
    "moe_ffn",
    "moe_m1_activate_rotate_kernel",
    "moe_m1_weighted_reduce_kernel",
    "moe_prefill_bucket_count_kernel",
    "moe_prefill_bucket_scatter_kernel",
    "moe_prefill_gather_kernel",
    "moe_splitk_reduce_kernel",
    "nvfp4_moe_batched_gemv_kernel",
    "nvfp4_moe_grouped_gemm_kernel",
    "nvfp4_moe_m1_gemv_kernel",
    "trellis_moe_gemv_kernel",
]
