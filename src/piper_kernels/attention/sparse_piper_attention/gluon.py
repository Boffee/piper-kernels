"""Paired-K128 Gluon kernel for sparse Piper Attention on SM120."""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook

from .triton import _PreparedSparsePiperAttention

_BLOCK_M = 64
_BLOCK_N = 64
_HEAD_DIM = 128
_LOG2_255 = 7.994353436858858

_GL_BLOCK_M = gl.constexpr(_BLOCK_M)
_GL_BLOCK_N = gl.constexpr(_BLOCK_N)
_GL_HEAD_DIM = gl.constexpr(_HEAD_DIM)
_GL_LOG2_255 = gl.constexpr(_LOG2_255)
_GL_VALUE_LOG_BOUND_CORRECTION = gl.constexpr(0.086085)


@gluon.jit
def _mark_uint8_int8_dot(value):
    return gl.inline_asm_elementwise(
        asm="piper_attention_u8s8_dot_marker $0, $1;",
        constraints="=r,r",
        args=[value],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _uint8_int8_mma(lhs, rhs, accumulator):
    gl.static_assert(lhs.dtype == gl.uint8, "lhs must be UINT8")
    gl.static_assert(rhs.dtype == gl.int8, "rhs must be INT8")
    lhs_bits = lhs.to(gl.int8, bitcast=True)
    return _mark_uint8_int8_dot(mma_v2(lhs_bits, rhs, accumulator))


@gluon.jit
def _packed_float32_to_uint8(values):
    return gl.inline_asm_elementwise(
        asm="""
        {
            .reg .s32 a, b, c, d;
            .reg .b32 lo;
            cvt.rzi.s32.f32 a, $1;
            cvt.rzi.s32.f32 b, $2;
            cvt.rzi.s32.f32 c, $3;
            cvt.rzi.s32.f32 d, $4;
            cvt.pack.sat.u8.s32.b32 lo, d, c, 0;
            cvt.pack.sat.u8.s32.b32 $0, b, a, lo;
        }
        """,
        constraints="=r,f,f,f,f",
        args=[values],
        dtype=gl.uint8,
        is_pure=True,
        pack=4,
    )


@gluon.jit
def _fma_fp32(lhs, rhs, addend):
    return gl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[lhs, rhs, addend],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _issue_tma(descriptor, offsets, shared, barrier):
    mbarrier.expect(barrier, descriptor.block_type.nbytes)
    tma.async_copy_global_to_shared(descriptor, offsets, barrier, shared)


@gluon.jit
def _issue_tma_pair(
    descriptor,
    offsets_0,
    offsets_1,
    shared_0,
    shared_1,
    barrier,
):
    mbarrier.expect(barrier, descriptor.block_type.nbytes * 2)
    tma.async_copy_global_to_shared(descriptor, offsets_0, barrier, shared_0)
    tma.async_copy_global_to_shared(descriptor, offsets_1, barrier, shared_1)


@gluon.jit
def _piper_probability_pair(
    query,
    key_shared_0,
    key_shared_1,
    query_scale,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    denominator,
    running_max,
    batch_head,
    start_n_0,
    start_n_1,
    has_second,
    sequence_tiles,
    logical_sequence_length,
    mma_layout: gl.constexpr,
    key_layout: gl.constexpr,
    probability_layout: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    mask_duplicate: gl.constexpr,
):
    """Advance one shared Piper coordinate over two independently scaled K64 tiles."""
    key_0 = key_shared_0.permute([1, 0]).load(key_layout)
    key_1 = key_shared_1.permute([1, 0]).load(key_layout)
    integer_scores_0 = mma_v2(
        query,
        key_0,
        gl.zeros([_GL_BLOCK_M, _GL_BLOCK_N], gl.int32, mma_layout),
    )
    integer_scores_1 = mma_v2(
        query,
        key_1,
        gl.zeros([_GL_BLOCK_M, _GL_BLOCK_N], gl.int32, mma_layout),
    )
    key_scale_0 = gl.load(key_scale_ptr + batch_head * sequence_tiles + start_n_0 // _GL_BLOCK_N)
    key_scale_1 = gl.load(key_scale_ptr + batch_head * sequence_tiles + start_n_1 // _GL_BLOCK_N)
    scores_0 = integer_scores_0.to(gl.float32) * (query_scale[:, None] * key_scale_0)
    scores_1 = integer_scores_1.to(gl.float32) * (query_scale[:, None] * key_scale_1)
    if mask_ragged_tail:
        column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
        offsets_n = gl.arange(0, _GL_BLOCK_N, column_layout)
        valid_keys_0 = start_n_0 + offsets_n < logical_sequence_length
        valid_keys_1 = start_n_1 + offsets_n < logical_sequence_length
        if mask_duplicate:
            valid_keys_1 &= has_second
        scores_0 = gl.where(valid_keys_0[None, :], scores_0, -float("inf"))
        scores_1 = gl.where(valid_keys_1[None, :], scores_1, -float("inf"))
    elif mask_duplicate:
        scores_1 = gl.where(has_second, scores_1, -float("inf"))

    value_scale_multiplier_0 = gl.load(
        value_scale_multiplier_ptr + batch_head * sequence_tiles + start_n_0 // _GL_BLOCK_N
    ).to(gl.float32)
    value_scale_multiplier_1 = gl.load(
        value_scale_multiplier_ptr + batch_head * sequence_tiles + start_n_1 // _GL_BLOCK_N
    ).to(gl.float32)
    multiplier_bits_0 = value_scale_multiplier_0.to(gl.int32, bitcast=True)
    multiplier_bits_1 = value_scale_multiplier_1.to(gl.int32, bitcast=True)
    value_log_scale_0 = multiplier_bits_0.to(gl.float32) * (1.0 / 8388608.0) - (
        127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION
    )
    value_log_scale_1 = multiplier_bits_1.to(gl.float32) * (1.0 / 8388608.0) - (
        127.0 + _GL_LOG2_255 - _GL_VALUE_LOG_BOUND_CORRECTION
    )
    block_max = gl.maximum(
        gl.max(scores_0, axis=1) + value_log_scale_0,
        gl.max(scores_1, axis=1) + value_log_scale_1,
    )
    next_max = gl.maximum(running_max, block_max)
    old_weight = gl.exp2(running_max - next_max)
    current_weight = gl.exp2(block_max - next_max)
    probabilities_0 = gl.exp2(scores_0 - block_max[:, None])
    probability_uint8_0 = _packed_float32_to_uint8(probabilities_0 * value_scale_multiplier_0 + 0.5)
    probabilities_1 = gl.exp2(scores_1 - block_max[:, None])
    probability_uint8_1 = _packed_float32_to_uint8(probabilities_1 * value_scale_multiplier_1 + 0.5)
    probability_uint8_0 = gl.convert_layout(probability_uint8_0, probability_layout)
    probability_uint8_1 = gl.convert_layout(probability_uint8_1, probability_layout)
    probability_sum_0 = gl.sum(probabilities_0, axis=1)
    probability_sum_1 = gl.sum(probabilities_1, axis=1)
    denominator = (
        denominator * old_weight + (probability_sum_0 + probability_sum_1) * current_weight
    )
    return (
        probability_uint8_0,
        probability_uint8_1,
        denominator,
        next_max,
        old_weight,
        current_weight,
    )


@gluon.jit
def _piper_pv_pair(
    probability_uint8_0,
    probability_uint8_1,
    value_shared_0,
    value_shared_1,
    accumulator,
    old_weight,
    current_weight,
    mma_layout: gl.constexpr,
    value_layout: gl.constexpr,
):
    """Combine two full-D128 integer PV partials under one numerator weight."""
    value_0 = value_shared_0.permute([1, 0]).load(value_layout)
    value_1 = value_shared_1.permute([1, 0]).load(value_layout)
    partial_0 = _uint8_int8_mma(
        probability_uint8_0,
        value_0,
        gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.int32, mma_layout),
    )
    partial_1 = _uint8_int8_mma(
        probability_uint8_1,
        value_1,
        gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.int32, mma_layout),
    )
    return _fma_fp32(
        (partial_0 + partial_1).to(gl.float32),
        current_weight[:, None],
        accumulator * old_weight[:, None],
    )


@gluon.jit
def _native_tile_start(
    route_base,
    tile_position,
    selected_sparse_tile_count,
    sparse_key_blocks,
    stride_rr,
):
    safe_route_position = gl.minimum(tile_position, selected_sparse_tile_count - 1)
    route = gl.load(route_base + safe_route_position * stride_rr).to(gl.int32)
    sparse_start = route * _GL_BLOCK_N
    dense_start = (
        sparse_key_blocks * _GL_BLOCK_N + (tile_position - selected_sparse_tile_count) * _GL_BLOCK_N
    )
    return gl.where(tile_position < selected_sparse_tile_count, sparse_start, dense_start)


@gluon.jit(
    do_not_specialize=[
        "logical_sequence_length",
        "sparse_key_blocks",
        "stride_rb",
        "stride_rq",
    ]
)
def _sparse_piper_attention_kernel(
    query_desc,
    key_desc,
    value_desc,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_mean_ptr,
    routes_ptr,
    keep_blocks_ptr,
    route_head_offsets_ptr,
    output_ptr,
    storage_sequence_length,
    logical_sequence_length,
    sparse_key_blocks,
    stride_rb,
    stride_rq,
    stride_rr,
    stride_ob,
    stride_oh,
    stride_on,
    heads,
    mask_ragged_tail: gl.constexpr,
    kernel_warps: gl.constexpr,
):
    """Pair native logical K64 tiles in one shared Piper probability coordinate."""
    query_block = gl.program_id(0)
    head = gl.program_id(1)
    batch = gl.program_id(2)
    batch_head = batch * heads + head
    start_m = query_block * _GL_BLOCK_M
    route_head_offset = gl.load(route_head_offsets_ptr + head)
    route_base = (
        routes_ptr + batch * stride_rb + query_block * stride_rq + route_head_offset * stride_rr
    )

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[kernel_warps, 1],
        instr_shape=[16, 8],
    )
    query_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    key_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    column_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)

    query_shared = gl.allocate_shared_memory(
        query_desc.dtype, [_GL_BLOCK_M, _GL_HEAD_DIM], query_desc.layout
    )
    key_shared_0 = gl.allocate_shared_memory(
        key_desc.dtype, [_GL_BLOCK_N, _GL_HEAD_DIM], key_desc.layout
    )
    key_shared_1 = gl.allocate_shared_memory(
        key_desc.dtype, [_GL_BLOCK_N, _GL_HEAD_DIM], key_desc.layout
    )
    value_shared_0 = gl.allocate_shared_memory(
        value_desc.dtype, [_GL_HEAD_DIM, _GL_BLOCK_N], value_desc.layout
    )
    value_shared_1 = gl.allocate_shared_memory(
        value_desc.dtype, [_GL_HEAD_DIM, _GL_BLOCK_N], value_desc.layout
    )
    query_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    key_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    value_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(query_barrier, count=1)
    mbarrier.init(key_barrier, count=1)
    mbarrier.init(value_barrier, count=1)

    selected_sparse_tile_count = gl.load(keep_blocks_ptr + head)
    sequence_tiles = storage_sequence_length // _GL_BLOCK_N
    dense_tile_count = sequence_tiles - sparse_key_blocks
    tile_count = selected_sparse_tile_count + dense_tile_count
    pair_count = gl.cdiv(tile_count, 2)
    initial_position_1 = gl.minimum(1, tile_count - 1)
    initial_n_0 = _native_tile_start(
        route_base,
        0,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
    )
    initial_n_1 = _native_tile_start(
        route_base,
        initial_position_1,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
    )

    _issue_tma(
        query_desc,
        [batch_head * storage_sequence_length + start_m, 0],
        query_shared,
        query_barrier,
    )
    _issue_tma_pair(
        key_desc,
        [batch_head * storage_sequence_length + initial_n_0, 0],
        [batch_head * storage_sequence_length + initial_n_1, 0],
        key_shared_0,
        key_shared_1,
        key_barrier,
    )
    _issue_tma_pair(
        value_desc,
        [batch_head * _GL_HEAD_DIM, initial_n_0],
        [batch_head * _GL_HEAD_DIM, initial_n_1],
        value_shared_0,
        value_shared_1,
        value_barrier,
    )
    mbarrier.wait(query_barrier, phase=0)
    query = query_shared.load(query_layout)

    offsets_m = gl.arange(0, _GL_BLOCK_M, row_layout)
    query_scale_stride = storage_sequence_length // 32
    query_scale = gl.load(
        query_scale_ptr + batch_head * query_scale_stride + (start_m + offsets_m) // 32
    )
    accumulator = gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.float32, mma_layout)
    denominator = gl.zeros([_GL_BLOCK_M], gl.float32, row_layout)
    running_max = gl.full([_GL_BLOCK_M], -float("inf"), gl.float32, row_layout)
    start_n_0 = initial_n_0
    start_n_1 = initial_n_1

    for pair_index in range(pair_count - 1):
        phase = pair_index & 1
        tile_position_0 = pair_index * 2
        mbarrier.wait(key_barrier, phase=phase)
        (
            probability_0,
            probability_1,
            denominator,
            running_max,
            old_weight,
            current_weight,
        ) = _piper_probability_pair(
            query,
            key_shared_0,
            key_shared_1,
            query_scale,
            key_scale_ptr,
            value_scale_multiplier_ptr,
            denominator,
            running_max,
            batch_head,
            start_n_0,
            start_n_1,
            True,
            sequence_tiles,
            logical_sequence_length,
            mma_layout,
            key_layout,
            probability_layout,
            mask_ragged_tail,
            False,
        )

        next_position_0 = tile_position_0 + 2
        next_position_1 = gl.minimum(next_position_0 + 1, tile_count - 1)
        next_n_0 = _native_tile_start(
            route_base,
            next_position_0,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
        )
        next_n_1 = _native_tile_start(
            route_base,
            next_position_1,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
        )
        gl.barrier()
        _issue_tma_pair(
            key_desc,
            [batch_head * storage_sequence_length + next_n_0, 0],
            [batch_head * storage_sequence_length + next_n_1, 0],
            key_shared_0,
            key_shared_1,
            key_barrier,
        )
        mbarrier.wait(value_barrier, phase=phase)
        accumulator = _piper_pv_pair(
            probability_0,
            probability_1,
            value_shared_0,
            value_shared_1,
            accumulator,
            old_weight,
            current_weight,
            mma_layout,
            value_layout,
        )
        gl.barrier()
        _issue_tma_pair(
            value_desc,
            [batch_head * _GL_HEAD_DIM, next_n_0],
            [batch_head * _GL_HEAD_DIM, next_n_1],
            value_shared_0,
            value_shared_1,
            value_barrier,
        )
        start_n_0 = next_n_0
        start_n_1 = next_n_1

    final_pair = pair_count - 1
    final_phase = final_pair & 1
    final_position_0 = final_pair * 2
    has_second = final_position_0 + 1 < tile_count
    mbarrier.wait(key_barrier, phase=final_phase)
    (
        probability_0,
        probability_1,
        denominator,
        running_max,
        old_weight,
        current_weight,
    ) = _piper_probability_pair(
        query,
        key_shared_0,
        key_shared_1,
        query_scale,
        key_scale_ptr,
        value_scale_multiplier_ptr,
        denominator,
        running_max,
        batch_head,
        start_n_0,
        start_n_1,
        has_second,
        sequence_tiles,
        logical_sequence_length,
        mma_layout,
        key_layout,
        probability_layout,
        mask_ragged_tail,
        True,
    )
    mbarrier.wait(value_barrier, phase=final_phase)
    accumulator = _piper_pv_pair(
        probability_0,
        probability_1,
        value_shared_0,
        value_shared_1,
        accumulator,
        old_weight,
        current_weight,
        mma_layout,
        value_layout,
    )

    offsets_d = gl.arange(0, _GL_HEAD_DIM, column_layout)
    output = accumulator / (gl.maximum(denominator, 1e-30) * 255.0)[:, None]
    value_mean = gl.load(value_mean_ptr + batch_head * _GL_HEAD_DIM + offsets_d).to(gl.float32)
    output += value_mean[None, :]
    output_offsets = (
        batch * stride_ob
        + head * stride_oh
        + (start_m + offsets_m[:, None]) * stride_on
        + offsets_d[None, :]
    )
    if mask_ragged_tail:
        valid_queries = start_m + offsets_m < logical_sequence_length
        gl.store(
            output_ptr + output_offsets,
            output.to(gl.bfloat16),
            mask=valid_queries[:, None],
        )
    else:
        gl.store(output_ptr + output_offsets, output.to(gl.bfloat16))

    mbarrier.invalidate(query_barrier)
    mbarrier.invalidate(key_barrier)
    mbarrier.invalidate(value_barrier)


def _make_gluon_descriptors(
    prepared: _PreparedSparsePiperAttention,
) -> tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor]:
    query = prepared.query
    key = prepared.key
    value = prepared.value
    query_layout = gl.NVMMASharedLayout.get_default_for([_BLOCK_M, _HEAD_DIM], gl.int8)
    key_layout = gl.NVMMASharedLayout.get_default_for([_BLOCK_N, _HEAD_DIM], gl.int8)
    value_layout = gl.NVMMASharedLayout.get_default_for([_HEAD_DIM, _BLOCK_N], gl.int8)
    batch_heads = int(query.shape[0] * query.shape[1])
    storage_sequence_length = int(query.shape[2])
    if key.shape != query.shape or value.shape != (
        query.shape[0],
        query.shape[1],
        query.shape[3],
        storage_sequence_length,
    ):
        raise ValueError("paired Gluon routed Piper requires equal Q/K/V sequence lengths")
    return (
        TensorDescriptor(
            query,
            [batch_heads * storage_sequence_length, _HEAD_DIM],
            [_HEAD_DIM, 1],
            [_BLOCK_M, _HEAD_DIM],
            query_layout,
        ),
        TensorDescriptor(
            key,
            [batch_heads * storage_sequence_length, _HEAD_DIM],
            [_HEAD_DIM, 1],
            [_BLOCK_N, _HEAD_DIM],
            key_layout,
        ),
        TensorDescriptor(
            value,
            [batch_heads * _HEAD_DIM, storage_sequence_length],
            [storage_sequence_length, 1],
            [_HEAD_DIM, _BLOCK_N],
            value_layout,
        ),
    )


def _launch_sparse_piper_attention(
    prepared: _PreparedSparsePiperAttention,
) -> None:
    """Launch paired K128 while retaining logical K64 routes and tile scales."""
    batch, heads, logical_sequence_length, head_dim = prepared.output.shape
    storage_sequence_length = prepared.query.shape[2]
    if (
        head_dim != _HEAD_DIM
        or storage_sequence_length % _BLOCK_M
        or (logical_sequence_length + _BLOCK_M - 1) // _BLOCK_M * _BLOCK_M
        != storage_sequence_length
    ):
        raise ValueError("paired Gluon routed Piper requires padded M64/D128 query storage")
    if prepared.value_scale_multiplier.shape[-1] != 1:
        raise ValueError("paired Gluon routed Piper requires one folded scale per K64 tile")
    with torch.cuda.device(prepared.query.device):
        install_uint8_int8_dot_hook()

    query_desc, key_desc, value_desc = _make_gluon_descriptors(prepared)
    routes = prepared.routes
    route_head_offsets = prepared.route_head_offsets
    stride_rb = routes.stride(0)
    stride_rq = routes.stride(1)
    stride_rr = routes.stride(2)
    _sparse_piper_attention_kernel[(storage_sequence_length // _BLOCK_M, heads, batch)](
        query_desc,
        key_desc,
        value_desc,
        prepared.query_scale,
        prepared.key_scale,
        prepared.value_scale_multiplier,
        prepared.value_mean,
        routes,
        prepared.keep_blocks,
        route_head_offsets,
        prepared.output,
        storage_sequence_length,
        logical_sequence_length,
        prepared.sparse_key_blocks,
        stride_rb,
        stride_rq,
        stride_rr,
        prepared.output.stride(0),
        prepared.output.stride(1),
        prepared.output.stride(2),
        heads,
        logical_sequence_length != storage_sequence_length,
        4,
        num_warps=4,
        num_stages=1,
    )
