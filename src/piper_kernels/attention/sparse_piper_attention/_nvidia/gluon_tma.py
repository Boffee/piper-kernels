"""Paired-K128 Gluon kernel for sparse Piper Attention with NVIDIA TMA loads.

On SM120 D128, V-mean restoration buffers 16 pairs (32 tiles) and flushes via
D64 BF16 tensor-core strips, reusing query shared memory after the initial load;
D64 keeps the exact per-pair FP32 scalar restoration path.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from __future__ import annotations

from dataclasses import dataclass

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper.gluon import pair_tile_offsets
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS, TILE_ROWS

from .._launch import _DO_NOT_SPECIALIZE_ARGUMENTS, validate_attention_launch
from .._prepared import _PreparedSparsePiperContext, _PreparedSparsePiperQuery
from . import policy
from ._recurrence import (
    _VALUE_MEAN_GROUP_TILES,
    K_BIAS_GROUP_TILES,
    accumulate_value_mean_step,
    build_k_bias_cache,
    compute_pair_k_bias,
    piper_probability_pair,
    piper_pv_pair,
    store_attention_output,
)

_BLOCK_N = TILE_ROWS

_GL_BLOCK_N = gl.constexpr(_BLOCK_N)
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)


@gluon.jit
def _issue_tma(descriptor, offsets, shared, barrier):
    mbarrier.expect(barrier, descriptor.block_type.nbytes)
    tma.async_load(descriptor, offsets, barrier, shared)


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
    tma.async_load(descriptor, offsets_0, barrier, shared_0)
    tma.async_load(descriptor, offsets_1, barrier, shared_1)


@gluon.jit
def _schedule_program_coords(
    head,
    local_query_block,
    heads,
    global_query_block_offset,
    sparse_query_blocks,
    group_dense_suffix: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
):
    """Group dense-suffix CTAs first across heads so sparse waves finish together."""
    if group_dense_suffix and has_dense_query_suffix:
        query_blocks = gl.num_programs(0)
        sparse = gl.minimum(
            gl.maximum(sparse_query_blocks - global_query_block_offset, 0), query_blocks
        )
        dense = query_blocks - sparse
        linear = head * query_blocks + local_query_block
        dense_programs = heads * dense
        sparse_linear = linear - dense_programs
        in_dense = linear < dense_programs
        head = gl.where(
            in_dense,
            linear // gl.maximum(dense, 1),
            sparse_linear // gl.maximum(sparse, 1),
        )
        local_query_block = gl.where(
            in_dense,
            sparse + linear % gl.maximum(dense, 1),
            sparse_linear % gl.maximum(sparse, 1),
        )
    return head, local_query_block


@gluon.jit
def _resolve_sparse_route_counts(
    head,
    head_keep_blocks_ptr,
    global_query_block,
    sparse_key_blocks,
    sparse_query_blocks,
    skip_dense_routing: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
):
    """Determine selected KV tile count and routing flag for one CTA."""
    if skip_dense_routing:
        routed_sparse_tile_count = sparse_key_blocks
        selected_sparse_tile_count = sparse_key_blocks
        use_sparse_routes = False
    else:
        routed_sparse_tile_count = gl.load(head_keep_blocks_ptr + head)
        if has_dense_query_suffix:
            use_sparse_routes = global_query_block < sparse_query_blocks
            selected_sparse_tile_count = gl.where(
                use_sparse_routes,
                routed_sparse_tile_count,
                sparse_key_blocks,
            )
        else:
            use_sparse_routes = True
            selected_sparse_tile_count = routed_sparse_tile_count
    return routed_sparse_tile_count, selected_sparse_tile_count, use_sparse_routes


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_ARGUMENTS)
def _sparse_piper_attention_kernel(
    query_desc,
    key_desc,
    value_desc,
    query_scale_ptr,
    key_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    value_scale_multiplier_ptr,
    value_mean_ptr,
    coarse_output_ptr,
    coarse_gate_ptr,
    block_lengths_ptr,
    routes_ptr,
    head_keep_blocks_ptr,
    route_head_offsets_ptr,
    output_ptr,
    query_block_offset,
    global_query_block_offset,
    query_storage_sequence_length,
    storage_sequence_length,
    logical_sequence_length,
    sparse_key_blocks,
    sparse_query_blocks,
    stride_rb,
    stride_rq,
    stride_rr,
    stride_ob,
    stride_oh,
    stride_on,
    stride_cb,
    stride_ch,
    stride_cq,
    stride_gb,
    stride_gh,
    stride_gn,
    heads,
    query_heads_per_kv_head: gl.constexpr,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
    block_m: gl.constexpr,
    mma_warps: gl.constexpr,
    skip_dense_routing: gl.constexpr,
    mask_output_tail: gl.constexpr,
    output_sequence_length,
):
    """Pair native logical K64 tiles in one shared Piper probability coordinate."""
    gl.static_assert(block_m == 64 or (block_m == 128 and head_dim == 64 and skip_dense_routing))
    gl.static_assert(mma_warps == 4 or (mma_warps == 2 and head_dim == 64 and block_m == 64))
    gl.static_assert(not apply_coarse_residual or block_m == 64)
    is_d128: gl.constexpr = head_dim == 128
    head, local_query_block = _schedule_program_coords(
        gl.program_id(1),
        gl.program_id(0) * (block_m // _GL_BLOCK_N),
        heads,
        global_query_block_offset,
        sparse_query_blocks,
        is_d128,
        has_dense_query_suffix,
    )
    query_block = query_block_offset + local_query_block
    global_query_block = global_query_block_offset + local_query_block
    batch = gl.program_id(2)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // query_heads_per_kv_head) + head // query_heads_per_kv_head
    start_m = query_block * _GL_BLOCK_N
    output_start_m = local_query_block * _GL_BLOCK_N
    if skip_dense_routing:  # noqa: SIM108
        route_head_offset = 0
    else:
        route_head_offset = gl.load(route_head_offsets_ptr + head)
    route_base = (
        routes_ptr + batch * stride_rb + query_block * stride_rq + route_head_offset * stride_rr
    )

    # The packed PV rescale is validated for each selected MMA register layout.
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[mma_warps, 1],
        instr_shape=[16, 8],
    )
    query_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    key_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)

    query_shared = gl.allocate_shared_memory(
        query_desc.dtype, [block_m, head_dim], query_desc.layout
    )
    key_shared_pair = gl.allocate_shared_memory(
        key_desc.dtype, [2 * _GL_BLOCK_N, head_dim], key_desc.layout
    )
    key_shared_0 = key_shared_pair.slice(0, _GL_BLOCK_N)
    key_shared_1 = key_shared_pair.slice(_GL_BLOCK_N, _GL_BLOCK_N)
    value_shared_pair = gl.allocate_shared_memory(
        value_desc.dtype, [2, head_dim, _GL_BLOCK_N], value_desc.layout
    )
    value_shared_0 = value_shared_pair.index(0)
    value_shared_1 = value_shared_pair.index(1)
    center_rows: gl.constexpr = (2 if is_d128 else 1) * K_BIAS_GROUP_TILES
    key_center_shared = gl.allocate_shared_memory(
        gl.int8,
        [center_rows, head_dim],
        gl.NVMMASharedLayout.get_default_for([center_rows, head_dim], gl.int8),
    )
    if is_d128:
        pair_max_shared = gl.allocate_shared_memory(
            gl.float32,
            [_GL_BLOCK_N, K_BIAS_GROUP_TILES],
            gl.SwizzledSharedLayout(1, 1, 1, [0, 1]),
        )
        tiles_shared = gl.allocate_shared_memory(
            gl.int32,
            [2, K_BIAS_GROUP_TILES],
            gl.SwizzledSharedLayout(1, 1, 1, [0]),
        )
    else:
        pair_max_shared = None
        tiles_shared = None
    query_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    key_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    value_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(query_barrier, count=1)
    mbarrier.init(key_barrier, count=1)
    mbarrier.init(value_barrier, count=1)
    gl.barrier()

    routed_sparse_tile_count, selected_sparse_tile_count, use_sparse_routes = (
        _resolve_sparse_route_counts(
            head,
            head_keep_blocks_ptr,
            global_query_block,
            sparse_key_blocks,
            sparse_query_blocks,
            skip_dense_routing,
            has_dense_query_suffix,
        )
    )
    sequence_tiles = storage_sequence_length // _GL_BLOCK_N
    dense_tile_count = sequence_tiles - sparse_key_blocks
    tile_count = selected_sparse_tile_count + dense_tile_count
    pair_count = gl.cdiv(tile_count, 2)
    initial_n_0, initial_n_1 = pair_tile_offsets(
        route_base,
        0,
        tile_count,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        _GL_BLOCK_N,
    )

    _issue_tma(
        query_desc,
        [batch_head * query_storage_sequence_length + start_m, 0],
        query_shared,
        query_barrier,
    )
    _issue_tma_pair(
        key_desc,
        [kv_batch_head * storage_sequence_length + initial_n_0, 0],
        [kv_batch_head * storage_sequence_length + initial_n_1, 0],
        key_shared_0,
        key_shared_1,
        key_barrier,
    )
    _issue_tma_pair(
        value_desc,
        [kv_batch_head * head_dim, initial_n_0],
        [kv_batch_head * head_dim, initial_n_1],
        value_shared_0,
        value_shared_1,
        value_barrier,
    )
    mbarrier.wait(query_barrier, phase=0)
    query = query_shared.load(query_layout)
    if is_d128:
        # Ensure all warps finish loading Q before reusing query_shared as FP32 mass scratch.
        gl.barrier()
        tile_masses_shared = query_shared.reinterpret(
            gl.float32,
            [_GL_BLOCK_N, _VALUE_MEAN_GROUP_TILES],
            gl.SwizzledSharedLayout(1, 1, 1, [0, 1]),
        )
        value_mean_workspace = (
            tile_masses_shared,
            pair_max_shared,
            tiles_shared,
            key_center_shared,
        )
    else:
        value_mean_workspace = None

    offsets_m = gl.arange(0, block_m, row_layout)
    query_scale_stride = query_storage_sequence_length // _GL_QUERY_SCALE_ROWS
    if block_m == 128:
        query_scale = gl.load(
            query_scale_ptr
            + batch_head * query_scale_stride
            + (start_m + offsets_m) // _GL_QUERY_SCALE_ROWS,
            mask=start_m + offsets_m < query_storage_sequence_length,
            other=0.0,
        )
    else:
        query_scale = gl.load(
            query_scale_ptr
            + batch_head * query_scale_stride
            + (start_m + offsets_m) // _GL_QUERY_SCALE_ROWS
        )
    accumulator = gl.zeros([block_m, head_dim], gl.float32, mma_layout)
    denominator = gl.zeros([block_m], gl.float32, row_layout)
    running_max = gl.full([block_m], -float("inf"), gl.float32, row_layout)
    bias_cache_0, bias_cache_1 = build_k_bias_cache(block_m, mma_layout)
    start_n_0 = initial_n_0
    start_n_1 = initial_n_1

    for pair_index in range(pair_count - 1):
        k_bias_0, k_bias_1, bias_cache_0, bias_cache_1 = compute_pair_k_bias(
            pair_index,
            bias_cache_0,
            bias_cache_1,
            query,
            query_scale,
            key_center_ptr,
            key_center_scale_ptr,
            key_center_shared,
            route_base,
            tile_count,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
            skip_dense_routing,
            kv_batch_head,
            sequence_tiles,
            head_dim,
            mma_warps,
            mma_layout,
            key_layout,
            row_layout,
            tiles_shared,
        )
        phase = pair_index & 1
        mbarrier.wait(key_barrier, phase=phase)
        (
            probability,
            denominator,
            running_max,
            old_weight,
            current_weight,
            mass_0,
            mass_1,
            block_max,
        ) = piper_probability_pair(
            query,
            key_shared_pair,
            query_scale,
            key_scale_ptr,
            value_scale_multiplier_ptr,
            block_lengths_ptr,
            k_bias_0,
            k_bias_1,
            denominator,
            running_max,
            kv_batch_head,
            start_n_0,
            start_n_1,
            True,
            sequence_tiles,
            logical_sequence_length,
            mma_layout,
            key_layout,
            probability_layout,
            mask_block_lengths,
            False,
            False,
            is_d128,
        )

        next_n_0, next_n_1 = pair_tile_offsets(
            route_base,
            pair_index + 1,
            tile_count,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
            skip_dense_routing,
            _GL_BLOCK_N,
        )
        gl.barrier()
        _issue_tma_pair(
            key_desc,
            [kv_batch_head * storage_sequence_length + next_n_0, 0],
            [kv_batch_head * storage_sequence_length + next_n_1, 0],
            key_shared_0,
            key_shared_1,
            key_barrier,
        )
        mbarrier.wait(value_barrier, phase=phase)
        accumulator = piper_pv_pair(
            probability,
            value_shared_pair,
            accumulator,
            old_weight,
            current_weight,
            mma_layout,
            value_layout,
            is_d128,
        )
        gl.barrier()
        _issue_tma_pair(
            value_desc,
            [kv_batch_head * head_dim, next_n_0],
            [kv_batch_head * head_dim, next_n_1],
            value_shared_0,
            value_shared_1,
            value_barrier,
        )
        accumulator = accumulate_value_mean_step(
            accumulator,
            value_mean_workspace,
            running_max,
            value_mean_ptr,
            mass_0,
            mass_1,
            block_max,
            current_weight,
            kv_batch_head,
            start_n_0,
            start_n_1,
            sequence_tiles,
            pair_index,
            tile_count,
            True,
            mma_layout,
        )
        start_n_0 = next_n_0
        start_n_1 = next_n_1

    final_pair = pair_count - 1
    k_bias_0, k_bias_1, _bias_cache_0, _bias_cache_1 = compute_pair_k_bias(
        final_pair,
        bias_cache_0,
        bias_cache_1,
        query,
        query_scale,
        key_center_ptr,
        key_center_scale_ptr,
        key_center_shared,
        route_base,
        tile_count,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        kv_batch_head,
        sequence_tiles,
        head_dim,
        mma_warps,
        mma_layout,
        key_layout,
        row_layout,
        tiles_shared,
    )
    final_phase = final_pair & 1
    final_position_0 = final_pair * 2
    has_second = final_position_0 + 1 < tile_count
    mbarrier.wait(key_barrier, phase=final_phase)
    (
        probability,
        denominator,
        running_max,
        old_weight,
        current_weight,
        mass_0,
        mass_1,
        block_max,
    ) = piper_probability_pair(
        query,
        key_shared_pair,
        query_scale,
        key_scale_ptr,
        value_scale_multiplier_ptr,
        block_lengths_ptr,
        k_bias_0,
        k_bias_1,
        denominator,
        running_max,
        kv_batch_head,
        start_n_0,
        start_n_1,
        has_second,
        sequence_tiles,
        logical_sequence_length,
        mma_layout,
        key_layout,
        probability_layout,
        mask_block_lengths,
        mask_ragged_tail,
        True,
        is_d128,
    )
    mbarrier.wait(value_barrier, phase=final_phase)
    accumulator = piper_pv_pair(
        probability,
        value_shared_pair,
        accumulator,
        old_weight,
        current_weight,
        mma_layout,
        value_layout,
        is_d128,
    )
    accumulator = accumulate_value_mean_step(
        accumulator,
        value_mean_workspace,
        running_max,
        value_mean_ptr,
        mass_0,
        mass_1,
        block_max,
        current_weight,
        kv_batch_head,
        start_n_0,
        start_n_1,
        sequence_tiles,
        final_pair,
        tile_count,
        False,
        mma_layout,
    )

    store_attention_output(
        accumulator,
        denominator,
        coarse_output_ptr,
        coarse_gate_ptr,
        output_ptr,
        batch,
        head,
        query_block,
        global_query_block,
        output_start_m,
        offsets_m,
        logical_sequence_length,
        output_sequence_length,
        stride_ob,
        stride_oh,
        stride_on,
        stride_cb,
        stride_ch,
        stride_cq,
        stride_gb,
        stride_gh,
        stride_gn,
        mma_layout,
        mask_ragged_tail,
        apply_coarse_residual,
        mask_output_tail,
    )

    # Every warp must finish its final waits before any warp invalidates the
    # shared barriers.
    gl.barrier()
    mbarrier.invalidate(query_barrier)
    mbarrier.invalidate(key_barrier)
    mbarrier.invalidate(value_barrier)


def _descriptor(data: torch.Tensor, shape: list[int], block_shape: list[int]) -> TensorDescriptor:
    with device_context(data.device):
        return TensorDescriptor(
            data,
            shape,
            [shape[1], 1],
            block_shape,
            gl.NVMMASharedLayout.get_default_for(block_shape, gl.int8),
        )


def bind_context(context: _PreparedSparsePiperContext) -> _ContextLauncher:
    """Prepare K/V descriptors once; Q descriptors belong to each query window."""
    if context.routes_per_query == 0 and context.key.shape[-1] != 64:
        raise ValueError("skip_dense_routing requires NVIDIA D64 attention")
    batch, heads, sequence, head_dim = context.key.shape
    return _ContextLauncher(
        context,
        _descriptor(context.key, [batch * heads * sequence, head_dim], [_BLOCK_N, head_dim]),
        _descriptor(context.value, [batch * heads * head_dim, sequence], [head_dim, _BLOCK_N]),
    )


@dataclass(frozen=True, slots=True)
class _ContextLauncher:
    context: _PreparedSparsePiperContext
    key_descriptor: TensorDescriptor
    value_descriptor: TensorDescriptor

    def __call__(
        self,
        query_state: _PreparedSparsePiperQuery,
        output: torch.Tensor,
        *,
        query_block_offset: int = 0,
        query_block_count: int | None = None,
        coarse_output: torch.Tensor | None = None,
        coarse_gate: torch.Tensor | None = None,
    ) -> None:
        """Execute a query window against the bound global context."""
        context = self.context
        launch = validate_attention_launch(
            context,
            query_state,
            output,
            query_block_offset,
            query_block_count,
            coarse_output,
            coarse_gate,
        )
        block_m, num_warps = policy.select_sm120_attention_schedule(
            launch.head_dim,
            launch.query_rows,
            launch.storage_sequence_length,
            skip_dense_routing=launch.skip_dense_routing,
            has_coarse_residual=launch.apply_coarse_residual,
            selected_key_rows=(
                context.routes_per_query // launch.heads * _BLOCK_N
                + launch.storage_sequence_length
                - launch.sparse_key_blocks * _BLOCK_N
            ),
        )
        query_desc = _descriptor(
            query_state.data,
            [launch.batch * launch.heads * launch.query_storage_sequence_length, launch.head_dim],
            [block_m, launch.head_dim],
        )
        with device_context(output.device):
            install_uint8_int8_dot_hook()
            grid = (
                (launch.query_rows + block_m - 1) // block_m,
                launch.heads,
                launch.batch,
            )
            _sparse_piper_attention_kernel[grid](
                query_desc,
                self.key_descriptor,
                self.value_descriptor,
                query_state.scale,
                context.key_scale,
                context.key_center,
                context.key_center_scale,
                context.value_scale_multiplier,
                context.value_mean,
                launch.coarse_output,
                launch.coarse_gate,
                launch.block_lengths,
                query_state.routes,
                context.head_keep_blocks,
                context.route_head_offsets,
                output,
                launch.query_block_offset,
                launch.global_query_block_offset,
                launch.query_storage_sequence_length,
                launch.storage_sequence_length,
                launch.logical_sequence_length,
                launch.sparse_key_blocks,
                launch.sparse_query_blocks,
                *launch.route_strides,
                *launch.output_strides,
                *launch.coarse_strides,
                *launch.gate_strides,
                launch.heads,
                launch.heads // context.key.shape[1],
                launch.head_dim,
                launch.mask_block_lengths,
                launch.mask_ragged_tail,
                launch.has_dense_query_suffix,
                launch.apply_coarse_residual,
                block_m,
                num_warps,
                launch.skip_dense_routing,
                launch.output_sequence_length % block_m != 0,
                launch.output_sequence_length,
                num_warps=num_warps,
                num_stages=1,
            )
