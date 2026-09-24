"""Paired-K128 Gluon kernel for sparse Piper Attention on SM120."""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper.gluon import tile_offset
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS, TILE_ROWS

from .._launch import _DO_NOT_SPECIALIZE_ARGUMENTS, validate_attention_launch
from .._prepared import _PreparedSparsePiperAttention
from . import policy
from ._recurrence import piper_probability_pair, piper_pv_pair, store_attention_output

_BLOCK_N = TILE_ROWS

_GL_BLOCK_N = gl.constexpr(_BLOCK_N)
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)


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


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_ARGUMENTS)
def _sparse_piper_attention_kernel(
    query_desc,
    key_desc,
    value_desc,
    query_scale_ptr,
    key_scale_ptr,
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
    head_groups: gl.constexpr,
    head_dim: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_ragged_tail: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
    ragged_tail_is_routed: gl.constexpr,
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
    local_query_block = gl.program_id(0) * (block_m // _GL_BLOCK_N)
    query_block = query_block_offset + local_query_block
    global_query_block = global_query_block_offset + local_query_block
    head = gl.program_id(1)
    batch = gl.program_id(2)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // head_groups) + head // head_groups
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
    query_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    key_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    value_barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(query_barrier, count=1)
    mbarrier.init(key_barrier, count=1)
    mbarrier.init(value_barrier, count=1)
    gl.barrier()

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
    sequence_tiles = storage_sequence_length // _GL_BLOCK_N
    dense_tile_count = sequence_tiles - sparse_key_blocks
    tile_count = selected_sparse_tile_count + dense_tile_count
    pair_count = gl.cdiv(tile_count, 2)
    initial_position_1 = gl.minimum(1, tile_count - 1)
    initial_n_0 = tile_offset(
        route_base,
        0,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        _GL_BLOCK_N,
    )
    initial_n_1 = tile_offset(
        route_base,
        initial_position_1,
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
    start_n_0 = initial_n_0
    start_n_1 = initial_n_1

    for pair_index in range(pair_count - 1):
        phase = pair_index & 1
        tile_position_0 = pair_index * 2
        mbarrier.wait(key_barrier, phase=phase)
        (
            probability,
            denominator,
            running_max,
            old_weight,
            current_weight,
        ) = piper_probability_pair(
            query,
            key_shared_pair,
            query_scale,
            key_scale_ptr,
            value_scale_multiplier_ptr,
            block_lengths_ptr,
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
            mask_ragged_tail and ragged_tail_is_routed,
            False,
        )

        next_position_0 = tile_position_0 + 2
        next_position_1 = gl.minimum(next_position_0 + 1, tile_count - 1)
        next_n_0 = tile_offset(
            route_base,
            next_position_0,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
            skip_dense_routing,
            _GL_BLOCK_N,
        )
        next_n_1 = tile_offset(
            route_base,
            next_position_1,
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
        start_n_0 = next_n_0
        start_n_1 = next_n_1

    final_pair = pair_count - 1
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
    ) = piper_probability_pair(
        query,
        key_shared_pair,
        query_scale,
        key_scale_ptr,
        value_scale_multiplier_ptr,
        block_lengths_ptr,
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
    )

    store_attention_output(
        accumulator,
        denominator,
        value_mean_ptr,
        coarse_output_ptr,
        coarse_gate_ptr,
        output_ptr,
        batch,
        head,
        kv_batch_head,
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


def _make_gluon_descriptors(
    prepared: _PreparedSparsePiperAttention,
    block_m: int = TILE_ROWS,
) -> tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor]:
    query = prepared.query.data
    key = prepared.context.key
    value = prepared.context.value
    head_dim = query.shape[-1]
    query_layout = gl.NVMMASharedLayout.get_default_for([block_m, head_dim], gl.int8)
    key_layout = gl.NVMMASharedLayout.get_default_for([_BLOCK_N, head_dim], gl.int8)
    value_layout = gl.NVMMASharedLayout.get_default_for([head_dim, _BLOCK_N], gl.int8)
    batch_heads = int(query.shape[0] * query.shape[1])
    kv_batch_heads = int(key.shape[0] * key.shape[1])
    query_storage_sequence_length = int(query.shape[2])
    storage_sequence_length = int(key.shape[2])
    if (
        key.shape[0] != query.shape[0]
        or key.shape[1] < 1
        or query.shape[1] % key.shape[1]
        or key.shape[3] != query.shape[3]
        or value.shape
        != (
            query.shape[0],
            key.shape[1],
            query.shape[3],
            storage_sequence_length,
        )
    ):
        raise ValueError("paired Gluon routed Piper requires compatible Q and K/V storage")
    with device_context(query.device):
        return (
            TensorDescriptor(
                query,
                [batch_heads * query_storage_sequence_length, head_dim],
                [head_dim, 1],
                [block_m, head_dim],
                query_layout,
            ),
            TensorDescriptor(
                key,
                [kv_batch_heads * storage_sequence_length, head_dim],
                [head_dim, 1],
                [_BLOCK_N, head_dim],
                key_layout,
            ),
            TensorDescriptor(
                value,
                [kv_batch_heads * head_dim, storage_sequence_length],
                [storage_sequence_length, 1],
                [head_dim, _BLOCK_N],
                value_layout,
            ),
        )


def _launch_sparse_piper_attention(
    prepared: _PreparedSparsePiperAttention,
    output: torch.Tensor,
    *,
    query_block_offset: int = 0,
    query_block_count: int | None = None,
    coarse_output: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
) -> None:
    """Launch one caller-owned query-block range over the complete K/V sequence.

    When present, ``coarse_gate`` contains exactly the local output rows
    covered by this launch. ``coarse_output`` is indexed within the prepared
    query storage, whose ``global_block_offset`` locates it in the sequence.
    """
    query_state = prepared.query
    context = prepared.context
    launch = validate_attention_launch(
        prepared, output, query_block_offset, query_block_count, coarse_output, coarse_gate
    )
    if launch.skip_dense_routing and launch.head_dim != 64:
        raise ValueError("skip_dense_routing requires NVIDIA D64 attention")
    block_m, num_warps = policy.select_attention_schedule(
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
    with device_context(output.device):
        install_uint8_int8_dot_hook()

        query_desc, key_desc, value_desc = _make_gluon_descriptors(prepared, block_m)
        grid = (
            (launch.query_rows + block_m - 1) // block_m,
            launch.heads,
            launch.batch,
        )
        _sparse_piper_attention_kernel[grid](
            query_desc,
            key_desc,
            value_desc,
            query_state.scale,
            context.key_scale,
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
            launch.ragged_tail_is_routed,
            block_m,
            num_warps,
            launch.skip_dense_routing,
            launch.output_sequence_length % block_m != 0,
            launch.output_sequence_length,
            num_warps=num_warps,
            num_stages=1,
        )
