"""Paired-K128 Gluon kernel for sparse Piper Attention on SM89.

Ada has no Tensor Memory Accelerator. Q, the two routed K64 tiles of each
pair, and their V tiles therefore reach shared memory through Ampere-style
``cp.async`` copies tracked by commit groups, instead of TMA descriptors and
mbarriers. The routed traversal, shared-memory layouts, paired-K128 recurrence,
and epilogue are the SM120 kernel's, so both targets share one numerical
contract.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper.gluon import tile_offset
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS, TILE_ROWS

from .._launch import _DO_NOT_SPECIALIZE_ARGUMENTS, validate_attention_launch
from .._prepared import _PreparedSparsePiperAttention
from . import policy
from ._recurrence import piper_probability_pair, piper_pv_pair, store_attention_output

_BLOCK_N = TILE_ROWS
# Every launch runs one Q64 tile per CTA with four MMA warps.
_NUM_WARPS = 4
# ``cp.async`` moves at most 16 bytes per thread and instruction.
_COPY_ALIGNMENT = 16

_GL_BLOCK_N = gl.constexpr(_BLOCK_N)
_GL_NUM_WARPS = gl.constexpr(_NUM_WARPS)
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)
_GL_COPY_BYTES = gl.constexpr(_COPY_ALIGNMENT)


@gluon.jit
def _copy_rows(base_ptr, rows, shared, copy_layout: gl.constexpr):
    """Copy INT8 rows of one contiguous feature width into shared memory."""
    width: gl.constexpr = shared.shape[1]
    features = gl.arange(0, width, gl.SliceLayout(0, copy_layout))
    async_copy.async_load(shared, base_ptr + rows[:, None] * width + features[None, :])


@gluon.jit
def _copy_key_pair(
    key_ptr,
    start_n_0,
    start_n_1,
    key_shared_pair,
    copy_layout: gl.constexpr,
):
    """Gather two routed K64 row tiles into one paired [K128, D] buffer.

    Each tile is copied from its own base pointer, so every thread reuses one
    set of row offsets instead of selecting a tile per row.
    """
    head_dim: gl.constexpr = key_shared_pair.shape[1]
    rows = gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(1, copy_layout))
    _copy_rows(
        key_ptr + start_n_0 * head_dim,
        rows,
        key_shared_pair.slice(0, _GL_BLOCK_N),
        copy_layout,
    )
    _copy_rows(
        key_ptr + start_n_1 * head_dim,
        rows,
        key_shared_pair.slice(_GL_BLOCK_N, _GL_BLOCK_N),
        copy_layout,
    )


@gluon.jit
def _copy_value_pair(
    value_ptr,
    storage_sequence_length,
    start_n_0,
    start_n_1,
    value_shared_pair,
    copy_layout: gl.constexpr,
):
    """Copy the transposed [D, K64] V tiles of one pair."""
    head_dim: gl.constexpr = value_shared_pair.shape[1]
    features = gl.arange(0, head_dim, gl.SliceLayout(1, copy_layout))
    keys = gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(0, copy_layout))
    offsets = features[:, None] * storage_sequence_length + keys[None, :]
    async_copy.async_load(value_shared_pair.index(0), value_ptr + start_n_0 + offsets)
    async_copy.async_load(value_shared_pair.index(1), value_ptr + start_n_1 + offsets)


@gluon.jit
def _pair_offsets(
    pair_index,
    tile_count,
    route_base,
    routed_sparse_tile_count,
    selected_sparse_tile_count,
    sparse_key_blocks,
    stride_rr,
    use_sparse_routes,
    skip_dense_routing: gl.constexpr,
):
    """Return both K/V tile offsets of a pair, duplicating a missing second tile."""
    position_0 = pair_index * 2
    position_1 = gl.minimum(position_0 + 1, tile_count - 1)
    start_n_0 = tile_offset(
        route_base,
        position_0,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        _GL_BLOCK_N,
    )
    start_n_1 = tile_offset(
        route_base,
        position_1,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
        _GL_BLOCK_N,
    )
    return start_n_0, start_n_1


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_ARGUMENTS)
def _sparse_piper_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
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
    skip_dense_routing: gl.constexpr,
    mask_output_tail: gl.constexpr,
    output_sequence_length,
):
    """Pair native logical K64 tiles in one shared Piper probability coordinate.

    The next pair's K tile is copied while the current pair's probabilities are
    computed, and its V tile while the current PV products accumulate, so one
    K and one V commit group are in flight across each wait. A CTA barrier
    after each wait publishes every thread's copies, and one before each
    reissue retires the reads of the buffer being overwritten.
    """
    local_query_block = gl.program_id(0)
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
    # Head-major storage can exceed signed 32-bit element offsets.
    query_base_ptr = query_ptr + batch_head.to(gl.int64) * query_storage_sequence_length * head_dim
    key_base_ptr = key_ptr + kv_batch_head.to(gl.int64) * storage_sequence_length * head_dim
    value_base_ptr = value_ptr + kv_batch_head.to(gl.int64) * head_dim * storage_sequence_length

    # The packed PV rescale is validated for each selected MMA register layout.
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[_GL_NUM_WARPS, 1],
        instr_shape=[16, 8],
    )
    query_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    key_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    # One 16-byte copy per thread and row segment, with rows spread over warps.
    row_copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, _GL_COPY_BYTES],
        [32 // (head_dim // _GL_COPY_BYTES), head_dim // _GL_COPY_BYTES],
        [_GL_NUM_WARPS, 1],
        [1, 0],
    )
    value_copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, _GL_COPY_BYTES],
        [32 // (_GL_BLOCK_N // _GL_COPY_BYTES), _GL_BLOCK_N // _GL_COPY_BYTES],
        [_GL_NUM_WARPS, 1],
        [1, 0],
    )
    # The SM120 kernel's TMA-compatible swizzles also serve ldmatrix here; Q and K
    # rows share one layout.
    row_shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [_GL_BLOCK_N, head_dim], gl.int8
    )
    value_shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [head_dim, _GL_BLOCK_N], gl.int8
    )

    query_shared = gl.allocate_shared_memory(gl.int8, [_GL_BLOCK_N, head_dim], row_shared_layout)
    key_shared_pair = gl.allocate_shared_memory(
        gl.int8, [2 * _GL_BLOCK_N, head_dim], row_shared_layout
    )
    value_shared_pair = gl.allocate_shared_memory(
        gl.int8, [2, head_dim, _GL_BLOCK_N], value_shared_layout
    )

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

    query_rows = start_m + gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(1, row_copy_layout))
    _copy_rows(query_base_ptr, query_rows, query_shared, row_copy_layout)
    async_copy.commit_group()
    start_n_0, start_n_1 = _pair_offsets(
        0,
        tile_count,
        route_base,
        routed_sparse_tile_count,
        selected_sparse_tile_count,
        sparse_key_blocks,
        stride_rr,
        use_sparse_routes,
        skip_dense_routing,
    )
    _copy_key_pair(key_base_ptr, start_n_0, start_n_1, key_shared_pair, row_copy_layout)
    async_copy.commit_group()
    _copy_value_pair(
        value_base_ptr,
        storage_sequence_length,
        start_n_0,
        start_n_1,
        value_shared_pair,
        value_copy_layout,
    )
    async_copy.commit_group()

    # Q is the oldest group; the first pair's K and V may remain in flight.
    async_copy.wait_group(2)
    gl.barrier()
    query = query_shared.load(query_layout)

    offsets_m = gl.arange(0, _GL_BLOCK_N, row_layout)
    query_scale_stride = query_storage_sequence_length // _GL_QUERY_SCALE_ROWS
    query_scale = gl.load(
        query_scale_ptr
        + batch_head * query_scale_stride
        + (start_m + offsets_m) // _GL_QUERY_SCALE_ROWS
    )
    accumulator = gl.zeros([_GL_BLOCK_N, head_dim], gl.float32, mma_layout)
    denominator = gl.zeros([_GL_BLOCK_N], gl.float32, row_layout)
    running_max = gl.full([_GL_BLOCK_N], -float("inf"), gl.float32, row_layout)

    for pair_index in range(pair_count - 1):
        # The consumed pair's K group is followed only by its V group.
        async_copy.wait_group(1)
        gl.barrier()
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
        next_n_0, next_n_1 = _pair_offsets(
            pair_index + 1,
            tile_count,
            route_base,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
            skip_dense_routing,
        )
        gl.barrier()
        _copy_key_pair(key_base_ptr, next_n_0, next_n_1, key_shared_pair, row_copy_layout)
        async_copy.commit_group()
        # The consumed pair's V group is followed only by the next pair's K group.
        async_copy.wait_group(1)
        gl.barrier()
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
        _copy_value_pair(
            value_base_ptr,
            storage_sequence_length,
            next_n_0,
            next_n_1,
            value_shared_pair,
            value_copy_layout,
        )
        async_copy.commit_group()
        start_n_0 = next_n_0
        start_n_1 = next_n_1

    has_second = (pair_count - 1) * 2 + 1 < tile_count
    async_copy.wait_group(1)
    gl.barrier()
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
    # No copy follows the final pair's V group.
    async_copy.wait_group(0)
    gl.barrier()
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


def _validate_copy_storage(prepared: _PreparedSparsePiperAttention) -> None:
    """Check the metadata that pointer-based 16-byte copies rely on."""
    query = prepared.query.data
    key = prepared.context.key
    value = prepared.context.value
    storage_sequence_length = key.shape[2]
    if (
        key.shape[0] != query.shape[0]
        or key.shape[1] < 1
        or query.shape[1] % key.shape[1]
        or key.shape[3] != query.shape[3]
        or value.shape != (query.shape[0], key.shape[1], query.shape[3], storage_sequence_length)
    ):
        raise ValueError("paired Gluon routed Piper requires compatible Q and K/V storage")
    if any(
        not tensor.is_contiguous() or tensor.data_ptr() % _COPY_ALIGNMENT
        for tensor in (query, key, value)
    ):
        raise ValueError("SM89 sparse Piper requires contiguous 16-byte-aligned Q/K/V storage")


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
    _validate_copy_storage(prepared)
    max_registers = policy.sm89_max_registers(launch.head_dim)
    compile_options = {} if max_registers is None else {"maxnreg": max_registers}
    with device_context(output.device):
        install_uint8_int8_dot_hook()
        grid = (launch.query_block_count, launch.heads, launch.batch)
        _sparse_piper_attention_kernel[grid](
            query_state.data,
            context.key,
            context.value,
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
            launch.skip_dense_routing,
            # Only a ragged final Q64 tile needs masked stores. Padded storage keeps
            # writing every row, exactly as SM120 does at Q64.
            launch.output_sequence_length % _BLOCK_N != 0,
            launch.output_sequence_length,
            num_warps=_NUM_WARPS,
            num_stages=1,
            **compile_options,
        )
