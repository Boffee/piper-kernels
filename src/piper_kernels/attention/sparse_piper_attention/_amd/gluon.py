"""Fused RDNA4 sparse Piper over the common quantized attention contract."""

# Gluon device parameters and layouts are not Python runtime values.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from dataclasses import dataclass

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper.gluon import tile_offset

from .._launch import _DO_NOT_SPECIALIZE_ARGUMENTS, validate_attention_launch
from .._prepared import _PreparedSparsePiperAttention, _PreparedSparsePiperContext
from ._fragments import (
    MMA_LAYOUT,
    concat_columns,
    pv_pair,
    qk_pair,
    query_fragments,
    rescale_numerator,
    softmax_fragment,
    split_columns,
)
from ._packing import (
    INVERSE_MULTIPLIER,
    KEY_SCALE,
    LOG_MULTIPLIER,
    LOG_MULTIPLIER_OVER_255,
    PARAMETER_COUNT,
    PackedContext,
    pack_context,
)


def _requires_64bit_query_offsets(
    query_storage_sequence_length: int,
    head_dim: int,
) -> bool:
    """Return whether packed Q word offsets exceed signed 32-bit range."""
    return query_storage_sequence_length * (head_dim // 8) > (1 << 31)


def _requires_64bit_context_offsets(
    storage_sequence_length: int,
    head_dim: int,
) -> bool:
    """Return whether local K/V byte offsets exceed unsigned 32-bit range."""
    return storage_sequence_length * head_dim > (1 << 32)


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_ARGUMENTS)
def _sparse_piper_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    parameters_ptr,
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
    head_dim: gl.constexpr,
    use_64bit_query_offsets: gl.constexpr,
    use_64bit_context_offsets: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    has_dense_query_suffix: gl.constexpr,
    apply_coarse_residual: gl.constexpr,
):
    # Keep global tensor bases in i64. Structural flags select fragment-local
    # address widths without making dynamic sequence lengths compile-time values.
    head = gl.program_id(1).to(gl.int64)
    batch = gl.program_id(2).to(gl.int64)
    batch_head = batch * heads + head
    local_query_block = gl.program_id(0)
    query_block = query_block_offset + local_query_block
    global_query_block = global_query_block_offset + local_query_block
    row_layout: gl.constexpr = gl.SliceLayout(2, MMA_LAYOUT)
    block_layout: gl.constexpr = gl.SliceLayout(1, row_layout)
    query_blocks = gl.full([1], query_block, gl.int32, block_layout)
    rows = gl.arange(0, 64, gl.SliceLayout(0, row_layout))
    # The two K64 tiles always produce 128 score columns, independently of D.
    columns = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
    features = gl.arange(0, head_dim, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
    route_head_offset = gl.load(route_head_offsets_ptr + head)
    route_base = (
        routes_ptr
        + batch * stride_rb
        + query_blocks.to(gl.int64) * stride_rq
        + route_head_offset * stride_rr
    )
    routed_sparse_tile_count = gl.load(head_keep_blocks_ptr + head)
    use_sparse_routes = global_query_block < sparse_query_blocks if has_dense_query_suffix else True
    selected_sparse_tile_count = gl.where(
        use_sparse_routes,
        routed_sparse_tile_count,
        sparse_key_blocks,
    )
    sequence_tiles = storage_sequence_length // 64
    dense_tile_count = sequence_tiles - sparse_key_blocks
    tile_count = selected_sparse_tile_count + dense_tile_count
    pair_count = gl.cdiv(tile_count, 2)
    query = query_fragments(
        query_ptr + batch_head * query_storage_sequence_length * head_dim,
        query_blocks,
        use_64bit_query_offsets,
        head_dim,
    )
    query_scale = gl.load(
        query_scale_ptr
        + batch_head * (query_storage_sequence_length // 32)
        + query_blocks[:, None] * 2
        + rows[None, :] // 32
    )
    numerator = gl.zeros([1, 64, head_dim], gl.float32, MMA_LAYOUT)
    denominator = gl.zeros([1, 64], gl.float32, row_layout)
    running_max = gl.full([1, 64], -float("inf"), gl.float32, row_layout)
    parameters = parameters_ptr + batch_head * sequence_tiles * PARAMETER_COUNT

    for pair in range(pair_count):
        tile_0 = tile_offset(
            route_base,
            pair * 2,
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
        )
        tile_1 = tile_offset(
            route_base,
            gl.minimum(pair * 2 + 1, tile_count - 1),
            routed_sparse_tile_count,
            selected_sparse_tile_count,
            sparse_key_blocks,
            stride_rr,
            use_sparse_routes,
        )
        scores = qk_pair(
            query,
            key_ptr + batch_head * storage_sequence_length * head_dim,
            tile_0,
            tile_1,
            use_64bit_context_offsets,
            head_dim,
        )
        parameters_0 = parameters + tile_0 * PARAMETER_COUNT
        parameters_1 = parameters + tile_1 * PARAMETER_COUNT
        key_scale_0 = gl.load(parameters_0 + KEY_SCALE)
        key_scale_1 = gl.load(parameters_1 + KEY_SCALE)
        scale_0 = query_scale * key_scale_0[:, None]
        scale_1 = query_scale * key_scale_1[:, None]
        if mask_block_lengths:
            length_0 = gl.load(block_lengths_ptr + tile_0)
            length_1 = gl.load(block_lengths_ptr + tile_1)
        else:
            length_0 = gl.minimum(64, logical_sequence_length - tile_0 * 64)
            length_1 = gl.minimum(64, logical_sequence_length - tile_1 * 64)
        complete = (
            (gl.sum(length_0, 0) == 64) & (gl.sum(length_1, 0) == 64) & (pair * 2 + 1 < tile_count)
        )
        if not complete:
            column_block_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(1, MMA_LAYOUT))
            valid_length_0 = gl.convert_layout(length_0, column_block_layout, assert_trivial=True)
            valid_length_1 = gl.convert_layout(length_1, column_block_layout, assert_trivial=True)
            valid = gl.where(
                columns[None, :] < 64,
                columns[None, :] < valid_length_0[:, None],
                (columns[None, :] - 64 < valid_length_1[:, None]) & (pair * 2 + 1 < tile_count),
            )
            column_scale_0 = gl.convert_layout(
                key_scale_0, column_block_layout, assert_trivial=True
            )
            column_scale_1 = gl.convert_layout(
                key_scale_1, column_block_layout, assert_trivial=True
            )
            scale = (
                gl.where(columns[None, :] < 64, column_scale_0[:, None], column_scale_1[:, None])[
                    :, None, :
                ]
                * query_scale[:, :, None]
            )
            # Invalid scores are -inf. Handle a zero scale before multiplication
            # so padded keys can never turn 0 * -inf into NaN.
            scores = gl.where(valid[:, None, :], gl.where(scale == 0, 0.0, scores), -float("inf"))
            scale_0 = gl.where(scale_0 == 0, 1.0, scale_0)
            scale_1 = gl.where(scale_1 == 0, 1.0, scale_1)
        scores_0, scores_1 = split_columns(scores)
        offset_0 = gl.load(parameters_0 + LOG_MULTIPLIER_OVER_255)
        offset_1 = gl.load(parameters_1 + LOG_MULTIPLIER_OVER_255)
        maximum_0 = gl.convert_layout(gl.max(scores_0, 2), row_layout)
        maximum_1 = gl.convert_layout(gl.max(scores_1, 2), row_layout)
        block_max = gl.maximum(
            gl.fma(maximum_0, scale_0, offset_0[:, None]),
            gl.fma(maximum_1, scale_1, offset_1[:, None]),
        )
        next_max = gl.maximum(running_max, block_max)
        old_weight = gl.exp2(running_max - next_max)
        current_weight = gl.exp2(block_max - next_max)
        numerator = rescale_numerator(numerator, old_weight)
        scores_00, scores_01 = split_columns(scores_0)
        scores_10, scores_11 = split_columns(scores_1)
        chunks = (scores_00, scores_01, scores_10, scores_11)
        inverse_0 = gl.load(parameters_0 + INVERSE_MULTIPLIER)
        inverse_1 = gl.load(parameters_1 + INVERSE_MULTIPLIER)
        log_0 = gl.load(parameters_0 + LOG_MULTIPLIER)
        log_1 = gl.load(parameters_1 + LOG_MULTIPLIER)
        packed, sums = (), ()
        for chunk in gl.static_range(4):
            words, total = softmax_fragment(
                chunks[chunk],
                block_max,
                scale_0 if chunk < 2 else scale_1,
                inverse_0 if chunk < 2 else inverse_1,
                log_0 if chunk < 2 else log_1,
            )
            packed += (words,)
            sums += (total,)
            with gl.amd.warp_pipeline_stage("softmax"):
                pass
        denominator = (
            denominator * old_weight + ((sums[0] + sums[1]) + (sums[2] + sums[3])) * current_weight
        )
        probabilities = concat_columns(
            concat_columns(packed[0], packed[1]), concat_columns(packed[2], packed[3])
        )
        numerator = pv_pair(
            probabilities,
            value_ptr + batch_head * storage_sequence_length * head_dim,
            tile_0,
            tile_1,
            numerator,
            current_weight,
            use_64bit_context_offsets,
        )
        running_max = next_max

    inverse_denominator = 1.0 / (gl.maximum(denominator, 1e-30) * 255.0)
    result = numerator * inverse_denominator[:, :, None]
    result += gl.load(value_mean_ptr + batch_head * head_dim + features)[None, None, :]
    output_rows = (local_query_block * 64 + rows).to(gl.int64)
    valid_rows = (global_query_block * 64 + rows < logical_sequence_length) | mask_block_lengths
    if apply_coarse_residual:
        coarse = gl.load(
            coarse_output_ptr
            + batch * stride_cb
            + head * stride_ch
            + query_block.to(gl.int64) * stride_cq
            + features
        )
        gate = gl.load(
            coarse_gate_ptr
            + batch * stride_gb
            + head * stride_gh
            + output_rows[None, :, None] * stride_gn
            + features[None, None, :],
            mask=valid_rows[None, :, None],
            other=0.0,
        ).to(gl.float32)
        result = gl.fma(gate, coarse[None, None, :], result)
    gl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + output_rows[None, :, None] * stride_on
        + features[None, None, :],
        result.to(output_ptr.dtype.element_ty),
        mask=valid_rows[None, :, None],
    )


@dataclass(frozen=True, slots=True)
class _ContextLauncher:
    packed: PackedContext

    def __call__(
        self,
        prepared: _PreparedSparsePiperAttention,
        output: torch.Tensor,
        *,
        query_block_offset: int = 0,
        query_block_count: int | None = None,
        coarse_output: torch.Tensor | None = None,
        coarse_gate: torch.Tensor | None = None,
    ) -> None:
        _launch_sparse_piper_attention(
            prepared,
            output,
            query_block_offset=query_block_offset,
            query_block_count=query_block_count,
            coarse_output=coarse_output,
            coarse_gate=coarse_gate,
            packed=self.packed,
        )


def _validate_context(context: _PreparedSparsePiperContext) -> None:
    """Reject unsupported execution modes before packing or launching."""
    if context.key.shape[-1] not in (64, 128):
        raise ValueError("AMD sparse Piper requires D64 or D128")
    if context.routes_per_query == 0:
        raise ValueError("AMD sparse Piper does not implement skip_dense_routing")


def bind_context(context: _PreparedSparsePiperContext) -> _ContextLauncher:
    """Pack immutable global K/V state once for any number of local Q launches."""
    _validate_context(context)
    return _ContextLauncher(pack_context(context))


def _launch_sparse_piper_attention(
    prepared: _PreparedSparsePiperAttention,
    output: torch.Tensor,
    *,
    query_block_offset: int = 0,
    query_block_count: int | None = None,
    coarse_output: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    packed: PackedContext | None = None,
) -> None:
    """Validate once, then execute with fresh or explicitly bound K/V packing."""
    if packed is not None and prepared.context is not packed.source:
        raise ValueError("bound sparse Piper launcher requires its original context")
    _validate_context(prepared.context)
    launch = validate_attention_launch(
        prepared, output, query_block_offset, query_block_count, coarse_output, coarse_gate
    )
    if packed is None:
        packed = pack_context(prepared.context)
    query_state, context = prepared.query, prepared.context
    use_64bit_query_offsets = _requires_64bit_query_offsets(
        launch.query_storage_sequence_length,
        launch.head_dim,
    )
    use_64bit_context_offsets = _requires_64bit_context_offsets(
        launch.storage_sequence_length,
        launch.head_dim,
    )
    with device_context(output.device):
        grid = (launch.query_block_count, launch.heads, launch.batch)
        _sparse_piper_attention_kernel[grid](
            query_state.data,
            context.key,
            packed.value,
            query_state.scale,
            packed.parameters,
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
            launch.head_dim,
            use_64bit_query_offsets,
            use_64bit_context_offsets,
            launch.mask_block_lengths,
            launch.has_dense_query_suffix,
            launch.apply_coarse_residual,
            num_warps=4,
            num_stages=1,
            llvm_fn_attrs=(("target-features", "+cumode"),),
        )
