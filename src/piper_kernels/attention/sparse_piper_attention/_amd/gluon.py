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

from .._launch import validate_attention_launch
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


@gluon.jit
def _tile_index(routes, position, routed_count, selected_count, sparse_blocks, use_routes):
    route = gl.load(routes + gl.minimum(position, routed_count - 1)).to(gl.int32)
    sparse_tile = gl.where(use_routes, route, position)
    return gl.where(
        position < selected_count, sparse_tile, sparse_blocks + position - selected_count
    )


@gluon.jit
def _sparse_piper_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    parameters_ptr,
    mean_ptr,
    coarse_ptr,
    gate_ptr,
    lengths_ptr,
    routes_ptr,
    keep_ptr,
    route_offsets_ptr,
    output_ptr,
    query_block_offset,
    global_block_offset,
    query_storage_length: gl.constexpr,
    storage_length: gl.constexpr,
    logical_length: gl.constexpr,
    sparse_blocks: gl.constexpr,
    sparse_query_blocks: gl.constexpr,
    stride_rb: gl.constexpr,
    stride_rq: gl.constexpr,
    stride_ob: gl.constexpr,
    stride_oh: gl.constexpr,
    stride_on: gl.constexpr,
    stride_cb: gl.constexpr,
    stride_ch: gl.constexpr,
    stride_cq: gl.constexpr,
    stride_gb: gl.constexpr,
    stride_gh: gl.constexpr,
    stride_gn: gl.constexpr,
    heads: gl.constexpr,
    has_lengths: gl.constexpr,
    has_dense_queries: gl.constexpr,
    has_coarse: gl.constexpr,
):
    # Promote global tensor bases and caller strides before multiplication,
    # not after an i32 overflow. Fragment helpers derive local address width
    # from storage size; UINT16 routes only bound the sparse prefix, not a dense suffix.
    head = gl.program_id(1).to(gl.int64)
    batch = gl.program_id(2).to(gl.int64)
    bh = batch * heads + head
    local_block = gl.program_id(0)
    query_block = query_block_offset + local_block
    global_block = global_block_offset + local_block
    row_layout: gl.constexpr = gl.SliceLayout(2, MMA_LAYOUT)
    block_layout: gl.constexpr = gl.SliceLayout(1, row_layout)
    blocks = gl.full([1], query_block, gl.int32, block_layout)
    rows = gl.arange(0, 64, gl.SliceLayout(0, row_layout))
    columns = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
    routes = (
        routes_ptr
        + batch * stride_rb
        + blocks.to(gl.int64) * stride_rq
        + gl.load(route_offsets_ptr + head)
    )
    routed_count = gl.load(keep_ptr + head)
    use_routes = global_block < sparse_query_blocks if has_dense_queries else True
    selected_count = gl.where(use_routes, routed_count, sparse_blocks)
    tile_count = selected_count + storage_length // 64 - sparse_blocks
    query = query_fragments(
        query_ptr + bh * query_storage_length * 128, blocks, query_storage_length
    )
    q_scale = gl.load(
        query_scale_ptr
        + bh * (query_storage_length // 32)
        + blocks[:, None] * 2
        + rows[None, :] // 32
    )
    numerator = gl.zeros([1, 64, 128], gl.float32, MMA_LAYOUT)
    denominator = gl.zeros([1, 64], gl.float32, row_layout)
    running_max = gl.full([1, 64], -float("inf"), gl.float32, row_layout)
    parameters = parameters_ptr + bh * (storage_length // 64) * PARAMETER_COUNT

    for pair in range(gl.cdiv(tile_count, 2)):
        tile_0 = _tile_index(
            routes, pair * 2, routed_count, selected_count, sparse_blocks, use_routes
        )
        tile_1 = _tile_index(
            routes,
            gl.minimum(pair * 2 + 1, tile_count - 1),
            routed_count,
            selected_count,
            sparse_blocks,
            use_routes,
        )
        scores = qk_pair(query, key_ptr + bh * storage_length * 128, tile_0, tile_1, storage_length)
        parameters_0 = parameters + tile_0 * PARAMETER_COUNT
        parameters_1 = parameters + tile_1 * PARAMETER_COUNT
        key_scale_0 = gl.load(parameters_0 + KEY_SCALE)
        key_scale_1 = gl.load(parameters_1 + KEY_SCALE)
        scale_0 = q_scale * key_scale_0[:, None]
        scale_1 = q_scale * key_scale_1[:, None]
        if has_lengths:
            length_0 = gl.load(lengths_ptr + tile_0)
            length_1 = gl.load(lengths_ptr + tile_1)
        else:
            length_0 = gl.minimum(64, logical_length - tile_0 * 64)
            length_1 = gl.minimum(64, logical_length - tile_1 * 64)
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
                * q_scale[:, :, None]
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
            value_ptr + bh * storage_length * 128,
            tile_0,
            tile_1,
            numerator,
            current_weight,
            storage_length,
        )
        running_max = next_max

    inverse_denominator = 1.0 / (gl.maximum(denominator, 1e-30) * 255.0)
    result = numerator * inverse_denominator[:, :, None]
    result += gl.load(mean_ptr + bh * 128 + columns)[None, None, :]
    output_rows = (local_block * 64 + rows).to(gl.int64)
    valid_rows = (global_block * 64 + rows < logical_length) | has_lengths
    if has_coarse:
        coarse = gl.load(
            coarse_ptr
            + batch * stride_cb
            + head * stride_ch
            + query_block.to(gl.int64) * stride_cq
            + columns
        )
        gate = gl.load(
            gate_ptr
            + batch * stride_gb
            + head * stride_gh
            + output_rows[None, :, None] * stride_gn
            + columns[None, None, :],
            mask=valid_rows[None, :, None],
            other=0.0,
        ).to(gl.float32)
        result = gl.fma(gate, coarse[None, None, :], result)
    gl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + output_rows[None, :, None] * stride_on
        + columns[None, None, :],
        result.to(gl.bfloat16),
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


def bind_context(context: _PreparedSparsePiperContext) -> _ContextLauncher:
    """Pack immutable global K/V state once for any number of local Q launches."""
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
    blocks, global_offset = validate_attention_launch(
        prepared, output, query_block_offset, query_block_count, coarse_output, coarse_gate
    )
    if packed is None:
        packed = pack_context(prepared.context)
    query, context = prepared.query, prepared.context
    batch, heads, query_length, _ = query.data.shape
    storage_length = context.key.shape[2]
    coarse_strides = (0, 0, 0) if coarse_output is None else coarse_output.stride()[:3]
    gate_strides = (
        (0, 0, 0)
        if coarse_gate is None
        else (coarse_gate.stride(0), coarse_gate.stride(2), coarse_gate.stride(1))
    )
    with device_context(output.device):
        _sparse_piper_attention_kernel[(blocks, heads, batch)](
            query.data,
            context.key,
            packed.value,
            query.scale,
            packed.parameters,
            context.value_mean,
            context.value_mean if coarse_output is None else coarse_output,
            output if coarse_gate is None else coarse_gate,
            context.head_keep_blocks if context.block_lengths is None else context.block_lengths,
            query.routes,
            context.head_keep_blocks,
            context.route_head_offsets,
            output,
            query_block_offset,
            global_offset,
            query_length,
            storage_length,
            context.logical_sequence_length,
            context.sparse_key_blocks,
            storage_length // 64
            if context.sparse_query_blocks is None
            else context.sparse_query_blocks,
            query.routes.stride(0),
            query.routes.stride(1),
            *output.stride()[:3],
            *coarse_strides,
            *gate_strides,
            heads,
            context.block_lengths is not None,
            context.sparse_query_blocks is not None,
            coarse_output is not None,
            num_warps=4,
            num_stages=1,
            llvm_fn_attrs=(("target-features", "+cumode"),),
        )
