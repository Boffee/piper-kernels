"""SM8x and SM120 Gluon kernels for the fused ConvRot INT8 sparse-Piper Q/K/V projections.

Each program projects 128 rows onto one D128 head with the shared projection
fragments. Q and K permute the weight rows so RoPE and the signed Hadamard stay
in registers; Q emits Q32 INT8 rows with routing summaries, and K stores BF16
rows and FP32 tile sums for the shared encoder, which centers K by its global mean.
Q/K programs run at the 255-register limit. V projects its tiles through the shared
value-tile fragment, then centers them by the projected V mean and scales each K64 tile.

The policy module selects shared Triton launchers for other shapes or unaligned operands.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MEAN_ROUTING
from piper_kernels.fusions.convrot_int8_centered_projection import triton as centered_projection
from piper_kernels.fusions.convrot_int8_projection._nvidia import fragments
from piper_kernels.fusions.convrot_int8_projection._nvidia._plan import NvidiaExecutionPlan

from .. import _kernels
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from .._layout import QUERY_SCALE_ROWS, TILE_ROWS

_HEAD_DIM = fragments.HEAD_DIM
# The projected V mean is one row per head; its launcher reuses the Triton kernel.
_MEAN_BLOCK_K = 128

_GL_HEAD_DIM = gl.constexpr(_HEAD_DIM)
_GL_BLOCK_M = gl.constexpr(fragments.BLOCK_M)
_GL_TILE_ROWS = gl.constexpr(TILE_ROWS)
_GL_QUERY_SCALE_ROWS = gl.constexpr(QUERY_SCALE_ROWS)
_GL_SCALE_EPSILON = gl.constexpr(1e-7)
_GL_INT8_RANGE = gl.constexpr(127.0)
_GL_P_UINT8_RANGE = gl.constexpr(255.0)
_GL_LOG2_E = gl.constexpr(1.4426950408889634)
_GL_FEATURE_LAYOUT = fragments.FEATURE_LAYOUT

# Sequence lengths and offsets vary per call; specializing them would compile one
# kernel per divisibility class. Storage extents arrive as K64 tile counts, so the
# kernels still know that row strides are multiples of 64.
_DO_NOT_SPECIALIZE = ("logical_sequence_length", "row_block_offset", "storage_tiles")
_DO_NOT_SPECIALIZE_QUERY = (*_DO_NOT_SPECIALIZE, "chunk_start", "query_sequence_end")


@gluon.jit
def _minmax(maximum_0, minimum_0, maximum_1, minimum_1):
    """Combine (maximum, minimum) pairs in one reduction."""
    return gl.maximum(maximum_0, maximum_1), gl.minimum(minimum_0, minimum_1)


@gluon.jit
def _valid_rows(rows, positions, row_end, block_lengths_ptr, mask_block_lengths: gl.constexpr):
    """Intersect the row window with each K64 block's valid prefix.

    Block lengths are read at the clamped positions, which stay inside the sequence.
    """
    if mask_block_lengths:
        return (rows < row_end) & (
            rows % _GL_TILE_ROWS < gl.load(block_lengths_ptr + positions // _GL_TILE_ROWS)
        )
    return rows < row_end


@gluon.jit
def _summarize_blocks(
    values,
    positions,
    features,
    first_row,
    row_end,
    block_lengths_ptr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Zero invalid rows and reduce each 64-row block to its routing summaries.

    Returns the masked tile with (primary, auxiliary) summaries: the block mean twice, or
    its maximum and minimum.
    """
    blocks: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    rows = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, _GL_FEATURE_LAYOUT))
    row_valid = _valid_rows(rows, positions, row_end, block_lengths_ptr, mask_block_lengths)
    valid = row_valid[:, None] & (features >= 0)[None, :]
    if mask_rows:
        values = gl.where(valid, values, 0.0)
    block_values = gl.reshape(values, [blocks, _GL_TILE_ROWS, _GL_HEAD_DIM])
    block_valid = gl.reshape(valid, [blocks, _GL_TILE_ROWS, _GL_HEAD_DIM])
    if mean_pool_summary:
        if mask_rows:
            block_sum = gl.sum(gl.where(block_valid, block_values, 0.0), axis=1)
            # Count rows, not row-feature pairs: a per-feature count would hold 32 more
            # registers per thread through the reduction.
            row_counts = gl.reshape(row_valid.to(gl.int32), [blocks, _GL_TILE_ROWS])
            valid_count = gl.convert_layout(
                gl.sum(row_counts, axis=1), gl.SliceLayout(1, block_sum.type.layout)
            )
            mean = block_sum / valid_count[:, None]
        else:
            mean = gl.sum(block_values, axis=1) / _GL_TILE_ROWS
        return values, mean, mean
    if mask_rows:
        maximum, minimum = gl.reduce(
            (
                gl.where(block_valid, block_values, -float("inf")),
                gl.where(block_valid, block_values, float("inf")),
            ),
            1,
            _minmax,
        )
    else:
        maximum, minimum = gl.reduce((block_values, block_values), 1, _minmax)
    return values, maximum, minimum


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_QUERY)
def _query_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    chunk_start,
    query_sequence_end,
    storage_tiles,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    softmax_scale: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a Q window tile and emit Q32 INT8 rows, scales, and Q64 summaries."""
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = chunk_start + row_block * _GL_BLOCK_M
    values, positions, features = fragments.project_rmsnorm_rope(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        batch,
        head,
        first_row,
        logical_sequence_length,
        input_features,
        rotary_dim,
        norm_epsilon,
        mask_rows,
    )

    values, maximum, minimum = _summarize_blocks(
        values,
        positions,
        features,
        first_row,
        query_sequence_end,
        block_lengths_ptr,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
    )
    summary = maximum
    if not mean_pool_summary:
        summary += minimum

    blocks: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    groups: gl.constexpr = _GL_BLOCK_M // _GL_QUERY_SCALE_ROWS
    smoothed = gl.reshape(
        fragments.signed_hadamard(values, features), [groups, _GL_QUERY_SCALE_ROWS, _GL_HEAD_DIM]
    )
    raw_scale = (
        gl.max(gl.max(gl.abs(smoothed), axis=2), axis=1) / _GL_INT8_RANGE + _GL_SCALE_EPSILON
    )
    if mask_rows:
        group_layout: gl.constexpr = raw_scale.type.layout  # pyright: ignore[reportAttributeAccessIssue]
        group_starts = first_row + gl.arange(0, groups, group_layout) * _GL_QUERY_SCALE_ROWS
        group_valid = _valid_rows(
            group_starts,
            gl.minimum(group_starts, logical_sequence_length - 1),
            query_sequence_end,
            block_lengths_ptr,
            mask_block_lengths,
        )
        raw_scale = gl.where(group_valid, raw_scale, 1.0)
        stored_scale = gl.where(group_valid, raw_scale * (softmax_scale * _GL_LOG2_E), 0.0)
    else:
        stored_scale = raw_scale * (softmax_scale * _GL_LOG2_E)
    quantized = fragments.round_to_int8(smoothed / raw_scale[:, None, None])

    batch_head = batch * heads + head
    storage_rows = storage_tiles * _GL_TILE_ROWS
    storage_groups = storage_rows // _GL_QUERY_SCALE_ROWS
    fragments.store_rows(
        query_ptr + batch_head.to(gl.int64) * storage_rows * _GL_HEAD_DIM,
        quantized,
        row_block * _GL_BLOCK_M,
        storage_rows,
        mask_rows,
    )
    fragments.store_scales(
        query_scale_ptr + batch_head * storage_groups,
        stored_scale,
        row_block * groups,
        storage_groups,
        mask_rows,
    )
    fragments.store_summaries(
        query_summary_ptr + batch_head * storage_tiles * _GL_HEAD_DIM,
        summary,
        row_block * blocks,
        storage_tiles,
        mask_rows,
    )


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE)
def _key_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    stored_ptr,
    partial_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    mean_pool_summary: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a K tile and emit BF16 rows, FP32 tile sums, and routing summaries.

    Summaries describe the FP32 transform; the sums describe the stored BF16 values,
    which the shared encoder centers by their global mean.
    """
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = row_block * _GL_BLOCK_M
    values, positions, features = fragments.project_rmsnorm_rope(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        batch,
        head,
        first_row,
        logical_sequence_length,
        input_features,
        rotary_dim,
        norm_epsilon,
        mask_rows,
    )

    values, key_summary, key_aux = _summarize_blocks(
        values,
        positions,
        features,
        first_row,
        logical_sequence_length,
        block_lengths_ptr,
        mean_pool_summary,
        mask_block_lengths,
        mask_rows,
    )

    fragments.store_centered_key(
        stored_ptr,
        partial_ptr,
        values,
        batch,
        heads,
        head,
        row_block,
        storage_tiles,
        _GL_TILE_ROWS,
        mask_rows,
    )
    batch_head = batch * heads + head
    tiles: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    first_tile = row_block * tiles
    tile_row = batch_head * storage_tiles
    fragments.store_summaries(
        key_summary_ptr + tile_row * _GL_HEAD_DIM, key_summary, first_tile, storage_tiles, mask_rows
    )
    if not mean_pool_summary:
        fragments.store_summaries(
            key_aux_ptr + tile_row * _GL_HEAD_DIM, key_aux, first_tile, storage_tiles, mask_rows
        )


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE)
def _value_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    value_mean_ptr,
    value_ptr,
    value_scale_ptr,
    block_mean_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    mask_block_lengths: gl.constexpr,
    emit_block_mean: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a V tile and emit centered, K64-scaled, transposed INT8 values."""
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    projection, sequence, tile_offsets, rows_in_tile, features = fragments.project_value_tiles(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        batch,
        head,
        row_block,
        logical_sequence_length,
        input_features,
        _GL_TILE_ROWS,
        mask_rows,
    )
    tiles: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    first_tile = row_block * tiles

    if mask_block_lengths:
        tile_length = gl.load(
            block_lengths_ptr + tile_offsets, mask=tile_offsets < storage_tiles, other=0
        )
        valid = rows_in_tile[None, :] < tile_length[:, None]
    else:
        valid = sequence < logical_sequence_length
    batch_head = batch * heads + head
    tile_row = batch_head * storage_tiles
    if emit_block_mean:
        if mask_rows:
            block_sum = gl.sum(gl.where(valid[:, :, None], projection, 0.0), axis=1)
            valid_count = gl.convert_layout(
                gl.maximum(gl.sum(valid.to(gl.int32), axis=1), 1),
                gl.SliceLayout(1, block_sum.type.layout),
            )
            block_mean = block_sum / valid_count[:, None]
        else:
            block_mean = gl.sum(projection, axis=1) / _GL_TILE_ROWS
        fragments.store_summaries(
            block_mean_ptr + tile_row * _GL_HEAD_DIM,
            block_mean,
            first_tile,
            storage_tiles,
            mask_rows,
        )

    value_mean = gl.load(value_mean_ptr + batch_head * _GL_HEAD_DIM + features)
    centered = projection - value_mean[None, None, :]
    if mask_rows:
        centered = gl.where(valid[:, :, None], centered, 0.0)
    value_scale = (
        gl.max(gl.max(gl.abs(centered), axis=2), axis=1) / _GL_INT8_RANGE + _GL_SCALE_EPSILON
    )
    quantized = fragments.round_to_int8(centered / value_scale[:, None, None])

    storage_rows = storage_tiles * _GL_TILE_ROWS
    fragments.store_value_codes(
        value_ptr + batch_head.to(gl.int64) * _GL_HEAD_DIM * storage_rows,
        quantized,
        sequence,
        features,
        tile_offsets,
        storage_tiles,
        mask_rows,
    )
    fragments.store_scales(
        value_scale_ptr + tile_row,
        value_scale * _GL_P_UINT8_RANGE,
        first_tile,
        storage_tiles,
        mask_rows,
    )


def project_query(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    chunk_start: int,
    chunk_rows: int,
    out: QueryOutput,
    bias: torch.Tensor | None = None,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill Q32 INT8 queries, scales, and Q64 summaries for a supported query window."""
    query, query_scale, query_summary = out
    batch, heads, storage_sequence_length, _head_dim = query.shape
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _query_kernel[(heads, row_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                norm_weight,
                cos,
                sin,
                query,
                query_scale,
                query_summary,
                block_lengths if mask_block_lengths else query_scale,
                input_qdata.shape[1],
                row_block_offset,
                chunk_start,
                chunk_start + chunk_rows,
                storage_sequence_length // TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        fragments.launch_row_blocks(chunk_rows, launch, mask_full_tiles=mask_block_lengths)


def project_key(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    out: KeyOutput,
    bias: torch.Tensor | None = None,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill centered K64 INT8 keys, scales, and routing summaries for supported operands."""
    key, key_scale, key_summary, key_aux = out
    batch, heads, storage_sequence_length, head_dim = key.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0:
        return
    stored, partials, mean = centered_projection.allocate_workspace(
        input_qdata, (batch, heads, storage_sequence_length, head_dim), tile_rows=TILE_ROWS
    )
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _key_kernel[(heads, row_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                norm_weight,
                cos,
                sin,
                stored,
                partials,
                key_summary,
                key_aux,
                block_lengths if mask_block_lengths else partials,
                input_qdata.shape[1],
                row_block_offset,
                storage_sequence_length // TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=mask_block_lengths,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        fragments.launch_row_blocks(sequence_length, launch, mask_full_tiles=mask_block_lengths)
        # Zeroed internal padding stays in the logical denominator, as in the shared path.
        centered_projection.finalize_mean(partials, sequence_length, out=mean)
        qk_quantization.prepare_key(
            stored.narrow(2, 0, sequence_length),
            mean,
            grouped=True,
            storage_key_length=storage_sequence_length,
            out=(key, key_scale),
        )


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    emit_block_mean: bool,
    value_format: ValueFormat,
    out: ValueOutput,
    bias: torch.Tensor | None = None,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill centered K64-scaled INT8 values, the projected mean, and block means."""
    if value_format is not ValueFormat.TRANSPOSED:
        raise ValueError("NVIDIA Gluon V projection requires transposed storage")
    value, value_scale_multiplier, value_mean, block_mean = out
    batch, heads, _head_dim, storage_sequence_length = value.shape
    mask_block_lengths = block_lengths is not None

    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _value_kernel[(heads, row_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                value_mean,
                value,
                value_scale_multiplier,
                block_mean,
                block_lengths if mask_block_lengths else value_mean,
                input_qdata.shape[1],
                row_block_offset,
                storage_sequence_length // TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                mask_block_lengths=mask_block_lengths,
                emit_block_mean=emit_block_mean,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        _kernels._project_prepared_input_mean_kernel[(heads, batch)](
            input_mean,
            weight_qdata,
            weight_scale,
            value_mean,
            bias_ptr=bias,
            input_features=input_qdata.shape[2],
            output_features=heads * _HEAD_DIM,
            block_n=_HEAD_DIM,
            block_k=_MEAN_BLOCK_K,
            num_warps=execution_plan.num_warps,
        )
        fragments.launch_row_blocks(
            input_qdata.shape[1], launch, mask_full_tiles=mask_block_lengths
        )
