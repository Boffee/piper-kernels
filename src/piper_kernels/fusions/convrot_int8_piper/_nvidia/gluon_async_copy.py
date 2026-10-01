"""SM89 Gluon kernels for the fused ConvRot INT8 dense-Piper Q/K/V projections.

Each program projects 128 rows onto one D128 head with the shared projection
fragments. Q quantizes its RMSNorm/RoPE tile after the signed Hadamard, with
per-thread or Q32 scales. Per-thread groups (rows 32b + 8i + t) lie in one
thread's registers, so their scales need no data exchange and are stored per row
from that layout. K stores BF16 rows and FP32 tile sums for the shared encoder,
which centers K by its global mean.

V keeps a 2x2 warp grid and quantizes each token. Its epilogue spreads rows
across lanes, so the transposed code stores and per-token metadata stores are
coalesced.

The policy module selects shared Triton launchers for other shapes or unaligned operands.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from __future__ import annotations

from collections.abc import Callable

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization
from piper_kernels.fusions.convrot_int8_centered_projection import triton as centered_projection
from piper_kernels.fusions.convrot_int8_projection._nvidia import fragments
from piper_kernels.fusions.convrot_int8_projection._nvidia._plan import NvidiaExecutionPlan
from piper_kernels.fusions.convrot_int8_projection.triton import project_prepared_input_mean_kernel
from piper_kernels.linear.convrot.int8 import _ops

from .._interfaces import KeyOutput, QueryOutput, ValueOutput

_BLOCK_M = fragments.BLOCK_M
_NUM_WARPS = fragments.NUM_WARPS
# K/V storage and K tile sums use K64 tiles.
_TILE_ROWS = 64
# The compact V mean projection needs more CTAs than the full matrix projection.
_MEAN_BLOCK_N = 32
_MEAN_BLOCK_K = 128

_GL_HEAD_DIM = gl.constexpr(fragments.HEAD_DIM)
_GL_BLOCK_M = gl.constexpr(_BLOCK_M)
_GL_NUM_WARPS = gl.constexpr(_NUM_WARPS)
_GL_TILE_ROWS = gl.constexpr(_TILE_ROWS)
_GL_QUERY_SCALE_ROWS = gl.constexpr(32)
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
    per_thread_scales: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a Q window tile and emit INT8 rows with per-row or Q32 base-2 scales.

    Rows past the window end store zero codes and scales.
    """
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = chunk_start + row_block * _GL_BLOCK_M
    values, _positions, features = fragments.project_rmsnorm_rope(
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
    if mask_rows:
        rows = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, _GL_FEATURE_LAYOUT))
        valid = (rows < query_sequence_end)[:, None] & (features >= 0)[None, :]
        values = gl.where(valid, values, 0.0)
    smoothed = fragments.signed_hadamard(values, features)

    batch_head = batch * heads + head
    storage_rows = storage_tiles * _GL_TILE_ROWS
    if per_thread_scales:
        grouped = gl.reshape(smoothed, [_GL_BLOCK_M // 32, 4, 8, _GL_HEAD_DIM])
        row_maximum = gl.max(gl.abs(grouped), axis=3)
        raw_scale = (
            gl.expand_dims(gl.max(row_maximum, axis=1), 1) / _GL_INT8_RANGE + _GL_SCALE_EPSILON
        )
        quantized = fragments.round_to_int8(grouped / gl.expand_dims(raw_scale, 3))
        row_scale, _ = gl.broadcast(raw_scale, row_maximum)
        row_scale = gl.reshape(row_scale, [_GL_BLOCK_M]) * (softmax_scale * _GL_LOG2_E)
        # Store from the row layout; moving the scales to another layout spills.
        local_rows = row_block * _GL_BLOCK_M + gl.arange(0, _GL_BLOCK_M, row_scale.type.layout)
        scale_pointers = query_scale_ptr + batch_head * storage_rows + local_rows
        if mask_rows:
            row_scale = gl.where(chunk_start + local_rows < query_sequence_end, row_scale, 0.0)
            gl.store(scale_pointers, row_scale, mask=local_rows < storage_rows)
        else:
            gl.store(scale_pointers, row_scale)
    else:
        groups: gl.constexpr = _GL_BLOCK_M // _GL_QUERY_SCALE_ROWS
        grouped = gl.reshape(smoothed, [groups, _GL_QUERY_SCALE_ROWS, _GL_HEAD_DIM])
        raw_scale = (
            gl.max(gl.max(gl.abs(grouped), axis=2), axis=1) / _GL_INT8_RANGE + _GL_SCALE_EPSILON
        )
        if mask_rows:
            group_layout: gl.constexpr = raw_scale.type.layout  # pyright: ignore[reportAttributeAccessIssue]
            group_starts = first_row + gl.arange(0, groups, group_layout) * _GL_QUERY_SCALE_ROWS
            group_valid = group_starts < query_sequence_end
            raw_scale = gl.where(group_valid, raw_scale, 1.0)
            stored_scale = gl.where(group_valid, raw_scale * (softmax_scale * _GL_LOG2_E), 0.0)
        else:
            stored_scale = raw_scale * (softmax_scale * _GL_LOG2_E)
        quantized = fragments.round_to_int8(grouped / raw_scale[:, None, None])
        storage_groups = storage_rows // _GL_QUERY_SCALE_ROWS
        fragments.store_scales(
            query_scale_ptr + batch_head * storage_groups,
            stored_scale,
            row_block * groups,
            storage_groups,
            mask_rows,
        )
    fragments.store_rows(
        query_ptr + batch_head.to(gl.int64) * storage_rows * _GL_HEAD_DIM,
        quantized,
        row_block * _GL_BLOCK_M,
        storage_rows,
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
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a K tile and emit BF16 rows and FP32 tile sums; padding rows are zero."""
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = row_block * _GL_BLOCK_M
    values, _positions, features = fragments.project_rmsnorm_rope(
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
    if mask_rows:
        rows = first_row + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, _GL_FEATURE_LAYOUT))
        valid = (rows < logical_sequence_length)[:, None] & (features >= 0)[None, :]
        values = gl.where(valid, values, 0.0)
    fragments.store_centered_key(
        stored_ptr,
        partial_ptr,
        values,
        batch,
        heads,
        head,
        row_block,
        first_row,
        storage_tiles,
        _GL_TILE_ROWS,
        mask_rows,
    )


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE)
def _value_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    mean_ptr,
    value_ptr,
    multiplier_ptr,
    log_ptr,
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    is_causal: gl.constexpr,
    mask_rows: gl.constexpr,
):
    """Project a V tile and emit transposed per-token INT8 values with their metadata.

    Non-causal V subtracts the projected input mean; causal V stays uncentered, and
    its first row block writes the zero mean.
    """
    head = gl.program_id(0)
    row_block = row_block_offset + gl.program_id(1)
    batch = gl.program_id(2)
    first_row = row_block * _GL_BLOCK_M
    input_row_start = batch * logical_sequence_length
    copy_positions = first_row + gl.arange(0, _GL_BLOCK_M, fragments.COPY_ROWS)
    if mask_rows:
        copy_positions = gl.minimum(copy_positions, logical_sequence_length - 1)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[2, 2], instr_shape=[16, 8]
    )
    accumulator = fragments.project_int8(
        input_ptr,
        weight_ptr,
        input_row_start + copy_positions,
        head * _GL_HEAD_DIM + gl.arange(0, _GL_HEAD_DIM, fragments.COPY_ROWS),
        gl.zeros([_GL_BLOCK_M, _GL_HEAD_DIM], gl.int32, mma_layout),
        input_features,
    )

    # [K64 tiles, rows, features] with lanes along rows for the transposed V stores.
    tiles: gl.constexpr = _GL_BLOCK_M // _GL_TILE_ROWS
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 32, 1], [tiles, 1, _GL_NUM_WARPS // tiles], [1, 2, 0]
    )
    tile_rows_layout: gl.constexpr = gl.SliceLayout(2, layout)
    tile_offsets = row_block * tiles + gl.arange(0, tiles, gl.SliceLayout(1, tile_rows_layout))
    rows_in_tile = gl.arange(0, _GL_TILE_ROWS, gl.SliceLayout(0, tile_rows_layout))
    sequence = tile_offsets[:, None] * _GL_TILE_ROWS + rows_in_tile[None, :]
    features = gl.arange(0, _GL_HEAD_DIM, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    positions = sequence
    if mask_rows:
        positions = gl.minimum(positions, logical_sequence_length - 1)
    input_scale = gl.load(input_scale_ptr + input_row_start + positions)
    weight_scale = gl.load(weight_scale_ptr + head * _GL_HEAD_DIM + features)
    accumulator = gl.convert_layout(
        gl.reshape(accumulator, [tiles, _GL_TILE_ROWS, _GL_HEAD_DIM]), layout
    )
    value = accumulator.to(gl.float32) * input_scale[:, :, None] * weight_scale[None, None, :]
    if bias_ptr is not None:
        bias = gl.load(bias_ptr + head * _GL_HEAD_DIM + features).to(gl.float32)
        value += bias[None, None, :]

    batch_head = batch * heads + head
    mean_pointers = mean_ptr + batch_head * _GL_HEAD_DIM + features
    if is_causal:
        if row_block == 0:
            gl.store(mean_pointers, gl.zeros_like(weight_scale))
    else:
        value -= gl.load(mean_pointers)[None, None, :]
    if mask_rows:
        value = gl.where((sequence < logical_sequence_length)[:, :, None], value, 0.0)
    value_scale = gl.max(gl.abs(value), axis=2) / _GL_INT8_RANGE + _GL_SCALE_EPSILON
    quantized = fragments.round_to_int8(value / value_scale[:, :, None])
    # NVIDIA attention reads FP16-rounded log scales.
    log_scale = gl.log2(value_scale).to(gl.float16).to(gl.float32)

    storage_rows = storage_tiles * _GL_TILE_ROWS
    value_pointers = (
        value_ptr
        + batch_head.to(gl.int64) * _GL_HEAD_DIM * storage_rows
        + features[None, None, :].to(gl.int64) * storage_rows
        + sequence[:, :, None]
    )
    metadata = batch_head * storage_rows + sequence
    multiplier = value_scale * _GL_P_UINT8_RANGE
    if mask_rows:
        in_storage = (tile_offsets < storage_tiles)[:, None]
        gl.store(value_pointers, quantized, mask=in_storage[:, :, None])
        gl.store(multiplier_ptr + metadata, multiplier, mask=in_storage)
        gl.store(log_ptr + metadata, log_scale, mask=in_storage)
    else:
        gl.store(value_pointers, quantized)
        gl.store(multiplier_ptr + metadata, multiplier)
        gl.store(log_ptr + metadata, log_scale)


def _launch_rows(sequence_rows: int, launch: Callable[..., None]) -> None:
    """Launch full 128-row tiles, then one masked tile for the remainder.

    Launch grids put heads on the fastest axis, so the programs that run together
    share their input rows in L2.
    """
    full_row_blocks = sequence_rows // _BLOCK_M
    if full_row_blocks:
        launch(full_row_blocks, 0, mask_rows=False)
    if sequence_rows % _BLOCK_M:
        launch(1, full_row_blocks, mask_rows=True)


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
    bias: torch.Tensor | None = None,
    *,
    chunk_start: int = 0,
    chunk_rows: int | None = None,
    out: QueryOutput,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill INT8 queries with Q32 or per-row scales for a supported query window."""
    query, query_scale = out
    batch, heads, storage_length, _head_dim = query.shape
    chunk_rows = input_qdata.shape[1] if chunk_rows is None else chunk_rows
    if batch == 0:
        return

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
                input_qdata.shape[1],
                row_block_offset,
                chunk_start,
                chunk_start + chunk_rows,
                storage_length // _TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                per_thread_scales=query_scale.shape[-1] == storage_length,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        _launch_rows(chunk_rows, launch)


def project_key(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    bias: torch.Tensor | None = None,
    *,
    out: KeyOutput,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill centered INT8 keys with K64 or per-key scales for supported operands."""
    key, key_scale = out
    batch, heads, storage_length, head_dim = key.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0:
        return
    stored, partials, mean = centered_projection.allocate_workspace(
        input_qdata, (batch, heads, storage_length, head_dim), tile_rows=_TILE_ROWS
    )

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
                sequence_length,
                row_block_offset,
                storage_length // _TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        _launch_rows(sequence_length, launch)
        centered_projection.finalize_mean(partials, sequence_length, out=mean)
        qk_quantization.prepare_key(
            stored.narrow(2, 0, sequence_length),
            mean,
            grouped=key_scale.shape[-1] != storage_length,
            storage_key_length=storage_length,
            out=out,
        )


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    is_causal: bool,
    out: ValueOutput,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Fill transposed per-token INT8 values, multipliers, logs, and the V mean."""
    value, multiplier, log_scale, mean = out
    batch, heads, head_dim, storage_length = value.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0:
        return

    with device_context(input_qdata.device):
        if not is_causal:
            represented_mean = _ops.dequantized_input_mean(input_qdata, input_scale)
            project_prepared_input_mean_kernel[(heads * head_dim // _MEAN_BLOCK_N, batch)](
                represented_mean,
                weight_qdata,
                weight_scale,
                mean,
                input_features=input_qdata.shape[2],
                output_features=heads * head_dim,
                block_n=_MEAN_BLOCK_N,
                block_k=_MEAN_BLOCK_K,
                bias_ptr=bias,
                num_warps=4,
            )

        def launch(row_blocks: int, row_block_offset: int, *, mask_rows: bool) -> None:
            _value_kernel[(heads, row_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                mean,
                value,
                multiplier,
                log_scale,
                sequence_length,
                row_block_offset,
                storage_length // _TILE_ROWS,
                input_features=input_qdata.shape[2],
                heads=heads,
                is_causal=is_causal,
                mask_rows=mask_rows,
                num_warps=execution_plan.num_warps,
            )

        _launch_rows(sequence_length, launch)
