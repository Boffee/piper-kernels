"""RDNA4 Gluon kernels and dispatch for fused ConvRot INT8 dense-Piper Q/K/V projections."""

# Gluon exposes low-level signatures and layouts that are not modeled by static type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    resolve_qk_quantization,
)
from piper_kernels.attention.piper_attention import _quantization as piper_quantization
from piper_kernels.fusions.convrot_int8_projection import triton as shared_projection
from piper_kernels.fusions.convrot_int8_projection._amd import fragments
from piper_kernels.fusions.convrot_int8_projection._amd import policy as amd_policy
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .. import triton as projection
from .._interfaces import KeyOutput, QueryOutput, ValueOutput

_DO_NOT_SPECIALIZE_VALUE = (*fragments.DO_NOT_SPECIALIZE, "chunk_count")


@gluon.jit
def _store_query_half(
    values,
    cos_ptr,
    sin_ptr,
    query_base_ptr,
    query_scale_ptr,
    batch_head_index,
    storage_tile,
    storage_tiles,
    seq_first_row,
    query_sequence_end,
    logical_sequence_length,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    softmax_scale: gl.constexpr,
    mask_rows: gl.constexpr,
):
    values, _row_valid, block_length = fragments.rope_and_mask_half(
        values,
        cos_ptr,
        sin_ptr,
        seq_first_row,
        query_sequence_end,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mask_rows,
        None,
        0,
        False,
    )
    fragments.quantize_and_store_query_half(
        values,
        query_base_ptr,
        query_scale_ptr,
        batch_head_index,
        storage_tile,
        storage_tiles,
        seq_first_row,
        query_sequence_end,
        block_length,
        head_dim,
        softmax_scale,
        False,
        mask_rows,
    )


@gluon.jit
def _store_key_half(
    values,
    cos_ptr,
    sin_ptr,
    stored_base_ptr,
    tile_index,
    storage_tiles,
    seq_first_row,
    logical_sequence_length,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    mask_rows: gl.constexpr,
):
    values, _row_valid, _block_length = fragments.rope_and_mask_half(
        values,
        cos_ptr,
        sin_ptr,
        seq_first_row,
        logical_sequence_length,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mask_rows,
        None,
        0,
        False,
    )
    fragments.store_head_major(
        stored_base_ptr,
        values.to(gl.bfloat16),
        tile_index,
        storage_tiles * fragments.GL_TILE_ROWS,
        head_dim,
    )


@gluon.jit
def _store_value_half(
    values,
    mean_ptr,
    value_base_ptr,
    multiplier_ptr,
    log_ptr,
    batch_head_index,
    tile_index,
    storage_tiles,
    chunk_count,
    chunk_index,
    seq_first_row,
    logical_sequence_length,
    head_dim: gl.constexpr,
    mask_rows: gl.constexpr,
):
    column_offsets = gl.arange(0, 128, gl.SliceLayout(0, fragments.TILE_HALF_LAYOUT))
    batch_head_64 = batch_head_index.to(gl.int64)
    chunk_64 = chunk_index.to(gl.int64)
    chunk_count_64 = chunk_count.to(gl.int64)
    storage_rows_64 = storage_tiles.to(gl.int64) * fragments.GL_TILE_ROWS
    if head_dim == 64:
        mean_offsets = (
            (batch_head_64 + (column_offsets // 64).to(gl.int64)) * chunk_count_64 + chunk_64
        ) * 64 + (column_offsets % 64)
    else:
        mean_offsets = (batch_head_64 * chunk_count_64 + chunk_64) * 128 + column_offsets
    chunk_mean = gl.load(mean_ptr + mean_offsets).to(gl.float32)
    centered = values - chunk_mean[None, :]
    if mask_rows:
        rows_local = gl.arange(0, 64, gl.SliceLayout(1, fragments.TILE_HALF_LAYOUT))
        row_valid = (seq_first_row + rows_local) < logical_sequence_length
        centered = gl.where(row_valid[:, None], centered, 0.0)

    if head_dim == 64:
        centered_heads = gl.reshape(centered, [64, 2, 64])
        residual_scale = fragments.compute_int8_scale(gl.max(gl.abs(centered_heads), axis=2))
        mean_floor = (
            gl.max(gl.abs(gl.reshape(chunk_mean, [2, 64])), axis=1)
            * fragments.GL_VALUE_MEAN_SCALE_FLOOR
        )
        mean_floor_bcast = gl.convert_layout(
            mean_floor, gl.SliceLayout(0, residual_scale.type.layout)
        )
        value_scale = gl.maximum(residual_scale, mean_floor_bcast[None, :])
        value_codes = gl.reshape(
            fragments.round_to_int8(centered_heads / value_scale[:, :, None]), [64, 128]
        )
        meta_layout: gl.constexpr = value_scale.type.layout
        meta_rows = gl.arange(0, 64, gl.SliceLayout(1, meta_layout))
        meta_heads = gl.arange(0, 2, gl.SliceLayout(0, meta_layout))
        meta_offsets = (batch_head_64 + meta_heads[None, :].to(gl.int64)) * storage_rows_64 + (
            tile_index * fragments.GL_TILE_ROWS + meta_rows[:, None]
        )
    else:
        residual_scale = fragments.compute_int8_scale(gl.max(gl.abs(centered), axis=1))
        mean_floor = gl.max(gl.abs(chunk_mean), axis=0) * fragments.GL_VALUE_MEAN_SCALE_FLOOR
        value_scale = gl.maximum(residual_scale, mean_floor)
        value_codes = fragments.round_to_int8(centered / value_scale[:, None])
        meta_rows = gl.arange(0, 64, value_scale.type.layout)
        meta_offsets = batch_head_64 * storage_rows_64 + (
            tile_index * fragments.GL_TILE_ROWS + meta_rows
        )

    gl.store(multiplier_ptr + meta_offsets, value_scale * fragments.GL_P_UINT8_RANGE)
    gl.store(log_ptr + meta_offsets, gl.log2(value_scale))
    fragments.store_amd_wmma_value_codes(
        value_base_ptr, value_codes, tile_index, storage_tiles, head_dim
    )


@gluon.jit(do_not_specialize=fragments.DO_NOT_SPECIALIZE_QUERY)
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
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    softmax_scale: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a Q window tile and emit INT8 queries and Q32 base-2 scales."""
    half_top, half_bot, batch_head_index, batch_head_stride, storage_tile_top, seq_first_row = (
        fragments.project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            norm_weight_ptr,
            row_block_offset,
            chunk_start,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            norm_epsilon,
            True,
            mask_rows,
            group_m,
        )
    )
    query_base_ptr = query_ptr + batch_head_stride * fragments.GL_TILE_ROWS * head_dim
    _store_query_half(
        half_top,
        cos_ptr,
        sin_ptr,
        query_base_ptr,
        query_scale_ptr,
        batch_head_index,
        storage_tile_top,
        storage_tiles,
        seq_first_row,
        query_sequence_end,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        softmax_scale,
        mask_rows,
    )
    storage_tile_bot = storage_tile_top + 1
    if (not mask_rows) or (
        (storage_tile_bot < storage_tiles)
        and (seq_first_row + fragments.GL_TILE_ROWS < query_sequence_end)
    ):
        _store_query_half(
            half_bot,
            cos_ptr,
            sin_ptr,
            query_base_ptr,
            query_scale_ptr,
            batch_head_index,
            storage_tile_bot,
            storage_tiles,
            seq_first_row + fragments.GL_TILE_ROWS,
            query_sequence_end,
            logical_sequence_length,
            head_dim,
            rotary_dim,
            softmax_scale,
            mask_rows,
        )


@gluon.jit(do_not_specialize=fragments.DO_NOT_SPECIALIZE)
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
    logical_sequence_length,
    row_block_offset,
    storage_tiles,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    rotary_dim: gl.constexpr,
    norm_epsilon: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a K tile and emit BF16 rows for K1024 chunk quantization."""
    half_top, half_bot, _batch_head_index, batch_head_stride, tile_top, seq_first_row = (
        fragments.project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            norm_weight_ptr,
            row_block_offset,
            0,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            norm_epsilon,
            True,
            mask_rows,
            group_m,
        )
    )
    stored_base_ptr = stored_ptr + batch_head_stride * fragments.GL_TILE_ROWS * head_dim
    _store_key_half(
        half_top,
        cos_ptr,
        sin_ptr,
        stored_base_ptr,
        tile_top,
        storage_tiles,
        seq_first_row,
        logical_sequence_length,
        head_dim,
        rotary_dim,
        mask_rows,
    )
    tile_bot = tile_top + 1
    if (not mask_rows) or (
        (tile_bot < storage_tiles)
        and (seq_first_row + fragments.GL_TILE_ROWS < logical_sequence_length)
    ):
        _store_key_half(
            half_bot,
            cos_ptr,
            sin_ptr,
            stored_base_ptr,
            tile_bot,
            storage_tiles,
            seq_first_row + fragments.GL_TILE_ROWS,
            logical_sequence_length,
            head_dim,
            rotary_dim,
            mask_rows,
        )


@gluon.jit(do_not_specialize=_DO_NOT_SPECIALIZE_VALUE)
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
    chunk_count,
    num_row_blocks,
    input_features: gl.constexpr,
    heads: gl.constexpr,
    head_dim: gl.constexpr,
    mask_rows: gl.constexpr,
    group_m: gl.constexpr,
):
    """Project a V tile, subtract its V1024 chunk mean, and emit AMD-WMMA INT8 values."""
    half_top, half_bot, batch_head_index, batch_head_stride, tile_top, seq_first_row = (
        fragments.project_dequantized_halves(
            input_ptr,
            input_scale_ptr,
            weight_ptr,
            weight_scale_ptr,
            bias_ptr,
            None,
            row_block_offset,
            0,
            logical_sequence_length,
            storage_tiles,
            num_row_blocks,
            input_features,
            heads,
            head_dim,
            1e-5,
            False,
            mask_rows,
            group_m,
        )
    )
    value_base_ptr = value_ptr + batch_head_stride * fragments.GL_TILE_ROWS * head_dim
    # Each 1024-token chunk is 8 consecutive 128-row tiles, so top and bot share one chunk.
    chunk_index = (tile_top * fragments.GL_TILE_ROWS) // 1024
    _store_value_half(
        half_top,
        mean_ptr,
        value_base_ptr,
        multiplier_ptr,
        log_ptr,
        batch_head_index,
        tile_top,
        storage_tiles,
        chunk_count,
        chunk_index,
        seq_first_row,
        logical_sequence_length,
        head_dim,
        mask_rows,
    )
    tile_bot = tile_top + 1
    if (not mask_rows) or (
        (tile_bot < storage_tiles)
        and (seq_first_row + fragments.GL_TILE_ROWS < logical_sequence_length)
    ):
        _store_value_half(
            half_bot,
            mean_ptr,
            value_base_ptr,
            multiplier_ptr,
            log_ptr,
            batch_head_index,
            tile_bot,
            storage_tiles,
            chunk_count,
            chunk_index,
            seq_first_row + fragments.GL_TILE_ROWS,
            logical_sequence_length,
            head_dim,
            mask_rows,
        )


def _iter_row_block_launches(rows: int) -> tuple[tuple[int, int, bool], ...]:
    """Return ``(row_blocks, row_block_offset, mask_rows)`` launches for dense row windows."""
    full_blocks = rows // amd_policy.GLUON_BLOCK_M
    launches: list[tuple[int, int, bool]] = []
    if full_blocks:
        launches.append((full_blocks, 0, False))
    if rows % amd_policy.GLUON_BLOCK_M:
        launches.append((1, full_blocks, True))
    return tuple(launches)


def _kernel_options(operation: amd_policy.ProjectionOperation) -> dict[str, object]:
    """Return RDNA4 Gluon kernel launch options for dense Q/K/V projections."""
    # Capping V at 6-8 waves/EU avoids L2 contention across warps (~2% faster on RDNA4).
    return fragments.build_kernel_options(waves_per_eu="6,8" if operation == "value" else None)


def default_execution_plan(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    *,
    operation: amd_policy.ProjectionOperation,
    head_dim: int,
    rotary_dim: int = 0,
    storage_sequence_length: int | None = None,
    per_thread_scales: bool = False,
) -> amd_policy.AmdExecutionPlan:
    """Resolve operand metadata and select RDNA4 Gluon or Triton execution."""
    sequence_length = input_qdata.shape[1]
    resolved_storage_sequence = (
        (sequence_length + amd_policy.TILE_ROWS - 1) // amd_policy.TILE_ROWS * amd_policy.TILE_ROWS
        if storage_sequence_length is None
        else storage_sequence_length
    )
    heads = (
        weight_qdata.shape[0] // head_dim
        if head_dim > 0 and weight_qdata.shape[0] % head_dim == 0
        else 0
    )
    return amd_policy.select_execution_plan(
        operation,
        head_dim=head_dim,
        input_features=input_qdata.shape[2],
        heads=heads,
        rotary_dim=rotary_dim,
        operands_aligned=(
            not (operation == "query" and per_thread_scales)
            and fragments.operands_aligned(
                input_qdata,
                weight_qdata,
                storage_sequence_length=resolved_storage_sequence,
                rotary_dim=rotary_dim,
                head_dim=head_dim,
                output_element_bytes=2 if operation == "key" else 1,
            )
        ),
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
    bias: torch.Tensor | None = None,
    *,
    chunk_start: int = 0,
    chunk_rows: int | None = None,
    out: QueryOutput,
    execution_plan: ProjectionExecutionPlan | None = None,
) -> None:
    """Project a query window using RDNA4 Gluon when supported, else shared Triton."""
    query, query_scale = out
    batch, heads, storage_length, head_dim = query.shape
    per_thread_scales = resolve_qk_quantization(query_scale, storage_length) == "per_thread"
    resolved_chunk_rows = input_qdata.shape[1] if chunk_rows is None else chunk_rows
    plan = (
        default_execution_plan(
            input_qdata,
            weight_qdata,
            operation="query",
            head_dim=head_dim,
            rotary_dim=cos.shape[1],
            storage_sequence_length=storage_length,
            per_thread_scales=per_thread_scales,
        )
        if execution_plan is None
        else execution_plan
    )
    if per_thread_scales or not (
        isinstance(plan, amd_policy.AmdExecutionPlan) and plan.kernel == "gluon"
    ):
        projection.project_query(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            norm_weight,
            cos,
            sin,
            norm_epsilon,
            softmax_scale,
            bias,
            chunk_start=chunk_start,
            chunk_rows=chunk_rows,
            execution_plan=plan,
            out=out,
        )
        return
    if batch == 0 or resolved_chunk_rows == 0:
        return
    col_blocks = (heads * head_dim) // 128
    options = _kernel_options("query")
    with device_context(input_qdata.device):
        for row_blocks, row_block_offset, mask_rows in _iter_row_block_launches(
            resolved_chunk_rows
        ):
            _query_kernel[(row_blocks * col_blocks, batch)](
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
                chunk_start + resolved_chunk_rows,
                storage_length // amd_policy.TILE_ROWS,
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mask_rows=mask_rows,
                group_m=plan.group_m,
                num_warps=plan.num_warps,
                **options,
            )


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
    execution_plan: ProjectionExecutionPlan | None = None,
) -> None:
    """Project K1024-centered keys using RDNA4 Gluon when supported, else shared Triton."""
    key_int8, key_scale, key_center = out
    batch, heads, storage_length, head_dim = key_int8.shape
    plan = (
        default_execution_plan(
            input_qdata,
            weight_qdata,
            operation="key",
            head_dim=head_dim,
            rotary_dim=cos.shape[1],
            storage_sequence_length=storage_length,
        )
        if execution_plan is None
        else execution_plan
    )
    if not (isinstance(plan, amd_policy.AmdExecutionPlan) and plan.kernel == "gluon"):
        projection.project_key(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            norm_weight,
            cos,
            sin,
            norm_epsilon,
            bias,
            execution_plan=plan,
            out=out,
        )
        return
    sequence_length = input_qdata.shape[1]
    if batch == 0 or sequence_length == 0:
        return
    stored = input_qdata.new_empty((batch, heads, storage_length, head_dim), dtype=torch.bfloat16)
    col_blocks = (heads * head_dim) // 128
    options = _kernel_options("key")
    with device_context(input_qdata.device):
        for row_blocks, row_block_offset, mask_rows in _iter_row_block_launches(sequence_length):
            _key_kernel[(row_blocks * col_blocks, batch)](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                norm_weight,
                cos,
                sin,
                stored,
                sequence_length,
                row_block_offset,
                storage_length // amd_policy.TILE_ROWS,
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                mask_rows=mask_rows,
                group_m=plan.group_m,
                num_warps=plan.num_warps,
                **options,
            )
        piper_quantization.prepare_chunk_centered_key(
            stored.narrow(2, 0, sequence_length),
            grouped=resolve_qk_quantization(key_scale, storage_length) == "per_warp",
            storage_key_length=storage_length,
            out=(key_int8, key_scale, key_center),
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
    execution_plan: ProjectionExecutionPlan | None = None,
) -> None:
    """Project V1024-centered AMD-WMMA values using RDNA4 Gluon when supported, else Triton."""
    value, multiplier, log_scale, mean = out
    batch, heads, head_dim, storage_length = value.shape
    plan = (
        default_execution_plan(
            input_qdata,
            weight_qdata,
            operation="value",
            head_dim=head_dim,
            storage_sequence_length=storage_length,
        )
        if execution_plan is None
        else execution_plan
    )
    if not (isinstance(plan, amd_policy.AmdExecutionPlan) and plan.kernel == "gluon"):
        projection.project_value(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            is_causal=is_causal,
            packed_amd=True,
            execution_plan=plan,
            out=out,
        )
        return
    sequence_length = input_qdata.shape[1]
    if batch == 0 or sequence_length == 0:
        return
    col_blocks = (heads * head_dim) // 128
    options = _kernel_options("value")
    with device_context(input_qdata.device):
        shared_projection.project_value_chunk_means(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            block_n=plan.resolve_value_mean_block_n(head_dim),
            block_k=plan.block_k,
            out=mean,
        )
        for row_blocks, row_block_offset, mask_rows in _iter_row_block_launches(sequence_length):
            _value_kernel[(row_blocks * col_blocks, batch)](
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
                storage_length // amd_policy.TILE_ROWS,
                mean.shape[2],
                row_blocks,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                mask_rows=mask_rows,
                group_m=plan.group_m,
                num_warps=plan.num_warps,
                **options,
            )
