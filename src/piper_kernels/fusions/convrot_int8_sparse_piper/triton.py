"""Shared fused Q/K/V Triton kernels and launchers; backends supply compute configurations."""

# Triton parameters, launch options, and constexpr args are not ordinary Python parameters.
# pyright: reportCallIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false

import torch
import triton
import triton.language as tl

from piper_kernels._triton.amd_compiler_workarounds import lds_only_fence_options
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper import triton as sparse_piper_kernels
from piper_kernels.attention.kernels.sparse_piper.layout import (
    QUERY_SCALE_ROWS,
    TILE_ROWS,
    ValueFormat,
)
from piper_kernels.attention.sparse_piper_attention._contracts import _MEAN_ROUTING
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan
from piper_kernels.fusions.convrot_int8_projection.triton import (
    project_rmsnorm_rope_tile,
    project_tile,
)
from piper_kernels.fusions.convrot_int8_projection.triton import (
    projection_tile_ids as _projection_tile_ids,
)

from ._interfaces import KeyOutput, QueryOutput, ValueOutput

_JIT_QUERY_SCALE_ROWS = tl.constexpr(QUERY_SCALE_ROWS)
_JIT_VALUE_TILE_ROWS = tl.constexpr(TILE_ROWS)


@triton.jit
def _assert_sparse_tile_contract(
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    rotary_dim: tl.constexpr = 0,
    validate_rotary: tl.constexpr = False,
):
    tl.static_assert(heads_per_program == 1 or heads_per_program == 2)  # noqa: PLR1714
    tl.static_assert(block_n == heads_per_program * head_dim)
    tl.static_assert(head_dim == 64 or head_dim == 128)  # noqa: PLR1714
    if validate_rotary:
        tl.static_assert(rotary_dim <= head_dim)
        tl.static_assert(rotary_dim % 2 == 0)


@triton.jit
def _convrot_project_rmsnorm_rope_quantize_query_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    rows,
    chunk_start,
    chunk_rows,
    logical_sequence_length,
    query_sequence_end,
    storage_sequence_length,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    softmax_scale: tl.constexpr,
    mean_pool_summary: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    aligned_projection: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    round_rsqrt_to_nearest: tl.constexpr = False,
    group_m: tl.constexpr = 0,
    bias_ptr=None,
):
    """Project a Q64 tile and emit Q32 INT8 plus route summaries."""
    tl.static_assert(block_m == 64)
    _assert_sparse_tile_contract(heads_per_program, head_dim, block_n, rotary_dim, True)

    storage_query_block, head_block = _projection_tile_ids(group_m)
    if mask_ragged_tail:
        storage_query_block = chunk_rows // block_m
    global_query_block = chunk_start // block_m + storage_query_block
    batch = tl.program_id(2)
    storage_sequence_offsets = storage_query_block * block_m + tl.arange(0, block_m)
    global_sequence_offsets = chunk_start + storage_sequence_offsets
    row_offsets = batch * logical_sequence_length + global_sequence_offsets
    projection_feature_offsets = tl.arange(0, block_n)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + projection_feature_offsets
    rope = project_rmsnorm_rope_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        row_offsets,
        weight_offsets,
        global_sequence_offsets,
        rows,
        query_sequence_end,
        input_features,
        heads * head_dim,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        aligned_projection,
        mask_ragged_tail,
        block_m,
        block_n,
        block_k,
        round_rsqrt_to_nearest,
        bias_ptr=bias_ptr,
    )

    sparse_piper_kernels.store_query_tile(
        rope,
        query_ptr,
        query_scale_ptr,
        query_summary_ptr,
        block_lengths_ptr,
        batch,
        heads,
        head_offsets,
        global_sequence_offsets,
        storage_sequence_offsets,
        query_sequence_end,
        storage_sequence_length,
        global_query_block,
        storage_query_block,
        softmax_scale,
        mean_pool_summary,
        mask_block_lengths,
        mask_ragged_tail,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_QUERY_SCALE_ROWS,
    )


@triton.jit
def _convrot_project_rmsnorm_rope_quantize_sparse_key_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    key_ptr,
    key_scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    rows,
    logical_sequence_length,
    storage_sequence_length,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    mean_pool_summary: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    aligned_projection: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    round_rsqrt_to_nearest: tl.constexpr = False,
    group_m: tl.constexpr = 0,
    bias_ptr=None,
):
    """Project K tiles and emit K64-centered INT8 keys, scales, centers, and route summaries."""
    tl.static_assert(block_m % _JIT_VALUE_TILE_ROWS == 0)
    _assert_sparse_tile_contract(heads_per_program, head_dim, block_n, rotary_dim, True)

    row_block, head_block = _projection_tile_ids(group_m)
    row_block += row_block_offset
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    row_offsets = batch * logical_sequence_length + sequence_offsets
    projection_feature_offsets = tl.arange(0, block_n)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + projection_feature_offsets
    rope = project_rmsnorm_rope_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        row_offsets,
        weight_offsets,
        sequence_offsets,
        rows,
        logical_sequence_length,
        input_features,
        heads * head_dim,
        heads_per_program,
        head_dim,
        rotary_dim,
        norm_epsilon,
        aligned_projection,
        mask_ragged_tail,
        block_m,
        block_n,
        block_k,
        round_rsqrt_to_nearest,
        bias_ptr=bias_ptr,
    )
    sparse_piper_kernels.store_key_tile(
        rope,
        key_ptr,
        key_scale_ptr,
        key_center_ptr,
        key_center_scale_ptr,
        key_summary_ptr,
        key_aux_ptr,
        block_lengths_ptr,
        batch,
        heads,
        head_offsets,
        sequence_offsets,
        logical_sequence_length,
        storage_sequence_length,
        row_block,
        mean_pool_summary,
        mask_block_lengths,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_VALUE_TILE_ROWS,
    )


@triton.jit
def _convrot_project_quantize_sparse_value_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    value_ptr,
    value_scale_ptr,
    value_mean_ptr,
    block_lengths_ptr,
    rows,
    logical_sequence_length,
    storage_sequence_length,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    aligned_projection: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    value_format: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr = 0,
    bias_ptr=None,
):
    """Project one or two heads over whole K64 tiles and emit sparse Piper's V format."""
    tl.static_assert(block_m % _JIT_VALUE_TILE_ROWS == 0)
    _assert_sparse_tile_contract(heads_per_program, head_dim, block_n)

    row_block, head_block = _projection_tile_ids(group_m)
    row_block += row_block_offset
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    row_offsets = batch * logical_sequence_length + sequence_offsets
    projection_feature_offsets = tl.arange(0, block_n)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + projection_feature_offsets
    projection = project_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        row_offsets,
        weight_offsets,
        rows,
        input_features,
        heads * head_dim,
        aligned_projection,
        block_m,
        block_n,
        block_k,
        bias_ptr=bias_ptr,
    )
    projection = tl.reshape(projection, (block_m, heads_per_program, head_dim))
    sparse_piper_kernels.store_value_tile(
        projection,
        value_ptr,
        value_scale_ptr,
        value_mean_ptr,
        block_lengths_ptr,
        batch,
        heads,
        head_offsets,
        sequence_offsets,
        logical_sequence_length,
        storage_sequence_length,
        row_block,
        mask_block_lengths,
        heads_per_program,
        head_dim,
        block_m,
        _JIT_VALUE_TILE_ROWS,
        value_format,
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
    execution_plan: ProjectionExecutionPlan,
    chunk_start: int,
    chunk_rows: int,
    out: QueryOutput,
    bias: torch.Tensor | None = None,
) -> None:
    """Launch Q32 quantization and Q64 summaries for a validated query window."""
    query, query_scale, query_summary = out
    batch, heads, storage_sequence_length, head_dim = query.shape
    sequence_length = input_qdata.shape[1]
    rotary_dim = cos.shape[1]
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else query_scale
    with device_context(input_qdata.device):
        for (
            grid,
            _row_block_offset,
            aligned_rows,
            aligned_projection,
        ) in execution_plan.iter_row_block_launches(batch, chunk_rows, input_qdata.shape[2], heads):
            _convrot_project_rmsnorm_rope_quantize_query_kernel[grid](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                norm_weight,
                cos,
                sin,
                query,
                query_scale,
                query_summary,
                block_lengths_ptr,
                batch * sequence_length,
                chunk_start,
                chunk_rows,
                sequence_length,
                chunk_start + chunk_rows,
                storage_sequence_length,
                input_features=input_qdata.shape[2],
                bias_ptr=bias,
                heads=heads,
                heads_per_program=execution_plan.heads_per_program,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                round_rsqrt_to_nearest=execution_plan.round_rsqrt_to_nearest,
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=has_block_lengths,
                mask_ragged_tail=not aligned_rows,
                aligned_projection=aligned_projection,
                block_m=execution_plan.block_m,
                block_n=head_dim * execution_plan.heads_per_program,
                block_k=execution_plan.block_k,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                num_stages=execution_plan.num_stages,
                **lds_only_fence_options(execution_plan.lds_only_fences),
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
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    execution_plan: ProjectionExecutionPlan,
    out: KeyOutput,
    bias: torch.Tensor | None = None,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    """Prepare K64-centered INT8 keys, scales, centers, and FP32 routing summaries."""
    key, key_scale, key_center, key_center_scale, key_summary, key_aux = out
    batch, heads, storage_sequence_length, head_dim = key.shape
    sequence_length = input_qdata.shape[1]
    rotary_dim = cos.shape[1]
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else key_scale
    block_n = head_dim * execution_plan.heads_per_program

    with device_context(input_qdata.device):
        for (
            grid,
            row_block_offset,
            aligned_rows,
            aligned_projection,
        ) in execution_plan.iter_row_block_launches(
            batch,
            sequence_length,
            input_qdata.shape[2],
            heads,
            start_row_block=start_row_block,
            num_row_blocks=num_row_blocks,
        ):
            _convrot_project_rmsnorm_rope_quantize_sparse_key_kernel[grid](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                norm_weight,
                cos,
                sin,
                key,
                key_scale,
                key_center,
                key_center_scale,
                key_summary,
                key_aux,
                block_lengths_ptr,
                batch * sequence_length,
                sequence_length,
                storage_sequence_length,
                row_block_offset,
                input_features=input_qdata.shape[2],
                heads=heads,
                heads_per_program=execution_plan.heads_per_program,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                norm_epsilon=norm_epsilon,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=has_block_lengths,
                mask_ragged_tail=not aligned_rows,
                aligned_projection=aligned_projection,
                block_m=execution_plan.block_m,
                block_n=block_n,
                block_k=execution_plan.block_k,
                round_rsqrt_to_nearest=execution_plan.round_rsqrt_to_nearest,
                group_m=execution_plan.group_m,
                bias_ptr=bias,
                num_warps=execution_plan.num_warps,
                num_stages=execution_plan.num_stages,
                **lds_only_fence_options(execution_plan.lds_only_fences),
            )


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    execution_plan: ProjectionExecutionPlan,
    value_format: ValueFormat,
    out: ValueOutput,
    bias: torch.Tensor | None = None,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    """Launch K64 tile-centered INT8 values, multipliers, and BF16 per-tile means."""
    value, value_scale_multiplier, value_mean = out
    batch, heads, storage_tiles, head_dim = value_mean.shape
    storage_sequence_length = storage_tiles * TILE_ROWS
    block_n = head_dim * execution_plan.heads_per_program
    sequence_length = input_qdata.shape[1]
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else value_scale_multiplier

    with device_context(input_qdata.device):
        for (
            grid,
            row_block_offset,
            _aligned_rows,
            aligned_projection,
        ) in execution_plan.iter_row_block_launches(
            batch,
            sequence_length,
            input_qdata.shape[2],
            heads,
            start_row_block=start_row_block,
            num_row_blocks=num_row_blocks,
        ):
            _convrot_project_quantize_sparse_value_kernel[grid](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                value,
                value_scale_multiplier,
                value_mean,
                block_lengths_ptr,
                batch * sequence_length,
                sequence_length,
                storage_sequence_length,
                row_block_offset,
                input_features=input_qdata.shape[2],
                bias_ptr=bias,
                heads=heads,
                heads_per_program=execution_plan.heads_per_program,
                head_dim=head_dim,
                aligned_projection=aligned_projection,
                mask_block_lengths=has_block_lengths,
                value_format=int(value_format),
                block_m=execution_plan.block_m,
                block_n=block_n,
                block_k=execution_plan.block_k,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                num_stages=execution_plan.num_stages,
                **lds_only_fence_options(execution_plan.lds_only_fences),
            )
