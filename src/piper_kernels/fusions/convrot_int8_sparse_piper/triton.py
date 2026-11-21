"""Shared fused Q/K/V launch mechanics; backends supply compute configurations."""

# Triton's launch options and constexpr function arguments are not ordinary Python parameters.
# pyright: reportCallIssue=false, reportArgumentType=false

from collections.abc import Callable

import torch
import triton

from piper_kernels._triton.amd_compiler_workarounds import lds_only_fence_options
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.attention.sparse_piper_attention._contracts import _MEAN_ROUTING
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from . import _kernels
from ._interfaces import KeyOutput, QueryOutput, ValueOutput
from ._layout import TILE_ROWS, resolve_row_block_launches


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

        def launch(row_block_count: int, *, mask_ragged_tail: bool) -> None:
            _kernels._convrot_project_rmsnorm_rope_quantize_query_kernel[
                (row_block_count, triton.cdiv(heads, execution_plan.heads_per_program), batch)
            ](
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
                mask_ragged_tail=mask_ragged_tail,
                aligned_projection=(
                    not mask_ragged_tail
                    and input_qdata.shape[2] % execution_plan.block_k == 0
                    and heads % execution_plan.heads_per_program == 0
                ),
                block_m=execution_plan.block_m,
                block_n=head_dim * execution_plan.heads_per_program,
                block_k=execution_plan.block_k,
                group_m=execution_plan.group_m,
                num_warps=execution_plan.num_warps,
                num_stages=execution_plan.num_stages,
                **lds_only_fence_options(execution_plan.lds_only_fences),
            )

        full_row_blocks = chunk_rows // execution_plan.block_m
        if full_row_blocks:
            launch(full_row_blocks, mask_ragged_tail=False)
        if chunk_rows % execution_plan.block_m:
            launch(1, mask_ragged_tail=True)


def _launch_full_and_tail_blocks(
    input_qdata: torch.Tensor,
    heads: int,
    execution_plan: ProjectionExecutionPlan,
    launch: Callable[[tuple[int, int, int], int, bool, bool], None],
    *,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    batch, sequence_length, input_features = input_qdata.shape
    head_blocks = triton.cdiv(heads, execution_plan.heads_per_program)
    aligned_shape = (
        input_features % execution_plan.block_k == 0
        and heads % execution_plan.heads_per_program == 0
    )
    for block_count, row_block_offset, aligned_rows in resolve_row_block_launches(
        sequence_length,
        execution_plan.block_m,
        start_row_block=start_row_block,
        num_row_blocks=num_row_blocks,
    ):
        launch(
            (block_count, head_blocks, batch),
            row_block_offset,
            aligned_rows,
            aligned_shape if aligned_rows else False,
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

        def launch(
            grid: tuple[int, int, int],
            row_block_offset: int,
            aligned_rows: bool,
            aligned_projection: bool,
        ) -> None:
            _kernels._convrot_project_rmsnorm_rope_quantize_sparse_key_kernel[grid](
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

        _launch_full_and_tail_blocks(
            input_qdata,
            heads,
            execution_plan,
            launch,
            start_row_block=start_row_block,
            num_row_blocks=num_row_blocks,
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

        def launch(
            grid: tuple[int, int, int],
            row_block_offset: int,
            _aligned_rows: bool,
            aligned_projection: bool,
        ) -> None:
            _kernels._convrot_project_quantize_sparse_value_kernel[grid](
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

        _launch_full_and_tail_blocks(
            input_qdata,
            heads,
            execution_plan,
            launch,
            start_row_block=start_row_block,
            num_row_blocks=num_row_blocks,
        )
