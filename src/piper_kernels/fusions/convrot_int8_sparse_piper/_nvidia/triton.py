"""NVIDIA launches over shared fused projection kernels and operand formats."""

# Triton's launch options and constexpr function arguments are not ordinary Python parameters.
# pyright: reportCallIssue=false, reportArgumentType=false

import torch
import triton
from triton.language.extra.cuda import libdevice

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MEAN_ROUTING

from .. import _kernels
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from .._layout import HEAD_DIM, TILE_ROWS

_QUERY_BLOCK_M = TILE_ROWS
_CONTEXT_BLOCK_M = 2 * TILE_ROWS
_BLOCK_K = 128
_HEADS_PER_PROGRAM = 2
_BLOCK_N = HEAD_DIM * _HEADS_PER_PROGRAM


def project_query(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
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
) -> None:
    """Launch Q32 quantization and Q64 summaries for a validated query window."""
    query, query_scale, query_summary = out
    batch, heads, storage_sequence_length, _ = query.shape
    sequence_length = input_qdata.shape[1]
    rotary_dim = cos.shape[1]
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else query_scale
    with device_context(input_qdata.device):

        def launch(row_block_count: int, *, mask_ragged_tail: bool) -> None:
            _kernels._convrot_project_rmsnorm_rope_quantize_query_kernel[
                (row_block_count, triton.cdiv(heads, _HEADS_PER_PROGRAM), batch)
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
                heads=heads,
                heads_per_program=_HEADS_PER_PROGRAM,
                head_dim=HEAD_DIM,
                rotary_dim=rotary_dim,
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=has_block_lengths,
                mask_ragged_tail=mask_ragged_tail,
                aligned_projection=(
                    not mask_ragged_tail
                    and input_qdata.shape[2] % _BLOCK_K == 0
                    and heads % _HEADS_PER_PROGRAM == 0
                ),
                block_m=_QUERY_BLOCK_M,
                block_n=_BLOCK_N,
                block_k=_BLOCK_K,
                rsqrt_fn=libdevice.rsqrt_rn,
                num_warps=8,
                num_stages=3,
            )

        full_row_blocks = chunk_rows // _QUERY_BLOCK_M
        if full_row_blocks:
            launch(full_row_blocks, mask_ragged_tail=False)
        if chunk_rows % _QUERY_BLOCK_M:
            launch(1, mask_ragged_tail=True)


def project_key(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    *,
    out: KeyOutput,
) -> None:
    """Launch K64 quantization and routing summaries for global key storage."""
    key, key_scale, key_summary, key_aux = out
    batch, heads, storage_sequence_length, _ = key.shape
    logical_sequence_length = input_qdata.shape[1]
    rotary_dim = cos.shape[1]
    mean_pool_summary = routing_mode == _MEAN_ROUTING
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else key_scale
    with device_context(input_qdata.device):

        def launch(row_block_count: int, row_block_offset: int, *, aligned_rows: bool) -> None:
            _kernels._convrot_project_quantize_key_kernel[
                (
                    row_block_count,
                    triton.cdiv(heads, _HEADS_PER_PROGRAM),
                    batch,
                )
            ](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                norm_weight,
                cos,
                sin,
                key,
                key_scale,
                key_summary,
                key_aux,
                block_lengths_ptr,
                batch * logical_sequence_length,
                logical_sequence_length,
                storage_sequence_length,
                row_block_offset,
                input_features=input_qdata.shape[2],
                heads=heads,
                heads_per_program=_HEADS_PER_PROGRAM,
                head_dim=HEAD_DIM,
                rotary_dim=rotary_dim,
                norm_epsilon=norm_epsilon,
                mean_pool_summary=mean_pool_summary,
                mask_block_lengths=has_block_lengths,
                aligned_projection=(
                    aligned_rows
                    and input_qdata.shape[2] % _BLOCK_K == 0
                    and heads % _HEADS_PER_PROGRAM == 0
                ),
                mask_ragged_tail=not aligned_rows,
                block_m=_CONTEXT_BLOCK_M,
                block_n=_BLOCK_N,
                block_k=_BLOCK_K,
                rsqrt_fn=libdevice.rsqrt_rn,
                num_warps=8,
                num_stages=3,
            )

        full_row_blocks = logical_sequence_length // _CONTEXT_BLOCK_M
        if full_row_blocks:
            launch(full_row_blocks, 0, aligned_rows=True)
        if logical_sequence_length % _CONTEXT_BLOCK_M:
            launch(1, full_row_blocks, aligned_rows=False)


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    emit_block_mean: bool,
    out: ValueOutput,
) -> None:
    """Launch projected means and centered tile-scaled INT8 values."""
    value, value_scale_multiplier, value_mean, block_mean = out
    batch, heads, _, storage_sequence_length = value.shape
    sequence_length = input_qdata.shape[1]
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else value_mean
    with device_context(input_qdata.device):
        _kernels._project_prepared_input_mean_kernel[
            (triton.cdiv(heads * HEAD_DIM, _BLOCK_N), batch)
        ](
            input_mean,
            weight_qdata,
            weight_scale,
            value_mean,
            input_features=input_qdata.shape[2],
            output_features=heads * HEAD_DIM,
            block_n=_BLOCK_N,
            block_k=_BLOCK_K,
            num_warps=8,
        )

        def launch(row_block_count: int, row_block_offset: int, *, aligned_rows: bool) -> None:
            _kernels._convrot_project_quantize_sparse_value_kernel[
                (
                    row_block_count,
                    triton.cdiv(heads, _HEADS_PER_PROGRAM),
                    batch,
                )
            ](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                value_mean,
                value,
                value_scale_multiplier,
                block_mean,
                block_lengths_ptr,
                batch * sequence_length,
                sequence_length,
                storage_sequence_length,
                row_block_offset,
                input_features=input_qdata.shape[2],
                heads=heads,
                heads_per_program=_HEADS_PER_PROGRAM,
                head_dim=HEAD_DIM,
                aligned_projection=(
                    aligned_rows
                    and input_qdata.shape[2] % _BLOCK_K == 0
                    and heads % _HEADS_PER_PROGRAM == 0
                ),
                mask_block_lengths=has_block_lengths,
                emit_block_mean=emit_block_mean,
                block_m=_CONTEXT_BLOCK_M,
                block_n=_BLOCK_N,
                block_k=_BLOCK_K,
                num_warps=8,
                num_stages=3,
            )

        full_row_blocks = sequence_length // _CONTEXT_BLOCK_M
        if full_row_blocks:
            launch(full_row_blocks, 0, aligned_rows=True)
        if sequence_length % _CONTEXT_BLOCK_M:
            launch(1, full_row_blocks, aligned_rows=False)
