"""Shared fused Q/K/V launch mechanics; backends supply compute configurations."""

# Triton's launch options and constexpr function arguments are not ordinary Python parameters.
# pyright: reportCallIssue=false, reportArgumentType=false

from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MEAN_ROUTING

from . import _kernels
from ._interfaces import KeyOutput, QueryOutput, ValueOutput


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    """Compute tiles do not change the shared attention scale/summary groups."""

    block_m: int
    block_k: int
    heads_per_program: int
    num_warps: int
    num_stages: int
    group_m: int = 0
    rsqrt_fn: Callable | None = None


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
    config: ProjectionConfig,
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
                (row_block_count, triton.cdiv(heads, config.heads_per_program), batch)
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
                heads_per_program=config.heads_per_program,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rsqrt_fn=config.rsqrt_fn,
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                mean_pool_summary=routing_mode == _MEAN_ROUTING,
                mask_block_lengths=has_block_lengths,
                mask_ragged_tail=mask_ragged_tail,
                aligned_projection=(
                    not mask_ragged_tail
                    and input_qdata.shape[2] % config.block_k == 0
                    and heads % config.heads_per_program == 0
                ),
                block_m=config.block_m,
                block_n=head_dim * config.heads_per_program,
                block_k=config.block_k,
                group_m=config.group_m,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )

        full_row_blocks = chunk_rows // config.block_m
        if full_row_blocks:
            launch(full_row_blocks, mask_ragged_tail=False)
        if chunk_rows % config.block_m:
            launch(1, mask_ragged_tail=True)


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
    config: ProjectionConfig,
    out: KeyOutput,
    bias: torch.Tensor | None = None,
) -> None:
    """Launch K64 quantization and routing summaries for global key storage."""
    key, key_scale, key_summary, key_aux = out
    batch, heads, storage_sequence_length, head_dim = key.shape
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
                    triton.cdiv(heads, config.heads_per_program),
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
                bias_ptr=bias,
                heads=heads,
                heads_per_program=config.heads_per_program,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rsqrt_fn=config.rsqrt_fn,
                norm_epsilon=norm_epsilon,
                mean_pool_summary=mean_pool_summary,
                mask_block_lengths=has_block_lengths,
                aligned_projection=(
                    aligned_rows
                    and input_qdata.shape[2] % config.block_k == 0
                    and heads % config.heads_per_program == 0
                ),
                mask_ragged_tail=not aligned_rows,
                block_m=config.block_m,
                block_n=head_dim * config.heads_per_program,
                block_k=config.block_k,
                group_m=config.group_m,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )

        full_row_blocks = logical_sequence_length // config.block_m
        if full_row_blocks:
            launch(full_row_blocks, 0, aligned_rows=True)
        if logical_sequence_length % config.block_m:
            launch(1, full_row_blocks, aligned_rows=False)


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    config: ProjectionConfig,
    emit_block_mean: bool,
    out: ValueOutput,
    bias: torch.Tensor | None = None,
) -> None:
    """Launch projected means and centered tile-scaled INT8 values."""
    value, value_scale_multiplier, value_mean, block_mean = out
    batch, heads, head_dim, storage_sequence_length = value.shape
    block_n = head_dim * config.heads_per_program
    sequence_length = input_qdata.shape[1]
    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else value_mean
    with device_context(input_qdata.device):
        _kernels._project_prepared_input_mean_kernel[
            (triton.cdiv(heads, config.heads_per_program), batch)
        ](
            input_mean,
            weight_qdata,
            weight_scale,
            value_mean,
            bias_ptr=bias,
            input_features=input_qdata.shape[2],
            output_features=heads * head_dim,
            block_n=block_n,
            block_k=config.block_k,
            num_warps=config.num_warps,
        )

        def launch(row_block_count: int, row_block_offset: int, *, aligned_rows: bool) -> None:
            _kernels._convrot_project_quantize_sparse_value_kernel[
                (
                    row_block_count,
                    triton.cdiv(heads, config.heads_per_program),
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
                bias_ptr=bias,
                heads=heads,
                heads_per_program=config.heads_per_program,
                head_dim=head_dim,
                aligned_projection=(
                    aligned_rows
                    and input_qdata.shape[2] % config.block_k == 0
                    and heads % config.heads_per_program == 0
                ),
                mask_block_lengths=has_block_lengths,
                emit_block_mean=emit_block_mean,
                block_m=config.block_m,
                block_n=block_n,
                block_k=config.block_k,
                group_m=config.group_m,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )

        full_row_blocks = sequence_length // config.block_m
        if full_row_blocks:
            launch(full_row_blocks, 0, aligned_rows=True)
        if sequence_length % config.block_m:
            launch(1, full_row_blocks, aligned_rows=False)
