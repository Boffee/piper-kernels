"""Dense Q/K/V launch mechanics; backends supply compute configurations."""

# Triton's launch options and constexpr arguments are not ordinary Python parameters.
# pyright: reportCallIssue=false, reportArgumentType=false

import torch
import triton

from piper_kernels._triton.amd_compiler_workarounds import lds_only_fence_options
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    resolve_qk_quantization,
)
from piper_kernels.attention.piper_attention import _quantization
from piper_kernels.fusions.convrot_int8_projection._plan import (
    ProjectionExecutionPlan,
    resolve_row_block_launches,
)
from piper_kernels.fusions.convrot_int8_projection.triton import project_value_chunk_means

from . import _kernels
from ._interfaces import KeyOutput, QueryOutput, ValueOutput


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
    execution_plan: ProjectionExecutionPlan,
    out: QueryOutput,
    chunk_start: int = 0,
    chunk_rows: int | None = None,
) -> None:
    """Emit INT8 Q directly from the shared FP32 Q/K projection.

    ``out`` holds Q32 scales, or per-row scales for per-thread quantization.
    """
    query, query_scale = out
    batch, heads, storage_length, head_dim = query.shape
    per_thread_scales = resolve_qk_quantization(query_scale, storage_length) == "per_thread"
    sequence_length = input_qdata.shape[1]
    chunk_rows = sequence_length if chunk_rows is None else chunk_rows
    if batch == 0:
        return
    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_ragged_tail: bool) -> None:
            _kernels._project_query_kernel[
                (row_blocks, triton.cdiv(heads, execution_plan.heads_per_program), batch)
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
                row_block_offset,
                chunk_start,
                chunk_rows,
                sequence_length,
                storage_length,
                batch,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                softmax_scale=softmax_scale,
                bias_ptr=bias,
                heads_per_program=execution_plan.heads_per_program,
                block_k=execution_plan.block_k,
                group_m=execution_plan.group_m,
                round_rsqrt_to_nearest=execution_plan.round_rsqrt_to_nearest,
                aligned_projection=(
                    not mask_ragged_tail
                    and input_qdata.shape[2] % execution_plan.block_k == 0
                    and heads % execution_plan.heads_per_program == 0
                ),
                mask_ragged_tail=mask_ragged_tail,
                per_thread_scales=per_thread_scales,
                num_warps=execution_plan.num_warps,
                num_stages=execution_plan.num_stages,
                **lds_only_fence_options(execution_plan.lds_only_fences),
            )

        for count, offset, aligned_rows in resolve_row_block_launches(chunk_rows, 64):
            launch(count, offset, mask_ragged_tail=not aligned_rows)


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
    execution_plan: ProjectionExecutionPlan,
    out: KeyOutput,
) -> None:
    """Project K once into BF16 storage, then quantize per 1024-token chunk."""
    batch, sequence_length, input_features = input_qdata.shape
    if batch == 0:
        return
    key_int8, key_scale, key_center = out
    _, heads, storage_length, head_dim = key_int8.shape
    stored = input_qdata.new_empty((batch, heads, sequence_length, head_dim), dtype=torch.bfloat16)
    with device_context(input_qdata.device):
        for count, offset, aligned_rows in resolve_row_block_launches(
            sequence_length, execution_plan.block_m
        ):
            _kernels._project_key_kernel[
                (count, triton.cdiv(heads, execution_plan.heads_per_program), batch)
            ](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                norm_weight,
                cos,
                sin,
                stored,
                batch * sequence_length,
                sequence_length,
                offset,
                input_features=input_features,
                heads=heads,
                heads_per_program=execution_plan.heads_per_program,
                head_dim=head_dim,
                rotary_dim=cos.shape[1],
                norm_epsilon=norm_epsilon,
                aligned_projection=(
                    aligned_rows
                    and input_features % execution_plan.block_k == 0
                    and heads % execution_plan.heads_per_program == 0
                ),
                mask_ragged_tail=not aligned_rows,
                block_m=execution_plan.block_m,
                block_n=head_dim * execution_plan.heads_per_program,
                block_k=execution_plan.block_k,
                round_rsqrt_to_nearest=execution_plan.round_rsqrt_to_nearest,
                group_m=execution_plan.group_m,
                bias_ptr=bias,
                num_warps=execution_plan.num_warps,
                num_stages=execution_plan.num_stages,
                **lds_only_fence_options(execution_plan.lds_only_fences),
            )
        _quantization.prepare_chunk_centered_key(
            stored,
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
    packed_amd: bool,
    execution_plan: ProjectionExecutionPlan,
    out: ValueOutput,
) -> None:
    """Project directly into dense V1024-centered per-token V for causal and non-causal paths."""
    del is_causal  # Both causal and non-causal dense Piper center each 1024-token chunk.
    value, multiplier, log_scale, mean = out
    batch, heads, head_dim, storage = value.shape
    sequence = input_qdata.shape[1]
    with device_context(input_qdata.device):
        project_value_chunk_means(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            block_n=execution_plan.resolve_value_mean_block_n(head_dim),
            block_k=execution_plan.block_k,
            out=mean,
        )
        for count, offset, aligned in resolve_row_block_launches(sequence, 64):
            _kernels._project_value_kernel[
                (count, triton.cdiv(heads, execution_plan.heads_per_program), batch)
            ](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                mean,
                value,
                multiplier,
                log_scale,
                bias,
                sequence,
                storage,
                mean.shape[2],
                batch,
                offset,
                input_features=input_qdata.shape[2],
                heads=heads,
                head_dim=head_dim,
                heads_per_program=execution_plan.heads_per_program,
                block_k=execution_plan.block_k,
                group_m=execution_plan.group_m,
                packed_amd=packed_amd,
                aligned_projection=(
                    aligned
                    and input_qdata.shape[2] % execution_plan.block_k == 0
                    and heads % execution_plan.heads_per_program == 0
                ),
                num_warps=execution_plan.num_warps,
                num_stages=execution_plan.num_stages,
                **lds_only_fence_options(execution_plan.lds_only_fences),
            )
