"""ConvRot INT8 projection directly into dense Piper's Q32 operand storage."""

# Triton device functions and launch options are not ordinary Python parameters.
# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false
# pyright: reportAssignmentType=false

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization
from piper_kernels.fusions.convrot_int8_sage_qk.triton import project_rmsnorm_rope_tile


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    """Target-specific compute settings; Q32 quantization is shared."""

    block_k: int
    heads_per_program: int
    num_warps: int
    num_stages: int
    group_m: int = 0
    round_rsqrt_to_nearest: bool = False


@triton.jit
def _project_query_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_ptr,
    query_scale_ptr,
    row_block_offset,
    sequence_length,
    storage_length,
    batch_size,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    norm_epsilon: tl.constexpr,
    softmax_scale: tl.constexpr,
    heads_per_program: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    round_rsqrt_to_nearest: tl.constexpr,
    aligned_projection: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
    bias_ptr=None,
):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = heads_per_program * head_dim
    row_block, head_block = tl.program_id(0), tl.program_id(1)
    if group_m:
        rows, head_blocks = tl.num_programs(0), tl.num_programs(1)
        program = row_block + head_block * rows
        group = program // (group_m * head_blocks)
        first_row = group * group_m
        group_rows = tl.minimum(rows - first_row, group_m)
        within_group = program % (group_m * head_blocks)
        row_block = first_row + within_group % group_rows
        head_block = within_group // group_rows
    row_block += row_block_offset
    batch = tl.program_id(2)
    sequence_offsets = row_block * block_m + tl.arange(0, block_m)
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    weight_offsets = head_block * block_n + tl.arange(0, block_n)
    transformed = project_rmsnorm_rope_tile(
        input_ptr,
        input_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        batch * sequence_length + sequence_offsets,
        weight_offsets,
        sequence_offsets,
        batch_size * sequence_length,
        sequence_length,
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
    group_offsets = tl.arange(0, 2)
    group_valid = head_offsets[:, None] < heads
    if mask_ragged_tail:
        transformed = tl.where(
            sequence_offsets[:, None, None] < sequence_length,
            transformed,
            0.0,
        )
        group_valid = group_valid & (
            row_block * block_m + group_offsets[None, :] * 32 < sequence_length
        )
    quantized, scale = qk_quantization.quantize_query_tile(
        transformed,
        group_valid,
        softmax_scale,
        heads_per_program,
        head_dim,
        block_m,
        32,
    )
    batch_heads = batch * heads + head_offsets.to(tl.int64)
    query_offsets = (
        batch_heads[:, None, None] * storage_length * head_dim
        + sequence_offsets[None, :, None] * head_dim
        + tl.arange(0, head_dim)[None, None, :]
    )
    tl.store(query_ptr + query_offsets, quantized, mask=head_offsets[:, None, None] < heads)
    scale_offsets = (
        batch_heads[:, None] * (storage_length // 32) + row_block * 2 + group_offsets[None, :]
    )
    tl.store(query_scale_ptr + scale_offsets, scale, mask=head_offsets[:, None] < heads)


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
    config: ProjectionConfig,
    out: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Fill only quantized Q and scales, keeping projection and transforms in FP32."""
    query, query_scale = out
    batch, heads, storage_length, head_dim = query.shape
    sequence_length = input_qdata.shape[1]
    if batch == 0:
        return
    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_ragged_tail: bool) -> None:
            _project_query_kernel[
                (row_blocks, triton.cdiv(heads, config.heads_per_program), batch)
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
                heads_per_program=config.heads_per_program,
                block_k=config.block_k,
                group_m=config.group_m,
                round_rsqrt_to_nearest=config.round_rsqrt_to_nearest,
                aligned_projection=(
                    not mask_ragged_tail
                    and input_qdata.shape[2] % config.block_k == 0
                    and heads % config.heads_per_program == 0
                ),
                mask_ragged_tail=mask_ragged_tail,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )

        full_blocks = sequence_length // 64
        if full_blocks:
            launch(full_blocks, 0, mask_ragged_tail=False)
        if sequence_length % 64:
            launch(1, full_blocks, mask_ragged_tail=True)
