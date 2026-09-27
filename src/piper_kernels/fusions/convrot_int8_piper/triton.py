"""Dense Q/K/V launch mechanics; backends supply compute configurations."""

# Triton's launch options and constexpr arguments are not ordinary Python parameters.
# pyright: reportCallIssue=false, reportArgumentType=false

import torch
import triton

from piper_kernels._triton.runtime import device_context
from piper_kernels.fusions.convrot_int8_projection.triton import (
    ProjectionConfig,
    project_prepared_input_mean_kernel,
)
from piper_kernels.fusions.convrot_int8_sage_qk.key import (
    project_key,  # noqa: F401 - shared launcher export
)
from piper_kernels.linear.convrot.int8 import _ops

from . import _kernels
from ._interfaces import QueryOutput, ValueOutput


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
    out: QueryOutput,
    chunk_start: int = 0,
    chunk_rows: int | None = None,
) -> None:
    """Emit Q32 INT8 directly from the shared FP32 Q/K projection."""
    query, query_scale = out
    batch, heads, storage_length, head_dim = query.shape
    sequence_length = input_qdata.shape[1]
    chunk_rows = sequence_length if chunk_rows is None else chunk_rows
    if batch == 0:
        return
    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_ragged_tail: bool) -> None:
            _kernels._project_query_kernel[
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

        full_blocks = chunk_rows // 64
        if full_blocks:
            launch(full_blocks, 0, mask_ragged_tail=False)
        if chunk_rows % 64:
            launch(1, full_blocks, mask_ragged_tail=True)


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    is_causal: bool,
    packed_amd: bool,
    mean_block_n: int | None,
    config: ProjectionConfig,
    out: ValueOutput,
) -> None:
    """Project directly into dense per-token V, with an optional global mean."""
    value, multiplier, log_scale, mean = out
    batch, heads, head_dim, storage = value.shape
    sequence = input_qdata.shape[1]
    mean_block_n = head_dim * config.heads_per_program if mean_block_n is None else mean_block_n
    with device_context(input_qdata.device):
        if not is_causal:
            represented_mean = _ops.dequantized_input_mean(input_qdata, input_scale)
            project_prepared_input_mean_kernel[
                (triton.cdiv(heads * head_dim, mean_block_n), batch)
            ](
                represented_mean,
                weight_qdata,
                weight_scale,
                mean,
                input_features=input_qdata.shape[2],
                output_features=heads * head_dim,
                block_n=mean_block_n,
                block_k=config.block_k,
                bias_ptr=bias,
                num_warps=4,
            )
        for count, offset, aligned in (
            (sequence // 64, 0, True),
            (int(sequence % 64 != 0), sequence // 64, False),
        ):
            if count:
                _kernels._project_value_kernel[
                    (count, triton.cdiv(heads, config.heads_per_program), batch)
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
                    batch,
                    offset,
                    input_features=input_qdata.shape[2],
                    heads=heads,
                    head_dim=head_dim,
                    heads_per_program=config.heads_per_program,
                    block_k=config.block_k,
                    group_m=config.group_m,
                    is_causal=is_causal,
                    packed_amd=packed_amd,
                    aligned_projection=(
                        aligned
                        and input_qdata.shape[2] % config.block_k == 0
                        and heads % config.heads_per_program == 0
                    ),
                    num_warps=config.num_warps,
                    num_stages=config.num_stages,
                )
