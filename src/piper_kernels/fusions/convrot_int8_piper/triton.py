"""Dense Q/K/V launch mechanics; backends supply compute configurations."""

# Triton's launch options and constexpr arguments are not ordinary Python parameters.
# pyright: reportCallIssue=false, reportArgumentType=false

import torch
import triton

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization
from piper_kernels.attention.piper_attention import _quantization
from piper_kernels.fusions.convrot_int8_projection.triton import (
    ProjectionConfig,
    project_prepared_input_mean_kernel,
)
from piper_kernels.linear.convrot.int8 import _ops

from . import _kernels
from ._interfaces import KeyOutput, QueryOutput, ValueOutput


def _project_qk(
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
    is_query: bool,
    chunk_start: int = 0,
    chunk_rows: int | None = None,
) -> None:
    """Emit Q32 codes or transformed K and tile sums from the same FP32 projection."""
    output, statistics = out
    batch, heads, storage_length, head_dim = output.shape
    sequence_length = input_qdata.shape[1]
    chunk_rows = sequence_length if chunk_rows is None else chunk_rows
    if batch == 0:
        return
    with device_context(input_qdata.device):

        def launch(row_blocks: int, row_block_offset: int, *, mask_ragged_tail: bool) -> None:
            _kernels._project_qk_kernel[
                (row_blocks, triton.cdiv(heads, config.heads_per_program), batch)
            ](
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                norm_weight,
                cos,
                sin,
                output,
                statistics,
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
                is_query=is_query,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )

        full_blocks = chunk_rows // 64
        if full_blocks:
            launch(full_blocks, 0, mask_ragged_tail=False)
        if chunk_rows % 64:
            launch(1, full_blocks, mask_ragged_tail=True)


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
    chunk_start: int = 0,
    chunk_rows: int | None = None,
    out: QueryOutput,
) -> None:
    """Emit Q32 INT8 directly from the shared FP32 Q/K projection."""
    _project_qk(
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
        config=config,
        out=out,
        is_query=True,
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
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
    config: ProjectionConfig,
    out: KeyOutput,
) -> None:
    """Reduce the global post-RoPE K mean before centered K64 quantization."""
    batch, sequence, _ = input_qdata.shape
    _, heads, storage, head_dim = out[0].shape
    transformed = input_qdata.new_empty((batch, heads, storage, head_dim), dtype=torch.float32)
    partials = input_qdata.new_empty((batch, heads, storage // 64, head_dim), dtype=torch.float32)
    mean = input_qdata.new_empty((batch, heads, head_dim), dtype=torch.float32)
    _project_qk(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        1.0,
        bias,
        config=config,
        out=(transformed, partials),
        is_query=False,
    )
    with device_context(input_qdata.device):
        _quantization._kv_mean_finalize_kernel[(batch * heads, triton.cdiv(head_dim, 64))](
            partials,
            partials,
            mean,
            mean,
            sequence,
            storage // 64,
            is_causal=True,
            head_dim=head_dim,
            block_chunks=triton.next_power_of_2(storage // 64),
            block_d=64,
            num_warps=4,
        )
        qk_quantization.prepare_key(
            transformed[:, :, :sequence],
            mean,
            grouped=True,
            storage_key_length=storage,
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
