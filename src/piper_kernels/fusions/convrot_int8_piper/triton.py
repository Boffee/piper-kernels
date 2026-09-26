"""ConvRot INT8 projection into dense Piper's Q32/K64 and per-token V operands."""

# Triton device functions and launch options are not ordinary Python parameters.
# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false
# pyright: reportAssignmentType=false, reportAttributeAccessIssue=false

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization
from piper_kernels.attention.piper_attention import _quantization
from piper_kernels.fusions.convrot_int8_projection.triton import (
    ProjectionConfig,
    project_prepared_input_mean_kernel,
    projection_tile_ids,
)
from piper_kernels.fusions.convrot_int8_sage_qk.triton import project_rmsnorm_rope_tile
from piper_kernels.linear.convrot.int8 import _ops
from piper_kernels.linear.convrot.int8._kernels import triton as matmul


@triton.jit
def _project_qk_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    statistics_ptr,
    row_block_offset,
    chunk_start,
    chunk_rows,
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
    is_query: tl.constexpr = True,
):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = heads_per_program * head_dim
    row_block, head_block = projection_tile_ids(group_m)
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
        batch * sequence_length + chunk_start + sequence_offsets,
        weight_offsets,
        chunk_start + sequence_offsets,
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
    if is_query:
        group_offsets = tl.arange(0, 2)
        group_valid = head_offsets[:, None] < heads
        if mask_ragged_tail:
            transformed = tl.where(
                sequence_offsets[:, None, None] < chunk_rows,
                transformed,
                0.0,
            )
            group_valid = group_valid & (
                row_block * block_m + group_offsets[None, :] * 32 < chunk_rows
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
        tl.store(output_ptr + query_offsets, quantized, mask=head_offsets[:, None, None] < heads)
        scale_offsets = (
            batch_heads[:, None] * (storage_length // 32) + row_block * 2 + group_offsets[None, :]
        )
        tl.store(statistics_ptr + scale_offsets, scale, mask=head_offsets[:, None] < heads)
    else:
        transformed = tl.where(sequence_offsets[:, None, None] < chunk_rows, transformed, 0.0)
        batch_heads = batch * heads + head_offsets.to(tl.int64)
        offsets = (
            batch_heads[None, :, None] * storage_length * head_dim
            + sequence_offsets[:, None, None] * head_dim
            + tl.arange(0, head_dim)[None, None, :]
        )
        tl.store(output_ptr + offsets, transformed, head_offsets[None, :, None] < heads)
        partial_offsets = (
            batch_heads[:, None] * (storage_length // 64) + row_block
        ) * head_dim + tl.arange(0, head_dim)[None, :]
        tl.store(
            statistics_ptr + partial_offsets, tl.sum(transformed, 0), head_offsets[:, None] < heads
        )


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
            _project_qk_kernel[(row_blocks, triton.cdiv(heads, config.heads_per_program), batch)](
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
    out: tuple[torch.Tensor, torch.Tensor],
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
    out: tuple[torch.Tensor, torch.Tensor],
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


@triton.jit
def _project_value_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    mean_ptr,
    value_ptr,
    multiplier_ptr,
    log_ptr,
    bias_ptr,
    sequence_length,
    storage_length,
    batch_size,
    row_block_offset,
    input_features: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    heads_per_program: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    is_causal: tl.constexpr,
    packed_amd: tl.constexpr,
    aligned_projection: tl.constexpr,
):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = heads_per_program * head_dim
    row_block, head_block = projection_tile_ids(group_m)
    row_block += row_block_offset
    batch = tl.program_id(2)
    rows = row_block * block_m + tl.arange(0, block_m)
    features = head_block * block_n + tl.arange(0, block_n)
    value = matmul.scaled_int8_matmul(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        batch * sequence_length + rows,
        features,
        batch_size * sequence_length,
        heads * head_dim,
        input_features,
        block_m,
        block_n,
        block_k,
        aligned_projection,
    )
    if bias_ptr is not None:
        value += tl.load(bias_ptr + features, features < heads * head_dim, 0).to(tl.float32)[
            None, :
        ]
    if not is_causal:
        mean = tl.load(
            mean_ptr + batch * heads * head_dim + features, features < heads * head_dim, 0
        )
        value -= mean[None, :]
    elif row_block == 0:
        tl.store(mean_ptr + batch * heads * head_dim + features, 0.0, features < heads * head_dim)
    value = tl.where(rows[:, None] < sequence_length, value, 0.0)
    value = value.reshape((block_m * heads_per_program, head_dim))
    codes, scales = _quantization.quantize_value_rows(value)
    codes = codes.reshape((block_m, heads_per_program, head_dim)).permute((1, 0, 2))
    scales = scales.reshape((block_m, heads_per_program)).T
    head_offsets = head_block * heads_per_program + tl.arange(0, heads_per_program)
    batch_heads = batch * heads + head_offsets.to(tl.int64)
    dim = tl.arange(0, head_dim)
    if packed_amd:
        token = tl.arange(0, block_m)
        packed_token = (token & ~24) | ((token & 8) << 1) | ((token & 16) >> 1)
        offsets = (
            (batch_heads[:, None, None] * (storage_length // 64) + row_block) * head_dim * 64
            + dim[None, None, :] * 64
            + packed_token[None, :, None]
        )
    else:
        offsets = (
            batch_heads[:, None, None] * head_dim + dim[None, None, :]
        ) * storage_length + rows[None, :, None]
    tl.store(value_ptr + offsets, codes, head_offsets[:, None, None] < heads)
    metadata = batch_heads[:, None] * storage_length + rows[None, :]
    tl.store(multiplier_ptr + metadata, scales * 255.0, head_offsets[:, None] < heads)
    logs = tl.log2(scales)
    if not packed_amd:
        logs = logs.to(tl.float16).to(tl.float32)
    tl.store(log_ptr + metadata, logs, head_offsets[:, None] < heads)


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
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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
                _project_value_kernel[(count, triton.cdiv(heads, config.heads_per_program), batch)](
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
