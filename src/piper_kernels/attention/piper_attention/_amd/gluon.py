"""RDNA4 dense Piper: per-token V scaling and a Q64/K64 recurrence."""

# Gluon device types and layouts are not Python runtime values.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportAttributeAccessIssue=false

from dataclasses import dataclass

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.piper._amd.fragments import (
    MMA_LAYOUT,
    concat_columns,
    four_fragments,
    pack_probabilities,
    pv_tiles,
    qk_tiles,
    query_fragments,
    rescale_numerator,
)
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization

from .._quantization import compute_kv_means
from .._validation import resolve_query_window, validate_output_buffer, validate_query_offset
from .triton import prepare_value


@gluon.jit
def _softmax_fragment(scores, maximum, scale, multiplier):
    """Pack one 16-column fragment while retaining unrounded denominator mass."""
    row_layout: gl.constexpr = maximum.type.layout
    maximum = gl.convert_layout(maximum, gl.SliceLayout(2, scores.type.layout))
    scale = gl.convert_layout(scale, maximum.type.layout)
    multiplier = gl.convert_layout(multiplier, gl.SliceLayout(1, scores.type.layout))
    probabilities = gl.exp2(gl.fma(scores, scale[:, :, None], -maximum[:, :, None]))
    total = gl.convert_layout(gl.sum(probabilities, 2), row_layout)
    return pack_probabilities(probabilities * multiplier[:, None, :]), total


@gluon.jit
def _softmax_tile(scores, block_max, scale, multiplier):
    """Keep K64 normalization, but consume FP32 probabilities in 16-column fragments."""
    chunks = four_fragments(scores)
    multiplier_0, multiplier_1 = gl.split(multiplier.reshape([1, 2, 32]).permute([0, 2, 1]))
    multiplier_00, multiplier_01 = gl.split(multiplier_0.reshape([1, 2, 16]).permute([0, 2, 1]))
    multiplier_10, multiplier_11 = gl.split(multiplier_1.reshape([1, 2, 16]).permute([0, 2, 1]))
    multipliers = (multiplier_00, multiplier_01, multiplier_10, multiplier_11)
    packed_chunks, sums = (), ()
    for chunk in gl.static_range(4):
        words, total = _softmax_fragment(chunks[chunk], block_max, scale, multipliers[chunk])
        packed_chunks += (words,)
        sums += (total,)
        with gl.amd.warp_pipeline_stage("softmax"):
            pass
    packed = concat_columns(
        concat_columns(packed_chunks[0], packed_chunks[1]),
        concat_columns(packed_chunks[2], packed_chunks[3]),
    )
    return packed, (sums[0] + sums[1]) + (sums[2] + sums[3])


@gluon.jit(
    do_not_specialize=[
        "query_start",
        "query_rows",
        "global_query_start",
        "key_length",
        "query_storage",
        "key_storage",
        "heads",
        "stride_ob",
        "stride_oh",
        "stride_om",
    ]
)
def _dense_piper_kernel(  # noqa: PLR0915
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    multiplier_ptr,
    log_scale_ptr,
    value_mean_ptr,
    output_ptr,
    query_start,
    query_rows,
    global_query_start,
    key_length,
    query_storage,
    key_storage,
    heads,
    stride_ob,
    stride_oh,
    stride_om,
    head_groups: gl.constexpr,
    head_dim: gl.constexpr,
    is_causal: gl.constexpr,
    wide_query_offsets: gl.constexpr,
    wide_context_offsets: gl.constexpr,
):
    head = gl.program_id(1).to(gl.int64)
    batch = gl.program_id(2).to(gl.int64)
    batch_head = batch * heads + head
    kv_batch_head = batch * (heads // head_groups) + head // head_groups
    query_block = gl.program_id(0)
    stored_query_block = query_start // 64 + query_block
    global_query_block = global_query_start // 64 + query_block
    row_layout: gl.constexpr = gl.SliceLayout(2, MMA_LAYOUT)
    block_layout: gl.constexpr = gl.SliceLayout(1, row_layout)
    column_layout: gl.constexpr = gl.SliceLayout(1, MMA_LAYOUT)
    blocks = gl.full([1], stored_query_block, gl.int32, block_layout)
    rows = gl.arange(0, 64, gl.SliceLayout(0, row_layout))
    columns = gl.arange(0, 64, gl.SliceLayout(0, column_layout))
    output_rows = query_block.to(gl.int64) * 64 + rows
    global_query_rows = global_query_start.to(gl.int64) + output_rows
    query = query_fragments(
        query_ptr + batch_head * query_storage * head_dim,
        blocks,
        wide_query_offsets,
        head_dim,
    )
    query_scale = gl.load(
        query_scale_ptr
        + batch_head * (query_storage // 32)
        + stored_query_block * 2
        + rows[None, :] // 32,
    )
    numerator = gl.zeros([1, 64, head_dim], gl.float32, MMA_LAYOUT)
    denominator = gl.zeros([1, 64], gl.float32, row_layout)
    running_max = gl.full([1, 64], -float("inf"), gl.float32, row_layout)
    tile_count = gl.cdiv(key_length, 64)
    if is_causal:
        tile_count = gl.minimum(tile_count, global_query_block + 1)
    for tile in range(tile_count):
        key_blocks = gl.full([1], tile, gl.int32, block_layout)
        scores = qk_tiles(
            query,
            key_ptr + kv_batch_head * key_storage * head_dim,
            key_blocks,
            key_blocks,
            wide_context_offsets,
            head_dim,
            two_tiles=False,
        )
        scale = query_scale * gl.load(key_scale_ptr + kv_batch_head * (key_storage // 64) + tile)
        key_rows = tile.to(gl.int64) * 64 + columns
        boundary = (tile + 1) * 64 > key_length
        if is_causal:
            boundary = boundary | (tile == global_query_block)
        if boundary:
            valid = key_rows[None, None, :] < key_length
            if is_causal:
                valid = valid & (key_rows[None, None, :] <= global_query_rows[None, :, None])
            # Fused scaling must not turn a masked zero-scale score into 0 * -inf.
            scores = gl.where(valid, gl.where(scale[:, :, None] == 0, 0.0, scores), -float("inf"))
            scale = gl.where(scale == 0, 1.0, scale)
        multiplier = gl.load(multiplier_ptr + kv_batch_head * key_storage + key_rows[None, :])
        log_scale = gl.load(log_scale_ptr + kv_batch_head * key_storage + key_rows[None, :])
        block_max = gl.max(gl.fma(scores, scale[:, :, None], log_scale[:, None, :]), 2)
        next_max = gl.maximum(running_max, block_max)
        old_weight = gl.exp2(running_max - next_max)
        current_weight = gl.exp2(block_max - next_max)
        packed, total = _softmax_tile(scores, block_max, scale, multiplier)
        denominator = denominator * old_weight + total * current_weight
        numerator = pv_tiles(
            packed,
            value_ptr + kv_batch_head * key_storage * head_dim,
            key_blocks,
            key_blocks,
            rescale_numerator(numerator, old_weight),
            current_weight,
            wide_context_offsets,
            two_tiles=False,
        )
        running_max = next_max
    output = numerator / (gl.maximum(denominator, 1e-30)[:, :, None] * 255.0)
    features = gl.arange(0, head_dim, gl.SliceLayout(0, column_layout))
    if not is_causal:
        mean = gl.load(value_mean_ptr + kv_batch_head * head_dim + features[None, :])
        output += mean[:, None, :]
    offsets = batch * stride_ob + head * stride_oh + output_rows[None, :, None] * stride_om
    gl.store(
        output_ptr + offsets + features[None, None, :],
        output,
        output_rows[None, :, None] < query_rows,
    )


@dataclass(frozen=True, slots=True)
class PreparedContext:
    """Reusable K/V operands whose storage scales with KV heads."""

    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor
    multiplier: torch.Tensor
    log_scale: torch.Tensor
    value_mean: torch.Tensor
    key_length: int
    is_causal: bool


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """Quantized Q with its unpadded logical shape and original floating dtype."""

    data: torch.Tensor
    scale: torch.Tensor
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    global_row_offset: int = 0


@dataclass(frozen=True, slots=True)
class PreparedAttention:
    """Prepared K/V and Q with one reusable output for full attention."""

    context: PreparedContext
    query: PreparedQuery
    output: torch.Tensor


def prepare_context(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
) -> PreparedContext:
    """Prepare validated K/V once, independently of Q and output storage."""
    key_storage = triton.cdiv(key.shape[2], 64) * 64
    with device_context(key.device):
        key_mean, value_mean = compute_kv_means(key, value, is_causal=is_causal)
        key_int8, key_scale = qk_quantization.prepare_key(
            key,
            key_mean,
            grouped=True,
            storage_key_length=key_storage,
        )
        packed_value, multiplier, log_scale = prepare_value(
            value,
            value_mean,
            is_causal=is_causal,
            storage_length=key_storage,
        )
    return PreparedContext(
        key=key_int8,
        value=packed_value,
        key_scale=key_scale,
        multiplier=multiplier,
        log_scale=log_scale,
        value_mean=value_mean,
        key_length=key.shape[2],
        is_causal=is_causal,
    )


def prepare_query(
    query: torch.Tensor,
    scale: float,
    *,
    global_row_offset: int = 0,
) -> PreparedQuery:
    """Prepare Q storage at an aligned global origin for full or ranged launches.

    Matching a full-sequence preparation requires complete Q32 scale groups,
    except for the actual sequence tail. Windows of existing prepared storage
    may end at any row.
    """
    validate_query_offset(global_row_offset, block_rows=64, name="global_row_offset")
    batch, heads, query_length, head_dim = query.shape
    query_storage = triton.cdiv(query_length, 64) * 64
    with device_context(query.device):
        query_int8, query_scale = qk_quantization.prepare_query(
            query,
            scale,
            grouped=True,
            storage_query_length=query_storage,
        )
    return PreparedQuery(
        data=query_int8,
        scale=query_scale,
        shape=(batch, heads, query_length, head_dim),
        dtype=query.dtype,
        global_row_offset=global_row_offset,
    )


def prepare_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> PreparedAttention:
    """Prepare operands and allocate the reusable output before timed launches."""
    context = prepare_context(key, value, is_causal=is_causal)
    prepared_query = prepare_query(query, scale)
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    return PreparedAttention(context=context, query=prepared_query, output=output)


def launch_attention_into(
    context: PreparedContext,
    query: PreparedQuery,
    output: torch.Tensor,
    *,
    query_start: int = 0,
    query_rows: int | None = None,
) -> torch.Tensor:
    """Fill a caller-owned output for an aligned window of prepared Q.

    Q and K/V must match in batch, head dimension, and device; Q heads must be
    divisible by K/V heads. Causal masking uses Q's global origin plus the local
    window start. query_start is local to prepared Q; query_rows=None selects
    its remaining logical rows. Output rows begin at zero and may use
    noncontiguous outer strides with contiguous features.
    """
    batch, heads, query_length, head_dim = query.shape
    rows = resolve_query_window(
        query_length,
        query_start=query_start,
        query_rows=query_rows,
        global_row_offset=query.global_row_offset,
        block_rows=64,
        key_length=context.key_length,
        is_causal=context.is_causal,
    )
    validate_output_buffer(
        output,
        shape=(batch, heads, rows, head_dim),
        dtype=query.dtype,
        device=query.data.device,
    )
    query_storage = query.data.shape[2]
    key_storage = context.key.shape[2]
    if batch == 0:
        return output
    with device_context(query.data.device):
        _dense_piper_kernel[(triton.cdiv(rows, 64), heads, batch)](
            query.data,
            context.key,
            context.value,
            query.scale,
            context.key_scale,
            context.multiplier,
            context.log_scale,
            context.value_mean,
            output,
            query_start,
            rows,
            query.global_row_offset + query_start,
            context.key_length,
            query_storage,
            key_storage,
            heads,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            heads // context.key.shape[1],
            head_dim,
            context.is_causal,
            query_storage * (head_dim // 8) > (1 << 31),
            key_storage * head_dim > (1 << 32),
            num_warps=4,
            num_stages=1,
            llvm_fn_attrs=(("target-features", "+cumode"),),
        )
    return output


def launch_attention(prepared: PreparedAttention) -> torch.Tensor:
    """Reuse the prepared output without allocating or preparing operands."""
    return launch_attention_into(prepared.context, prepared.query, prepared.output)


def run_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    if query.shape[0] == 0:
        return torch.empty_like(query, memory_format=torch.contiguous_format)
    return launch_attention(prepare_attention(query, key, value, scale, is_causal))
