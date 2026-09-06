"""One-pass ConvRot INT8 projection and sparse-Piper INT8 query preparation."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

# Triton device functions cannot carry ordinary Python type annotations.
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_kernels,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    validate_routing_mode,
)
from piper_kernels.fusions.convrot_int8_sage_qk.triton import (
    project_rmsnorm_rope_tile,
    validate_qk_projection_inputs,
)

from ._layout import (
    HEAD_DIM,
    QUERY_SCALE_ROWS,
    TILE_ROWS,
    padded_sequence_length,
    validate_block_lengths,
)

_BLOCK_M = TILE_ROWS
_BLOCK_K = 128
_HEADS_PER_PROGRAM = 2
_BLOCK_N = HEAD_DIM * _HEADS_PER_PROGRAM
_JIT_QUERY_SCALE_ROWS = tl.constexpr(QUERY_SCALE_ROWS)


@triton.jit
def _convrot_project_rmsnorm_rope_quantize_query_kernel(  # noqa: PLR0913, PLR0917
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
):
    """Project one Q64/two-head tile and emit Q32 INT8 plus route summaries."""
    tl.static_assert(block_m == 64)
    tl.static_assert(heads_per_program == 2)
    tl.static_assert(block_n == heads_per_program * head_dim)
    tl.static_assert(head_dim == 128)
    tl.static_assert(rotary_dim <= head_dim)
    tl.static_assert(rotary_dim % 2 == 0)

    storage_query_block = tl.program_id(0)
    if mask_ragged_tail:
        storage_query_block = chunk_rows // block_m
    global_query_block = chunk_start // block_m + storage_query_block
    head_block = tl.program_id(1)
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


def _validate_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
    softmax_scale: float,
) -> tuple[int, int, int, int]:
    result = validate_qk_projection_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        name="Q",
    )
    if result[1] < TILE_ROWS:
        raise ValueError(f"Q projection requires at least {TILE_ROWS} sequence rows")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Q projection softmax scale must be finite and positive")
    return result


def _launch_query_projection_range(  # noqa: PLR0913, PLR0917
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
    block_lengths: torch.Tensor | None = None,
    *,
    chunk_start: int = 0,
    chunk_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_routing_mode(routing_mode)
    batch, sequence_length, heads, rotary_dim = _validate_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        softmax_scale=softmax_scale,
    )
    if chunk_rows is None:
        chunk_rows = sequence_length
    if (
        isinstance(chunk_start, bool)
        or not isinstance(chunk_start, int)
        or isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_start < 0
        or chunk_rows < 1
        or chunk_start % TILE_ROWS
        or chunk_start + chunk_rows > sequence_length
    ):
        raise ValueError("Q projection range must be a nonempty aligned sequence window")
    storage_sequence_length = padded_sequence_length(chunk_rows)
    validate_block_lengths(block_lengths, sequence_length, input_qdata.device)
    query = torch.empty(
        (batch, heads, storage_sequence_length, HEAD_DIM),
        device=input_qdata.device,
        dtype=torch.int8,
    )
    query_scale = torch.empty(
        (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
        device=input_qdata.device,
        dtype=torch.float32,
    )
    query_summary = torch.empty(
        (batch, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM),
        device=input_qdata.device,
        dtype=torch.float32,
    )

    has_block_lengths = block_lengths is not None
    block_lengths_ptr = block_lengths if has_block_lengths else query_scale

    with device_context(input_qdata.device):

        def launch(row_block_count: int, *, mask_ragged_tail: bool) -> None:
            _convrot_project_rmsnorm_rope_quantize_query_kernel[
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
                block_m=_BLOCK_M,
                block_n=_BLOCK_N,
                block_k=_BLOCK_K,
                num_warps=8,
                num_stages=3,
            )

        full_row_blocks = chunk_rows // _BLOCK_M
        if full_row_blocks:
            launch(full_row_blocks, mask_ragged_tail=False)
        if chunk_rows % _BLOCK_M:
            launch(1, mask_ragged_tail=True)
        return query, query_scale, query_summary


def _launch_query_projection(  # noqa: PLR0913, PLR0917
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
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the complete query storage for the public standalone boundary."""
    return _launch_query_projection_range(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        routing_mode,
        block_lengths,
    )


@torch.library.custom_op("piper_kernels::convrot_int8_sparse_piper_project_query", mutates_args=())
def _project_query_op(  # noqa: PLR0913, PLR0917
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
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_query_projection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        routing_mode,
        block_lengths,
    )


@_project_query_op.register_fake
def _project_query_op_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _norm_weight: torch.Tensor,
    cos: torch.Tensor,
    _sin: torch.Tensor,
    _norm_epsilon: float,
    _softmax_scale: float,
    _routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence_length, _input_features = input_qdata.shape
    storage_sequence_length = padded_sequence_length(sequence_length)
    heads = weight_qdata.shape[0] // HEAD_DIM
    return (
        input_qdata.new_empty((batch, heads, storage_sequence_length, HEAD_DIM)),
        input_qdata.new_empty(
            (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
            dtype=torch.float32,
        ),
        cos.new_empty((batch, heads, storage_sequence_length // TILE_ROWS, HEAD_DIM)),
    )
