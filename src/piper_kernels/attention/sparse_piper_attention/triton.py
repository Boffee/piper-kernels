"""Shared Triton preparation for the quantized sparse-Piper tensor contract."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels._grid import select_block_grid
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_kernels,
)
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS, ValueFormat

from ._contracts import valid_block_rows
from ._prepared import _PreparedSparsePiperOperands

_BLOCK_N = 64
# Sequence lengths, strides, and head counts must not become JIT keys: the
# separate quantization and summary kernels they replace compile once per
# layout, and a fused pass may not compile more often than that.
_FUSED_DO_NOT_SPECIALIZE = (
    "logical_sequence_length",
    "storage_sequence_length",
    "stride_qb",
    "stride_qh",
    "stride_qn",
    "heads",
    "heads_inner",
)
# Measured warps per K64 tile. Head width is already a specialization key.
_FUSED_QK_WARPS = {64: 1, 128: 2}


@triton.jit(
    do_not_specialize=(
        "key_length",
        "stride_vb",
        "stride_vh",
        "stride_vn",
        "heads",
        "heads_inner",
    )
)
def _quantize_value_per_tile_kernel(
    value_ptr,
    value_mean_ptr,
    value_scale_ptr,
    output_ptr,
    block_lengths_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    heads,
    heads_inner,
    value_format: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    mask_block_lengths: tl.constexpr,
):
    """Per-K64 tile-center and quantize one storage tile while masking the tail."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    key_block = tl.where(heads_inner != 0, pid1, pid0).to(tl.int64)
    head = tl.where(heads_inner != 0, pid0, pid1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    batch_head = batch * heads + head
    value, valid_rows, offsets_n, offsets_d = _load_row_tile(
        value_ptr,
        key_block,
        head,
        batch,
        stride_vb,
        stride_vh,
        stride_vn,
        block_lengths_ptr,
        key_length,
        mask_block_lengths,
        head_dim,
        block_n,
    )
    quantized, value_scale_multiplier, value_mean = sparse_piper_kernels.quantize_value_tile(  # pyright: ignore[reportGeneralTypeIssues]
        tl.reshape(value, (block_n, 1, head_dim)),
        valid_rows,
        tl.constexpr(1),
        head_dim,
        block_n,
        block_n,
    )
    quantized = tl.reshape(quantized, (block_n, head_dim))
    value_scale_multiplier = tl.reshape(value_scale_multiplier, ())
    value_mean = tl.reshape(value_mean, (head_dim,))
    tile_count = tl.cdiv(key_length, block_n)
    tl.store(
        value_mean_ptr + (batch_head * tile_count + key_block) * head_dim + offsets_d,
        value_mean,
    )
    tl.store(
        value_scale_ptr + batch_head * tile_count + key_block,
        value_scale_multiplier,
    )
    tl.store(
        output_ptr
        + sparse_piper_kernels.value_storage_offsets(
            batch_head,
            offsets_n[:, None],
            offsets_d[None, :],
            tile_count * block_n,
            head_dim,
            value_format,
        ),
        quantized,
    )


@triton.jit
def _load_row_tile(
    tensor_ptr,
    block,
    head,
    batch,
    stride_b,
    stride_h,
    stride_n,
    block_lengths_ptr,
    logical_sequence_length,
    mask_block_lengths: tl.constexpr,
    head_dim: tl.constexpr,
    block_rows: tl.constexpr,
):
    """Load one FP32 row tile; padded and ragged rows read as zero."""
    rows = tl.arange(0, block_rows)
    features = tl.arange(0, head_dim)
    sequence_offsets = block * block_rows + rows
    if mask_block_lengths:
        valid_rows = rows < tl.load(block_lengths_ptr + block)
    else:
        valid_rows = sequence_offsets < logical_sequence_length
    values = tl.load(
        tensor_ptr
        + batch.to(tl.int64) * stride_b
        + head.to(tl.int64) * stride_h
        + sequence_offsets[:, None].to(tl.int64) * stride_n
        + features[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    ).to(tl.float32)
    return values, valid_rows, sequence_offsets, features


@triton.jit(do_not_specialize=_FUSED_DO_NOT_SPECIALIZE)
def _quantize_query_with_summary_kernel(
    query_ptr,
    query_int8_ptr,
    query_scale_ptr,
    query_summary_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    storage_sequence_length,
    stride_qb,
    stride_qh,
    stride_qn,
    heads,
    heads_inner,
    softmax_scale: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Quantize one Q64 tile and emit its min/max routing summary from the same load."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    query_block = tl.where(heads_inner != 0, pid1, pid0).to(tl.int64)
    head = tl.where(heads_inner != 0, pid0, pid1)
    batch = tl.program_id(2)
    values, valid_rows, sequence_offsets, features = _load_row_tile(
        query_ptr,
        query_block,
        head,
        batch,
        stride_qb,
        stride_qh,
        stride_qn,
        block_lengths_ptr,
        logical_sequence_length,
        mask_block_lengths,
        head_dim,
        block_m,
    )
    summary, _auxiliary = sparse_piper_kernels.summarize_block_tiles(
        tl.reshape(values, (1, 1, block_m, head_dim)),
        tl.reshape(valid_rows, (1, block_m)),
        tl.constexpr(False),
        tl.constexpr(True),
    )
    head_block = (batch * heads + head).to(tl.int64)
    tl.store(
        query_summary_ptr
        + (head_block * (storage_sequence_length // block_m) + query_block) * head_dim
        + features,
        tl.reshape(summary, (head_dim,)),
    )

    # Quantization sees only the logical length, as the separate quantizer did:
    # padded storage zeroes invalid rows but still scales every row group.
    groups = tl.arange(0, block_m // scale_rows)
    group_valid = query_block * block_m + groups * scale_rows < logical_sequence_length
    quantized, stored_scale = qk_quantization.quantize_query_tile(
        tl.reshape(values, (block_m, 1, head_dim)),
        tl.reshape(group_valid, (1, block_m // scale_rows)),
        softmax_scale,
        tl.constexpr(1),
        head_dim,
        block_m,
        scale_rows,
    )
    tl.store(
        query_int8_ptr
        + head_block * storage_sequence_length * head_dim
        + sequence_offsets[:, None].to(tl.int64) * head_dim
        + features[None, :],
        tl.reshape(quantized, (block_m, head_dim)),
    )
    tl.store(
        query_scale_ptr
        + head_block * (storage_sequence_length // scale_rows)
        + query_block * (block_m // scale_rows)
        + groups,
        tl.reshape(stored_scale, (block_m // scale_rows,)),
    )


_LaunchPreparationKernels = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float, _PreparedSparsePiperOperands],
    None,
]


def _launch_triton_fused_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    operands: _PreparedSparsePiperOperands,
) -> None:
    """Launch fused Q/K quantization + min/max summary Triton kernels."""
    assert operands.query_summary is not None
    assert operands.key_summary is not None
    assert operands.key_aux is not None
    heads = query.shape[1]
    kv_heads = key.shape[1]
    head_dim = query.shape[-1]
    sequence_length = operands.logical_sequence_length
    storage_sequence_length = operands.key.shape[2]
    blocks = storage_sequence_length // _BLOCK_N
    block_lengths = operands.block_lengths
    lengths_or_placeholder = block_lengths if block_lengths is not None else operands.key_scale
    query_grid, query_heads_inner = select_block_grid(query, blocks)
    key_grid, key_heads_inner = select_block_grid(key, blocks)
    with device_context(query.device):
        _quantize_query_with_summary_kernel[query_grid](
            query,
            operands.query,
            operands.query_scale,
            operands.query_summary,
            lengths_or_placeholder,
            sequence_length,
            storage_sequence_length,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            heads,
            query_heads_inner,
            scale,
            mask_block_lengths=block_lengths is not None,
            head_dim=head_dim,
            block_m=_BLOCK_N,
            scale_rows=QUERY_SCALE_ROWS,
            num_warps=_FUSED_QK_WARPS[head_dim],
        )
        qk_quantization.quantize_key_tile_centered_per_block_kernel[key_grid](
            key,
            operands.key,
            operands.key_scale,
            operands.key_center,
            operands.key_center_scale,
            lengths_or_placeholder,
            sequence_length,
            storage_sequence_length,
            blocks,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            operands.key.stride(0),
            operands.key.stride(1),
            operands.key.stride(2),
            operands.key_scale.stride(0),
            operands.key_scale.stride(1),
            kv_heads,
            key_heads_inner,
            mask_block_lengths=block_lengths is not None,
            head_dim=head_dim,
            block_n=_BLOCK_N,
            key_summary_ptr=operands.key_summary,
            key_aux_ptr=operands.key_aux,
            sparse_key_blocks=operands.sparse_key_blocks,
            emit_summaries=True,
            num_warps=_FUSED_QK_WARPS[head_dim],
        )


def _launch_triton_value(
    combined_value: torch.Tensor,
    value_format: ValueFormat,
    operands: _PreparedSparsePiperOperands,
) -> None:
    """Launch per-K64-tile Triton V mean-centering and packing kernel."""
    _batch, kv_heads, logical_sequence_length, head_dim = combined_value.shape
    tile_count = operands.value_scale_multiplier.shape[2]
    block_lengths = operands.block_lengths
    value_grid, value_heads_inner = select_block_grid(combined_value, tile_count)
    with device_context(combined_value.device):
        _quantize_value_per_tile_kernel[value_grid](
            combined_value,
            operands.value_mean,
            operands.value_scale_multiplier,
            operands.value,
            block_lengths if block_lengths is not None else operands.value_scale_multiplier,
            logical_sequence_length,
            combined_value.stride(0),
            combined_value.stride(1),
            combined_value.stride(2),
            kv_heads,
            value_heads_inner,
            value_format=int(value_format),
            head_dim=head_dim,
            block_n=_BLOCK_N,
            mask_block_lengths=block_lengths is not None,
            num_warps=4,
        )


def _prepare_sparse_piper_operands(
    query: torch.Tensor,
    scale: float,
    *,
    sparse_key_blocks: int,
    combined_key: torch.Tensor,
    combined_value: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
    emit_summaries: bool = False,
    value_format: ValueFormat = ValueFormat.TRANSPOSED,
    launch_kernels: _LaunchPreparationKernels | None = None,
) -> _PreparedSparsePiperOperands:
    """Allocate sparse-Piper storage once and dispatch backend quantization kernels.

    Mask padded rows in the mean and V kernels. The fused Q/K pass and custom
    launchers also mask padding, while the separate Triton Q/K quantizer still
    needs zeroed copies.
    """
    batch, kv_heads, logical_sequence_length, head_dim = combined_key.shape
    heads = query.shape[1]
    tile_count = (logical_sequence_length + _BLOCK_N - 1) // _BLOCK_N
    storage_sequence_length = tile_count * _BLOCK_N

    if launch_kernels is None and not emit_summaries and batch > 0 and tile_count > 0:
        if block_lengths is not None:
            valid_rows = valid_block_rows(block_lengths).reshape(-1)
            valid_rows = valid_rows[None, None, :, None]
            query = torch.where(valid_rows, query, 0)
        query_int8, query_scale = qk_quantization.prepare_query(
            query,
            scale,
            grouped=True,
            storage_query_length=storage_sequence_length,
        )
        key_int8, key_scale, key_center, key_center_scale = (
            qk_quantization.prepare_tile_centered_key(
                combined_key,
                storage_key_length=storage_sequence_length,
                block_lengths=block_lengths,
            )
        )
        query_summary = key_summary = key_aux = None
    else:
        query_int8 = torch.empty(
            (batch, heads, storage_sequence_length, head_dim),
            device=query.device,
            dtype=torch.int8,
        )
        query_scale = torch.empty(
            (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
            device=query.device,
            dtype=torch.float32,
        )
        key_int8 = torch.empty(
            (batch, kv_heads, storage_sequence_length, head_dim),
            device=combined_key.device,
            dtype=torch.int8,
        )
        key_scale = torch.empty(
            (batch, kv_heads, tile_count),
            device=combined_key.device,
            dtype=torch.float32,
        )
        key_center = torch.empty(
            (batch, kv_heads, tile_count, 2, head_dim),
            device=combined_key.device,
            dtype=torch.int8,
        )
        key_center_scale = torch.empty(
            (batch, kv_heads, tile_count, 2),
            device=combined_key.device,
            dtype=torch.float32,
        )
        query_summary = (
            torch.empty(
                (batch, heads, tile_count, head_dim),
                device=query.device,
                dtype=torch.float32,
            )
            if emit_summaries
            else None
        )
        key_summary = (
            torch.empty(
                (batch, kv_heads, sparse_key_blocks, head_dim),
                device=combined_key.device,
                dtype=torch.float32,
            )
            if emit_summaries
            else None
        )
        key_aux = torch.empty_like(key_summary) if key_summary is not None else None

    value_int8 = torch.empty(
        value_format.shape(batch, kv_heads, storage_sequence_length, head_dim),
        device=combined_value.device,
        dtype=torch.int8,
    )
    value_scale_multiplier = torch.empty(
        (batch, kv_heads, tile_count, 1),
        device=combined_value.device,
        dtype=torch.float32,
    )
    value_mean = torch.empty(
        (batch, kv_heads, tile_count, head_dim),
        device=combined_value.device,
        dtype=torch.bfloat16,
    )
    operands = _PreparedSparsePiperOperands(
        key=key_int8,
        value=value_int8,
        value_format=value_format,
        key_scale=key_scale,
        key_center=key_center,
        key_center_scale=key_center_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_mean=value_mean,
        query=query_int8,
        query_scale=query_scale,
        block_lengths=block_lengths,
        sparse_key_blocks=sparse_key_blocks,
        sparse_query_blocks=sparse_query_blocks,
        logical_sequence_length=logical_sequence_length,
        query_summary=query_summary,
        key_summary=key_summary,
        key_aux=key_aux,
    )
    if launch_kernels is not None:
        if batch > 0 and tile_count > 0:
            launch_kernels(query, combined_key, combined_value, scale, operands)
        return operands
    if emit_summaries and batch > 0 and tile_count > 0:
        _launch_triton_fused_qk(query, combined_key, scale, operands)
    if combined_value.numel() > 0 and tile_count > 0:
        _launch_triton_value(combined_value, value_format, operands)
    return operands
