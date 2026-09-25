"""Shared Triton preparation for the quantized sparse-Piper tensor contract."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.sparse_piper import (
    triton as sparse_piper_kernels,
)
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS
from piper_kernels.attention.piper_attention import _quantization as piper_quantization

from ._block_layout import valid_block_rows, validate_sparse_query_blocks
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
    "stride_kb",
    "stride_kh",
    "stride_kn",
    "heads",
)
# Measured warps per K64 tile. Head width is already a specialization key.
_FUSED_QK_WARPS = {64: 1, 128: 2}


@triton.jit
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
    stride_ob,
    stride_oh,
    stride_od,
    stride_ok,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    mask_block_lengths: tl.constexpr,
):
    """Quantize one K64 storage tile while masking the logical V tail."""
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid_rows = offsets_n < key_length
    if mask_block_lengths:
        valid_rows &= offsets_n - key_block * block_n < tl.load(block_lengths_ptr + key_block)
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    ).to(tl.float32)
    value_mean = tl.load(value_mean_ptr + batch_head * head_dim + offsets_d)
    quantized, value_scale_multiplier = sparse_piper_kernels.quantize_value_tile(  # pyright: ignore[reportGeneralTypeIssues]
        tl.reshape(value, (block_n, 1, head_dim)),
        tl.reshape(value_mean, (1, head_dim)),
        valid_rows,
        tl.constexpr(1),
        head_dim,
        block_n,
        block_n,
    )
    quantized = tl.reshape(quantized, (block_n, head_dim))
    value_scale_multiplier = tl.reshape(value_scale_multiplier, ())
    tile_count = tl.cdiv(key_length, block_n)
    tl.store(
        value_scale_ptr + batch_head * tile_count + key_block,
        value_scale_multiplier,
    )
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_d[None, :] * stride_od
        + offsets_n[:, None] * stride_ok,
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
    softmax_scale: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Quantize one Q64 tile and emit its min/max routing summary from the same load."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
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


@triton.jit(do_not_specialize=_FUSED_DO_NOT_SPECIALIZE)
def _quantize_key_with_summary_kernel(
    key_ptr,
    key_mean_ptr,
    key_int8_ptr,
    key_scale_ptr,
    key_summary_ptr,
    key_aux_ptr,
    block_lengths_ptr,
    logical_sequence_length,
    storage_sequence_length,
    stride_kb,
    stride_kh,
    stride_kn,
    heads,
    mask_block_lengths: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Quantize one centered K64 tile and emit min/max routing summaries of the raw tile."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    values, valid_rows, sequence_offsets, features = _load_row_tile(
        key_ptr,
        key_block,
        head,
        batch,
        stride_kb,
        stride_kh,
        stride_kn,
        block_lengths_ptr,
        logical_sequence_length,
        mask_block_lengths,
        head_dim,
        block_n,
    )
    head_block = (batch * heads + head).to(tl.int64)

    # Routing summarizes raw K; quantization encodes K centered by its mean.
    summary, auxiliary = sparse_piper_kernels.summarize_block_tiles(
        tl.reshape(values, (1, 1, block_n, head_dim)),
        tl.reshape(valid_rows, (1, block_n)),
        tl.constexpr(False),
        tl.constexpr(False),
    )
    summary_offsets = (
        head_block * (storage_sequence_length // block_n) + key_block
    ) * head_dim + features
    tl.store(key_summary_ptr + summary_offsets, tl.reshape(summary, (head_dim,)))
    tl.store(key_aux_ptr + summary_offsets, tl.reshape(auxiliary, (head_dim,)))

    # Centering sees only the logical length, as the separate quantizer did, so a
    # zeroed padded row still centers to -mean and shares its tile's scale.
    mean = tl.load(key_mean_ptr + head_block * head_dim + features)
    logical_rows = sequence_offsets < logical_sequence_length
    centered = tl.where(logical_rows[:, None], values - mean[None, :], 0.0)
    quantized, key_scale = qk_quantization.quantize_key_tile(
        tl.reshape(centered, (block_n, 1, head_dim)),
        tl.constexpr(1),
        head_dim,
        block_n,
        block_n,
    )
    tl.store(
        key_int8_ptr
        + head_block * storage_sequence_length * head_dim
        + sequence_offsets[:, None].to(tl.int64) * head_dim
        + features[None, :],
        tl.reshape(quantized, (block_n, head_dim)),
    )
    tl.store(
        key_scale_ptr + head_block * (storage_sequence_length // block_n) + key_block,
        tl.reshape(key_scale, ()),
    )


def _prepare_query_key_with_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
    key_mean: torch.Tensor,
    scale: float,
    *,
    storage_sequence_length: int,
    block_lengths: torch.Tensor | None,
) -> tuple[qk_quantization.PreparedInt8QueryKey, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize Q/K and emit their min/max K64 summaries in one pass over each operand.

    The separate summary pass reads Q and K a second time. Emitting both from
    one load halves those reads and removes two launches; the quantization and
    summary arithmetic are the shared primitives, so results are unchanged.
    """
    batch, heads, sequence_length, head_dim = query.shape
    kv_heads = key.shape[1]
    blocks = storage_sequence_length // _BLOCK_N
    query_int8 = torch.empty(
        (batch, heads, storage_sequence_length, head_dim), device=query.device, dtype=torch.int8
    )
    query_scale = torch.empty(
        (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS),
        device=query.device,
        dtype=torch.float32,
    )
    key_int8 = torch.empty(
        (batch, kv_heads, storage_sequence_length, head_dim), device=key.device, dtype=torch.int8
    )
    key_scale = torch.empty((batch, kv_heads, blocks), device=key.device, dtype=torch.float32)
    query_summary = torch.empty(
        (batch, heads, blocks, head_dim), device=query.device, dtype=torch.float32
    )
    key_primary = torch.empty(
        (batch, kv_heads, blocks, head_dim), device=key.device, dtype=torch.float32
    )
    key_aux = torch.empty_like(key_primary)
    lengths_or_placeholder = block_lengths if block_lengths is not None else key_mean
    with device_context(query.device):
        _quantize_query_with_summary_kernel[(blocks, heads, batch)](
            query,
            query_int8,
            query_scale,
            query_summary,
            lengths_or_placeholder,
            sequence_length,
            storage_sequence_length,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            heads,
            scale,
            mask_block_lengths=block_lengths is not None,
            head_dim=head_dim,
            block_m=_BLOCK_N,
            scale_rows=QUERY_SCALE_ROWS,
            num_warps=_FUSED_QK_WARPS[head_dim],
        )
        _quantize_key_with_summary_kernel[(blocks, kv_heads, batch)](
            key,
            key_mean,
            key_int8,
            key_scale,
            key_primary,
            key_aux,
            lengths_or_placeholder,
            sequence_length,
            storage_sequence_length,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            kv_heads,
            mask_block_lengths=block_lengths is not None,
            head_dim=head_dim,
            block_n=_BLOCK_N,
            num_warps=_FUSED_QK_WARPS[head_dim],
        )
    prepared = qk_quantization.PreparedInt8QueryKey(
        query=query_int8,
        key=key_int8,
        query_scale=query_scale,
        key_scale=key_scale,
    )
    return prepared, query_summary, key_primary, key_aux


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
) -> _PreparedSparsePiperOperands:
    """Quantize Q/K/V, optionally emitting min/max summaries from the Q/K pass.

    Mask padded rows in the mean and V kernels. The fused Q/K pass also masks
    padding, while the separate Q/K quantizer still needs zeroed copies.
    """
    if (
        combined_key.shape[0] != query.shape[0]
        or combined_key.shape[2:] != query.shape[2:]
        or combined_key.shape[1] < 1
        or query.shape[1] % combined_key.shape[1]
        or combined_value.shape != combined_key.shape
        or not 1 <= sparse_key_blocks <= query.shape[2] // _BLOCK_N
    ):
        raise ValueError("combined Q/K/V and the sparse-prefix K64 count must agree")
    if combined_key.stride(-1) != 1 or combined_value.stride(-1) != 1:
        raise ValueError("combined K/V feature dimensions must be contiguous")

    batch, kv_heads, logical_sequence_length, head_dim = combined_key.shape
    if block_lengths is not None and (
        logical_sequence_length % _BLOCK_N
        or block_lengths.shape != (logical_sequence_length // _BLOCK_N,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != query.device
        or not block_lengths.is_contiguous()
    ):
        raise ValueError("padded sparse Piper requires one contiguous device INT32 length per K64")
    if block_lengths is not None and not emit_summaries:
        valid_rows = valid_block_rows(block_lengths).reshape(-1)
        valid_rows = valid_rows[None, None, :, None]
        query = torch.where(valid_rows, query, 0)
        combined_key = torch.where(valid_rows, combined_key, 0)
    tile_count = (logical_sequence_length + _BLOCK_N - 1) // _BLOCK_N
    storage_sequence_length = tile_count * _BLOCK_N
    validate_sparse_query_blocks(
        sparse_query_blocks,
        query_blocks=tile_count,
        context="sparse Piper",
    )

    key_mean, value_mean = piper_quantization.compute_kv_means(
        combined_key,
        combined_value,
        is_causal=False,
        block_lengths=block_lengths,
    )
    query_summary = key_summary = key_aux = None
    if emit_summaries:
        prepared_qk, query_summary, key_summary, key_aux = _prepare_query_key_with_summaries(
            query,
            combined_key,
            key_mean,
            scale,
            storage_sequence_length=storage_sequence_length,
            block_lengths=block_lengths,
        )
    else:
        prepared_qk = qk_quantization.prepare_query_key(
            query,
            combined_key,
            key_mean,
            scale,
            grouped=True,
            storage_key_length=storage_sequence_length,
            storage_query_length=storage_sequence_length,
        )
    value_int8 = torch.empty(
        (batch, kv_heads, head_dim, storage_sequence_length),
        device=combined_value.device,
        dtype=torch.int8,
    )
    value_scale_multiplier = torch.empty(
        (batch, kv_heads, tile_count, 1),
        device=combined_value.device,
        dtype=torch.float32,
    )

    with device_context(query.device):
        _quantize_value_per_tile_kernel[(tile_count, batch * kv_heads)](
            combined_value,
            value_mean,
            value_scale_multiplier,
            value_int8,
            block_lengths if block_lengths is not None else value_mean,
            logical_sequence_length,
            combined_value.stride(0),
            combined_value.stride(1),
            combined_value.stride(2),
            value_int8.stride(0),
            value_int8.stride(1),
            value_int8.stride(2),
            value_int8.stride(3),
            heads=kv_heads,
            head_dim=head_dim,
            block_n=_BLOCK_N,
            mask_block_lengths=block_lengths is not None,
            num_warps=4,
        )
    return _PreparedSparsePiperOperands(
        key=prepared_qk.key,
        value=value_int8,
        key_scale=prepared_qk.key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_mean=value_mean,
        query=prepared_qk.query,
        query_scale=prepared_qk.query_scale,
        block_lengths=block_lengths,
        sparse_key_blocks=sparse_key_blocks,
        sparse_query_blocks=sparse_query_blocks,
        logical_sequence_length=logical_sequence_length,
        query_summary=query_summary,
        key_summary=key_summary,
        key_aux=key_aux,
    )
