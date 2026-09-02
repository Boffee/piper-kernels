"""Triton K64 summaries shared by sparse Piper routing policies."""

# Triton's JIT launcher options and tensor return types are not represented in
# its Python signatures.
# pyright: reportCallIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels.attention.kernels.sparse_piper import triton as sparse_piper_kernels

from ._routes import _MEAN_ROUTING, validate_routing_mode

_BLOCK_ROWS = 64
_HEAD_DIM = 128


@triton.jit(
    do_not_specialize=[
        "logical_block_count",
        "logical_row_count",
        "stride_ib",
        "stride_ih",
        "stride_il",
        "stride_ir",
    ]
)
def _block_summary_kernel(  # noqa: PLR0913, PLR0917
    input_ptr: torch.Tensor,
    output_primary_ptr: torch.Tensor,
    output_aux_ptr: torch.Tensor,
    block_lengths_ptr: torch.Tensor,
    logical_block_count: int,
    logical_row_count: int,
    stride_ib: int,
    stride_ih: int,
    stride_il: int,
    stride_ir: int,
    block_rows: tl.constexpr,
    head_dim: tl.constexpr,
    heads: tl.constexpr,
    query_summary: tl.constexpr,
    mean_pool_summary: tl.constexpr,
    mask_block_lengths: tl.constexpr,
    mask_ragged_tail: tl.constexpr,
) -> None:
    block = tl.program_id(0)
    batch_head = block // logical_block_count
    logical_block = block % logical_block_count
    batch = batch_head // heads
    head = batch_head % heads
    row_offsets = tl.arange(0, block_rows)[:, None]
    feature_offsets = tl.arange(0, head_dim)[None, :]
    input_offsets = (
        batch * stride_ib
        + head * stride_ih
        + logical_block * stride_il
        + row_offsets * stride_ir
        + feature_offsets
    )
    if mask_block_lengths:
        valid_rows = row_offsets < tl.load(block_lengths_ptr + logical_block)
    elif mask_ragged_tail:
        valid_rows = row_offsets < logical_row_count - logical_block * block_rows
    else:
        valid_rows = row_offsets < block_rows
    values = tl.load(input_ptr + input_offsets, mask=valid_rows, other=0.0).to(tl.float32)

    primary, auxiliary = sparse_piper_kernels.summarize_block_tiles(
        tl.reshape(values, (1, 1, block_rows, head_dim)),
        tl.reshape(valid_rows, (1, block_rows)),
        mean_pool_summary,
        query_summary,
    )
    primary = tl.reshape(primary, (head_dim,))
    auxiliary = tl.reshape(auxiliary, (head_dim,))

    output_offsets = block * head_dim + tl.arange(0, head_dim)
    tl.store(output_primary_ptr + output_offsets, primary)
    if not mean_pool_summary and not query_summary:
        tl.store(output_aux_ptr + output_offsets, auxiliary)


def sequence_block_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize compact or valid-front padded Q/K blocks on CUDA."""
    validate_routing_mode(routing_mode)
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("optimized sequence summaries require rank-four Q/K tensors")
    if (
        query.shape[:2] != key.shape[:2]
        or query.shape[-1] != _HEAD_DIM
        or key.shape[-1] != _HEAD_DIM
        or query.shape[2] < 1
        or key.shape[2] < _BLOCK_ROWS
    ):
        raise ValueError("optimized summaries require nonempty ragged Q/K with D128 K")
    if query.device.type != "cuda" or query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("optimized summaries require CUDA BF16/FP16 inputs")
    if key.device != query.device or key.dtype != query.dtype:
        raise ValueError("optimized summary Q/K sequences must share a device and dtype")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("optimized summaries require contiguous feature dimensions")

    batch, heads, query_rows, _head_dim = query.shape
    key_rows = key.shape[2]
    if block_lengths is not None and (
        query_rows % _BLOCK_ROWS
        or block_lengths.shape != (query_rows // _BLOCK_ROWS,)
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != query.device
        or not block_lengths.is_contiguous()
        or key_rows % _BLOCK_ROWS
        or key_rows // _BLOCK_ROWS > block_lengths.numel()
    ):
        raise ValueError(
            "optimized padded summaries require one contiguous device INT32 length per query K64"
        )
    query_blocks = (query_rows + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    key_blocks = (key_rows + _BLOCK_ROWS - 1) // _BLOCK_ROWS
    query_summary = torch.empty(
        (batch, heads, query_blocks, _HEAD_DIM),
        device=query.device,
        dtype=torch.float32,
    )
    key_primary = torch.empty(
        (batch, heads, key_blocks, _HEAD_DIM),
        device=key.device,
        dtype=torch.float32,
    )
    mean_pool_summary = routing_mode == _MEAN_ROUTING
    key_aux = (
        torch.empty((batch, heads, 0, _HEAD_DIM), device=key.device, dtype=torch.float32)
        if mean_pool_summary
        else torch.empty_like(key_primary)
    )

    def launch(
        sequence: torch.Tensor,
        primary: torch.Tensor,
        aux: torch.Tensor,
        *,
        query: bool,
    ) -> None:
        logical_rows = sequence.shape[2]
        logical_blocks = (logical_rows + _BLOCK_ROWS - 1) // _BLOCK_ROWS
        _block_summary_kernel[(batch * heads * logical_blocks,)](
            sequence,
            primary,
            aux,
            sequence if block_lengths is None else block_lengths,
            logical_block_count=logical_blocks,
            logical_row_count=logical_rows,
            stride_ib=sequence.stride(0),
            stride_ih=sequence.stride(1),
            stride_il=_BLOCK_ROWS * sequence.stride(2),
            stride_ir=sequence.stride(2),
            block_rows=_BLOCK_ROWS,
            head_dim=_HEAD_DIM,
            heads=heads,
            query_summary=query,
            mean_pool_summary=mean_pool_summary,
            mask_block_lengths=block_lengths is not None,
            mask_ragged_tail=block_lengths is None and logical_rows % _BLOCK_ROWS != 0,
            num_warps=4,
            num_stages=1,
        )

    launch(query, query_summary, query_summary, query=True)
    launch(key, key_primary, key_aux, query=False)
    return query_summary, key_primary, key_aux


__all__ = ["sequence_block_summaries"]
