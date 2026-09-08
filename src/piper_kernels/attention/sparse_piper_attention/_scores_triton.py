"""FP32 minmax summary scoring without an auxiliary global score matrix."""

# Triton's JIT launch options are not represented in its Python signatures.
# pyright: reportCallIssue=false, reportGeneralTypeIssues=false

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context

_BLOCK_M = 32
_BLOCK_N = 64
_BLOCK_K = 16


@triton.jit
def _minmax_scores_kernel(  # noqa: PLR0913, PLR0917
    query_ptr: torch.Tensor,
    primary_ptr: torch.Tensor,
    auxiliary_ptr: torch.Tensor,
    output_ptr: torch.Tensor,
    query_blocks: int,
    key_blocks: int,
    heads: int,
    query_batch_stride: int,
    query_head_stride: int,
    query_block_stride: int,
    primary_batch_stride: int,
    primary_head_stride: int,
    primary_block_stride: int,
    auxiliary_batch_stride: int,
    auxiliary_head_stride: int,
    auxiliary_block_stride: int,
    score_scale: float,
    has_score_scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    batch_head = tl.program_id(2).to(tl.int64)
    batch, head = batch_head // heads, batch_head % heads
    rows = (tl.program_id(0) * block_m + tl.arange(0, block_m)).to(tl.int64)
    columns = (tl.program_id(1) * block_n + tl.arange(0, block_n)).to(tl.int64)
    features = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, 2 * block_n), tl.float32)
    for start in range(0, 128, block_k):
        offsets = start + features
        query = tl.load(
            query_ptr
            + batch * query_batch_stride
            + head * query_head_stride
            + rows[:, None] * query_block_stride
            + offsets[None, :],
            mask=rows[:, None] < query_blocks,
            other=0.0,
        )
        primary = tl.load(
            primary_ptr
            + batch * primary_batch_stride
            + head * primary_head_stride
            + columns[None, :] * primary_block_stride
            + offsets[:, None],
            mask=columns[None, :] < key_blocks,
            other=0.0,
        )
        auxiliary = tl.load(
            auxiliary_ptr
            + batch * auxiliary_batch_stride
            + head * auxiliary_head_stride
            + columns[None, :] * auxiliary_block_stride
            + offsets[:, None],
            mask=columns[None, :] < key_blocks,
            other=0.0,
        )
        # Interleave paired keys in registers so one dot reuses each query tile.
        keys = tl.reshape(tl.join(primary, auxiliary), (block_k, 2 * block_n))
        accumulator = tl.dot(query, keys, accumulator, input_precision="ieee")
    primary_scores, auxiliary_scores = tl.split(tl.reshape(accumulator, (block_m, block_n, 2)))
    if has_score_scale:
        primary_scores *= score_scale
        auxiliary_scores *= score_scale
    scores = tl.maximum(primary_scores, auxiliary_scores, propagate_nan=tl.PropagateNan.ALL)
    tl.store(
        output_ptr
        + batch_head * query_blocks * key_blocks
        + rows[:, None] * key_blocks
        + columns[None, :],
        scores,
        mask=(rows[:, None] < query_blocks) & (columns[None, :] < key_blocks),
    )


def minmax_scores(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    *,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Contract validated D128 summaries into one contiguous FP32 score tensor."""
    batch, heads, query_blocks, _ = query_summary.shape
    key_blocks = key_primary.shape[2]
    output = query_summary.new_empty((batch, heads, query_blocks, key_blocks))
    with device_context(query_summary.device):
        _minmax_scores_kernel[
            (triton.cdiv(query_blocks, _BLOCK_M), triton.cdiv(key_blocks, _BLOCK_N), batch * heads)
        ](
            query_summary,
            key_primary,
            key_aux,
            output,
            query_blocks,
            key_blocks,
            heads,
            *query_summary.stride()[:3],
            *key_primary.stride()[:3],
            *key_aux.stride()[:3],
            1.0 if score_scale is None else score_scale,
            has_score_scale=score_scale is not None,
            block_m=_BLOCK_M,
            block_n=_BLOCK_N,
            block_k=_BLOCK_K,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["minmax_scores"]
