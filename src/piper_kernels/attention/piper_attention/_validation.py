"""Metadata checks for dense Piper query windows, Q/K scales, and output storage."""

import torch

from piper_kernels.attention.kernels.qk_quantization.int8.sage.reference import (
    QKQuantizationGranularity,
)


def validate_qk_quantization(granularity: str) -> QKQuantizationGranularity:
    """Accept the Sage Q/K scale granularities that quantized Piper operands may use."""
    if granularity == "per_thread":
        return "per_thread"
    if granularity == "per_warp":
        return "per_warp"
    raise ValueError(f"unknown Q/K quantization granularity: {granularity}")


def query_scale_length(storage_length: int, granularity: QKQuantizationGranularity) -> int:
    """Count FP32 scales per head of Q64-padded Q: one per Q32 group or per row."""
    return storage_length if granularity == "per_thread" else storage_length // 32


def key_scale_length(storage_length: int, granularity: QKQuantizationGranularity) -> int:
    """Count FP32 scales per head of K64-padded K: one per K64 group or per key."""
    return storage_length if granularity == "per_thread" else storage_length // 64


def scale_granularity(scales: torch.Tensor, storage_length: int) -> QKQuantizationGranularity:
    """Read Q/K granularity from scale metadata: one scale per stored row is per thread."""
    return "per_thread" if scales.ndim == 3 and scales.shape[2] == storage_length else "per_warp"


def validate_query_offset(offset: int, *, block_rows: int, name: str) -> None:
    """Keep prepared query origins and launch starts aligned to query tiles."""
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError(f"Piper Attention {name} must be an integer")
    if offset < 0 or offset % block_rows:
        raise ValueError(f"Piper Attention {name} must be a nonnegative multiple of {block_rows}")


def resolve_query_window(
    query_length: int,
    *,
    query_start: int,
    query_rows: int | None,
    global_row_offset: int,
    block_rows: int,
    key_length: int,
    is_causal: bool,
) -> int:
    """Resolve a local Q range while preserving its global causal coordinates."""
    validate_query_offset(query_start, block_rows=block_rows, name="query_start")
    validate_query_offset(global_row_offset, block_rows=block_rows, name="global_row_offset")
    if query_rows is not None and (isinstance(query_rows, bool) or not isinstance(query_rows, int)):
        raise TypeError("Piper Attention query_rows must be an integer or None")
    rows = query_length - query_start if query_rows is None else query_rows
    if rows < 1 or query_start + rows > query_length:
        raise ValueError("Piper Attention query window must fit the prepared query")
    if is_causal and global_row_offset + query_start + rows > key_length:
        raise ValueError("Piper Attention causal query window must fit the key sequence")
    return rows


def validate_output_buffer(
    output: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate padded or permuted dense layouts with contiguous head features."""
    if (
        output.layout is not torch.strided
        or output.shape != shape
        or output.dtype is not dtype
        or output.device != device
        or output.stride(-1) != 1
    ):
        raise ValueError(
            "Piper Attention output must match the query shape, dtype, and device "
            "and use strided storage with contiguous head features"
        )
    # Permuted dimensions and padding between rows, heads, or batches are valid.
    # Sorting strides proves disjoint storage using only host metadata.
    span = 1
    for stride, size in sorted(zip(output.stride(), shape, strict=True)):
        if size > 1:
            if stride < span:
                raise ValueError(
                    "Piper Attention output must use nonoverlapping padded or permuted storage"
                )
            span = stride * (size - 1) + span
