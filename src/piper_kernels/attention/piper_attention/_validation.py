"""Metadata checks and prepared operand containers for dense Piper Attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ._nvidia.policy import PiperAttentionExecutionPlan


@dataclass(frozen=True, slots=True)
class PreparedContext:
    """Reusable K/V operands shared by NVIDIA and AMD dense Piper backends.

    ``key`` and ``value`` hold tensors (or NVIDIA ``TensorDescriptor`` objects
    when ``execution_plan.use_tensor_descriptors`` is enabled). ``execution_plan``
    is set for NVIDIA contexts and ``None`` for AMD RDNA4 contexts.
    """

    key: Any
    value: Any
    key_scale: torch.Tensor
    key_center: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_log_scale: torch.Tensor
    value_mean: torch.Tensor
    key_length: int
    is_causal: bool
    execution_plan: PiperAttentionExecutionPlan | None = None
    padded_kv: bool = False


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """Prepared Q with its unpadded logical shape and original floating dtype.

    With fused query quantization, ``data`` is floating-point Q, ``scale`` is
    ``None``, and ``softmax_scale`` is applied when the kernel quantizes each
    tile. ``descriptor`` holds an optional NVIDIA ``TensorDescriptor``.
    """

    data: torch.Tensor
    scale: torch.Tensor | None
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    descriptor: Any = None
    global_row_offset: int = 0
    softmax_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class PreparedAttention:
    """Prepared K/V and Q paired with one reusable output for full attention."""

    context: PreparedContext
    query: PreparedQuery
    output: torch.Tensor


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
