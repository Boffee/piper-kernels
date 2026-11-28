"""Metadata checks and prepared operand containers for dense Piper Attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

if TYPE_CHECKING:
    from ._nvidia.policy import PiperAttentionExecutionPlan

type WindowMode = Literal["none", "causal", "windowed"]


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
    window_left: int | None = None
    window_right: int | None = None


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


def normalize_window_bounds(
    is_causal: bool,
    window_left: int | None,
    window_right: int | None,
) -> tuple[int | None, int | None, WindowMode]:
    """Collapse ``(is_causal, window_left, window_right)`` into effective bounds and mode."""
    eff_left = window_left
    eff_right: int | None = 0 if is_causal else window_right
    if eff_left is None and eff_right is None:
        return None, None, "none"
    right_is_zero = (
        statically_known_true(eff_right == 0)
        if isinstance(eff_right, torch.SymInt)
        else eff_right == 0
    )
    if eff_left is None and right_is_zero:
        return None, 0, "causal"
    return eff_left, eff_right, "windowed"


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


def validate_lse_buffer(
    lse: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    device: torch.device,
) -> None:
    """Validate float32 LSE storage with contiguous query rows."""
    if (
        lse.layout is not torch.strided
        or lse.shape != shape
        or lse.dtype is not torch.float32
        or lse.device != device
        or lse.stride(-1) != 1
    ):
        raise ValueError(
            "Piper Attention lse must match [batch, heads, rows], float32, query device, "
            "and contiguous query rows"
        )
    span = 1
    for stride, size in sorted(zip(lse.stride(), shape, strict=True)):
        if size > 1:
            if stride < span:
                raise ValueError("Piper Attention lse must use nonoverlapping storage")
            span = stride * (size - 1) + span
