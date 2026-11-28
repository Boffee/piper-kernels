"""Validation and backend selection for Piper Attention."""

from typing import Literal, overload

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels._validation import validate_nonnegative_int
from piper_kernels.attention._validation import validate_attention_inputs

from . import _backend
from .reference import reference_piper_attention

_INT64_MAX = (1 << 63) - 1


def _validate_op_window_bound(value: int, name: str) -> None:
    message = f"Piper Attention {name} must be -1 or a nonnegative integer"
    if isinstance(value, bool) or not isinstance(value, (int, torch.SymInt)):
        raise TypeError(f"{message}, got {type(value).__name__}")
    if torch.compiler.is_compiling() or isinstance(value, torch.SymInt):
        torch._check(value >= -1, lambda: message)
        torch._check(value <= _INT64_MAX, lambda: message)
        return
    if value < -1 or value > _INT64_MAX:
        raise ValueError(f"{message}, got {value}")


def _decode_op_window(window_left: int, window_right: int) -> tuple[int | None, int | None]:
    _validate_op_window_bound(window_left, "window_left")
    _validate_op_window_bound(window_right, "window_right")
    return (
        None if window_left == -1 else window_left,
        None if window_right == -1 else window_right,
    )


@torch.library.custom_op("piper_kernels::piper_attention", mutates_args=())
def _triton_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    window_left: int = -1,
    window_right: int = -1,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the selected native preparation and integer-PV recurrence."""
    eff_left, eff_right = _decode_op_window(window_left, window_right)
    backend = _backend.select_backend(AcceleratorTarget.from_device(query.device))
    if backend is None:
        raise RuntimeError(f"native Piper Attention is unavailable on {query.device}")
    result = backend(
        query,
        key,
        value,
        scale,
        is_causal,
        window_left=eff_left,
        window_right=eff_right,
        return_lse=return_lse,
    )
    if return_lse:
        assert isinstance(result, tuple)
        return result
    assert isinstance(result, torch.Tensor)
    return result, query.new_empty((0,), dtype=torch.float32)


@_triton_piper_attention.register_fake
def _triton_piper_attention_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
    window_left: int = -1,
    window_right: int = -1,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_op_window_bound(window_left, "window_left")
    _validate_op_window_bound(window_right, "window_right")
    out = torch.empty_like(query, memory_format=torch.contiguous_format)
    lse = (
        torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        if return_lse
        else query.new_empty((0,), dtype=torch.float32)
    )
    return out, lse


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    is_causal: bool,
    window_left: int | None,
    window_right: int | None,
    return_lse: bool,
) -> tuple[float, int, int]:
    if not isinstance(is_causal, bool):
        raise TypeError("Piper Attention is_causal must be a bool")
    if not isinstance(return_lse, bool):
        raise TypeError("Piper Attention return_lse must be a bool")
    converted_scale = validate_attention_inputs(
        "Piper Attention",
        query,
        key,
        value,
        scale,
        is_causal,
        allow_gqa=True,
    )
    resolved_left = (
        -1
        if window_left is None
        else validate_nonnegative_int(
            window_left,
            "Piper Attention window_left must be a nonnegative integer",
        )
    )
    resolved_right = (
        -1
        if window_right is None
        else validate_nonnegative_int(
            window_right,
            "Piper Attention window_right must be a nonnegative integer",
        )
    )
    return converted_scale, resolved_left, resolved_right


def _supports_triton(target: AcceleratorTarget) -> bool:
    return _backend.select_backend(target) is not None


@overload
def piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    window_left: int | None = None,
    window_right: int | None = None,
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    window_left: int | None = None,
    window_right: int | None = None,
    return_lse: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    window_left: int | None = None,
    window_right: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    window_left: int | None = None,
    window_right: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute dense, causal, or windowed quantized attention, including grouped queries.

    ``query``, ``key``, and ``value`` use strided
    ``[batch, heads, sequence, head_dim]`` tensors on the same device, with
    matching FP16 or BF16 dtype, batch size, and head dimension of 64 or 128.
    The final dimension must be contiguous; outer dimensions may be strided.
    Key and value shapes match. Head counts and sequence lengths must be positive;
    query heads must be an integer multiple of KV heads. Query head ``h`` uses
    KV head ``h // (query_heads // kv_heads)`` without repeating K/V storage.

    ``scale`` must be finite and positive and defaults to ``head_dim**-0.5``.
    Non-causal calls allow different query and key lengths. ``is_causal=True``
    requires equal lengths and caps the effective right window bound at ``0``.
    ``window_left`` and ``window_right`` are inclusive non-negative token
    distances (or ``None`` for unlimited): query position ``q_pos`` attends
    key position ``k_pos`` iff ``q_pos - window_left <= k_pos <= q_pos + eff_right``.
    Queries whose visible key intersection is empty return ``0.0`` output and
    ``-inf`` LSE.

    Pass ``return_lse=True`` to receive ``(output, lse)`` with FP32 natural-log
    ``lse`` of shape ``[batch, query_heads, query_sequence]``. Without
    ``return_lse``, returns a tensor with the query's shape, dtype, and device,
    without mutating the inputs. This operator is inference-only: detach inputs
    or use ``torch.no_grad()`` / ``torch.inference_mode()`` when they require
    gradients.

    INT8 Q/K/V and UINT8 probabilities approximate floating-point attention.
    K and V are centered independently per 1024-token chunk across all modes,
    making KV preparation token-sharding friendly across 1024-token shard
    boundaries while eliminating global cross-token reductions.
    Native backends cover NVIDIA SM8x/SM12x and AMD RDNA4 (gfx1200/gfx1201)
    with the required backend installed. Other configurations use a slower
    reference for the same quantized algorithm, not exact floating-point SDPA.
    Cross-backend bitwise equality is not promised.
    """
    converted_scale, resolved_left, resolved_right = _validate_inputs(
        query,
        key,
        value,
        scale,
        is_causal,
        window_left,
        window_right,
        return_lse,
    )
    target = AcceleratorTarget.from_device(query.device)
    if _supports_triton(target):
        output, lse = _triton_piper_attention(
            query,
            key,
            value,
            converted_scale,
            is_causal,
            resolved_left,
            resolved_right,
            return_lse,
        )
        return (output, lse) if return_lse else output
    return reference_piper_attention(
        query,
        key,
        value,
        converted_scale,
        is_causal,
        window_left=window_left,
        window_right=window_right,
        return_lse=return_lse,
    )


__all__ = ["piper_attention"]
