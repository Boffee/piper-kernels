"""Validation and backend selection for Piper Attention."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention._validation import validate_attention_inputs

from . import _backend
from .reference import reference_piper_attention


@torch.library.custom_op("piper_kernels::piper_attention", mutates_args=())
def _triton_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Run the selected native preparation and integer-PV recurrence."""
    backend = _backend.select_backend(AcceleratorTarget.from_device(query.device))
    if backend is None:
        raise RuntimeError(f"native Piper Attention is unavailable on {query.device}")
    return backend(query, key, value, scale, is_causal)


@_triton_piper_attention.register_fake
def _triton_piper_attention_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    is_causal: bool,
) -> float:
    return validate_attention_inputs(
        "Piper Attention",
        query,
        key,
        value,
        scale,
        is_causal,
        allow_gqa=True,
    )


def _supports_triton(target: AcceleratorTarget) -> bool:
    return _backend.select_backend(target) is not None


def piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Compute dense or causal quantized attention, including grouped queries.

    ``query``, ``key``, and ``value`` use strided
    ``[batch, heads, sequence, head_dim]`` tensors on the same device, with
    matching FP16 or BF16 dtype, batch size, and head dimension of 64 or 128.
    The final dimension must be contiguous; outer dimensions may be strided.
    Key and value shapes match. Head counts and sequence lengths must be positive;
    query heads must be an integer multiple of KV heads. Query head ``h`` uses
    KV head ``h // (query_heads // kv_heads)`` without repeating K/V storage.

    ``scale`` must be finite and positive and defaults to ``head_dim**-0.5``.
    Non-causal calls allow different query and key lengths. ``is_causal=True``
    requires equal lengths and lets query row ``i`` attend key rows through
    ``i``; offset-causal decoding, arbitrary masks, and dropout are unsupported.

    Returns a tensor with the query's shape, dtype, and device, without mutating
    the inputs. This operator is inference-only: detach inputs or use
    ``torch.no_grad()`` / ``torch.inference_mode()`` when they require gradients.

    INT8 Q/K/V and UINT8 probabilities approximate floating-point attention.
    K and V are centered independently per 1024-token chunk in both non-causal
    and causal attention, making KV preparation token-sharding friendly across
    1024-token shard boundaries while eliminating global cross-token reductions.
    Native backends cover NVIDIA SM8x/SM12x and AMD RDNA4 (gfx1200/gfx1201)
    with the required backend installed. Other configurations use a slower
    reference for the same quantized algorithm, not exact floating-point SDPA.
    Cross-backend bitwise equality is not promised.
    """
    converted_scale = _validate_inputs(query, key, value, scale, is_causal)
    target = AcceleratorTarget.from_device(query.device)
    if _supports_triton(target):
        return _triton_piper_attention(
            query,
            key,
            value,
            converted_scale,
            is_causal,
        )
    return reference_piper_attention(
        query,
        key,
        value,
        converted_scale,
        is_causal,
    )


__all__ = ["piper_attention"]
