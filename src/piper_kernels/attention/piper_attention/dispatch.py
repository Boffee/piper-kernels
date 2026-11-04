"""Validation and backend selection for Piper Attention."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention._validation import validate_attention_inputs

from . import _backend
from .reference import reference_piper_attention

try:
    from .triton import triton_piper_attention as _triton_piper_attention
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _triton_piper_attention = None


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
    return _triton_piper_attention is not None and _backend.select_backend(target) is not None


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
    Non-causal calls center V and restore its mean in the result; causal calls
    leave V uncentered so future values cannot affect earlier quantization.
    Native backends cover NVIDIA SM8x/SM12x and AMD RDNA4 (gfx1200/gfx1201)
    with the required backend installed. Other configurations use a slower
    reference for the same quantized algorithm, not exact floating-point SDPA.
    Cross-backend bitwise equality is not promised.
    """
    converted_scale = _validate_inputs(query, key, value, scale, is_causal)
    target = AcceleratorTarget.from_device(query.device)
    if _supports_triton(target):
        assert _triton_piper_attention is not None
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
