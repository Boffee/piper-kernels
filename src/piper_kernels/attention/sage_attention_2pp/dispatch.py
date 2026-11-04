"""Validation and backend selection for SageAttention2++."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention._validation import validate_attention_inputs

from .reference import reference_sage_attention_2pp

try:
    from .triton import triton_sage_attention_2pp as _triton_sage_attention_2pp
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_sage_attention_2pp = None


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    is_causal: bool,
) -> float:
    return validate_attention_inputs(
        "SageAttention2++",
        query,
        key,
        value,
        scale,
        is_causal,
    )


def _supports_triton(target: AcceleratorTarget) -> bool:
    return _triton_sage_attention_2pp is not None and target.supports_fp8_fp16_mma


def sage_attention_2pp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Compute dense or causal attention with canonical SageAttention2++ 8+8.

    ``query``, ``key``, and ``value`` use strided
    ``[batch, heads, sequence, head_dim]`` tensors on the same device, with
    matching FP16 or BF16 dtype, batch size, head count, and head dimension of
    64 or 128. The final dimension must be contiguous; outer dimensions may
    be strided. Key and value shapes match. Head counts and sequence lengths
    must be positive. Grouped-query attention is unsupported.

    ``scale`` must be finite and positive and defaults to ``head_dim**-0.5``.
    Non-causal calls allow different query and key lengths. ``is_causal=True``
    requires equal lengths and lets query row ``i`` attend key rows through
    ``i``; offset-causal decoding, arbitrary masks, and dropout are unsupported.

    Returns a tensor with the query's shape, dtype, and device, without mutating
    the inputs. This operator is inference-only: detach inputs or use
    ``torch.no_grad()`` / ``torch.inference_mode()`` when they require gradients.

    INT8 Q/K, FP8 probabilities/V, and FP16 tile accumulation approximate
    floating-point attention. The optimized backend requires NVIDIA FP8 tensor
    cores with FP16 accumulation and the Triton backend installed. Other
    configurations use a slower reference for the same quantized algorithm,
    not exact floating-point SDPA. Cross-backend bitwise equality is not promised.
    """
    converted_scale = _validate_inputs(query, key, value, scale, is_causal)
    target = AcceleratorTarget.from_device(query.device)
    if _supports_triton(target):
        assert _triton_sage_attention_2pp is not None
        return _triton_sage_attention_2pp(query, key, value, converted_scale, is_causal)
    return reference_sage_attention_2pp(query, key, value, converted_scale, is_causal)
