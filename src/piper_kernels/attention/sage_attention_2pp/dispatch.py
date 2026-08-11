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
    """Run canonical SageAttention2++ 8+8 forward attention.

    Query, key, and value use ``[batch, heads, sequence, head_dim]`` layout
    with FP16 or BF16 elements and contiguous head dimensions. Query and key
    sequence lengths may differ for non-causal attention. ``scale`` defaults
    to ``head_dim**-0.5``.

    The optimized backend requires NVIDIA FP8 tensor cores with FP16
    accumulation. Architecture-specific schedules are selected independently.
    Other devices use the portable quantized reference, which is intended for
    correctness rather than performance. This is an inference-only operator.
    """
    converted_scale = _validate_inputs(query, key, value, scale, is_causal)
    target = AcceleratorTarget.from_device(query.device)
    if _supports_triton(target):
        assert _triton_sage_attention_2pp is not None
        return _triton_sage_attention_2pp(query, key, value, converted_scale, is_causal)
    return reference_sage_attention_2pp(query, key, value, converted_scale, is_causal)
