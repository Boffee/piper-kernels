"""Validation and backend selection for Piper Attention."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention._validation import validate_attention_inputs

from .reference import reference_piper_attention

try:
    from .triton import triton_piper_attention as _triton_piper_attention
except ModuleNotFoundError as exc:
    if exc.name != "triton":
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
    )


def _supports_triton(target: AcceleratorTarget) -> bool:
    return _triton_piper_attention is not None and target.supports_uint8_int8_mma


def piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run Piper Attention's key-scaled UINT8-P/INT8-V forward attention.

    Query, key, and value use ``[batch, heads, sequence, head_dim]`` layout
    with FP16 or BF16 elements and contiguous head dimensions. Query and key
    sequence lengths may differ for non-causal attention. ``scale`` defaults
    to ``head_dim**-0.5``.

    The algorithm retains Sage-style INT8 QK and FP32 online softmax, then
    quantizes each V row with its own signed-INT8 scale and folds that scale
    into a UINT8 probability operand. Non-causal calls center V by its
    sequence-wide per-feature mean and restore that mean in the epilogue;
    causal calls remain uncentered so future V rows cannot affect earlier
    outputs through quantization.

    The optimized backend supports NVIDIA SM8x and consumer Blackwell SM12x,
    where the packaged compiler extension can select mixed-sign MMAv2. Other
    devices use the portable quantized reference. This is an inference-only
    operator.
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
