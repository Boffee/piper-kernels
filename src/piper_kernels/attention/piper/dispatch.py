"""Validation and backend selection for Piper Attention."""

import math

import torch

from piper_kernels._triton.targets import supports_uint8_int8_mma

from .reference import reference_piper_attention

try:
    from .triton import triton_piper_attention as _triton_piper_attention
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_piper_attention = None

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_HEAD_DIMS = (64, 128)


def _validate_inputs(  # noqa: PLR0912
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    is_causal: bool,
) -> float:
    tensors = {"query": query, "key": key, "value": value}
    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise ValueError(
                f"Piper Attention {name} must have shape "
                f"[batch, heads, sequence, head_dim], got {tuple(tensor.shape)}"
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Piper Attention {name} must use float16 or bfloat16, got {tensor.dtype}"
            )
        if tensor.layout is not torch.strided:
            raise ValueError(f"Piper Attention {name} must use strided layout")
        if tensor.stride(-1) != 1:
            raise ValueError(f"Piper Attention {name}'s head dimension must be contiguous")

    if query.device != key.device or query.device != value.device:
        raise ValueError(
            "Piper Attention query, key, and value must share a device, "
            f"got {query.device}/{key.device}/{value.device}"
        )
    if query.dtype is not key.dtype or query.dtype is not value.dtype:
        raise ValueError(
            "Piper Attention query, key, and value must share a dtype, "
            f"got {query.dtype}/{key.dtype}/{value.dtype}"
        )
    if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
        raise ValueError(
            "Piper Attention currently requires equal batch and head dimensions, got "
            f"{query.shape[:2]}/{key.shape[:2]}/{value.shape[:2]}"
        )
    if key.shape[2] != value.shape[2]:
        raise ValueError(
            f"Piper Attention key/value lengths must match, got {key.shape[2]}/{value.shape[2]}"
        )
    if query.shape[3] != key.shape[3] or key.shape[3] != value.shape[3]:
        raise ValueError(
            "Piper Attention head dimensions must match, got "
            f"{query.shape[3]}/{key.shape[3]}/{value.shape[3]}"
        )
    if query.shape[3] not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            "Piper Attention currently supports head dimensions 64 and 128, "
            f"got {query.shape[3]}"
        )
    if query.shape[2] == 0 or key.shape[2] == 0:
        raise ValueError("Piper Attention does not accept empty query or key sequences")
    if is_causal and query.shape[2] != key.shape[2]:
        raise ValueError(
            "Causal Piper Attention currently requires equal query and key lengths, got "
            f"{query.shape[2]}/{key.shape[2]}"
        )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors.values()):
        raise RuntimeError(
            "Piper Attention is an inference-only operator and does not support autograd"
        )

    converted_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(converted_scale) or converted_scale <= 0:
        raise ValueError(f"Piper Attention scale must be finite and positive, got {scale}")
    return converted_scale


def _supports_triton(device: torch.device) -> bool:
    return _triton_piper_attention is not None and supports_uint8_int8_mma(device)


def _default_center_value(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool,
) -> bool:
    """Select the measured centered-V region without assuming it helps every target."""
    if not _supports_triton(query.device):
        return False
    capability = torch.cuda.get_device_capability(query.device)
    return (
        capability[0] == 12
        and not is_causal
        and query.shape[-1] == 128
        and query.shape[2] >= 1024
        and key.shape[2] >= 1024
    )


def piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    center_value: bool | None = None,
) -> torch.Tensor:
    """Run Piper's key-scaled UINT8-P/INT8-V forward attention.

    Query, key, and value use ``[batch, heads, sequence, head_dim]`` layout
    with FP16 or BF16 elements and contiguous head dimensions. Query and key
    sequence lengths may differ for non-causal attention. ``scale`` defaults
    to ``head_dim**-0.5``.

    The algorithm retains Sage-style INT8 QK and FP32 online softmax, then
    quantizes each V row with its own signed-INT8 scale and folds that scale
    into a UINT8 probability operand. ``center_value=True`` subtracts the
    sequence-wide per-feature V mean before quantization and restores it in
    the epilogue. ``None`` enables centering only in its measured long,
    non-causal SM12x D128 region.

    The optimized backend supports NVIDIA SM8x and consumer Blackwell SM12x,
    where the packaged compiler extension can select mixed-sign MMAv2. Other
    devices use the portable quantized reference. This is an inference-only
    operator.
    """
    converted_scale = _validate_inputs(query, key, value, scale, is_causal)
    selected_centering = (
        _default_center_value(query, key, is_causal)
        if center_value is None
        else bool(center_value)
    )
    if _supports_triton(query.device):
        assert _triton_piper_attention is not None
        return _triton_piper_attention(
            query,
            key,
            value,
            converted_scale,
            is_causal,
            selected_centering,
        )
    return reference_piper_attention(
        query,
        key,
        value,
        converted_scale,
        is_causal,
        center_value=selected_centering,
    )
