"""Validation and backend selection for SageAttention2++."""

import math

import torch

from .reference import reference_sage_attention

try:
    from .backends.triton import triton_sage_attention as _triton_sage_attention
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_sage_attention = None

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
                f"SageAttention {name} must have shape [batch, heads, sequence, head_dim], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"SageAttention {name} must use float16 or bfloat16, got {tensor.dtype}"
            )
        if tensor.layout is not torch.strided:
            raise ValueError(f"SageAttention {name} must use strided layout")
        if tensor.stride(-1) != 1:
            raise ValueError(f"SageAttention {name}'s head dimension must be contiguous")

    if query.device != key.device or query.device != value.device:
        raise ValueError(
            "SageAttention query, key, and value must share a device, "
            f"got {query.device}/{key.device}/{value.device}"
        )
    if query.dtype is not key.dtype or query.dtype is not value.dtype:
        raise ValueError(
            "SageAttention query, key, and value must share a dtype, "
            f"got {query.dtype}/{key.dtype}/{value.dtype}"
        )
    if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
        raise ValueError(
            "SageAttention currently requires equal batch and head dimensions, got "
            f"{query.shape[:2]}/{key.shape[:2]}/{value.shape[:2]}"
        )
    if key.shape[2] != value.shape[2]:
        raise ValueError(
            f"SageAttention key/value lengths must match, got {key.shape[2]}/{value.shape[2]}"
        )
    if query.shape[3] != key.shape[3] or key.shape[3] != value.shape[3]:
        raise ValueError(
            "SageAttention head dimensions must match, got "
            f"{query.shape[3]}/{key.shape[3]}/{value.shape[3]}"
        )
    if query.shape[3] not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"SageAttention currently supports head dimensions 64 and 128, got {query.shape[3]}"
        )
    if query.shape[2] == 0 or key.shape[2] == 0:
        raise ValueError("SageAttention does not accept empty query or key sequences")
    if is_causal and query.shape[2] != key.shape[2]:
        raise ValueError(
            "Causal SageAttention currently requires equal query and key lengths, got "
            f"{query.shape[2]}/{key.shape[2]}"
        )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors.values()):
        raise RuntimeError(
            "SageAttention is an inference-only operator and does not support autograd"
        )

    converted_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(converted_scale) or converted_scale <= 0:
        raise ValueError(f"SageAttention scale must be finite and positive, got {scale}")
    return converted_scale


def _supports_triton(device: torch.device) -> bool:
    if _triton_sage_attention is None or device.type != "cuda":
        return False
    capability = torch.cuda.get_device_capability(device)
    return capability == (8, 9) or capability[0] == 12


def sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run canonical SageAttention2++ 8+8 forward attention.

    Inputs use ``[batch, heads, sequence, head_dim]`` layout. The optimized
    backend targets Ada SM89 and consumer Blackwell SM12x. Other devices use
    the portable quantized reference, which is intended for correctness rather
    than performance.
    """
    converted_scale = _validate_inputs(query, key, value, scale, is_causal)
    if _supports_triton(query.device):
        assert _triton_sage_attention is not None
        return _triton_sage_attention(query, key, value, converted_scale, is_causal)
    return reference_sage_attention(query, key, value, converted_scale, is_causal)
