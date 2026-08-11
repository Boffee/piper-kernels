"""Shared validation for full-attention operators."""

import math
from collections.abc import Mapping

import torch

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_HEAD_DIMS = (64, 128)


def _validate_tensor(operator: str, name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 4:
        raise ValueError(
            f"{operator} {name} must have shape [batch, heads, sequence, head_dim], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"{operator} {name} must use float16 or bfloat16, got {tensor.dtype}")
    if tensor.layout is not torch.strided:
        raise ValueError(f"{operator} {name} must use strided layout")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{operator} {name}'s head dimension must be contiguous")


def _validate_tensor_relationships(
    operator: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
) -> None:
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"{operator} query, key, and value must share a device, "
            f"got {query.device}/{key.device}/{value.device}"
        )
    if query.dtype is not key.dtype or query.dtype is not value.dtype:
        raise ValueError(
            f"{operator} query, key, and value must share a dtype, "
            f"got {query.dtype}/{key.dtype}/{value.dtype}"
        )
    if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
        raise ValueError(
            f"{operator} currently requires equal batch and head dimensions, got "
            f"{query.shape[:2]}/{key.shape[:2]}/{value.shape[:2]}"
        )
    if key.shape[2] != value.shape[2]:
        raise ValueError(
            f"{operator} key/value lengths must match, got {key.shape[2]}/{value.shape[2]}"
        )
    if query.shape[3] != key.shape[3] or key.shape[3] != value.shape[3]:
        raise ValueError(
            f"{operator} head dimensions must match, got "
            f"{query.shape[3]}/{key.shape[3]}/{value.shape[3]}"
        )
    if query.shape[3] not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"{operator} currently supports head dimensions 64 and 128, got {query.shape[3]}"
        )
    if query.shape[2] == 0 or key.shape[2] == 0:
        raise ValueError(f"{operator} does not accept empty query or key sequences")
    if is_causal and query.shape[2] != key.shape[2]:
        raise ValueError(
            f"Causal {operator} currently requires equal query and key lengths, got "
            f"{query.shape[2]}/{key.shape[2]}"
        )


def validate_attention_inputs(
    operator: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    is_causal: bool,
) -> float:
    """Validate a common attention contract and return the effective scale."""
    tensors: Mapping[str, torch.Tensor] = {
        "query": query,
        "key": key,
        "value": value,
    }
    for name, tensor in tensors.items():
        _validate_tensor(operator, name, tensor)
    _validate_tensor_relationships(
        operator,
        query,
        key,
        value,
        is_causal=is_causal,
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors.values()):
        raise RuntimeError(
            f"{operator} is an inference-only operator and does not support autograd"
        )

    converted_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(converted_scale) or converted_scale <= 0:
        raise ValueError(f"{operator} scale must be finite and positive, got {scale}")
    return converted_scale
