"""Compiler-visible dense Piper boundary shared by NVIDIA and AMD."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from . import _backend


@torch.library.custom_op("piper_kernels::piper_attention", mutates_args=())
def triton_piper_attention(
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


@triton_piper_attention.register_fake
def _triton_piper_attention_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
