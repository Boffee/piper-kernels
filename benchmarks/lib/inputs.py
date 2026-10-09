"""Reproducible inputs and bounded sample selection for benchmarks."""

from collections.abc import Sequence

import torch


def normal_tensor(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """Use CPU-generated inputs so equal seeds mean equal values across vendors."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    if scale != 1.0:
        value.mul_(scale)
    return value.to(dtype=dtype).to(device=device)


def sample_indices(length: int, *, count: int = 64, device: torch.device) -> torch.Tensor:
    """Cover both boundaries and evenly spaced interior positions deterministically."""
    return (
        torch.linspace(0, length - 1, min(length, count), dtype=torch.float64)
        .round()
        .to(
            device=device,
            dtype=torch.int64,
        )
    )
