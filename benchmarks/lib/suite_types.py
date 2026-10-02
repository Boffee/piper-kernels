"""The small adapter contract shared by benchmark operation families."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .quality import QualityMetrics


@dataclass(frozen=True)
class QualityCheck:
    """Reference coverage in scalar elements and the error bound for the comparison."""

    metrics: QualityMetrics
    reference: str
    sample_count: int
    total_count: int
    relative_l2_limit: float
    comparisons: Mapping[str, QualityMetrics] = field(default_factory=dict)


@dataclass
class Operation:
    """Resident inputs/weights; run includes all required per-call preparation."""

    run: Callable[[], torch.Tensor]
    check: Callable[[torch.Tensor], QualityCheck]
    configuration: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Implementation:
    """Defer allocation and compilation until this implementation is measured."""

    name: str
    build: Callable[[], Operation]
    unsupported_reason: str | None = None


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
