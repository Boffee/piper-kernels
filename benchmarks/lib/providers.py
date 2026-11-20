"""Unified deferred implementation and runnable operation harness for benchmarks and tuners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from .quality import QualityCheck


def _no_sync() -> None:
    return None


@dataclass
class Operation[OutputT = torch.Tensor]:
    """Resident inputs/weights; run executes the measured operation call."""

    run: Callable[[], OutputT]
    check: Callable[[OutputT], QualityCheck] | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
    synchronize: Callable[[], None] = _no_sync


@dataclass(frozen=True)
class Implementation[OperationT = Operation[torch.Tensor]]:
    """Defer allocation and compilation until this implementation is measured."""

    name: str
    build: Callable[[], OperationT]
    unsupported_reason: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)
