"""Operations over shared sparse-Piper tensors, independent of kernel layouts."""

from dataclasses import dataclass
from typing import Protocol

import torch

from ._prepared import _PreparedSparsePiperAttention


class PrepareAttention(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        routes: torch.Tensor,
        head_keep_blocks: torch.Tensor,
        scale: float,
        *,
        sparse_key_blocks: int,
        route_head_offsets: torch.Tensor,
        combined_key: torch.Tensor,
        combined_value: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        sparse_query_blocks: int | None = None,
    ) -> _PreparedSparsePiperAttention: ...


class LaunchAttention(Protocol):
    def __call__(
        self,
        prepared: _PreparedSparsePiperAttention,
        output: torch.Tensor,
        *,
        query_block_offset: int = 0,
        query_block_count: int | None = None,
        coarse_output: torch.Tensor | None = None,
        coarse_gate: torch.Tensor | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class AttentionBackend:
    """Preparation and execution over the common quantized tensor contract.

    A launcher owns any accelerator-specific packing, descriptors, and schedule;
    callers only provide quantized tensors, routes, and logical query ranges.
    """

    prepare: PrepareAttention
    launch: LaunchAttention


class SelectRoutes(Protocol):
    def __call__(
        self,
        scores: torch.Tensor,
        routes: torch.Tensor,
        head_keep_blocks: torch.Tensor,
        route_head_offsets: torch.Tensor,
        *,
        query_block_offset: int,
    ) -> None: ...


class SequenceSummaries(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        routing_mode: int,
        block_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
