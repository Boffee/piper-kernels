"""Operations over shared sparse-Piper tensors, independent of kernel layouts."""

from dataclasses import dataclass
from typing import Protocol

import torch

from ._prepared import _PreparedSparsePiperAttention, _PreparedSparsePiperContext


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


class BindAttention(Protocol):
    def __call__(self, context: _PreparedSparsePiperContext) -> LaunchAttention: ...


@dataclass(frozen=True, slots=True)
class AttentionBackend:
    """Preparation and execution over the common quantized tensor contract.

    A launcher owns any accelerator-specific packing, descriptors, and schedule;
    callers only provide quantized tensors, routes, and logical query ranges.
    """

    prepare: PrepareAttention
    launch: LaunchAttention
    bind: BindAttention | None = None
    # Selected for this backend and head width; applies only to full-keep calls.
    skip_dense_routing: bool = False

    def bind_context(self, context: _PreparedSparsePiperContext) -> LaunchAttention:
        """Own backend preparation once for an immutable K/V context lifetime."""
        return self.launch if self.bind is None else self.bind(context)


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


class MinmaxScores(Protocol):
    def __call__(
        self,
        query_summary: torch.Tensor,
        key_primary: torch.Tensor,
        key_aux: torch.Tensor,
        *,
        score_scale: float | None = None,
    ) -> torch.Tensor: ...


class SequenceSummaries(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        routing_mode: int,
        block_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
