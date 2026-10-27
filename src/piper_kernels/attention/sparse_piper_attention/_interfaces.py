"""Producer storage requirements and bound execution for sparse-Piper backends."""

from dataclasses import dataclass
from typing import Protocol

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat

from ._prepared import (
    _PreparedSparsePiperContext,
    _PreparedSparsePiperOperands,
    _PreparedSparsePiperQuery,
)


class PrepareOperands(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        scale: float,
        *,
        sparse_key_blocks: int,
        combined_key: torch.Tensor,
        combined_value: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        sparse_query_blocks: int | None = None,
        emit_summaries: bool = False,
    ) -> _PreparedSparsePiperOperands: ...


class LaunchAttention(Protocol):
    """Run validated query windows against an immutable context.

    Query preparation validates compatibility with ``context``. Output and
    epilogue metadata are validated per launch; context metadata is bound once.
    ``coarse_gate`` covers the local output rows; ``coarse_output`` covers the
    prepared query storage, located globally by ``query_state.global_block_offset``.
    """

    @property
    def context(self) -> _PreparedSparsePiperContext: ...

    def __call__(
        self,
        query_state: _PreparedSparsePiperQuery,
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
    """The producer storage contract and a mandatory context binding operation.

    Binding owns immutable K/V tensors and backend launch state. Each execution
    accepts only window-local Q, routes, output, and optional coarse epilogue.
    """

    value_format: ValueFormat
    prepare: PrepareOperands
    bind: BindAttention
    # Selected for this backend and head width; applies only to full-keep calls.
    skip_dense_routing: bool = False

    def bind_context(self, context: _PreparedSparsePiperContext) -> LaunchAttention:
        """Validate the producer's format and own its immutable context."""
        if context.value_format is not self.value_format:
            raise ValueError(f"sparse Piper backend requires {self.value_format.name} V storage")
        return self.bind(context)


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
