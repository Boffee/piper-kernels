"""Typed execution contracts for dense Piper projection producers."""

from typing import Protocol

import torch

type QueryOutput = tuple[torch.Tensor, torch.Tensor]
type KeyOutput = tuple[torch.Tensor, torch.Tensor]
type ValueOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ProjectionBackend(Protocol):
    """Fill caller-owned native operands; implementations own temporary workspaces.

    All storage is contiguous on one device. Q is local to its aligned global
    input window; K/V cover the complete sequence. Q/K storage is padded to 64
    rows with Q32/K64 FP32 scales. V uses target-native codes and per-token
    FP32 multipliers/logs, plus a per-head mean.
    """

    def project_query(  # noqa: PLR0913
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        norm_weight: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        norm_epsilon: float,
        softmax_scale: float,
        bias: torch.Tensor | None = None,
        *,
        chunk_start: int = 0,
        chunk_rows: int | None = None,
        out: QueryOutput,
    ) -> None: ...

    def project_key(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        norm_weight: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        norm_epsilon: float,
        bias: torch.Tensor | None = None,
        *,
        out: KeyOutput,
    ) -> None: ...

    def project_value(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        is_causal: bool,
        out: ValueOutput,
    ) -> None: ...
