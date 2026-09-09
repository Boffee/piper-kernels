"""Operation contracts for fused projections into shared sparse-attention storage."""

from typing import Protocol

import torch

type QueryOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
type KeyOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
type ValueOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ProjectionBackend(Protocol):
    """Consume validated prepared inputs and fill caller-owned Q/K/V storage.

    INT8 projection accumulates in INT32. Intermediate precision, compute tiles,
    fusion boundaries, launch counts, and device primitives belong to the
    implementation, not graph rewrites or query-chunk orchestration.
    They do not change Q32/K64/V64 scale groups or 64-row routing summaries.

    Inputs and caller-owned outputs are contiguous on the same device. B is
    batch size, H is the head count, and S is the output sequence capacity,
    padded to a multiple of 64. Only Q uses chunk-local output storage;
    input rows, RoPE positions, and optional valid-prefix block lengths remain
    global. Output tuples contain INT8 data followed by FP32 metadata.
    """

    def project_query(  # noqa: PLR0913, PLR0917
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
        routing_mode: int,
        block_lengths: torch.Tensor | None,
        *,
        chunk_start: int,
        chunk_rows: int,
        out: QueryOutput,
    ) -> None:
        """Fill (Q[B,H,S,D], scales[B,H,S/32], summaries[B,H,S/64,D]), D=64 or 128.

        Read the global window [chunk_start, chunk_start + chunk_rows) and
        write it starting at local row zero, with neutral tail padding.
        """

    def project_key(  # noqa: PLR0913
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        norm_weight: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        norm_epsilon: float,
        routing_mode: int,
        block_lengths: torch.Tensor | None,
        *,
        out: KeyOutput,
    ) -> None:
        """Fill (K[B,H,S,D], scales[B,H,S/64], summaries, auxiliary), D=64 or 128.

        Minmax routing uses two [B,H,S/64,D] summary tensors. Mean routing
        uses one summary tensor and an empty auxiliary [B,H,0,D].
        """

    def project_value(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        input_mean: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        block_lengths: torch.Tensor | None,
        *,
        emit_block_mean: bool,
        out: ValueOutput,
    ) -> None:
        """Fill (V[B,H,D,S], multipliers[B,H,S/64,1], mean, block_mean), D=64 or 128.

        The projected global mean is [B,H,D]. When emit_block_mean is true,
        block_mean is [B,H,S/64,D]; otherwise it aliases mean and must not
        be written separately. V is centered using the projected global mean.
        """
