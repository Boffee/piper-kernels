"""Operation contracts for fused projections into shared sparse-attention storage."""

from typing import Protocol

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat

type QueryOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
type KeyOutput = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
type ValueOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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
        bias: torch.Tensor | None = None,
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
        bias: torch.Tensor | None = None,
    ) -> None:
        """Fill (K, scales, center, center_scales, summaries, auxiliary), D=64 or 128.

        Minmax routing uses two [B,H,S/64,D] summary tensors. Mean routing
        uses one summary tensor and an empty auxiliary [B,H,0,D]. K is
        tile-centered by its K64 INT8 residual mean (two signed terms per tile);
        routing summaries describe the uncentered FP32 transform values before
        BF16 K storage.
        """

    def project_value(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        block_lengths: torch.Tensor | None,
        *,
        value_format: ValueFormat,
        out: ValueOutput,
        bias: torch.Tensor | None = None,
    ) -> None:
        """Fill (V, multipliers[B,H,S/64,1], value_mean[B,H,S/64,D]), D=64 or 128.

        V is allocated in value_format and written directly in that final format.
        Each K64 tile of V is centered around its BF16 per-tile mean
        ``value_mean``; padded tail blocks write zero means.
        """
