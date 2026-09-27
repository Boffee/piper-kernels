"""SM89 fused projections: Gluon kernels for D128 heads, shared Triton launchers otherwise."""

import torch

from .. import triton as projection
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from .._layout import TILE_ROWS
from . import gluon

# Triton configuration for the shapes that the Gluon kernels do not cover. SM120's tiles need more
# than SM89's 99 KiB of shared memory (K) or spill (V's 128x256 accumulator). Tiles of 64 rows by
# one head keep four warps within the register budget, and grouping eight row blocks reuses input
# rows across heads.
_CONFIG = projection.ProjectionConfig(
    block_m=TILE_ROWS,
    block_k=128,
    heads_per_program=1,
    num_warps=4,
    num_stages=3,
    group_m=8,
    round_rsqrt_to_nearest=True,
)


def project_query(  # noqa: PLR0913, PLR0917
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
    """Project a query window with the Gluon kernel when it covers the operands."""
    arguments = (
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        routing_mode,
        block_lengths,
    )
    window = {"chunk_start": chunk_start, "chunk_rows": chunk_rows, "out": out, "bias": bias}
    if gluon.supports_projection(input_qdata, out[0].shape[3], cos.shape[1]):
        gluon.project_query(*arguments, **window)
    else:
        projection.project_query(*arguments, config=_CONFIG, **window)


def project_key(  # noqa: PLR0913
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
    """Project keys with the Gluon kernel when it covers the operands."""
    arguments = (
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        routing_mode,
        block_lengths,
    )
    if gluon.supports_projection(input_qdata, out[0].shape[3], cos.shape[1]):
        gluon.project_key(*arguments, out=out, bias=bias)
    else:
        projection.project_key(*arguments, config=_CONFIG, out=out, bias=bias)


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    emit_block_mean: bool,
    out: ValueOutput,
    bias: torch.Tensor | None = None,
) -> None:
    """Project values with the Gluon kernel when it covers the operands."""
    arguments = (input_qdata, input_scale, input_mean, weight_qdata, weight_scale, block_lengths)
    if gluon.supports_projection(input_qdata, out[0].shape[2]):
        gluon.project_value(*arguments, emit_block_mean=emit_block_mean, out=out, bias=bias)
    else:
        projection.project_value(
            *arguments, config=_CONFIG, emit_block_mean=emit_block_mean, out=out, bias=bias
        )
