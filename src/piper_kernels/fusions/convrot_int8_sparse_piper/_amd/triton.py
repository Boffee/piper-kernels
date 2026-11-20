"""RDNA4 sparse Q/K/V dispatch over Gluon and shared Triton projection kernels."""

from typing import Literal

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .. import triton as projection
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from .._layout import TILE_ROWS, padded_sequence_length
from . import gluon, policy


def default_execution_plan(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    *,
    operation: policy.ProjectionOperation,
    head_dim: int,
    rotary_dim: int = 0,
    storage_sequence_length: int | None = None,
    value_format: ValueFormat = ValueFormat.AMD_WMMA,
) -> policy.AmdExecutionPlan:
    """Resolve operand metadata and select RDNA4 Gluon or Triton execution."""
    resolved_storage_sequence = (
        padded_sequence_length(input_qdata.shape[1])
        if storage_sequence_length is None
        else storage_sequence_length
    )
    heads = (
        weight_qdata.shape[0] // head_dim
        if head_dim > 0 and weight_qdata.shape[0] % head_dim == 0
        else 0
    )
    return policy.select_execution_plan(
        operation,
        head_dim=head_dim,
        input_features=input_qdata.shape[2],
        heads=heads,
        rotary_dim=rotary_dim,
        operands_aligned=gluon.operands_aligned(
            input_qdata,
            weight_qdata,
            storage_sequence_length=resolved_storage_sequence,
            rotary_dim=rotary_dim,
            head_dim=head_dim,
        ),
        value_format=value_format,
    )


def project_query(
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
    execution_plan: ProjectionExecutionPlan | None = None,
) -> None:
    """Project a query window using the RDNA4 Gluon kernel or Triton fallback."""
    plan = (
        default_execution_plan(
            input_qdata,
            weight_qdata,
            operation="query",
            head_dim=out[0].shape[3],
            rotary_dim=cos.shape[1],
            storage_sequence_length=out[0].shape[2],
        )
        if execution_plan is None
        else execution_plan
    )
    launch = (
        gluon.project_query
        if isinstance(plan, policy.AmdExecutionPlan) and plan.kernel == "gluon"
        else projection.project_query
    )
    launch(
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
        execution_plan=plan,
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
        out=out,
        bias=bias,
    )


def partial_row_block_size(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    *,
    operation: Literal["key", "value"],
    head_dim: int,
    rotary_dim: int = 0,
    value_format: ValueFormat = ValueFormat.TRANSPOSED,
) -> int:
    """Return RDNA4's row block height for partial K/V launches."""
    return default_execution_plan(
        input_qdata,
        weight_qdata,
        operation=operation,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        storage_sequence_length=padded_sequence_length(input_qdata.shape[1]),
        value_format=value_format,
    ).block_m


def project_key(
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
    execution_plan: ProjectionExecutionPlan | None = None,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    """Project keys using the RDNA4 Gluon kernel or Triton fallback."""
    plan = (
        default_execution_plan(
            input_qdata,
            weight_qdata,
            operation="key",
            head_dim=out[0].shape[3],
            rotary_dim=cos.shape[1],
            storage_sequence_length=out[0].shape[2],
        )
        if execution_plan is None
        else execution_plan
    )
    launch = (
        gluon.project_key
        if isinstance(plan, policy.AmdExecutionPlan) and plan.kernel == "gluon"
        else projection.project_key
    )
    launch(
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
        execution_plan=plan,
        out=out,
        bias=bias,
        start_row_block=start_row_block,
        num_row_blocks=num_row_blocks,
    )


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    block_lengths: torch.Tensor | None,
    *,
    value_format: ValueFormat,
    out: ValueOutput,
    bias: torch.Tensor | None = None,
    execution_plan: ProjectionExecutionPlan | None = None,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    """Project values using the RDNA4 Gluon kernel or Triton fallback."""
    plan = (
        default_execution_plan(
            input_qdata,
            weight_qdata,
            operation="value",
            head_dim=out[2].shape[-1],
            storage_sequence_length=out[2].shape[2] * TILE_ROWS,
            value_format=value_format,
        )
        if execution_plan is None
        else execution_plan
    )
    launch = (
        gluon.project_value
        if (
            isinstance(plan, policy.AmdExecutionPlan)
            and plan.kernel == "gluon"
            and value_format is ValueFormat.AMD_WMMA
        )
        else projection.project_value
    )
    launch(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        block_lengths,
        execution_plan=plan,
        value_format=value_format,
        out=out,
        bias=bias,
        start_row_block=start_row_block,
        num_row_blocks=num_row_blocks,
    )
