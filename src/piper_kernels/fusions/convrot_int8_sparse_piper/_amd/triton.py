"""Bind shared projection launchers to RDNA4 policy."""

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .. import triton as projection
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from . import policy


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
    """Project a query window using the RDNA4 Q execution plan."""
    plan = (
        policy.select_execution_plan("query", head_dim=out[0].shape[3])
        if execution_plan is None
        else execution_plan
    )
    projection.project_query(
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
) -> None:
    """Project keys using the RDNA4 K execution plan."""
    plan = (
        policy.select_execution_plan("key", head_dim=out[0].shape[3])
        if execution_plan is None
        else execution_plan
    )
    projection.project_key(
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
) -> None:
    """Select the RDNA4 V projection plan from head width unless overridden."""
    plan = (
        policy.select_execution_plan("value", head_dim=out[2].shape[-1])
        if execution_plan is None
        else execution_plan
    )
    projection.project_value(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        block_lengths,
        execution_plan=plan,
        value_format=value_format,
        out=out,
        bias=bias,
    )
