"""Dispatch the async-copy projection backend from validated operand metadata."""

from functools import partial

import torch

from piper_kernels.linear.convrot.int8._nvidia import gluon_async_copy as linear_gluon

from .. import triton as projection
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from . import gluon_async_copy, policy
from ._plan import NvidiaExecutionPlan


def default_execution_plan(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    *,
    head_dim: int,
    rotary_dim: int = 0,
) -> NvidiaExecutionPlan:
    """Resolve the async-copy backend's policy from validated operand metadata."""
    return policy.select_execution_plan(
        input_features=input_qdata.shape[2],
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        operands_aligned=linear_gluon.operands_aligned(input_qdata, weight_qdata),
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
    """Project a query window, with the Gluon kernel when it covers the operands."""
    plan = default_execution_plan(
        input_qdata, weight_qdata, head_dim=out[0].shape[3], rotary_dim=cos.shape[1]
    )
    launch = (
        gluon_async_copy.project_query
        if plan.execution_plan is None
        else partial(projection.project_query, execution_plan=plan.execution_plan)
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
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
        out=out,
        bias=bias,
    )


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
    """Project keys, with the Gluon kernel when it covers the operands."""
    plan = default_execution_plan(
        input_qdata, weight_qdata, head_dim=out[0].shape[3], rotary_dim=cos.shape[1]
    )
    launch = (
        gluon_async_copy.project_key
        if plan.execution_plan is None
        else partial(projection.project_key, execution_plan=plan.execution_plan)
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
        out=out,
        bias=bias,
    )


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
    """Project values, with the Gluon kernel when it covers the operands."""
    plan = default_execution_plan(input_qdata, weight_qdata, head_dim=out[0].shape[2])
    launch = (
        gluon_async_copy.project_value
        if plan.execution_plan is None
        else partial(projection.project_value, execution_plan=plan.execution_plan)
    )
    launch(
        input_qdata,
        input_scale,
        input_mean,
        weight_qdata,
        weight_scale,
        block_lengths,
        emit_block_mean=emit_block_mean,
        out=out,
        bias=bias,
    )
