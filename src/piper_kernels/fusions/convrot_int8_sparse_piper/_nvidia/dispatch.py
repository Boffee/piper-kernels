"""NVIDIA sparse Q/K/V dispatch over Gluon and shared Triton projection kernels."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._nvidia import gluon_async_copy as linear_gluon

from .. import triton as projection
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from . import gluon_async_copy, policy
from ._plan import NvidiaExecutionPlan, ProjectionOperation


def default_execution_plan(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    *,
    operation: ProjectionOperation,
    head_dim: int,
    rotary_dim: int = 0,
    target: AcceleratorTarget | None = None,
) -> NvidiaExecutionPlan:
    """Resolve operand metadata, accepting an explicit target for offline tuning."""
    target = AcceleratorTarget.from_device(input_qdata.device) if target is None else target
    return policy.select_execution_plan(
        target,
        operation=operation,
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
    execution_plan: NvidiaExecutionPlan | None = None,
) -> None:
    """Project a query window, with the Gluon kernel when it covers the operands."""
    plan = (
        execution_plan
        if execution_plan is not None
        else default_execution_plan(
            input_qdata,
            weight_qdata,
            operation="query",
            head_dim=out[0].shape[3],
            rotary_dim=cos.shape[1],
        )
    )
    launch = (
        gluon_async_copy.project_query
        if plan.kernel == "gluon_async_copy"
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
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
        out=out,
        bias=bias,
        execution_plan=plan,
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
    execution_plan: NvidiaExecutionPlan | None = None,
) -> None:
    """Project keys, with the Gluon kernel when it covers the operands."""
    plan = (
        execution_plan
        if execution_plan is not None
        else default_execution_plan(
            input_qdata,
            weight_qdata,
            operation="key",
            head_dim=out[0].shape[3],
            rotary_dim=cos.shape[1],
        )
    )
    launch = (
        gluon_async_copy.project_key
        if plan.kernel == "gluon_async_copy"
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
        out=out,
        bias=bias,
        execution_plan=plan,
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
    execution_plan: NvidiaExecutionPlan | None = None,
) -> None:
    """Project values, with the Gluon kernel when it covers the operands."""
    plan = (
        execution_plan
        if execution_plan is not None
        else default_execution_plan(
            input_qdata, weight_qdata, operation="value", head_dim=out[0].shape[2]
        )
    )
    launch = (
        gluon_async_copy.project_value
        if plan.kernel == "gluon_async_copy"
        else projection.project_value
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
        execution_plan=plan,
    )
