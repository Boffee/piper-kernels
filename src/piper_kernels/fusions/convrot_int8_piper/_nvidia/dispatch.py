"""NVIDIA dense Q/K/V dispatch over Gluon and shared Triton projection kernels."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._nvidia._plan import (
    NvidiaExecutionPlan,
    ProjectionOperation,
)
from piper_kernels.linear.convrot.int8._nvidia import gluon_async_copy as linear_gluon

from .. import triton as projection
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from . import gluon_async_copy, policy


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


def project_query(  # noqa: PLR0913
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
        bias,
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
        out=out,
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
    bias: torch.Tensor | None = None,
    *,
    out: KeyOutput,
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
        bias,
        out=out,
        execution_plan=plan,
    )


def project_value(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    is_causal: bool,
    out: ValueOutput,
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
    if plan.kernel == "gluon_async_copy":
        gluon_async_copy.project_value(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            is_causal=is_causal,
            out=out,
            execution_plan=plan,
        )
        return
    projection.project_value(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        is_causal=is_causal,
        packed_amd=policy.PACKED_VALUE,
        mean_block_n=policy.VALUE_MEAN_BLOCK_N,
        out=out,
        execution_plan=plan,
    )
