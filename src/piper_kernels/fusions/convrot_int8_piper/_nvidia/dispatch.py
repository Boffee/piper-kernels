"""NVIDIA dense Q/K/V dispatch over the shared Triton projection kernels."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_projection._plan import ProjectionExecutionPlan

from .. import triton as projection
from .._interfaces import KeyOutput, QueryOutput, ValueOutput
from . import policy


def default_execution_plan(
    input_qdata: torch.Tensor, *, target: AcceleratorTarget | None = None
) -> ProjectionExecutionPlan:
    """Resolve the operand device, accepting an explicit target for offline tuning."""
    target = AcceleratorTarget.from_device(input_qdata.device) if target is None else target
    return policy.select_execution_plan(target)


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
    execution_plan: ProjectionExecutionPlan | None = None,
) -> None:
    """Project a query window with the target's resolved plan."""
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
        bias,
        chunk_start=chunk_start,
        chunk_rows=chunk_rows,
        out=out,
        execution_plan=(
            execution_plan if execution_plan is not None else default_execution_plan(input_qdata)
        ),
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
    execution_plan: ProjectionExecutionPlan | None = None,
) -> None:
    """Project keys with the target's resolved plan."""
    projection.project_key(
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
        execution_plan=(
            execution_plan if execution_plan is not None else default_execution_plan(input_qdata)
        ),
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
    execution_plan: ProjectionExecutionPlan | None = None,
) -> None:
    """Project values with the target's resolved plan."""
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
        execution_plan=(
            execution_plan if execution_plan is not None else default_execution_plan(input_qdata)
        ),
    )
