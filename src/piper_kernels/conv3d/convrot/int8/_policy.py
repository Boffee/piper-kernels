"""Schedules, execution plans, and target policy for ConvRot INT8 Conv3D.

Kept Triton-free so CPU imports, validation, and offline compilation/benchmark
planning never depend on Triton. Vendor-specific SM120/SM8x and RDNA4 policies
live in ``_nvidia/policy.py`` and ``_amd/policy.py`` while sharing the kernels in
``triton.py``.
"""

from __future__ import annotations

import math

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._amd import policy as _amd_policy
from ._nvidia import policy as _nvidia_policy
from ._schedule import (
    ConvolutionExecutionPlan,
    ConvolutionSchedule,
    Padding,
    PreparationSchedule,
    output_shape,
)

_amd_preparation_schedule = _amd_policy.preparation_schedule


def supports_target(target: AcceleratorTarget) -> bool:
    """Return whether ``target`` has a tuned native Conv3D schedule."""
    return _nvidia_policy.supports_target(target) or _amd_policy.supports_target(target)


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    channels: int,
    outputs: int,
    input_rows: int,
    output_rows: int,
    output_height: int,
    weight_aligned: bool,
    group_norm: bool,
    convolution_schedule: ConvolutionSchedule | None = None,
) -> ConvolutionExecutionPlan:
    """Resolve preparation, convolution, and weight-load choices once before launch.

    An explicit ``convolution_schedule`` lets tuning reuse production preparation
    and recompute weight-descriptor eligibility for the candidate's column width.
    """
    if _nvidia_policy.supports_target(target):
        return _nvidia_policy.select_execution_plan(
            target,
            channels=channels,
            outputs=outputs,
            input_rows=input_rows,
            output_rows=output_rows,
            output_height=output_height,
            weight_aligned=weight_aligned,
            group_norm=group_norm,
            convolution_schedule=convolution_schedule,
        )
    if _amd_policy.supports_target(target):
        return _amd_policy.select_execution_plan(
            target,
            channels=channels,
            outputs=outputs,
            output_rows=output_rows,
            convolution_schedule=convolution_schedule,
        )
    raise ValueError(f"ConvRot INT8 convolution has no native policy for {target}")


def default_execution_plan(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    group_norm: bool,
    padding: Padding,
    target: AcceleratorTarget | None = None,
    convolution_schedule: ConvolutionSchedule | None = None,
) -> ConvolutionExecutionPlan:
    """Resolve the production plan from validated operand metadata."""
    target = AcceleratorTarget.from_device(input.device) if target is None else target
    batch, channels, frames, height, width = input.shape
    outputs = weight_qdata.shape[0]
    shape = output_shape(input.shape, outputs, stride, padding)
    return select_execution_plan(
        target,
        channels=channels,
        outputs=outputs,
        input_rows=batch * frames * height * width,
        output_rows=batch * math.prod(shape[2:]),
        output_height=shape[3],
        weight_aligned=weight_qdata.data_ptr() % 16 == 0,
        group_norm=group_norm,
        convolution_schedule=convolution_schedule,
    )


__all__ = [
    "ConvolutionExecutionPlan",
    "ConvolutionSchedule",
    "Padding",
    "PreparationSchedule",
    "default_execution_plan",
    "output_shape",
    "select_execution_plan",
    "supports_target",
]
