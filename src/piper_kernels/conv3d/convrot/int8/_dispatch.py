"""Resolve ConvRot INT8 convolution plans from validated operand metadata."""

import math

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._interfaces import ConvolutionPolicy
from ._plan import ConvolutionExecutionPlan, ConvolutionSchedule
from ._validation import _output_shape


def default_execution_plan(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    policy: ConvolutionPolicy,
    group_norm: bool,
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    target: AcceleratorTarget | None = None,
    convolution_schedule: ConvolutionSchedule | None = None,
) -> ConvolutionExecutionPlan:
    """Use production policy, with explicit target/tile overrides for offline tuning."""
    target = AcceleratorTarget.from_device(input.device) if target is None else target
    batch, channels, frames, height, width = input.shape
    outputs = weight_qdata.shape[0]
    output_shape = _output_shape(
        input.shape, outputs, stride, symmetric_spatial_padding, right_spatial_padding
    )
    return policy.select_execution_plan(
        target,
        channels=channels,
        outputs=outputs,
        input_rows=batch * frames * height * width,
        output_rows=batch * math.prod(output_shape[2:]),
        output_height=output_shape[3],
        weight_aligned=weight_qdata.data_ptr() % 16 == 0,
        group_norm=group_norm,
        convolution_schedule=convolution_schedule,
    )
