"""NVIDIA ConvRot INT8 convolution dispatch over shared Triton launchers."""

import torch

from .. import triton as convolution
from .._dispatch import default_execution_plan
from . import policy


def conv3d(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Launch with the SM8x or SM120 execution plan of the input's device."""
    plan = default_execution_plan(
        input,
        weight_qdata,
        stride,
        policy=policy,
        group_norm=False,
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
    )
    return convolution.conv3d(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride,
        execution_plan=plan,
        accelerator_backend="cuda",
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
        residual=residual,
    )


def group_norm_silu_conv3d(  # noqa: PLR0913, PLR0917
    input: torch.Tensor,  # noqa: A002
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_groups: int,
    norm_epsilon: float,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    symmetric_spatial_padding: bool,
    right_spatial_padding: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Launch with the SM8x or SM120 execution plan of the input's device."""
    plan = default_execution_plan(
        input,
        weight_qdata,
        stride,
        policy=policy,
        group_norm=True,
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
    )
    return convolution.group_norm_silu_conv3d(
        input,
        norm_weight,
        norm_bias,
        norm_groups,
        norm_epsilon,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride,
        execution_plan=plan,
        accelerator_backend="cuda",
        symmetric_spatial_padding=symmetric_spatial_padding,
        right_spatial_padding=right_spatial_padding,
        residual=residual,
    )
