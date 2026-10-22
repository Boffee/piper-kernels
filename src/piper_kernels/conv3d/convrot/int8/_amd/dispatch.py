"""RDNA4 ConvRot INT8 convolution dispatch over shared Triton launchers."""

import torch

from .. import triton as convolution
from .._dispatch import default_execution_plan
from .._padding import Padding
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
    padding: Padding,
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Launch with the RDNA4 execution plan of the input's device."""
    plan = default_execution_plan(
        input,
        weight_qdata,
        stride,
        policy=policy,
        group_norm=False,
        padding=padding,
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
        accelerator_backend="hip",
        padding=padding,
        spatial_reflection=spatial_reflection,
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
    padding: Padding,
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Launch with the RDNA4 execution plan of the input's device."""
    plan = default_execution_plan(
        input,
        weight_qdata,
        stride,
        policy=policy,
        group_norm=True,
        padding=padding,
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
        accelerator_backend="hip",
        padding=padding,
        spatial_reflection=spatial_reflection,
        residual=residual,
    )
