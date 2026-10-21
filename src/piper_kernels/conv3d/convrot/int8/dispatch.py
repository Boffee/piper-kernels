"""Validated storage-level entry points for ConvRot INT8 convolutions.

Custom operators remain portable because padding fusion can emit them on both
CPU and GPU. Runtime backend selection belongs to those stable operators; fake
implementations share the metadata checks used here.
"""

import torch

from . import _ops
from ._padding import Padding
from ._validation import _validate_common, _validate_inference, _validate_norm


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
    spatial_reflection: bool = False,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply raw ConvRot storage with explicit padding and FP16 output.

    This is the internal storage-level ABI. Callers normally use F.conv3d with
    a ConvRotInt8Tensor weight, or the causal convenience functions.
    """
    _validate_inference((input, weight_scale, input_scale, bias, residual))
    _validate_common(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        list(stride),
        padding,
        spatial_reflection,
        residual,
    )
    return _ops._conv3d_op(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        list(stride),
        list(padding),
        spatial_reflection,
        residual,
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
    spatial_reflection: bool = False,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply framewise GroupNorm/SiLU and convolution to validated raw storage."""
    _validate_inference((input, norm_weight, norm_bias, weight_scale, input_scale, bias, residual))
    _validate_common(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        list(stride),
        padding,
        spatial_reflection,
        residual,
    )
    _validate_norm(input, norm_weight, norm_bias, norm_groups, norm_epsilon)
    return _ops._group_norm_silu_conv3d_op(
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
        list(stride),
        list(padding),
        spatial_reflection,
        residual,
    )
