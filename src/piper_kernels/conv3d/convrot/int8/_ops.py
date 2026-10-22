"""Stable ConvRot INT8 convolution schemas with runtime backend selection.

Storage dispatch validates eager calls; fake implementations share its metadata
contracts during compilation. Runtime bodies also serve compiler rewrites.
"""

from __future__ import annotations

from typing import cast

import torch

from . import _backend, reference
from ._padding import Padding
from ._validation import (
    _validate_common,
    _validate_norm,
)


@torch.library.custom_op(
    "piper_kernels::convrot_int8_conv3d",
    mutates_args=(),
)
def _conv3d_op(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: list[int],
    padding: list[int],
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    backend = _backend.select_backend(input) or reference
    return backend.conv3d(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        cast(tuple[int, int, int], tuple(stride)),
        padding=cast(Padding, tuple(padding)),
        spatial_reflection=spatial_reflection,
        residual=residual,
    )


@_conv3d_op.register_fake
def _conv3d_fake(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    input_scale: torch.Tensor,
    stride: list[int],
    padding: list[int],
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    shape = _validate_common(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride,
        cast(Padding, tuple(padding)),
        spatial_reflection,
        residual,
    )
    return input.new_empty(shape, dtype=torch.float16)


@torch.library.custom_op(
    "piper_kernels::convrot_int8_group_norm_silu_conv3d",
    mutates_args=(),
)
def _group_norm_silu_conv3d_op(  # noqa: PLR0913, PLR0917
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
    stride: list[int],
    padding: list[int],
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    backend = _backend.select_backend(input) or reference
    return backend.group_norm_silu_conv3d(
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
        cast(tuple[int, int, int], tuple(stride)),
        padding=cast(Padding, tuple(padding)),
        spatial_reflection=spatial_reflection,
        residual=residual,
    )


@_group_norm_silu_conv3d_op.register_fake
def _group_norm_silu_conv3d_fake(  # noqa: PLR0913, PLR0917
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
    stride: list[int],
    padding: list[int],
    spatial_reflection: bool,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    shape = _validate_common(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        input_scale,
        stride,
        cast(Padding, tuple(padding)),
        spatial_reflection,
        residual,
    )
    _validate_norm(input, norm_weight, norm_bias, norm_groups, norm_epsilon)
    return input.new_empty(shape, dtype=torch.float16)
