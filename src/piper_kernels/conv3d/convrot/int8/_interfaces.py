"""Execution contracts for ConvRot INT8 convolution backends and shared launches."""

from typing import Protocol

import torch

from ._plan import ConvolutionPlan, PreparationPlan


class ConvolutionPolicy(Protocol):
    def convolution_plan(self, channels: int, outputs: int, rows: int) -> ConvolutionPlan: ...

    def preparation_plan(
        self, channels: int, rows: int, *, group_norm: bool
    ) -> PreparationPlan: ...

    def use_weight_descriptor(
        self, channels: int, outputs: int, height: int, block_n: int, *, aligned: bool
    ) -> bool: ...


class ConvolutionBackend(Protocol):
    """Callers supply operands; backends own preparation and convolution policy."""

    def conv3d(
        self,
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
    ) -> torch.Tensor: ...

    def group_norm_silu_conv3d(  # noqa: PLR0913, PLR0917
        self,
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
    ) -> torch.Tensor: ...
