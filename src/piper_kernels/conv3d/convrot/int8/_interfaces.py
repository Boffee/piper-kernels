"""Execution contracts for ConvRot INT8 convolution backends and shared launches."""

from typing import Protocol

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._plan import ConvolutionExecutionPlan, ConvolutionSchedule


class ConvolutionPolicy(Protocol):
    """Select a complete execution plan using only target and operand metadata."""

    def select_execution_plan(
        self,
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
    ) -> ConvolutionExecutionPlan: ...


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
