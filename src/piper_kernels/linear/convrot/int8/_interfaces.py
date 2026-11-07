"""Execution contracts shared by ConvRot INT8 custom ops and implementations."""

from collections.abc import Callable
from typing import Protocol

import torch

from piper_kernels._residual import ResidualAdd

from ._plan import LinearExecutionPlan

type PreparedInput = tuple[torch.Tensor, torch.Tensor]
type SecondProjection = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


class PreparationBackend(Protocol):
    """Input preparation can run without an optimized matrix implementation."""

    def prepare_input(
        self,
        input: torch.Tensor,  # noqa: A002
        group_size: int,
        activation_fn: str | None = None,
        input_scale: torch.Tensor | None = None,
        *,
        out: PreparedInput | None = None,
    ) -> PreparedInput: ...


class LinearBackend(PreparationBackend, Protocol):
    """A compatible linear, preparation, and projection implementation.

    Backends own target inspection and launch-plan selection. Ordinary callers
    supply mathematical operands and optional buffers, not execution plans.

    Prepared inputs contain contiguous INT8 data of shape ``[..., K]`` and FP32
    row scales of shape ``[...]``. Preparation depends on input width, dtype,
    group size, activation, and optional static input scale, never on output width.
    Projections apply row scales after INT32 accumulation and preserve the
    existing bias and logical-dtype rounding contract.

    Caller-owned preparation buffers are contiguous and returned unchanged.
    Projection outputs may have a row stride but must be column-contiguous;
    paired projections write adjacent columns for two equal-shaped weights.
    Neither operation may write beyond the supplied output views.
    """

    def linear(
        self,
        input: torch.Tensor,  # noqa: A002
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
        activation_fn: str | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def linear_prepared(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        logical_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
        second_projection: SecondProjection | None = None,
    ) -> torch.Tensor: ...

    def linear_prepared_residual(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        residual_add: ResidualAdd,
        *,
        row_offset: int = 0,
        projection_scratch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project one chunk and update the corresponding residual output rows.

        The owning fusion validates ``residual_add`` once. ``row_offset`` locates
        this chunk in its flattened rows; the result has the input's leading shape
        and aliases ``residual_add.output``. The residual owns the output dtype.
        Both native and standalone epilogues preserve projection rounding before
        FP32 residual arithmetic and the final cast.

        Optional ``projection_scratch`` is a column-contiguous chunk buffer with
        the projection's shape, dtype, and device, disjoint from live operands.
        Only the standalone fallback uses it; the backend owns that decision.
        """
        ...

    def default_execution_plan(
        self,
        weight_qdata: torch.Tensor,
    ) -> LinearExecutionPlan: ...

    def execute_prepared_linear(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        logical_dtype: torch.dtype,
        execution_plan: LinearExecutionPlan,
        *,
        out: torch.Tensor | None = None,
        second_projection: SecondProjection | None = None,
    ) -> torch.Tensor: ...


type DequantizedMean = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]
