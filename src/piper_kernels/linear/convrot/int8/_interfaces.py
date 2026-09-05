"""Execution contracts shared by ConvRot INT8 custom ops and implementations."""

from collections.abc import Callable
from typing import Protocol

import torch

type PreparedInput = tuple[torch.Tensor, torch.Tensor]
type SecondProjection = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


class PreparationBackend(Protocol):
    """Input preparation can run without an optimized matrix implementation."""

    def prepare_input(
        self,
        input: torch.Tensor,  # noqa: A002
        group_size: int,
        activation_fn: str | None = None,
        *,
        out: PreparedInput | None = None,
    ) -> PreparedInput: ...


class LinearBackend(PreparationBackend, Protocol):
    """A compatible linear, preparation, and projection implementation.

    Prepared inputs contain contiguous INT8 data of shape ``[..., K]`` and FP32
    row scales of shape ``[...]``. Preparation depends on input width, dtype,
    group size, and activation, never on the consuming weight or output width.
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


type Add = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, float, int | None], None]
type Addmm = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, float, int | None],
    None,
]
type GGUFConvert = Callable[[torch.Tensor, int, int, torch.dtype, torch.Tensor, torch.Tensor], None]
type DequantizedMean = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]
