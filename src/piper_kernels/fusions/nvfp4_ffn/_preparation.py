"""Activation-independent standard NVFP4 source preparation for fused FFNs."""

from dataclasses import dataclass

import torch

from piper_kernels._triton import nvfp4 as nvfp4_primitives
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend


@dataclass(frozen=True, slots=True)
class StandardSourcePreparation:
    """Prepare ordinary NVFP4 inputs shared by one or more source projections."""

    high_first: bool

    @property
    def group_size(self) -> None:
        """Standard NVFP4 does not rotate features."""
        return None

    def dynamic_scale(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
    ) -> torch.Tensor:
        return nvfp4_primitives.dynamic_scale(input)

    def prepare(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return nvfp4_backend.prepare_static_out(
            input,
            per_tensor_scale,
            out,
            high_first=self.high_first,
        )


__all__ = ["StandardSourcePreparation"]
