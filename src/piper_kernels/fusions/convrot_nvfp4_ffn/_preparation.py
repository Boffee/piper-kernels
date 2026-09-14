"""Activation-independent ConvRot NVFP4 source preparation for fused FFNs."""

from dataclasses import dataclass

import torch

from piper_kernels.linear.convrot.nvfp4 import triton as convrot_backend
from piper_kernels.weights.convrot._rotation import validate_group_size


@dataclass(frozen=True, slots=True)
class ConvRotSourcePreparation:
    """Prepare one input shared by ConvRot NVFP4 source projections."""

    group_size: int
    high_first: bool

    def __post_init__(self) -> None:
        validate_group_size(self.group_size)

    def dynamic_scale(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
    ) -> torch.Tensor:
        return convrot_backend.dynamic_scale(input, self.group_size)

    def prepare(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return convrot_backend.prepare_static_out(
            input,
            per_tensor_scale,
            self.group_size,
            out,
            high_first=self.high_first,
        )


__all__ = ["ConvRotSourcePreparation"]
