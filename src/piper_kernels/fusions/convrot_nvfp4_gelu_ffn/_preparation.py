"""FP32 GELU preparation inside a bounded ConvRot NVFP4 FFN chunk."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from piper_kernels.linear.convrot.nvfp4 import triton as convrot_backend
from piper_kernels.weights.convrot._rotation import validate_group_size


@dataclass(frozen=True, slots=True)
class ConvRotGELUPreparation:
    """Apply GELU, rotate, and prepare ConvRot NVFP4 down inputs."""

    source_projection_count: ClassVar[int] = 1
    group_size: int
    high_first: bool

    def __post_init__(self) -> None:
        validate_group_size(self.group_size)

    def prepare(
        self,
        projections: torch.Tensor,
        activation_per_tensor_scale: torch.Tensor | None,
        dynamic_activation_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        validated_input = convrot_backend._validate_input(
            projections,
            self.group_size,
            "gelu_tanh",
        )
        per_tensor_scale = (
            convrot_backend._prepare_dynamic_scale(
                validated_input,
                self.group_size,
                activation_fn="gelu_tanh",
            )
            if dynamic_activation_scale
            else activation_per_tensor_scale
        )
        assert per_tensor_scale is not None
        qdata, scale = convrot_backend._prepare_static_storage(
            validated_input,
            per_tensor_scale,
            self.group_size,
            activation_fn="gelu_tanh",
            high_first=self.high_first,
        )
        return qdata, scale, per_tensor_scale


__all__ = ["ConvRotGELUPreparation"]
