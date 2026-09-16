"""FP32 GELU preparation for a bounded standard NVFP4 FFN chunk."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.linear.nvfp4 import triton as nvfp4_backend


def _dynamic_gelu_scale(projections: torch.Tensor) -> torch.Tensor:
    """Calculate the dynamic scale in FP32 without materializing GELU."""
    return per_tensor_amax_to_scale(F.gelu(projections.float(), approximate="tanh").abs().amax())


dynamic_gelu_scale = torch.compile(_dynamic_gelu_scale, fullgraph=True)


@dataclass(frozen=True, slots=True)
class StandardGELUPreparation:
    """Apply GELU and prepare ordinary NVFP4 down-projection inputs."""

    source_projection_count: ClassVar[int] = 1
    high_first: bool

    @property
    def group_size(self) -> None:
        """Standard NVFP4 does not rotate features."""
        return None

    def prepare(
        self,
        projections: torch.Tensor,
        activation_per_tensor_scale: torch.Tensor | None,
        dynamic_activation_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_tensor_scale = (
            dynamic_gelu_scale(projections)
            if dynamic_activation_scale
            else activation_per_tensor_scale
        )
        assert per_tensor_scale is not None
        qdata, scale = nvfp4_backend._prepare_static_storage(
            projections,
            per_tensor_scale,
            activation_fn="gelu_tanh",
            high_first=self.high_first,
        )
        return qdata, scale, per_tensor_scale


__all__ = ["StandardGELUPreparation", "dynamic_gelu_scale"]
