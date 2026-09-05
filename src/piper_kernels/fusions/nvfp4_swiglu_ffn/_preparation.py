"""Standard NVFP4 activation preparation shared by plain and mixed FFNs."""

from dataclasses import dataclass

import torch

from piper_kernels.linear.nvfp4 import triton as nvfp4_backend

from . import _core


@dataclass(frozen=True, slots=True)
class StandardPreparation:
    """Ordinary NVFP4 preparation used by the shared chunked runner."""

    source_high_first: bool
    down_high_first: bool

    def dynamic_source_scale(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
    ) -> torch.Tensor:
        return nvfp4_backend.dynamic_scale(input)

    def prepare_source(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return nvfp4_backend.prepare_static_out(
            input,
            per_tensor_scale,
            out,
            high_first=self.source_high_first,
        )

    def prepare_down(
        self,
        projections: torch.Tensor,
        activation_per_tensor_scale: torch.Tensor | None,
        dynamic_activation_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_tensor_scale = (
            _core.dynamic_swiglu_scale(projections)
            if dynamic_activation_scale
            else activation_per_tensor_scale
        )
        assert per_tensor_scale is not None
        return nvfp4_backend.prepare_static(
            projections,
            per_tensor_scale,
            swiglu=True,
            high_first=self.down_high_first,
        )
