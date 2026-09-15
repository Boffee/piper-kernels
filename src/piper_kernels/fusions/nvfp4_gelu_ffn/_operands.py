"""Canonical NVFP4 operands for the public GELU custom operations."""

from __future__ import annotations

import torch

from piper_kernels.fusions.nvfp4_ffn import _core


def linear_operands(  # noqa: PLR0913, PLR0917 - explicit custom-op projection operands
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_high_first: bool,
) -> tuple[_core.LinearOperands, _core.LinearOperands]:
    """Return up and down operands in the public schema's order."""
    return (
        _core.LinearOperands(
            up_weight_qdata,
            up_weight_scale,
            up_weight_per_tensor_scale,
            up_activation_per_tensor_scale,
            up_bias,
            up_dynamic_activation_scale,
            up_high_first,
        ),
        _core.LinearOperands(
            down_weight_qdata,
            down_weight_scale,
            down_weight_per_tensor_scale,
            down_activation_per_tensor_scale,
            down_bias,
            down_dynamic_activation_scale,
            down_high_first,
        ),
    )


__all__ = ["linear_operands"]
