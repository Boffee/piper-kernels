"""Canonical NVFP4 operands for the public SwiGLU custom operations."""

from __future__ import annotations

import torch

from piper_kernels.fusions.nvfp4_ffn import _core


def linear_operands(  # noqa: PLR0913, PLR0917 - explicit custom-op projection operands
    gate_weight_qdata: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_weight_per_tensor_scale: torch.Tensor | None,
    gate_activation_per_tensor_scale: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    gate_dynamic_activation_scale: bool,
    gate_high_first: bool,
    value_weight_qdata: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_weight_per_tensor_scale: torch.Tensor | None,
    value_activation_per_tensor_scale: torch.Tensor | None,
    value_bias: torch.Tensor | None,
    value_dynamic_activation_scale: bool,
    value_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_high_first: bool,
) -> tuple[_core.LinearOperands, _core.LinearOperands, _core.LinearOperands]:
    """Return gate, value, and down operands in the public schema's order."""
    return (
        _core.LinearOperands(
            gate_weight_qdata,
            gate_weight_scale,
            gate_weight_per_tensor_scale,
            gate_activation_per_tensor_scale,
            gate_bias,
            gate_dynamic_activation_scale,
            gate_high_first,
        ),
        _core.LinearOperands(
            value_weight_qdata,
            value_weight_scale,
            value_weight_per_tensor_scale,
            value_activation_per_tensor_scale,
            value_bias,
            value_dynamic_activation_scale,
            value_high_first,
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
