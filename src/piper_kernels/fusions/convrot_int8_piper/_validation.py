"""Metadata validation shared by dense Piper's Q and K projection producers."""

import torch

from piper_kernels.fusions.convrot_int8_sage_qk._validation import validate_qk_projection_inputs
from piper_kernels.fusions.projected_qk._validation import resolve_head_dim


def validate_qk_inputs(  # noqa: PLR0913
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    norm_epsilon: float,
    name: str,
    bias: torch.Tensor | None,
    head_dim: int | None,
) -> tuple[int, int, int, int]:
    """Add dense projection dimensions and inference checks to the shared Q/K contract."""
    batch, sequence_length, heads = validate_qk_projection_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        name=name,
        head_dim=head_dim,
        bias=bias,
    )
    if heads < 1 or input_qdata.shape[2] < 1:
        raise ValueError(f"{name} projection requires nonempty head and input dimensions")
    if torch.is_grad_enabled() and any(
        operand is not None and operand.requires_grad
        for operand in (input_scale, weight_scale, norm_weight, cos, sin, bias)
    ):
        raise RuntimeError(f"{name} projection is inference-only")
    return batch, sequence_length, heads, resolve_head_dim(norm_weight, head_dim)
