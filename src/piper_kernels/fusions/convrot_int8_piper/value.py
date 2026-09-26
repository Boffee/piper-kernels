"""ConvRot INT8 projection directly into dense per-token V storage."""

import torch

from piper_kernels.fusions.convrot_int8_projection._validation import validate_projection_inputs
from piper_kernels.fusions.projected_qk._validation import resolve_head_dim

from . import _backend


def _validate_inputs(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    head_dim: int,
    is_causal: bool,
) -> tuple[int, int, int, int]:
    head_dim = resolve_head_dim(None, head_dim)
    if not isinstance(is_causal, bool):
        raise TypeError("dense Piper V is_causal must be boolean")
    batch, sequence, heads = validate_projection_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        head_dim=head_dim,
        name="dense Piper V",
    )
    if sequence < 1 or input_qdata.shape[2] < 1 or heads < 1:
        raise ValueError("dense Piper V input dimensions must be positive")
    if torch.is_grad_enabled() and any(
        operand is not None and operand.requires_grad
        for operand in (input_scale, weight_scale, bias)
    ):
        raise RuntimeError("dense Piper V projection is inference-only")
    return batch, sequence, heads, head_dim


def _outputs(
    input_qdata: torch.Tensor, shape: tuple[int, int, int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence, heads, head_dim = shape
    storage = (sequence + 63) // 64 * 64
    return (
        input_qdata.new_empty((batch, heads, head_dim, storage)),
        input_qdata.new_empty((batch, heads, storage), dtype=torch.float32),
        input_qdata.new_empty((batch, heads, storage), dtype=torch.float32),
        input_qdata.new_empty((batch, heads, head_dim), dtype=torch.float32),
    )


@torch.library.custom_op("piper_kernels::convrot_int8_piper_project_value", mutates_args=())
def _project_value_op(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    head_dim: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return native V codes, per-token multipliers/logs, and the non-causal mean.

    Storage is K64 padded. On RDNA4, codes use the packed WMMA tile order in
    the same allocation shape. NVIDIA log scales retain FP16 rounding. Causal
    means are zero; no sequence-wide V reduction runs for causal attention.
    """
    shape = _validate_inputs(
        input_qdata, input_scale, weight_qdata, weight_scale, bias, head_dim, is_causal
    )
    if shape[0] == 0:
        return _outputs(input_qdata, shape)
    backend = _backend.select_projection_backend(input_qdata, head_dim=head_dim)
    if backend is None:
        raise ValueError(f"dense Piper V projection is unavailable on {input_qdata.device}")
    output = _outputs(input_qdata, shape)
    backend.project_value(
        input_qdata, input_scale, weight_qdata, weight_scale, bias, is_causal=is_causal, out=output
    )
    return output


@_project_value_op.register_fake
def _project_value_fake(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    head_dim: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = _validate_inputs(
        input_qdata, input_scale, weight_qdata, weight_scale, bias, head_dim, is_causal
    )
    return _outputs(input_qdata, shape)
