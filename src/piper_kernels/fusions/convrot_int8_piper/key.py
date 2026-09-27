"""ConvRot INT8 K projection with a global post-RMSNorm/RoPE mean."""

import torch

from . import _backend
from ._interfaces import KeyOutput
from ._validation import validate_qk_inputs


def _new_outputs(input_qdata: torch.Tensor, shape: tuple[int, int, int, int]) -> KeyOutput:
    batch, sequence, heads, head_dim = shape
    storage = (sequence + 63) // 64 * 64
    return (
        input_qdata.new_empty((batch, heads, storage, head_dim)),
        input_qdata.new_empty((batch, heads, storage // 64), dtype=torch.float32),
    )


@torch.library.custom_op("piper_kernels::convrot_int8_piper_project_key", mutates_args=())
def _project_key_op(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce K64 codes after centering the complete transformed sequence."""
    shape = validate_qk_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        name="dense Piper K",
        bias=bias,
        head_dim=head_dim,
    )
    if shape[0] == 0:
        return _new_outputs(input_qdata, shape)
    backend = _backend.require_projection_backend(input_qdata, head_dim=shape[-1])
    output = _new_outputs(input_qdata, shape)
    backend.project_key(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        bias,
        out=output,
    )

    return output


@_project_key_op.register_fake
def _project_key_op_fake(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = validate_qk_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        name="dense Piper K",
        bias=bias,
        head_dim=head_dim,
    )
    return _new_outputs(input_qdata, shape)
