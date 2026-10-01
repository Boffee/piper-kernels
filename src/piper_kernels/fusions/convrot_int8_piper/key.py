"""ConvRot INT8 K projection with a global post-RMSNorm/RoPE mean."""

import torch

from piper_kernels.attention.piper_attention._validation import (
    key_scale_length,
    validate_qk_quantization,
)

from . import _backend
from ._interfaces import KeyOutput
from ._validation import validate_qk_inputs


def _new_outputs(
    input_qdata: torch.Tensor, shape: tuple[int, int, int, int], qk_quantization: str
) -> KeyOutput:
    batch, sequence, heads, head_dim = shape
    storage = (sequence + 63) // 64 * 64
    scales = key_scale_length(storage, validate_qk_quantization(qk_quantization))
    return (
        input_qdata.new_empty((batch, heads, storage, head_dim)),
        input_qdata.new_empty((batch, heads, scales), dtype=torch.float32),
    )


@torch.library.custom_op("piper_kernels::convrot_int8_piper_project_key", mutates_args=())
def _project_key_op(  # noqa: PLR0913
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
    qk_quantization: str = "per_warp",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce INT8 K after centering the complete transformed sequence.

    ``qk_quantization`` selects K64 scales (``per_warp``) or per-thread groups
    stored per key (``per_thread``), matching the target's quantized attention.
    """
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
        return _new_outputs(input_qdata, shape, qk_quantization)
    backend = _backend.require_projection_backend(input_qdata, head_dim=shape[-1])
    output = _new_outputs(input_qdata, shape, qk_quantization)
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
def _project_key_op_fake(  # noqa: PLR0913
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
    qk_quantization: str = "per_warp",
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
    return _new_outputs(input_qdata, shape, qk_quantization)
