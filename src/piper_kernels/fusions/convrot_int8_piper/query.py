"""One-pass ConvRot INT8 projection and dense Piper Q32 preparation."""

import math

import torch

from . import _backend
from ._interfaces import QueryOutput
from ._validation import validate_qk_inputs


def _validate_inputs(  # noqa: PLR0913, PLR0917
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    bias: torch.Tensor | None,
    head_dim: int | None,
) -> tuple[int, int, int, int]:
    shape = validate_qk_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon=norm_epsilon,
        name="dense Piper Q",
        head_dim=head_dim,
        bias=bias,
    )
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("dense Piper Q projection softmax scale must be finite and positive")
    return shape


def _new_outputs(
    input_qdata: torch.Tensor,
    shape: tuple[int, int, int, int],
) -> QueryOutput:
    batch, sequence_length, heads, head_dim = shape
    storage_length = (sequence_length + 63) // 64 * 64
    return (
        input_qdata.new_empty((batch, heads, storage_length, head_dim)),
        input_qdata.new_empty((batch, heads, storage_length // 32), dtype=torch.float32),
    )


def _launch_query_projection(  # noqa: PLR0913
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> QueryOutput:
    """Project the complete Q sequence into padded Q32 data and base-2 scales."""
    shape = _validate_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        bias,
        head_dim,
    )
    if shape[0] == 0:
        return _new_outputs(input_qdata, shape)
    backend = _backend.require_projection_backend(input_qdata, head_dim=shape[-1])
    output = _new_outputs(input_qdata, shape)
    backend.project_query(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        bias,
        out=output,
    )
    return output


@torch.library.custom_op("piper_kernels::convrot_int8_piper_project_query", mutates_args=())
def _project_query_op(  # noqa: PLR0913
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse projection, FP32 RMSNorm/RoPE, and signed-Hadamard Q32 quantization."""
    return _launch_query_projection(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        bias,
        head_dim=head_dim,
    )


@_project_query_op.register_fake
def _project_query_op_fake(  # noqa: PLR0913
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_epsilon: float,
    softmax_scale: float,
    bias: torch.Tensor | None = None,
    *,
    head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = _validate_inputs(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        norm_weight,
        cos,
        sin,
        norm_epsilon,
        softmax_scale,
        bias,
        head_dim,
    )
    return _new_outputs(input_qdata, shape)
