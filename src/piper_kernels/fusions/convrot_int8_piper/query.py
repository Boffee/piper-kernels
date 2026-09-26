"""One-pass ConvRot INT8 projection and dense Piper Q32 preparation."""

import math

import torch

from piper_kernels.fusions.convrot_int8_sage_qk._validation import validate_qk_projection_inputs
from piper_kernels.fusions.projected_qk._validation import resolve_head_dim

from . import _backend


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
    batch, sequence_length, heads = validate_qk_projection_inputs(
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
    if heads < 1 or input_qdata.shape[2] < 1:
        raise ValueError("dense Piper Q projection requires nonempty head and input dimensions")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("dense Piper Q projection softmax scale must be finite and positive")
    if torch.is_grad_enabled() and any(
        operand is not None and operand.requires_grad
        for operand in (input_scale, weight_scale, norm_weight, cos, sin, bias)
    ):
        raise RuntimeError("dense Piper Q projection is inference-only")
    return batch, sequence_length, heads, resolve_head_dim(norm_weight, head_dim)


def _new_outputs(
    input_qdata: torch.Tensor,
    shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    backend = _backend.select_projection_backend(input_qdata, head_dim=shape[-1])
    if backend is None:
        raise ValueError(f"dense Piper Q projection is unavailable on {input_qdata.device}")
    output = _new_outputs(input_qdata, shape)
    backend(
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
