"""Stable ConvRot INT8 custom-op schemas with runtime implementation selection.

Fake implementations describe shared tensor contracts without selecting hardware.
Runtime bodies resolve implementations again, including after compiler rewrites.
"""

import torch

from piper_kernels.linear._input_activations import input_activation_width

from . import _backend


@torch.library.custom_op("piper_kernels::convrot_int8_linear", mutates_args=())
def linear(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    activation_fn: str | None = None,
) -> torch.Tensor:
    """Dispatch the stable linear operation to a supported implementation."""
    return _backend.require_linear_backend(input).linear(
        input, weight_qdata, weight_scale, bias, group_size, activation_fn
    )


@linear.register_fake
def _linear_fake(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    _group_size: int,
    _activation_fn: str | None = None,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], weight_qdata.shape[0]))


@torch.library.custom_op("piper_kernels::convrot_int8_prepare_input", mutates_args=())
def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch the stable prepare_input operation to a supported implementation."""
    return _backend.require_linear_backend(input).prepare_input(input, group_size, activation_fn)


@prepare_input.register_fake
def _prepare_input_fake(
    input: torch.Tensor,  # noqa: A002
    _group_size: int,
    activation_fn: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_width = input.shape[-1] // input_activation_width(activation_fn)
    return (
        input.new_empty((*input.shape[:-1], input_width), dtype=torch.int8),
        input.new_empty(input.shape[:-1], dtype=torch.float32),
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_dequantized_input_mean",
    mutates_args=(),
)
def dequantized_input_mean(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch the stable dequantized_input_mean operation to a supported implementation."""
    execute = _backend.select_dequantized_mean(input_qdata)
    if execute is None:
        raise ValueError(f"ConvRot INT8 optimized mean is unavailable on {input_qdata.device}")
    return execute(input_qdata, input_scale, block_lengths)


@dequantized_input_mean.register_fake
def _dequantized_input_mean_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _block_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    return input_qdata.new_empty(
        (input_qdata.shape[0], input_qdata.shape[2]),
        dtype=torch.float32,
    )


@torch.library.custom_op("piper_kernels::convrot_int8_linear_prepared", mutates_args=())
def linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    """Dispatch the stable linear_prepared operation to a supported implementation."""
    return _backend.require_linear_backend(input_qdata).linear_prepared(
        input_qdata, input_scale, weight_qdata, weight_scale, bias, logical_dtype
    )


@linear_prepared.register_fake
def _linear_prepared_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    return input_qdata.new_empty(
        (*input_qdata.shape[:-1], weight_qdata.shape[0]),
        dtype=logical_dtype,
    )


@torch.library.custom_op(
    "piper_kernels::convrot_int8_addmm_",
    mutates_args=("qdata", "scale"),
)
def addmm_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    group_size: int,
    beta: float,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Dispatch the stable addmm_ operation to a supported implementation."""
    execute = _backend.select_addmm(qdata)
    if execute is None:
        raise ValueError(f"ConvRot INT8 optimized addmm_ is unavailable on {qdata.device}")
    execute(qdata, scale, mat1, mat2, group_size, beta, alpha, rounding_seed)


@addmm_.register_fake
def _addmm_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _mat1: torch.Tensor,
    _mat2: torch.Tensor,
    _group_size: int,
    _beta: float,
    _alpha: float,
    _rounding_seed: int | None = None,
) -> None:
    return None


@torch.library.custom_op(
    "piper_kernels::convrot_int8_add_",
    mutates_args=("qdata", "scale"),
)
def add_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    update: torch.Tensor,
    group_size: int,
    alpha: float,
    rounding_seed: int | None = None,
) -> None:
    """Dispatch the stable add_ operation to a supported implementation."""
    execute = _backend.select_add(qdata)
    if execute is None:
        raise ValueError(f"ConvRot INT8 optimized add_ is unavailable on {qdata.device}")
    execute(qdata, scale, update, group_size, alpha, rounding_seed)


@add_.register_fake
def _add_fake(
    _qdata: torch.Tensor,
    _scale: torch.Tensor,
    _update: torch.Tensor,
    _group_size: int,
    _alpha: float,
    _rounding_seed: int | None = None,
) -> None:
    return None


__all__ = ["add_", "addmm_", "dequantized_input_mean", "linear", "linear_prepared", "prepare_input"]
