"""Semantic NVFP4 W4A4 operators with reusable activation preparation."""

from __future__ import annotations

from typing import cast

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.linear import _input_activations as input_activations

from . import triton as nvfp4_triton
from ._typing import NVFP4Storage

_BLOCK_SIZE = 16


def _prepare_static_tensors(
    input: torch.Tensor,  # noqa: A002
    per_tensor_scale: torch.Tensor,
    activation_fn: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activated = input_activations.apply_input_activation(input, activation_fn)
    prepared = cast(
        NVFP4Storage,
        TorchAONVFP4Tensor.to_nvfp4(
            activated.reshape(-1, activated.shape[-1]),
            block_size=_BLOCK_SIZE,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        ),
    )
    return prepared.qdata, prepared.scale, per_tensor_scale.clone()


def _prepare_dynamic_tensors(
    input: torch.Tensor,  # noqa: A002
    activation_fn: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activated = input_activations.apply_input_activation(input, activation_fn)
    flattened = activated.reshape(-1, activated.shape[-1])
    per_tensor_scale = per_tensor_amax_to_scale(flattened.abs().amax())
    prepared = cast(
        NVFP4Storage,
        TorchAONVFP4Tensor.to_nvfp4(
            flattened,
            block_size=_BLOCK_SIZE,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        ),
    )
    return prepared.qdata, prepared.scale, per_tensor_scale


_compiled_prepare_static = torch.compile(_prepare_static_tensors, fullgraph=True)
_compiled_prepare_dynamic = torch.compile(_prepare_dynamic_tensors, fullgraph=True)


def _scale_result(
    result: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    return (result.float() * global_scale.float()).to(result.dtype)


def _scale_result_and_add_bias(
    result: torch.Tensor,
    global_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return (result.float() * global_scale.float() + bias.float()).to(result.dtype)


_compiled_scale_result = torch.compile(_scale_result, fullgraph=True)
_compiled_scale_result_and_add_bias = torch.compile(
    _scale_result_and_add_bias,
    fullgraph=True,
)


def _prepare_compiled(
    input: torch.Tensor,  # noqa: A002
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_activations.validate_input_activation(activation_fn)
    if dynamic_activation_scale:
        return _compiled_prepare_dynamic(input, activation_fn)
    if activation_per_tensor_scale is None:
        raise ValueError("static NVFP4 activation preparation requires a per-tensor scale")
    if activation_fn in (None, "swiglu"):
        return nvfp4_triton.prepare_static(
            input,
            activation_per_tensor_scale,
            swiglu=activation_fn == "swiglu",
        )
    return _compiled_prepare_static(input, activation_per_tensor_scale, activation_fn)


def _execute_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    result = torch._scaled_mm(
        input_qdata.view(torch.float4_e2m1fn_x2),
        weight_qdata.t().view(torch.float4_e2m1fn_x2),
        input_scale.view(torch.float8_e4m3fn),
        weight_scale.view(torch.float8_e4m3fn),
        out_dtype=logical_dtype,
    )
    global_scale = input_per_tensor_scale
    if weight_per_tensor_scale is not None:
        global_scale = global_scale * weight_per_tensor_scale
    if bias is None:
        return _compiled_scale_result(result, global_scale)
    return _compiled_scale_result_and_add_bias(result, global_scale, bias)


@torch.library.custom_op("piper_kernels::nvfp4_linear", mutates_args=())
def linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_activation_scale: bool,
) -> torch.Tensor:
    """Quantize an activation and apply a standard swizzled NVFP4 W4A4 weight."""
    input_qdata, input_scale, input_per_tensor_scale = _prepare_compiled(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
    )
    result = _execute_prepared(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        input.dtype,
    )
    return result.reshape(*input.shape[:-1], weight_qdata.shape[0])


@linear.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _linear_fake(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _activation_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    _dynamic_activation_scale: bool,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], weight_qdata.shape[0]))


@torch.library.custom_op("piper_kernels::nvfp4_prepare_input", mutates_args=())
def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional portable activation and prepare it for NVFP4 projections."""
    return _prepare_compiled(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
        activation_fn,
    )


@prepare_input.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _prepare_input_fake(
    input: torch.Tensor,  # noqa: A002
    activation_per_tensor_scale: torch.Tensor | None,
    _dynamic_activation_scale: bool,
    activation_fn: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = input.numel() // input.shape[-1]
    features = input.shape[-1] // input_activations.input_activation_width(activation_fn)
    scale_rows = ((rows + 127) // 128) * 32
    scale_columns = ((features + 63) // 64) * 16
    per_tensor_scale = (
        activation_per_tensor_scale.new_empty(())
        if activation_per_tensor_scale is not None
        else input.new_empty((), dtype=torch.float32)
    )
    return (
        input.new_empty((rows, features // 2), dtype=torch.uint8),
        input.new_empty((scale_rows, scale_columns), dtype=torch.float8_e4m3fn),
        per_tensor_scale,
    )


@torch.library.custom_op("piper_kernels::nvfp4_linear_prepared", mutates_args=())
def linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply one NVFP4 weight to an activation prepared by the matching operator."""
    return _execute_prepared(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        logical_dtype,
    )


@linear_prepared.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _linear_prepared_fake(
    input_qdata: torch.Tensor,
    _input_scale: torch.Tensor,
    _input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
) -> torch.Tensor:
    return input_qdata.new_empty(
        (input_qdata.shape[0], weight_qdata.shape[0]),
        dtype=logical_dtype,
    )


__all__ = ["linear", "linear_prepared", "prepare_input"]
