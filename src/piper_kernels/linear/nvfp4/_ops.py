"""Semantic NVFP4 W4A4 operators with reusable activation preparation."""

from __future__ import annotations

from typing import cast

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.linear import _input_activations as input_activations

from . import _layout
from . import triton as nvfp4_triton
from ._typing import NVFP4Storage


def _prepare_static_tensors(
    input: torch.Tensor,  # noqa: A002
    per_tensor_scale: torch.Tensor,
    activation_fn: str | None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activated = input_activations.apply_input_activation(input, activation_fn)
    prepared = cast(
        NVFP4Storage,
        TorchAONVFP4Tensor.to_nvfp4(
            activated.reshape(-1, activated.shape[-1]),
            block_size=_layout.BLOCK_SIZE,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        ),
    )
    qdata = prepared.qdata
    if high_first:
        qdata = _layout.swap_packed_pairs(qdata)
    return qdata, prepared.scale, per_tensor_scale.clone()


def _prepare_dynamic_tensors(
    input: torch.Tensor,  # noqa: A002
    activation_fn: str | None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activated = input_activations.apply_input_activation(input, activation_fn)
    flattened = activated.reshape(-1, activated.shape[-1])
    per_tensor_scale = per_tensor_amax_to_scale(flattened.abs().amax())
    prepared = cast(
        NVFP4Storage,
        TorchAONVFP4Tensor.to_nvfp4(
            flattened,
            block_size=_layout.BLOCK_SIZE,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        ),
    )
    qdata = prepared.qdata
    if high_first:
        qdata = _layout.swap_packed_pairs(qdata)
    return qdata, prepared.scale, per_tensor_scale


def _prepare_dynamic_swiglu_tensors(
    input: torch.Tensor,  # noqa: A002
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _prepare_dynamic_tensors(input, "swiglu", high_first)


def _prepare_dynamic_gelu_tanh_tensors(
    input: torch.Tensor,  # noqa: A002
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _prepare_dynamic_tensors(input, "gelu_tanh", high_first)


_compiled_prepare_static = torch.compile(_prepare_static_tensors, fullgraph=True)
_compiled_prepare_dynamic_swiglu = torch.compile(_prepare_dynamic_swiglu_tensors, fullgraph=True)
_compiled_prepare_dynamic_gelu_tanh = torch.compile(
    _prepare_dynamic_gelu_tanh_tensors,
    fullgraph=True,
)


def _compiled_prepare_dynamic(
    input: torch.Tensor,  # noqa: A002
    activation_fn: str | None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare dynamic activations without generalizing across unrelated layouts."""
    if activation_fn is None:
        per_tensor_scale = nvfp4_triton.dynamic_scale(input)
        qdata, scale = nvfp4_triton._prepare_static_storage(
            input,
            per_tensor_scale,
            swiglu=False,
            source_global_scale=None,
            source_bias=None,
            high_first=high_first,
        )
        return qdata, scale, per_tensor_scale
    if activation_fn == "swiglu":
        return _compiled_prepare_dynamic_swiglu(input, high_first)
    assert activation_fn == "gelu_tanh"
    return _compiled_prepare_dynamic_gelu_tanh(input, high_first)


def _scale_result_to(
    result: torch.Tensor,
    global_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return (result.float() * global_scale.float()).to(output_dtype)


def _scale_result_and_add_bias_to(
    result: torch.Tensor,
    global_scale: torch.Tensor,
    bias: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return (result.float() * global_scale.float() + bias.float()).to(output_dtype)


def _scale_result(
    result: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    return _scale_result_to(result, global_scale, result.dtype)


def _scale_result_and_add_bias(
    result: torch.Tensor,
    global_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _scale_result_and_add_bias_to(result, global_scale, bias, result.dtype)


def _scale_fp16_result(
    result: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    return _scale_result_to(result, global_scale, torch.float16)


def _scale_fp16_result_and_add_bias(
    result: torch.Tensor,
    global_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _scale_result_and_add_bias_to(result, global_scale, bias, torch.float16)


_compiled_scale_result = torch.compile(_scale_result, fullgraph=True)
_compiled_scale_result_and_add_bias = torch.compile(
    _scale_result_and_add_bias,
    fullgraph=True,
)
_compiled_scale_fp16_result = torch.compile(_scale_fp16_result, fullgraph=True)
_compiled_scale_fp16_result_and_add_bias = torch.compile(
    _scale_fp16_result_and_add_bias,
    fullgraph=True,
)


def _prepare_compiled(
    input: torch.Tensor,  # noqa: A002
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_activations.validate_input_activation(activation_fn)
    if dynamic_activation_scale:
        return _compiled_prepare_dynamic(input, activation_fn, high_first)
    if activation_per_tensor_scale is None:
        raise ValueError("static NVFP4 activation preparation requires a per-tensor scale")
    if activation_fn in (None, "swiglu"):
        return nvfp4_triton.prepare_static(
            input,
            activation_per_tensor_scale,
            swiglu=activation_fn == "swiglu",
            high_first=high_first,
        )
    return _compiled_prepare_static(
        input,
        activation_per_tensor_scale,
        activation_fn,
        high_first,
    )


def _apply_mixed_bias_epilogue(
    result: torch.Tensor,
    global_scale: torch.Tensor,
    bias: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply a mixed-dtype affine without specializing a shared compiled graph."""
    output = (
        result if result.dtype is output_dtype else torch.empty_like(result, dtype=output_dtype)
    )
    nvfp4_triton.apply_projection_epilogue(result, global_scale, bias, output)
    return output


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
    accumulator_dtype = torch.bfloat16 if logical_dtype is torch.float16 else logical_dtype
    result = torch._scaled_mm(
        input_qdata.view(torch.float4_e2m1fn_x2),
        weight_qdata.t().view(torch.float4_e2m1fn_x2),
        input_scale.view(torch.float8_e4m3fn),
        weight_scale.view(torch.float8_e4m3fn),
        out_dtype=accumulator_dtype,
    )
    global_scale = input_per_tensor_scale
    if weight_per_tensor_scale is not None:
        global_scale = global_scale * weight_per_tensor_scale
    if bias is not None and bias.dtype is not logical_dtype:
        return _apply_mixed_bias_epilogue(result, global_scale, bias, logical_dtype)
    if logical_dtype is torch.float16:
        if bias is None:
            return _compiled_scale_fp16_result(result, global_scale)
        return _compiled_scale_fp16_result_and_add_bias(result, global_scale, bias)
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
    high_first: bool = False,
) -> torch.Tensor:
    """Quantize an activation and apply a standard swizzled NVFP4 W4A4 weight."""
    input_qdata, input_scale, input_per_tensor_scale = _prepare_compiled(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
        high_first=high_first,
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
    _high_first: bool = False,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], weight_qdata.shape[0]))


@torch.library.custom_op("piper_kernels::nvfp4_prepare_input", mutates_args=())
def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an optional portable activation and prepare it for NVFP4 projections."""
    return _prepare_compiled(
        input,
        activation_per_tensor_scale,
        dynamic_activation_scale,
        activation_fn,
        high_first,
    )


@prepare_input.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _prepare_input_fake(
    input: torch.Tensor,  # noqa: A002
    activation_per_tensor_scale: torch.Tensor | None,
    _dynamic_activation_scale: bool,
    activation_fn: str | None = None,
    _high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = input.numel() // input.shape[-1]
    features = input.shape[-1] // input_activations.input_activation_width(activation_fn)
    qdata_shape = _layout.qdata_shape(rows, features)
    scale_shape = _layout.scale_shape(rows, features)
    per_tensor_scale = (
        activation_per_tensor_scale.new_empty(())
        if activation_per_tensor_scale is not None
        else input.new_empty((), dtype=torch.float32)
    )
    return (
        input.new_empty(qdata_shape, dtype=torch.uint8),
        input.new_empty(scale_shape, dtype=torch.float8_e4m3fn),
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
