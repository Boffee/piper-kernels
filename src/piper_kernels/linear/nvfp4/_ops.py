"""Semantic NVFP4 linear operators with reusable activation preparation."""

from __future__ import annotations

import torch

from piper_kernels import _input_activations as input_activations
from piper_kernels.weights.convrot._rotation import validate_group_size

from . import _validation, reference
from ._storage import prepared_input_metadata


def _prepare_compiled(
    input: torch.Tensor,  # noqa: A002
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    activation_fn: str | None = None,
    high_first: bool = False,
    *,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the activation and pack NVFP4 input in the requested rotation basis."""
    if input.numel() == 0:
        if group_size is not None:
            validate_group_size(group_size)
        return reference.prepare_input(
            input,
            activation_per_tensor_scale,
            dynamic_activation_scale,
            activation_fn,
            high_first,
            group_size=group_size or 0,
        )
    if group_size is None:
        from . import triton as nvfp4_triton  # noqa: PLC0415

        input_activations.validate_input_activation(activation_fn)
        if dynamic_activation_scale:
            return nvfp4_triton.prepare_dynamic(input, activation_fn, high_first=high_first)
        return nvfp4_triton.prepare_static(
            input, activation_per_tensor_scale, activation_fn, high_first
        )

    from piper_kernels.linear.convrot.nvfp4 import triton as convrot_triton  # noqa: PLC0415

    if dynamic_activation_scale:
        return convrot_triton.prepare_dynamic(
            input, group_size, activation_fn, high_first=high_first
        )
    if activation_per_tensor_scale is None:
        activation_per_tensor_scale = input.new_ones((), dtype=torch.float32)
    return convrot_triton.prepare_static(
        input, activation_per_tensor_scale, group_size, activation_fn, high_first
    )


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
    output = torch.empty(
        (input_qdata.shape[0], weight_qdata.shape[0]),
        device=input_qdata.device,
        dtype=logical_dtype,
    )
    if input_qdata.shape[0] == 0:
        return output
    from . import _projection  # noqa: PLC0415

    return _projection.matmul_prepared_chunk_affine_out(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        0,
        input_qdata.shape[0],
        output,
    )


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
    weight_only: bool = False,
) -> torch.Tensor:
    """Apply a swizzled NVFP4 weight (W4A16 when ``weight_only`` is true, otherwise W4A4)."""
    if input.ndim == 0 or input.numel() == 0 or weight_only:
        shape = _validation.validate_semantic_linear(
            input,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            activation_per_tensor_scale,
            bias,
            dynamic_activation_scale,
            "NVFP4 linear",
            allow_empty=True,
            weight_only=weight_only,
        )
        if shape.rows == 0:
            return input.new_empty((*input.shape[:-1], shape.output_features))
        from . import _w4a16_triton  # noqa: PLC0415

        return _w4a16_triton.linear_w4a16(
            input,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            bias,
            high_first=high_first,
        )
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
    activation_per_tensor_scale: torch.Tensor | None,
    _bias: torch.Tensor | None,
    dynamic_activation_scale: bool,
    _high_first: bool = False,
    weight_only: bool = False,
) -> torch.Tensor:
    if input.ndim == 0:
        raise ValueError("NVFP4 linear input must be a non-scalar strided floating tensor")
    if weight_only and (activation_per_tensor_scale is not None or dynamic_activation_scale):
        raise ValueError("NVFP4 linear weight-only execution does not accept activation scales")
    return input.new_empty((*input.shape[:-1], weight_qdata.shape[0]))


prepare_input = torch.library.custom_op(
    "piper_kernels::nvfp4_prepare_input", _prepare_compiled, mutates_args=()
)


@prepare_input.register_fake
def _prepare_input_fake(
    input: torch.Tensor,  # noqa: A002
    activation_per_tensor_scale: torch.Tensor | None,
    _dynamic_activation_scale: bool,
    activation_fn: str | None = None,
    _high_first: bool = False,
    *,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size is not None:
        validate_group_size(group_size)
        if input.ndim == 0:
            raise ValueError("ConvRot NVFP4 input must be non-scalar")
    return prepared_input_metadata(
        input, activation_per_tensor_scale, activation_fn, group_size=group_size
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
