"""Storage validation shared by semantic and prepared NVFP4 consumers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear import _bias

from . import _layout

_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True, slots=True)
class LinearShape:
    """Logical dimensions of one validated NVFP4 linear."""

    rows: int | torch.SymInt
    input_features: int | torch.SymInt
    output_features: int | torch.SymInt


def _dimension_key(dimension: int | torch.SymInt) -> tuple[str, int | str]:
    return ("static", dimension) if isinstance(dimension, int) else ("symbolic", str(dimension))


def _shape_matches(
    actual: torch.Size,
    expected: tuple[int | torch.SymInt, ...],
) -> bool:
    return len(actual) == len(expected) and all(
        _dimension_key(left) == _dimension_key(right)
        for left, right in zip(actual, expected, strict=True)
    )


def _valid_scalar(value: torch.Tensor | None) -> bool:
    return value is None or (value.shape == () and value.dtype is torch.float32)


def _validate_input_features(
    input_features: int | torch.SymInt,
    name: str,
) -> None:
    if isinstance(input_features, int) and (
        input_features < _layout.BLOCK_SIZE or input_features % _layout.BLOCK_SIZE
    ):
        raise ValueError(
            f"{name} input features must be a positive multiple of {_layout.BLOCK_SIZE}"
        )


def _validate_device(device: torch.device, name: str) -> None:
    if device.type == "meta":
        return
    if device.type != "cuda" or not AcceleratorTarget.from_device(device).is_cuda_capability(12, 0):
        raise ValueError(f"{name} requires exact NVIDIA SM120")


def validate_activation_scale(
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    device: torch.device,
    name: str,
) -> None:
    """Validate one NVFP4 activation's global-scale configuration."""
    if not isinstance(dynamic_activation_scale, bool):
        raise ValueError(f"{name} dynamic activation scale flag must be boolean")
    if not _valid_scalar(activation_per_tensor_scale) or (
        not dynamic_activation_scale and activation_per_tensor_scale is None
    ):
        raise ValueError(f"{name} static activation scale must be an FP32 scalar")
    if activation_per_tensor_scale is not None and activation_per_tensor_scale.device != device:
        raise ValueError(f"{name} activation scale must share the input device")
    if activation_per_tensor_scale is not None and (
        activation_per_tensor_scale.layout is not torch.strided
        or not activation_per_tensor_scale.is_contiguous()
    ):
        raise ValueError(f"{name} activation scale must be a contiguous strided tensor")


def validate_weight(
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    input_features: int | torch.SymInt,
    device: torch.device,
    name: str,
) -> int | torch.SymInt:
    _validate_input_features(input_features, name)
    packed_input_features = input_features // 2
    if (
        weight_qdata.ndim != 2
        or weight_qdata.dtype is not torch.uint8
        or _dimension_key(weight_qdata.shape[1]) != _dimension_key(packed_input_features)
    ):
        raise ValueError(f"{name} weight has an incompatible packed UINT8 layout")
    output_features = weight_qdata.shape[0]
    if (
        not _shape_matches(
            weight_scale.shape,
            _layout.scale_shape(output_features, input_features),
        )
        or weight_scale.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError(f"{name} weight scale has an incompatible swizzled layout")
    if not _valid_scalar(weight_per_tensor_scale):
        raise ValueError(f"{name} weight per-tensor scale must be an FP32 scalar")
    if bias is not None:
        if not _shape_matches(bias.shape, (output_features,)):
            raise ValueError(f"{name} bias must contain one value per output feature")
        _bias.validate_dtype(bias, name)
    operands = [weight_qdata, weight_scale]
    operands.extend(operand for operand in (weight_per_tensor_scale, bias) if operand is not None)
    if any(operand.device != device for operand in operands):
        raise ValueError(f"{name} operands must share a device")
    if any(
        operand.layout is not torch.strided or not operand.is_contiguous() for operand in operands
    ):
        raise ValueError(f"{name} weight operands must be contiguous strided tensors")
    return output_features


def validate_semantic_linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    dynamic_activation_scale: bool,
    name: str,
    *,
    allow_empty: bool = False,
) -> LinearShape:
    """Validate an unprepared semantic NVFP4 linear."""
    if input.ndim == 0 or input.dtype not in _LOGICAL_DTYPES or input.layout is not torch.strided:
        raise ValueError(f"{name} input must be a non-scalar strided floating tensor")
    input_features = input.shape[-1]
    _validate_input_features(input_features, name)
    rows = math.prod(input.shape[:-1])
    if not allow_empty and isinstance(rows, int) and rows < 1:
        raise ValueError(f"{name} input must contain at least one row")
    validate_activation_scale(
        activation_per_tensor_scale,
        dynamic_activation_scale,
        input.device,
        name,
    )
    output_features = validate_weight(
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        input_features=input_features,
        device=input.device,
        name=name,
    )
    _validate_device(input.device, name)
    return LinearShape(rows, input_features, output_features)


def validate_prepared_linear(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
    name: str,
) -> LinearShape:
    """Validate one canonical prepared NVFP4 linear."""
    if input_qdata.ndim != 2 or input_qdata.dtype is not torch.uint8:
        raise ValueError(f"{name} input must be a two-dimensional packed UINT8 tensor")
    rows, packed_input_features = input_qdata.shape
    input_features = 2 * packed_input_features
    if isinstance(rows, int) and rows < 1:
        raise ValueError(f"{name} input must contain at least one row")
    _validate_input_features(input_features, name)
    if (
        not _shape_matches(
            input_scale.shape,
            _layout.scale_shape(rows, input_features),
        )
        or input_scale.dtype is not torch.float8_e4m3fn
    ):
        raise ValueError(f"{name} input scale has an incompatible swizzled layout")
    if input_per_tensor_scale.shape != () or input_per_tensor_scale.dtype is not torch.float32:
        raise ValueError(f"{name} input per-tensor scale must be an FP32 scalar")
    if logical_dtype not in _LOGICAL_DTYPES:
        raise ValueError(f"{name} logical dtype must be FP16, BF16, or FP32")
    output_features = validate_weight(
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        bias,
        input_features=input_features,
        device=input_qdata.device,
        name=name,
    )
    inputs = (input_qdata, input_scale, input_per_tensor_scale)
    if any(value.device != input_qdata.device for value in inputs):
        raise ValueError(f"{name} prepared inputs must share a device")
    if any(value.layout is not torch.strided or not value.is_contiguous() for value in inputs):
        raise ValueError(f"{name} prepared inputs must be contiguous strided tensors")
    _validate_device(input_qdata.device, name)
    return LinearShape(rows, input_features, output_features)


__all__ = [
    "LinearShape",
    "validate_activation_scale",
    "validate_prepared_linear",
    "validate_semantic_linear",
    "validate_weight",
]
