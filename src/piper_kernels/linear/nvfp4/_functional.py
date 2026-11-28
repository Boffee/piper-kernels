"""Functional NVFP4 linear dispatch and operand eligibility."""

from collections.abc import Callable
from typing import Any, cast

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._dispatch import apply_linear_autocast, bind_linear_arguments
from piper_kernels.weights._dispatch import QuantizedWeight
from piper_kernels.weights._views import require_untransposed
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor, _layout


def _supports_common_nvfp4_linear(
    input: object,  # noqa: A002
    weight: PiperNVFP4Tensor,
    *,
    require_swizzled: bool = True,
) -> bool:
    if not isinstance(input, torch.Tensor) or isinstance(input, QuantizedWeight):
        return False
    tensor_input = cast(torch.Tensor, input)
    return (
        tensor_input.ndim > 0
        and tensor_input.dtype is weight.dtype
        and tensor_input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and tensor_input.layout is torch.strided
        and weight.ndim == 2
        and weight.block_size == _layout.BLOCK_SIZE
        and (weight.is_swizzled_scales if require_swizzled else True)
        and (weight.per_tensor_scale is None or weight.per_tensor_scale.ndim == 0)
    )


def _canonical_weight_scale(weight: PiperNVFP4Tensor) -> torch.Tensor:
    if weight.ndim == 2 and weight.is_swizzled_scales:
        expected = _layout.scale_shape(weight.shape[0], weight.shape[1])
        if tuple(weight.scale.shape) == (expected[0] * expected[1],):
            return weight.scale.view(expected)
    return weight.scale


def supports_semantic_linear(input: object, weight: PiperNVFP4Tensor) -> bool:  # noqa: A002
    if not _supports_common_nvfp4_linear(input, weight):
        return False
    tensor_input = cast(torch.Tensor, input)
    return tensor_input.device.type in ("cuda", "meta") and weight.act_quant in (
        "static",
        "dynamic",
    )


def supports_weight_only_linear(input: object, weight: PiperNVFP4Tensor) -> bool:  # noqa: A002
    """Return whether ``input`` and ``weight`` qualify for weight-only W4A16 execution."""
    if not _supports_common_nvfp4_linear(input, weight):
        return False
    tensor_input = cast(torch.Tensor, input)
    if weight.act_quant is not None or weight.act_per_tensor_scale is not None:
        return False
    return tensor_input.device.type in ("cuda", "meta")


def linear_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    input, weight, bias = bind_linear_arguments(args, kwargs)  # noqa: A001
    if (
        not isinstance(input, torch.Tensor)
        or isinstance(input, QuantizedWeight)
        or not isinstance(weight, PiperNVFP4Tensor)
    ):
        raise TypeError("NVFP4 linear requires a dense tensor input and PiperNVFP4Tensor weight")
    require_untransposed(weight, "linear")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(f"NVFP4 linear bias must be a tensor or None, got {type(bias).__name__}")

    converted_input, converted_weight, bias = apply_linear_autocast(input, weight, bias)
    assert isinstance(converted_weight, PiperNVFP4Tensor)
    weight = converted_weight
    weight_scale = _canonical_weight_scale(weight)
    if supports_semantic_linear(converted_input, weight):
        from . import _ops  # noqa: PLC0415

        return _ops.linear(
            converted_input,
            weight.qdata,
            weight_scale,
            weight.per_tensor_scale,
            weight.act_per_tensor_scale,
            bias,
            weight.act_quant == "dynamic",
            weight.high_first,
        )
    if supports_weight_only_linear(converted_input, weight):
        from . import _ops  # noqa: PLC0415

        return _ops.linear(
            converted_input,
            weight.qdata,
            weight_scale,
            weight.per_tensor_scale,
            None,
            bias,
            False,
            weight.high_first,
            True,
        )
    if weight.act_quant is None:
        if weight.act_per_tensor_scale is not None:
            raise ValueError("NVFP4 weight-only execution does not accept activation scales")
        if not _supports_common_nvfp4_linear(converted_input, weight, require_swizzled=False):
            raise ValueError(
                "NVFP4 linear requires matching floating-point dense activation and "
                "2-D block-16 weight"
            )
        if converted_input.device != weight.device:
            raise ValueError("NVFP4 linear operands must share a device")
        if (
            converted_input.device.type == "cuda"
            and not AcceleratorTarget.from_device(converted_input.device).supports_nvfp4_w4a16
        ):
            raise ValueError(
                "NVFP4 linear requires NVIDIA SM12x "
                "(use ConvRot INT8 or ConvRot Trellis on other GPUs)"
            )
        if weight.is_swizzled_scales:
            from . import reference  # noqa: PLC0415

            return reference.linear_w4a16(
                converted_input,
                weight.qdata,
                weight_scale,
                weight.per_tensor_scale,
                bias,
                high_first=weight.high_first,
            )
        return torch.nn.functional.linear(converted_input, weight.dequantize(), bias)
    if _supports_common_nvfp4_linear(converted_input, weight):
        from . import reference  # noqa: PLC0415

        if converted_input.device != weight.device:
            raise ValueError("NVFP4 linear operands must share a device")
        return reference.linear(
            converted_input,
            weight.qdata,
            weight_scale,
            weight.per_tensor_scale,
            weight.act_per_tensor_scale,
            bias,
            weight.act_quant == "dynamic",
            weight.high_first,
        )
    raise ValueError(
        "NVFP4 linear requires matching floating-point dense activation and "
        "canonical block-16 NVFP4 weight"
    )
