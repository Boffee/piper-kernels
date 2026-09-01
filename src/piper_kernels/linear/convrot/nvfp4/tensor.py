"""TorchAO-compatible NVFP4 weight in a grouped ConvRot basis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

from piper_kernels.linear._dispatch import bind_linear_arguments
from piper_kernels.linear.convrot._rotation import (
    rotate_groups,
    validate_group_size,
)
from piper_kernels.linear.nvfp4._typing import NVFP4Storage
from piper_kernels.linear.nvfp4.tensor import (
    PiperNVFP4Tensor,
    supports_semantic_linear,
)

from . import _addmm, _ops


class ConvRotNVFP4Tensor(PiperNVFP4Tensor):
    """Standard NVFP4 weight storage carrying its grouped rotation metadata."""

    tensor_attribute_names: ClassVar[list[str]] = [
        *PiperNVFP4Tensor.tensor_attribute_names,
        "group_size",
    ]
    group_size: int

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: int,
        orig_dtype: torch.dtype,
        group_size: int,
        per_tensor_scale: torch.Tensor | None = None,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        use_triton_kernel: bool = False,
        act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None = None,
    ) -> ConvRotNVFP4Tensor:
        validate_group_size(group_size)
        if orig_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("ConvRot NVFP4 logical dtype must be FP16 or BF16")
        tensor = super().__new__(
            cls,
            qdata,
            scale,
            block_size,
            orig_dtype,
            per_tensor_scale,
            act_per_tensor_scale,
            is_swizzled_scales,
            use_triton_kernel,
            act_quant_kwargs,
        )
        if tensor.ndim != 2:
            raise ValueError("ConvRot NVFP4 weight must be two-dimensional")
        if tensor.shape[-1] % group_size:
            raise ValueError(
                f"ConvRot NVFP4 weight features {tensor.shape[-1]} must be divisible "
                f"by group size {group_size}"
            )
        return cast(ConvRotNVFP4Tensor, tensor)

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: int,
        orig_dtype: torch.dtype,
        group_size: int,
        per_tensor_scale: torch.Tensor | None = None,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        use_triton_kernel: bool = False,
        act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None = None,
    ) -> None:
        super().__init__(
            qdata,
            scale,
            block_size,
            orig_dtype,
            per_tensor_scale,
            act_per_tensor_scale,
            is_swizzled_scales,
            use_triton_kernel,
            act_quant_kwargs,
        )
        self.group_size = group_size

    @classmethod
    def from_torchao(
        cls,
        tensor: TorchAONVFP4Tensor,
        *,
        group_size: int | None = None,
    ) -> ConvRotNVFP4Tensor:
        """Attach ConvRot metadata to canonical TorchAO NVFP4 storage without copying."""
        if group_size is None:
            raise TypeError("ConvRot NVFP4 wrapping requires a group size")
        validate_group_size(group_size)
        if isinstance(tensor, cls):
            if tensor.group_size != group_size:
                raise ValueError(
                    f"ConvRot NVFP4 tensor uses group size {tensor.group_size}, not {group_size}"
                )
            return tensor
        storage = cast(NVFP4Storage, tensor)
        return cls(
            storage.qdata,
            storage.scale,
            storage.block_size,
            storage.orig_dtype,
            group_size,
            storage.per_tensor_scale,
            storage.act_per_tensor_scale,
            storage.is_swizzled_scales,
            storage.use_triton_kernel,
            storage.act_quant_kwargs,
        )

    def _stable_hash_for_caching(self) -> str:
        """Include the rotation group in AOTAutograd's persistent-cache key."""
        return repr((super()._stable_hash_for_caching(), self.group_size))

    def _rebuild_with_orig_dtype(self, orig_dtype: torch.dtype) -> ConvRotNVFP4Tensor:
        """Preserve the concrete wrapper and rotation metadata across conversion."""
        return type(self)(
            self.qdata,
            self.scale,
            self.block_size,
            orig_dtype,
            self.group_size,
            self.per_tensor_scale,
            self.act_per_tensor_scale,
            self.is_swizzled_scales,
            self.use_triton_kernel,
            self.act_quant_kwargs,
        )

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Recover the logical weight in the unrotated basis."""
        return rotate_groups(super().dequantize(output_dtype), self.group_size)

    def addmm_(
        self,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta: int | float | complex = 1,
        alpha: int | float | complex = 1,
        rounding_seed: int | None = None,
    ) -> ConvRotNVFP4Tensor:
        """Update and requantize in place, optionally using stochastic rounding.

        ``rounding_seed`` accepts the full unsigned 64-bit range. Supplying it
        makes terminal E2M1 code selection reproducible for a fixed device and
        backend without consuming the process-global random-number generator.
        """
        if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
            raise TypeError("ConvRot NVFP4 addmm_ matrices must be tensors")
        _addmm.addmm_(
            self,
            self.group_size,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
            rounding_seed=rounding_seed,
        )
        return self


def _supports_convrot_linear(input: object, weight: ConvRotNVFP4Tensor) -> bool:  # noqa: A002
    return (
        supports_semantic_linear(input, weight)
        and isinstance(input, torch.Tensor)
        and input.dtype in (torch.float16, torch.bfloat16)
        and input.ndim > 0
        and input.shape[-1] % weight.group_size == 0
    )


def convrot_nvfp4_linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    weight: ConvRotNVFP4Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a canonical NVFP4 weight and activation in the same ConvRot basis."""
    if not isinstance(input, torch.Tensor) or not isinstance(weight, ConvRotNVFP4Tensor):
        raise TypeError(
            "ConvRot NVFP4 linear requires a tensor input and ConvRotNVFP4Tensor weight"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"ConvRot NVFP4 linear bias must be a tensor or None, got {type(bias).__name__}"
        )
    if not _supports_convrot_linear(input, weight):
        raise ValueError("ConvRot NVFP4 linear requires canonical SM120 NVFP4 operands")
    quantization = weight.act_quant_kwargs
    assert quantization is not None
    return _ops.linear(
        input,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
        bias,
        quantization.use_dynamic_per_tensor_scale,
        weight.group_size,
    )


@ConvRotNVFP4Tensor.implements(torch.ops.aten.linear.default)
@ConvRotNVFP4Tensor.implements_torch_function(torch.nn.functional.linear)
def _convrot_nvfp4_linear_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    input, weight, bias = bind_linear_arguments(args, kwargs)  # noqa: A001
    if not isinstance(input, torch.Tensor) or not isinstance(weight, ConvRotNVFP4Tensor):
        raise TypeError(
            "ConvRot NVFP4 linear dispatch requires a tensor input and ConvRotNVFP4Tensor weight"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"ConvRot NVFP4 linear bias must be a tensor or None, got {type(bias).__name__}"
        )
    return convrot_nvfp4_linear(input, weight, bias)


@ConvRotNVFP4Tensor.implements(torch.ops.aten.addmm_.default)
def _convrot_nvfp4_addmm_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ConvRotNVFP4Tensor:
    weight, mat1, mat2 = args
    if not isinstance(weight, ConvRotNVFP4Tensor):
        raise TypeError(
            f"ConvRot NVFP4 addmm_ weight must be ConvRotNVFP4Tensor, got {type(weight)}"
        )
    if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
        raise TypeError("ConvRot NVFP4 addmm_ matrices must be tensors")
    _addmm.addmm_(
        weight,
        weight.group_size,
        mat1,
        mat2,
        beta=kwargs.get("beta", 1),
        alpha=kwargs.get("alpha", 1),
    )
    return weight


__all__ = ["ConvRotNVFP4Tensor", "convrot_nvfp4_linear"]
