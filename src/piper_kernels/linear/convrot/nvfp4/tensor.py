"""TorchAO-compatible NVFP4 weight in a grouped ConvRot basis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
)

from piper_kernels.linear._dispatch import apply_linear_autocast, bind_linear_arguments
from piper_kernels.linear.convrot._rotation import (
    rotate_groups,
    validate_group_size,
)
from piper_kernels.linear.nvfp4._typing import NVFP4Storage
from piper_kernels.linear.nvfp4.tensor import (
    PiperNVFP4Tensor,
    _quantize_hp,
    supports_semantic_linear,
)


class ConvRotNVFP4Tensor(PiperNVFP4Tensor):
    """Standard NVFP4 weight storage carrying its grouped rotation metadata."""

    tensor_attribute_names: ClassVar[list[str]] = [
        *PiperNVFP4Tensor.tensor_attribute_names,
        "group_size",
    ]
    group_size: int

    def __new__(  # noqa: PLR0913, PLR0917
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
        high_first: bool = False,
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
            high_first,
        )
        if tensor.ndim != 2:
            raise ValueError("ConvRot NVFP4 weight must be two-dimensional")
        if tensor.shape[-1] % group_size:
            raise ValueError(
                f"ConvRot NVFP4 weight features {tensor.shape[-1]} must be divisible "
                f"by group size {group_size}"
            )
        return cast(ConvRotNVFP4Tensor, tensor)

    def __init__(  # noqa: PLR0913, PLR0917
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
        high_first: bool = False,
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
            high_first,
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
            high_first=getattr(storage, "high_first", False),
        )

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        *,
        block_size: int = 16,
        per_tensor_scale: torch.Tensor | None = None,
        compute_per_tensor_scale: bool = False,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        use_triton_kernel: bool = False,
        act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None = None,
        group_size: int | None = None,
    ) -> ConvRotNVFP4Tensor:
        """Rotate and quantize a high-precision weight into ConvRot NVFP4 storage.

        The NVFP4 arguments otherwise match TorchAO's :meth:`NVFP4Tensor.to_nvfp4`
        builder. ``compute_per_tensor_scale=True`` derives the optional global weight
        scale from the rotated weight, avoiding both a second rotation in the caller and
        an incorrect scale from the logical basis. ``hp_tensor`` is detached before
        conversion; checkpoint quantization is an inference transform and does not retain
        an autograd graph.
        """
        if group_size is None:
            raise TypeError("ConvRot NVFP4 quantization requires a group size")
        return cls.from_torchao(
            _quantize_hp(
                rotate_groups(hp_tensor.detach(), group_size),
                block_size=block_size,
                per_tensor_scale=per_tensor_scale,
                compute_per_tensor_scale=compute_per_tensor_scale,
                act_per_tensor_scale=act_per_tensor_scale,
                is_swizzled_scales=is_swizzled_scales,
                use_triton_kernel=use_triton_kernel,
                act_quant_kwargs=act_quant_kwargs,
            ),
            group_size=group_size,
        )

    @classmethod
    def from_gguf(  # noqa: PLR0913
        cls,
        data: torch.Tensor,
        *,
        quant_type: int | None = None,
        logical_dtype: torch.dtype = torch.bfloat16,
        block_size: int = 16,
        per_tensor_scale: torch.Tensor | None = None,
        compute_per_tensor_scale: bool = False,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        use_triton_kernel: bool = False,
        act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None = None,
        group_size: int | None = None,
        high_first: bool = False,
    ) -> ConvRotNVFP4Tensor:
        """Decode packed GGUF storage directly into ConvRot NVFP4 storage.

        Exact SM120 conversion keeps decoded values in registers. Deriving a
        per-tensor scale performs one small amax pass followed by the packing
        pass, without allocating a dense weight.
        """
        if group_size is None:
            raise TypeError("ConvRot NVFP4 GGUF conversion requires a group size")
        if block_size != 16:
            raise ValueError("ConvRot NVFP4 GGUF conversion requires block size 16")
        from ._gguf import convert  # noqa: PLC0415

        qdata, scale, effective_per_tensor_scale = convert(
            data,
            quant_type=quant_type,
            logical_dtype=logical_dtype,
            group_size=group_size,
            per_tensor_scale=per_tensor_scale,
            compute_per_tensor_scale=compute_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            high_first=high_first,
        )
        return cls(
            qdata,
            scale,
            block_size,
            logical_dtype,
            group_size,
            effective_per_tensor_scale,
            act_per_tensor_scale,
            is_swizzled_scales,
            use_triton_kernel,
            act_quant_kwargs,
            high_first,
        )

    def copy_from_gguf_(
        self,
        data: torch.Tensor,
        *,
        quant_type: int | None = None,
        compute_per_tensor_scale: bool = False,
    ) -> ConvRotNVFP4Tensor:
        """Refill this tensor from compatible packed GGUF storage in place."""
        if compute_per_tensor_scale and self.per_tensor_scale is None:
            raise ValueError("recomputing the NVFP4 scale requires existing scale storage")
        from ._gguf import convert  # noqa: PLC0415

        convert(
            data,
            quant_type=quant_type,
            logical_dtype=self.orig_dtype,
            group_size=self.group_size,
            per_tensor_scale=None if compute_per_tensor_scale else self.per_tensor_scale,
            compute_per_tensor_scale=compute_per_tensor_scale,
            is_swizzled_scales=self.is_swizzled_scales,
            high_first=self.high_first,
            out=(self.qdata, self.scale),
            per_tensor_scale_out=self.per_tensor_scale if compute_per_tensor_scale else None,
        )
        return self

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
            high_first=self.high_first,
        )

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Recover the logical weight in the unrotated basis."""
        return rotate_groups(super().dequantize(output_dtype), self.group_size)

    def _update_group_size(self) -> int:
        """Merge updates in this weight's stored ConvRot basis."""
        return self.group_size


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
    converted_input, converted_weight, bias = apply_linear_autocast(input, weight, bias)
    assert isinstance(converted_weight, ConvRotNVFP4Tensor)
    weight = converted_weight
    if not _supports_convrot_linear(converted_input, weight):
        raise ValueError("ConvRot NVFP4 linear requires canonical SM120 NVFP4 operands")
    quantization = weight.act_quant_kwargs
    assert quantization is not None
    from . import _ops  # noqa: PLC0415

    return _ops.linear(
        converted_input,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
        bias,
        quantization.use_dynamic_per_tensor_scale,
        weight.group_size,
        weight.high_first,
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


__all__ = ["ConvRotNVFP4Tensor", "convrot_nvfp4_linear"]
