"""NVFP4 weight in a grouped ConvRot basis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor

from piper_kernels.weights._dispatch import require_untransposed
from piper_kernels.weights.convrot._rotation import rotate_groups, validate_group_size
from piper_kernels.weights.nvfp4 import _layout
from piper_kernels.weights.nvfp4.tensor import (
    NVFP4ActQuant,
    PiperNVFP4Tensor,
    _quantize_hp_tensors,
    _validate_nvfp4_hp_inputs,
)


class ConvRotNVFP4Tensor(PiperNVFP4Tensor):
    """Standard NVFP4 weight storage carrying its grouped rotation metadata."""

    _update_format_name: ClassVar[str] = "ConvRot NVFP4"
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
        dtype: torch.dtype,
        group_size: int,
        per_tensor_scale: torch.Tensor | None = None,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        act_quant: NVFP4ActQuant | None = None,
        high_first: bool = False,
        *,
        _validate: bool = True,
    ) -> ConvRotNVFP4Tensor:
        if _validate:
            validate_group_size(group_size)
        tensor = super().__new__(
            cls,
            qdata,
            scale,
            block_size,
            dtype,
            per_tensor_scale=per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
            high_first=high_first,
            _validate=_validate,
        )
        if _validate:
            if tensor.ndim != 2:
                raise ValueError("ConvRot NVFP4 weight must be two-dimensional")
            features = tensor.shape[-2] if tensor.transposed else tensor.shape[-1]
            if features % group_size:
                raise ValueError(
                    f"ConvRot NVFP4 weight features {features} must be divisible "
                    f"by group size {group_size}"
                )
        convrot = cast(ConvRotNVFP4Tensor, tensor)
        convrot.group_size = group_size
        return convrot

    @classmethod
    def _validate_storage_inputs(  # type: ignore[override]
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
        group_size: int,
        **kwargs,
    ) -> None:
        validate_group_size(group_size)
        if qdata.ndim != 2:
            raise ValueError("ConvRot NVFP4 weight must be two-dimensional")
        super()._validate_storage_inputs(
            qdata, scale, block_size, dtype, allowed_ndims=(2,), **kwargs
        )
        features = qdata.shape[1] * 2
        if features % group_size:
            raise ValueError(
                f"ConvRot NVFP4 weight features {features} must be divisible "
                f"by group size {group_size}"
            )

    @classmethod
    def from_quantized(  # type: ignore[override]
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        *,
        group_size: int,
        logical_dtype: torch.dtype = torch.bfloat16,
        block_size: int = _layout.BLOCK_SIZE,
        per_tensor_scale: torch.Tensor | None = None,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        act_quant: NVFP4ActQuant | None = None,
        high_first: bool = False,
    ) -> ConvRotNVFP4Tensor:
        """Wrap packed rotated NVFP4 checkpoint storage zero-copy when contiguous."""
        cls._validate_storage_inputs(
            qdata,
            scale,
            block_size,
            logical_dtype,
            group_size,
            per_tensor_scale=per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
            high_first=high_first,
            require_contiguous=False,
        )
        return cls(
            qdata.contiguous(),
            scale.contiguous(),
            block_size,
            logical_dtype,
            group_size,
            per_tensor_scale=None if per_tensor_scale is None else per_tensor_scale.contiguous(),
            act_per_tensor_scale=(
                None if act_per_tensor_scale is None else act_per_tensor_scale.contiguous()
            ),
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
            high_first=high_first,
            _validate=False,
        )

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        *,
        block_size: int = _layout.BLOCK_SIZE,
        per_tensor_scale: torch.Tensor | None = None,
        compute_per_tensor_scale: bool = False,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        act_quant: NVFP4ActQuant | None = None,
        group_size: int | None = None,
    ) -> ConvRotNVFP4Tensor:
        """Rotate and quantize a high-precision weight into ConvRot NVFP4 storage.

        ``compute_per_tensor_scale=True`` derives the optional global weight scale
        from the rotated weight, avoiding both a second rotation in the caller and
        an incorrect scale from the logical basis. ``hp_tensor`` is detached
        before conversion; checkpoint quantization is an inference transform and
        does not retain an autograd graph.
        """
        if group_size is None:
            raise TypeError("ConvRot NVFP4 quantization requires a group size")
        validate_group_size(group_size)
        if hp_tensor.ndim != 2:
            raise ValueError("ConvRot NVFP4 weight must be two-dimensional")
        _validate_nvfp4_hp_inputs(
            hp_tensor,
            block_size=block_size,
            per_tensor_scale=per_tensor_scale,
            compute_per_tensor_scale=compute_per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
        )
        rotated = rotate_groups(hp_tensor.detach().float(), group_size)
        if (
            not compute_per_tensor_scale
            and rotated.device.type != "meta"
            and not isinstance(rotated, FakeTensor)
            and not torch.compiler.is_compiling()
            and detect_fake_mode() is None
            and not bool(torch.isfinite(rotated).all())
        ):
            raise ValueError("ConvRot NVFP4 rotation produced non-finite weights")
        qdata, scales, effective_per_tensor_scale = _quantize_hp_tensors(
            rotated,
            block_size=block_size,
            per_tensor_scale=per_tensor_scale,
            compute_per_tensor_scale=compute_per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
            validate=False,
        )
        return cls(
            qdata,
            scales,
            block_size,
            hp_tensor.dtype,
            group_size,
            per_tensor_scale=effective_per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
            _validate=False,
        )

    @classmethod
    def from_gguf(  # noqa: PLR0913
        cls,
        data: torch.Tensor,
        *,
        quant_type: int | None = None,
        logical_dtype: torch.dtype = torch.bfloat16,
        block_size: int = _layout.BLOCK_SIZE,
        per_tensor_scale: torch.Tensor | None = None,
        compute_per_tensor_scale: bool = False,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        act_quant: NVFP4ActQuant | None = None,
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
        validate_group_size(group_size)
        _layout.validate_nvfp4_options(
            block_size=block_size,
            dtype=logical_dtype,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
            high_first=high_first,
        )
        if block_size != _layout.BLOCK_SIZE:
            raise ValueError("ConvRot NVFP4 GGUF conversion requires block size 16")
        if compute_per_tensor_scale and per_tensor_scale is not None:
            raise ValueError("NVFP4 from_gguf cannot both compute and receive a per-tensor scale")
        _layout.validate_nvfp4_global_scales(
            per_tensor_scale,
            act_per_tensor_scale,
            device=data.device,
            ndim=2,
            experts=0,
        )
        from piper_kernels.weights.convrot.nvfp4._gguf import convert  # noqa: PLC0415

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
            per_tensor_scale=effective_per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
            high_first=high_first,
            _validate=False,
        )

    def copy_from_gguf_(
        self,
        data: torch.Tensor,
        *,
        quant_type: int | None = None,
        compute_per_tensor_scale: bool = False,
    ) -> ConvRotNVFP4Tensor:
        """Refill this tensor from compatible packed GGUF storage in place."""
        require_untransposed(self, "copy_from_gguf_")
        if compute_per_tensor_scale and self.per_tensor_scale is None:
            raise ValueError("recomputing the NVFP4 scale requires existing scale storage")
        from piper_kernels.weights.convrot.nvfp4._gguf import convert  # noqa: PLC0415

        convert(
            data,
            quant_type=quant_type,
            logical_dtype=self.dtype,
            group_size=self.group_size,
            per_tensor_scale=None if compute_per_tensor_scale else self.per_tensor_scale,
            compute_per_tensor_scale=compute_per_tensor_scale,
            is_swizzled_scales=self.is_swizzled_scales,
            high_first=self.high_first,
            out=(self.qdata, self.scale),
            per_tensor_scale_out=self.per_tensor_scale if compute_per_tensor_scale else None,
        )
        return self

    def _validate_storage_for_export(self) -> None:
        require_untransposed(self, "storage export")
        validate_group_size(self.group_size)
        if self.ndim != 2:
            raise ValueError("ConvRot NVFP4 weight must be two-dimensional")
        _layout.validate_nvfp4_weight(self, allowed_ndims=(2,))
        if self.shape[-1] % self.group_size:
            raise ValueError(
                f"ConvRot NVFP4 weight features {self.shape[-1]} must be divisible "
                f"by group size {self.group_size}"
            )

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Recover the logical weight in the unrotated basis."""
        if self.transposed:
            return self._transpose().dequantize(output_dtype).t()
        return rotate_groups(super().dequantize(output_dtype), self.group_size)

    def _update_group_size(self) -> int:
        """Merge updates in this weight's stored ConvRot basis."""
        return self.group_size


__all__ = ["ConvRotNVFP4Tensor"]


@ConvRotNVFP4Tensor.implements(torch.ops.aten.linear.default)
@ConvRotNVFP4Tensor.implements_torch_function(torch.nn.functional.linear)
def _linear_dispatch(
    func: Callable[..., torch.Tensor],
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    from piper_kernels.linear.convrot.nvfp4._functional import linear_dispatch  # noqa: PLC0415

    return linear_dispatch(func, types, args, kwargs)
