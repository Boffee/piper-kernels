"""Packed NVFP4 weight storage with Piper quantized tensor semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Literal, Self, cast

import torch
from torch._guards import detect_fake_mode
from torch._prims_common import make_contiguous_strides_for
from torch._subclasses.fake_tensor import FakeTensor

from piper_kernels.weights._dispatch import QuantizedWeight, require_untransposed
from piper_kernels.weights._fsdp import FSDPWeightMixin, register_fsdp_ops
from piper_kernels.weights._matmul import register_matrix_ops
from piper_kernels.weights._update import UpdatableWeightMixin, register_update_ops
from piper_kernels.weights._views import register_view_ops
from piper_kernels.weights.nvfp4 import _layout
from piper_kernels.weights.nvfp4._quantization import (
    dequantize_nvfp4,
    quantize_nvfp4,
    unswizzle_nvfp4_scales,
)

type NVFP4ActQuant = Literal["static", "dynamic"]


def _validate_nvfp4_hp_inputs(
    source: torch.Tensor,
    *,
    block_size: int,
    per_tensor_scale: torch.Tensor | None,
    compute_per_tensor_scale: bool,
    act_per_tensor_scale: torch.Tensor | None,
    is_swizzled_scales: bool,
    act_quant: NVFP4ActQuant | None,
) -> None:
    if source.ndim not in (2, 3):
        raise ValueError(f"NVFP4 high-precision weight must be 2-D or 3-D, got {source.ndim}-D")
    _layout.validate_nvfp4_options(
        block_size=block_size,
        dtype=source.dtype,
        is_swizzled_scales=is_swizzled_scales,
        act_quant=act_quant,
        high_first=False,
    )
    if block_size != _layout.BLOCK_SIZE:
        raise ValueError(f"NVFP4 requires block_size={_layout.BLOCK_SIZE}")
    if compute_per_tensor_scale and per_tensor_scale is not None:
        raise ValueError("NVFP4 from_hp cannot both compute and receive a per-tensor scale")
    _layout.validate_nvfp4_global_scales(
        per_tensor_scale,
        act_per_tensor_scale,
        device=source.device,
        ndim=source.ndim,
        experts=source.shape[0] if source.ndim == 3 else 0,
    )


def _quantize_hp_tensors(
    hp_tensor: torch.Tensor,
    *,
    block_size: int = _layout.BLOCK_SIZE,
    per_tensor_scale: torch.Tensor | None = None,
    compute_per_tensor_scale: bool = False,
    act_per_tensor_scale: torch.Tensor | None = None,
    is_swizzled_scales: bool = False,
    act_quant: NVFP4ActQuant | None = None,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize one detached high-precision weight with optional derived global scale."""
    source = hp_tensor.detach()
    if validate:
        _validate_nvfp4_hp_inputs(
            source,
            block_size=block_size,
            per_tensor_scale=per_tensor_scale,
            compute_per_tensor_scale=compute_per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
        )
    if compute_per_tensor_scale:
        amax = source.float().abs().amax()
        if (
            source.device.type != "meta"
            and not isinstance(source, FakeTensor)
            and not torch.compiler.is_compiling()
            and detect_fake_mode() is None
            and not bool(torch.isfinite(amax))
        ):
            raise ValueError("cannot quantize an NVFP4 weight with non-finite values")
        per_tensor_scale = _layout.per_tensor_amax_to_scale(amax).clamp_min(
            _layout.MIN_WEIGHT_GLOBAL_SCALE
        )
    qdata, scales = quantize_nvfp4(source, per_tensor_scale, swizzled=is_swizzled_scales)
    return qdata, scales, per_tensor_scale


class PiperNVFP4Tensor(FSDPWeightMixin, UpdatableWeightMixin, QuantizedWeight):  # pyright: ignore[reportIncompatibleMethodOverride]
    """NVFP4 weight storage, quantization metadata, and tensor dispatch hooks."""

    _format_name: ClassVar[str] = "NVFP4"
    _update_format_name: ClassVar[str] = "NVFP4"

    tensor_data_names: ClassVar[list[str]] = ["qdata", "scale"]
    tensor_attribute_names: ClassVar[list[str]] = ["block_size", "dtype"]
    optional_tensor_data_names: ClassVar[list[str]] = ["per_tensor_scale", "act_per_tensor_scale"]
    optional_tensor_attribute_names: ClassVar[list[str]] = [
        "is_swizzled_scales",
        "act_quant",
        "high_first",
    ]

    qdata: torch.Tensor
    scale: torch.Tensor
    block_size: int
    per_tensor_scale: torch.Tensor | None = None
    act_per_tensor_scale: torch.Tensor | None = None
    is_swizzled_scales: bool = False
    act_quant: NVFP4ActQuant | None = None
    high_first: bool = False

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
        per_tensor_scale: torch.Tensor | None = None,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        act_quant: NVFP4ActQuant | None = None,
        high_first: bool = False,
        *,
        _validate: bool = True,
    ) -> PiperNVFP4Tensor:
        if _validate:
            _layout.validate_nvfp4_options(
                block_size=block_size,
                dtype=dtype,
                is_swizzled_scales=is_swizzled_scales,
                act_quant=act_quant,
                high_first=high_first,
            )
            if qdata.layout is not torch.strided or scale.layout is not torch.strided:
                raise ValueError("NVFP4 weight storage must use strided layout")
            if qdata.ndim not in (2, 3):
                raise ValueError(f"NVFP4 qdata must be 2-D or 3-D, got {qdata.ndim}-D")
            if per_tensor_scale is not None and per_tensor_scale.ndim not in (0, 3):
                raise ValueError("NVFP4 per-tensor scale must be scalar or per-expert")
            if act_per_tensor_scale is not None and act_per_tensor_scale.ndim != 0:
                raise ValueError("NVFP4 activation scale must be a scalar tensor")
        transposed = qdata.stride(-2) < qdata.stride(-1)
        shape = _layout.tensor_size_fp4x2_to_hp(tuple(qdata.shape), transposed=transposed)
        # Reconstruct logical strides independently of packed storage strides so
        # transposes expose reversed logical strides to DTensor and AOTAutograd.
        canonical_shape = (*shape[:-2], shape[-1], shape[-2]) if transposed else shape
        strides = make_contiguous_strides_for(cast(Any, canonical_shape))
        if transposed:
            strides = (*strides[:-2], strides[-1], strides[-2])
        tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            dtype=dtype,
            device=qdata.device,
            requires_grad=False,
        )
        tensor.qdata = qdata
        tensor.scale = scale
        tensor.block_size = block_size
        tensor.per_tensor_scale = per_tensor_scale
        tensor.act_per_tensor_scale = act_per_tensor_scale
        tensor.is_swizzled_scales = is_swizzled_scales
        tensor.act_quant = act_quant
        tensor.high_first = high_first
        return cast(PiperNVFP4Tensor, tensor)

    @property
    def transposed(self) -> bool:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Whether the packed feature axis precedes the output-row axis."""
        return self.qdata.stride(-2) < self.qdata.stride(-1)

    def _transpose(self) -> Self:
        """Transpose packed views without changing their quantized interpretation."""
        names, metadata = self.__tensor_flatten__()
        tensors = {name: getattr(self, name) for name in names}
        tensors["qdata"] = self.qdata.t()
        tensors["scale"] = self.scale.t()
        return type(self).__tensor_unflatten__(tensors, metadata, None, None)

    @classmethod
    def _validate_storage_inputs(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
        **kwargs,
    ) -> None:
        _layout.validate_nvfp4_storage(qdata, scale, block_size=block_size, dtype=dtype, **kwargs)

    @classmethod
    def _from_storage_components(
        cls,
        required_tensors: list[torch.Tensor],
        optional_tensors: dict[str, torch.Tensor | None],
        required_attrs: list[Any],
        optional_attrs: dict[str, Any],
    ) -> Self:
        qdata, scale = required_tensors
        block_size, dtype, *extra_attrs = required_attrs
        cls._validate_storage_inputs(
            qdata,
            scale,
            block_size,
            dtype,
            *extra_attrs,
            **optional_tensors,
            **optional_attrs,
            require_contiguous=True,
        )
        return cls(
            qdata,
            scale,
            block_size,
            dtype,
            *extra_attrs,
            **optional_tensors,
            **optional_attrs,
            _validate=False,
        )

    @classmethod
    def from_quantized(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        *,
        logical_dtype: torch.dtype = torch.bfloat16,
        block_size: int = _layout.BLOCK_SIZE,
        per_tensor_scale: torch.Tensor | None = None,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        act_quant: NVFP4ActQuant | None = None,
        high_first: bool = False,
    ) -> PiperNVFP4Tensor:
        """Wrap packed NVFP4 checkpoint storage zero-copy when contiguous."""
        _layout.validate_nvfp4_storage(
            qdata,
            scale,
            block_size=block_size,
            dtype=logical_dtype,
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
    ) -> PiperNVFP4Tensor:
        """Quantize a detached high-precision weight into Piper NVFP4 storage.

        Piper's portable encoding retains E4M3 subnormal scales without a
        smallest-normal floor. For a 3-D expert batch, each matrix has
        independently padded swizzled scales; an optional per-expert global scale
        has shape ``[experts, 1, 1]``. ``compute_per_tensor_scale=True`` derives
        the optional global weight scale from ``hp_tensor`` and handles an
        all-zero weight without producing a zero global scale.
        """
        qdata, scales, effective_per_tensor_scale = _quantize_hp_tensors(
            hp_tensor,
            block_size=block_size,
            per_tensor_scale=per_tensor_scale,
            compute_per_tensor_scale=compute_per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
        )
        return cls(
            qdata,
            scales,
            block_size,
            hp_tensor.dtype,
            per_tensor_scale=effective_per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            act_quant=act_quant,
            _validate=False,
        )

    def _validate_storage_for_export(self) -> None:
        require_untransposed(self, "storage export")
        _layout.validate_nvfp4_weight(self)

    def get_hp_scales(self) -> torch.Tensor:
        """Recover canonical FP32 scales, removing swizzled tile padding when enabled."""
        if self.transposed:
            return self._transpose().get_hp_scales()
        scales = unswizzle_nvfp4_scales(
            self.scale,
            self.shape,
            swizzled=self.is_swizzled_scales,
            block_size=self.block_size,
        )
        return scales if self.per_tensor_scale is None else self.per_tensor_scale * scales

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize either packed-pair order to the logical weight."""
        target_dtype = self.dtype if output_dtype is None else output_dtype
        if target_dtype not in _layout.SUPPORTED_LOGICAL_DTYPES:
            raise ValueError(
                "NVFP4 dequantization output dtype must be float16, bfloat16, or float32, "
                f"got {target_dtype}"
            )
        if self.transposed:
            return self._transpose().dequantize(target_dtype).t()
        return dequantize_nvfp4(
            self.qdata,
            self.scale,
            per_tensor_scale=self.per_tensor_scale,
            swizzled=self.is_swizzled_scales,
            high_first=self.high_first,
            block_size=self.block_size,
            output_dtype=target_dtype,
        )

    def _update_group_size(self) -> int:
        """Use the unrotated basis for ordinary NVFP4 updates."""
        return 0

    def _update_target(self, _operation: str) -> tuple[Any, tuple[object, ...]]:
        from piper_kernels.weights.nvfp4 import _update  # noqa: PLC0415

        return _update, (self, self._update_group_size())


register_view_ops(PiperNVFP4Tensor)
register_fsdp_ops(PiperNVFP4Tensor)
register_matrix_ops(PiperNVFP4Tensor)
register_update_ops(PiperNVFP4Tensor, format_name=PiperNVFP4Tensor._update_format_name)


__all__ = ["NVFP4ActQuant", "PiperNVFP4Tensor"]


@PiperNVFP4Tensor.implements(torch.ops.aten.linear.default)
@PiperNVFP4Tensor.implements_torch_function(torch.nn.functional.linear)
def _linear_dispatch(
    func: Callable[..., torch.Tensor],
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    from piper_kernels.linear.nvfp4._functional import linear_dispatch  # noqa: PLC0415

    return linear_dispatch(func, types, args, kwargs)
