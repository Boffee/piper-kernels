"""Packed NVFP4 weight storage with TorchAO-compatible tensor semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Self, cast

import torch
from torch._prims_common import make_contiguous_strides_for
from torchao.prototype.mx_formats.mx_tensor import tensor_size_fp4x2_to_hp
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor as TorchAONVFP4Tensor
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import from_blocked
from torchao.utils import TorchAOBaseTensor

from piper_kernels.weights._dispatch import (
    register_conversion_ops,
    register_inference_mode_dispatch,
    to_quantized_weight,
)
from piper_kernels.weights._matmul import register_matrix_ops
from piper_kernels.weights._update import register_update_ops, run_add_, run_addmm_
from piper_kernels.weights._views import register_view_ops, require_untransposed
from piper_kernels.weights.nvfp4 import _layout
from piper_kernels.weights.nvfp4._quantization import quantize_nvfp4
from piper_kernels.weights.nvfp4._typing import NVFP4Storage


def _quantize_hp(
    hp_tensor: torch.Tensor,
    *,
    block_size: int = 16,
    per_tensor_scale: torch.Tensor | None = None,
    compute_per_tensor_scale: bool = False,
    act_per_tensor_scale: torch.Tensor | None = None,
    is_swizzled_scales: bool = False,
    use_triton_kernel: bool = False,
    act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None = None,
) -> TorchAONVFP4Tensor:
    """Quantize one detached high-precision weight with optional derived global scale."""
    if compute_per_tensor_scale and per_tensor_scale is not None:
        raise ValueError("NVFP4 from_hp cannot both compute and receive a per-tensor scale")
    source = hp_tensor.detach()
    if compute_per_tensor_scale:
        amax = source.float().abs().amax()
        if not bool(torch.isfinite(amax)):
            raise ValueError("cannot quantize an NVFP4 weight with non-finite values")
        per_tensor_scale = per_tensor_amax_to_scale(amax).clamp_min(_layout.MIN_WEIGHT_GLOBAL_SCALE)
    if use_triton_kernel:
        return TorchAONVFP4Tensor.to_nvfp4(
            source,
            block_size=block_size,
            per_tensor_scale=per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            use_triton_kernel=True,
            act_quant_kwargs=act_quant_kwargs,
        )
    if block_size != _layout.BLOCK_SIZE:
        raise ValueError("NVFP4 requires block_size=16")
    qdata, scales = quantize_nvfp4(source, per_tensor_scale, swizzled=is_swizzled_scales)
    return TorchAONVFP4Tensor(
        qdata,
        scales,
        block_size,
        source.dtype,
        per_tensor_scale,
        act_per_tensor_scale,
        is_swizzled_scales,
        False,
        act_quant_kwargs,
    )


class PiperNVFP4Tensor(TorchAONVFP4Tensor):
    """NVFP4 weight storage, quantization metadata, and tensor dispatch hooks."""

    __torch_function__ = classmethod(TorchAOBaseTensor.__torch_function__.__func__)
    optional_tensor_attribute_names: ClassVar[list[str]] = [
        *TorchAONVFP4Tensor.optional_tensor_attribute_names,
        "high_first",
    ]
    qdata: torch.Tensor
    scale: torch.Tensor
    block_size: int
    orig_dtype: torch.dtype
    per_tensor_scale: torch.Tensor | None
    act_per_tensor_scale: torch.Tensor | None
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None
    high_first: bool

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: int,
        orig_dtype: torch.dtype,
        per_tensor_scale: torch.Tensor | None = None,
        act_per_tensor_scale: torch.Tensor | None = None,
        is_swizzled_scales: bool = False,
        use_triton_kernel: bool = False,
        act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None = None,
        high_first: bool = False,
    ) -> PiperNVFP4Tensor:
        if type(high_first) is not bool:
            raise TypeError(f"NVFP4 high_first must be bool, got {type(high_first).__name__}")
        transposed = qdata.stride(-2) < qdata.stride(-1)
        shape = tensor_size_fp4x2_to_hp(qdata.shape, not transposed)
        # TorchAO reconstructs every wrapper with contiguous logical strides.
        # A transpose must also expose the reversed logical strides to DTensor
        # and AOTAutograd, independently of the packed storage's strides.
        canonical_shape = (*shape[:-2], shape[-1], shape[-2]) if transposed else shape
        strides = make_contiguous_strides_for(canonical_shape)
        if transposed:
            strides = (*strides[:-2], strides[-1], strides[-2])
        tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            dtype=orig_dtype,
            device=qdata.device,
            requires_grad=False,
        )
        if per_tensor_scale is not None and per_tensor_scale.ndim not in (0, 3):
            raise ValueError("NVFP4 per-tensor scale must be scalar or per-expert")
        tensor.qdata = qdata
        tensor.scale = scale
        tensor.block_size = block_size
        tensor.orig_dtype = orig_dtype
        tensor.per_tensor_scale = per_tensor_scale
        tensor.act_per_tensor_scale = act_per_tensor_scale
        tensor.is_swizzled_scales = is_swizzled_scales
        tensor.use_triton_kernel = use_triton_kernel
        tensor.act_quant_kwargs = act_quant_kwargs
        tensor.high_first = high_first
        return cast(PiperNVFP4Tensor, tensor)

    @property
    def transposed(self) -> bool:
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
    ) -> PiperNVFP4Tensor:
        """Quantize a detached high-precision weight into Piper NVFP4 storage.

        The arguments otherwise match TorchAO's :meth:`NVFP4Tensor.to_nvfp4`
        builder, but Piper's portable encoding retains E4M3 subnormal scales
        instead of imposing TorchAO's smallest-normal floor. For a 3-D expert
        batch, each matrix has independently padded swizzled scales; an optional
        per-expert global scale has shape ``[experts, 1, 1]``. Selecting
        ``use_triton_kernel=True`` delegates quantization to TorchAO's optional
        backend and retains its encoding behavior.
        ``compute_per_tensor_scale=True`` derives the optional global
        weight scale from ``hp_tensor`` and handles an all-zero weight without
        producing a zero global scale.
        """
        return cls.from_torchao(
            _quantize_hp(
                hp_tensor,
                block_size=block_size,
                per_tensor_scale=per_tensor_scale,
                compute_per_tensor_scale=compute_per_tensor_scale,
                act_per_tensor_scale=act_per_tensor_scale,
                is_swizzled_scales=is_swizzled_scales,
                use_triton_kernel=use_triton_kernel,
                act_quant_kwargs=act_quant_kwargs,
            )
        )

    @classmethod
    def from_torchao(cls, tensor: TorchAONVFP4Tensor) -> PiperNVFP4Tensor:
        """Wrap existing TorchAO NVFP4 storage without copying it."""
        if isinstance(tensor, cls):
            return tensor
        storage = cast(NVFP4Storage, tensor)
        return cls(
            storage.qdata,
            storage.scale,
            storage.block_size,
            storage.orig_dtype,
            storage.per_tensor_scale,
            storage.act_per_tensor_scale,
            storage.is_swizzled_scales,
            storage.use_triton_kernel,
            storage.act_quant_kwargs,
            high_first=getattr(storage, "high_first", False),
        )

    def _stable_hash_for_caching(self) -> str:
        """Return a metadata fingerprint for AOTAutograd's persistent cache."""
        return repr(
            (
                type(self).__qualname__,
                tuple(self.shape),
                self.stride(),
                str(self.device),
                str(self.dtype),
                self.block_size,
                tuple(self.qdata.shape),
                self.qdata.stride(),
                tuple(self.scale.shape),
                self.scale.stride(),
                self.is_swizzled_scales,
                self.use_triton_kernel,
                self.act_quant_kwargs,
                self.high_first,
                None if self.per_tensor_scale is None else tuple(self.per_tensor_scale.shape),
                None
                if self.act_per_tensor_scale is None
                else tuple(self.act_per_tensor_scale.shape),
            )
        )

    def _validate_memory_format(self, memory_format: object) -> None:
        if memory_format not in (None, torch.preserve_format):
            require_untransposed(self, "to with a different memory format")

    def _rebuild_with_logical_dtype(self, dtype: torch.dtype) -> PiperNVFP4Tensor:
        """Rebuild the concrete semantic wrapper while preserving its storage."""
        return type(self)(
            self.qdata,
            self.scale,
            self.block_size,
            dtype,
            self.per_tensor_scale,
            self.act_per_tensor_scale,
            self.is_swizzled_scales,
            self.use_triton_kernel,
            self.act_quant_kwargs,
            high_first=self.high_first,
        )

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize either packed-pair order to the logical weight."""
        if self.transposed:
            return self._transpose().dequantize(output_dtype).t()
        result = super().dequantize(output_dtype)
        if not self.high_first:
            return result
        pairs = result.reshape(*result.shape[:-1], result.shape[-1] // 2, 2)
        return pairs.flip(-1).reshape(result.shape)

    def get_hp_scales(self) -> torch.Tensor:
        """Recover canonical scales, with each expert's tile padding removed separately."""
        if self.transposed:
            return self._transpose().get_hp_scales()
        if self.ndim == 3 and self.is_swizzled_scales:
            scales = torch.stack(
                [
                    from_blocked(expert, self.shape[-2], self.shape[-1] // self.block_size)
                    for expert in self.scale
                ]
            ).float()
            return scales if self.per_tensor_scale is None else scales * self.per_tensor_scale
        return super().get_hp_scales()

    def _update_group_size(self) -> int:
        """Use the unrotated basis for ordinary NVFP4 updates."""
        return 0

    # Quantized updates require real coefficients; Tensor also accepts complex/symbolic scalars.
    def addmm_(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
        rounding_seed: int | None = None,
    ) -> Self:
        """Update and requantize in place, optionally using stochastic rounding.

        ``alpha`` and ``beta`` must be finite real numbers; integers are accepted.
        ``rounding_seed`` accepts the full unsigned 64-bit range. Supplying it
        makes terminal E2M1 code selection reproducible for a fixed device and
        backend without consuming the process-global random-number generator.
        """
        from piper_kernels.weights.nvfp4 import _update  # noqa: PLC0415

        return run_addmm_(
            self,
            mat1,
            mat2,
            operation=_update._operation(self, "addmm_"),
            apply=lambda left, right: _update.addmm_(
                self,
                self._update_group_size(),
                left,
                right,
                beta=beta,
                alpha=alpha,
                rounding_seed=rounding_seed,
            ),
        )

    # Tensor.add_ also types complex and None, which quantized updates reject.
    def add_(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: object,
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
        rounding_seed: int | None = None,
    ) -> Self:
        """Requantize ``beta * weight + alpha * other`` in place.

        ``alpha`` and ``beta`` must be finite real numbers; integers are accepted.
        ``beta`` is a Piper extension to Tensor.add_; its default preserves
        ordinary addition. Rounding follows the ``addmm_`` seed contract.
        """
        from piper_kernels.weights.nvfp4 import _update  # noqa: PLC0415

        return run_add_(
            self,
            other,
            operation=_update._operation(self, "add_"),
            apply=lambda update: _update.add_(
                self,
                self._update_group_size(),
                update,
                beta=beta,
                alpha=alpha,
                rounding_seed=rounding_seed,
            ),
        )

    def to(self, *args: object, **kwargs: object) -> Self:
        """Preserve explicit-copy semantics hidden by ``aten._to_copy``."""
        return to_quantized_weight(
            self,
            super().to,
            args,
            kwargs,
            format_name="NVFP4",
        )


register_inference_mode_dispatch(PiperNVFP4Tensor)
register_view_ops(PiperNVFP4Tensor)
register_matrix_ops(PiperNVFP4Tensor)
register_conversion_ops(PiperNVFP4Tensor, format_name="NVFP4")
register_update_ops(PiperNVFP4Tensor, format_name="NVFP4")


__all__ = ["PiperNVFP4Tensor"]


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
