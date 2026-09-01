"""TorchAO-compatible NVFP4 tensor with a stable semantic linear operator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_linear as torchao_nvfp4_linear
from torchao.utils import TorchAOBaseTensor

from piper_kernels.linear._dispatch import bind_linear_arguments

from . import _layout, _ops
from ._typing import NVFP4Storage

# TorchAO divides the global reciprocal by the FP8 block scale. Keep that
# intermediate finite even when the weight is zero or extremely small.
_MIN_PER_TENSOR_SCALE = torch.finfo(torch.float32).tiny / torch.finfo(torch.float8_e4m3fn).tiny


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
        per_tensor_scale = per_tensor_amax_to_scale(amax).clamp_min(_MIN_PER_TENSOR_SCALE)
    return TorchAONVFP4Tensor.to_nvfp4(
        source,
        block_size=block_size,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=is_swizzled_scales,
        use_triton_kernel=use_triton_kernel,
        act_quant_kwargs=act_quant_kwargs,
    )


class PiperNVFP4Tensor(TorchAONVFP4Tensor):
    """TorchAO NVFP4 storage whose standard W4A4 linears remain semantic in FX."""

    __torch_function__ = classmethod(TorchAOBaseTensor.__torch_function__.__func__)
    qdata: torch.Tensor
    scale: torch.Tensor
    block_size: int
    orig_dtype: torch.dtype
    per_tensor_scale: torch.Tensor | None
    act_per_tensor_scale: torch.Tensor | None
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: QuantizeTensorToNVFP4Kwargs | None

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
        builder. ``compute_per_tensor_scale=True`` derives the optional global
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
                None if self.per_tensor_scale is None else tuple(self.per_tensor_scale.shape),
                None
                if self.act_per_tensor_scale is None
                else tuple(self.act_per_tensor_scale.shape),
            )
        )

    def _rebuild_with_orig_dtype(self, orig_dtype: torch.dtype) -> PiperNVFP4Tensor:
        """Rebuild the concrete semantic wrapper while preserving its storage."""
        return type(self)(
            self.qdata,
            self.scale,
            self.block_size,
            orig_dtype,
            self.per_tensor_scale,
            self.act_per_tensor_scale,
            self.is_swizzled_scales,
            self.use_triton_kernel,
            self.act_quant_kwargs,
        )


@PiperNVFP4Tensor.implements(torch.ops.aten._to_copy.default)
def _nvfp4_to_copy(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> PiperNVFP4Tensor:
    """Preserve the semantic wrapper across autocast and device movement."""
    tensor = args[0]
    assert isinstance(tensor, PiperNVFP4Tensor)
    arguments = dict(kwargs)
    dtype = arguments.pop("dtype", tensor.orig_dtype)
    device = arguments.pop("device", tensor.device)
    non_blocking = arguments.pop("non_blocking", False)
    arguments.pop("copy", None)
    arguments.pop("memory_format", None)
    arguments.pop("layout", None)
    arguments.pop("pin_memory", None)
    if arguments:
        raise NotImplementedError(f"unsupported NVFP4 conversion arguments: {arguments}")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"NVFP4 logical dtype must be floating point, got {dtype}")
    moved = cast(
        PiperNVFP4Tensor,
        tensor._apply_fn_to_data(
            lambda value: func(value, device=device, non_blocking=non_blocking)
        ),
    )
    if dtype is not moved.orig_dtype:
        moved = moved._rebuild_with_orig_dtype(dtype)
    return cast(
        PiperNVFP4Tensor,
        return_and_correct_aliasing(func, args, kwargs, moved),
    )


def supports_semantic_linear(input: object, weight: PiperNVFP4Tensor) -> bool:  # noqa: A002
    if not isinstance(input, torch.Tensor):
        return False
    tensor_input = cast(torch.Tensor, input)
    if isinstance(input, TorchAONVFP4Tensor):
        return False
    quantization = weight.act_quant_kwargs
    return (
        tensor_input.ndim > 0
        and tensor_input.dtype is weight.orig_dtype
        and tensor_input.device.type in ("cuda", "meta")
        and weight.block_size == _layout.BLOCK_SIZE
        and weight.is_swizzled_scales
        and not weight.use_triton_kernel
        and quantization is not None
        and quantization.block_size == _layout.BLOCK_SIZE
        and quantization.is_swizzled_scales
        and not quantization.use_triton_kernel
        and (quantization.use_dynamic_per_tensor_scale or weight.act_per_tensor_scale is not None)
        and (weight.per_tensor_scale is None or weight.per_tensor_scale.ndim == 0)
    )


@PiperNVFP4Tensor.implements(torch.ops.aten.linear.default)
@PiperNVFP4Tensor.implements_torch_function(torch.nn.functional.linear)
def _nvfp4_linear_dispatch(
    func: Callable[..., torch.Tensor],
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    input, weight, bias = bind_linear_arguments(args, kwargs)  # noqa: A001
    if (
        not torch.compiler.is_compiling()
        or not isinstance(weight, PiperNVFP4Tensor)
        or not supports_semantic_linear(input, weight)
    ):
        return torchao_nvfp4_linear(func, types, args, kwargs)
    if bias is not None and not isinstance(bias, torch.Tensor):
        return torchao_nvfp4_linear(func, types, args, kwargs)
    assert isinstance(input, torch.Tensor)
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
    )


__all__ = ["PiperNVFP4Tensor"]
