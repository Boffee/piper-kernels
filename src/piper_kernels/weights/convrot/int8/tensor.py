"""Rotated INT8 W8A8 tensor subclass."""

from collections.abc import Callable
from typing import Any, ClassVar, Self

import torch
from torchao.utils import TorchAOBaseTensor

from piper_kernels.weights._dispatch import (
    register_conversion_ops,
    register_inference_mode_dispatch,
    to_quantized_weight,
)
from piper_kernels.weights._matmul import register_matrix_ops
from piper_kernels.weights._update import register_update_ops, run_add_, run_addmm_
from piper_kernels.weights._views import register_view_ops, require_untransposed
from piper_kernels.weights.convrot.int8._quantization import (
    dequantize_weight,
    quantize_weight,
    validate_activation_scale,
    validate_storage,
)


class ConvRotInt8Tensor(TorchAOBaseTensor):
    """INT8 rotated weight with per-output scale and logical floating dtype.

    Use :meth:`from_quantized` for existing quantized storage or :meth:`from_hp`
    to rotate and quantize a floating-point weight. Linear weights are 2-D;
    causal 3x3x3 convolution weights have logical shape ``[out, in, 3, 3, 3]``
    and packed shape ``[out, 3, 3, 3, in]``. Both use one scale per output.

    Convolution execution requires a finite positive FP32 scalar tensor in
    ``act_per_tensor_scale``, on the weight device. It is checkpoint storage,
    moved and serialized with the weight, as in ConvRot NVFP4. Linear execution
    uses this scale when present, or dynamic per-row scaling when it is None.
    Calibrate the input after any activation and rotation for the chosen group size.
    Weight conversion and dequantization alone do not need an activation scale.
    """

    tensor_data_names: ClassVar[list[str]] = ["qdata", "scale"]
    tensor_attribute_names: ClassVar[list[str]] = ["group_size", "dtype"]
    optional_tensor_data_names: ClassVar[list[str]] = ["act_per_tensor_scale"]
    act_per_tensor_scale: torch.Tensor | None = None
    optional_tensor_attribute_names: ClassVar[list[str]] = ["transposed"]
    transposed: bool = False

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
        act_per_tensor_scale: torch.Tensor | None = None,
        transposed: bool = False,
    ) -> "ConvRotInt8Tensor":
        validate_storage(qdata, scale, group_size, dtype)
        validate_activation_scale(act_per_tensor_scale, qdata.device)
        if type(transposed) is not bool:
            raise TypeError("ConvRot INT8 transposed must be bool")
        if qdata.ndim == 5 and transposed:
            raise NotImplementedError("ConvRot INT8 transpose requires a 2-D weight")
        shape = (
            (qdata.shape[0], qdata.shape[4], *qdata.shape[1:4]) if qdata.ndim == 5 else qdata.shape
        )
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape[::-1] if transposed else shape,
            strides=(1, max(1, qdata.shape[1])) if transposed else None,
            device=qdata.device,
            dtype=dtype,
            requires_grad=False,
        )

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
        act_per_tensor_scale: torch.Tensor | None = None,
        transposed: bool = False,
    ) -> None:
        super().__init__()
        if self.dtype is not dtype:
            raise RuntimeError(f"ConvRot wrapper dtype mismatch: {self.dtype} != {dtype}")
        self.qdata = qdata
        self.scale = scale
        self.group_size = group_size
        self.act_per_tensor_scale = act_per_tensor_scale
        self.transposed = transposed

    def _transpose(self) -> Self:
        """Reverse the logical axes while retaining canonical rowwise storage."""
        self._require_matrix("transpose")
        return type(self)(
            self.qdata,
            self.scale,
            self.group_size,
            self.dtype,
            self.act_per_tensor_scale,
            not self.transposed,
        )

    @classmethod
    def from_quantized(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        *,
        group_size: int,
        logical_dtype: torch.dtype = torch.bfloat16,
        act_per_tensor_scale: torch.Tensor | None = None,
    ) -> "ConvRotInt8Tensor":
        """Build a weight from quantized storage and canonicalize its layout.

        ``qdata`` contains the rotated INT8 weight and ``scale`` contains one
        float32 value per output channel. A flat scale or an ``[out, 1]``
        scale is accepted. ``logical_dtype`` controls the wrapper's floating
        dtype and default dequantization dtype. Convolution qdata has shape
        ``[out, 3, 3, 3, in]``. Contiguous checkpoint storage is reused without
        copying; noncontiguous storage is canonicalized. The optional static
        activation scale is retained directly and must be finite and positive.
        """
        if qdata.ndim in (2, 5):
            out_features = qdata.shape[0]
            valid_scale_shapes = ((out_features,), (out_features, 1))
            if tuple(scale.shape) not in valid_scale_shapes:
                raise ValueError(
                    "ConvRot INT8 from_quantized scale must have shape "
                    f"({out_features},) or ({out_features}, 1), got {tuple(scale.shape)}"
                )
        return cls(
            qdata.contiguous(),
            (scale.view(-1, 1) if scale.ndim == 1 else scale).contiguous(),
            group_size,
            logical_dtype,
            act_per_tensor_scale,
        )

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        *,
        group_size: int,
        act_per_tensor_scale: torch.Tensor | None = None,
    ) -> "ConvRotInt8Tensor":
        """Rotate and quantize a high-precision weight into ConvRot INT8 storage."""
        source = hp_tensor.detach()
        qdata, scale = quantize_weight(source, group_size)
        return cls(qdata, scale, group_size, source.dtype, act_per_tensor_scale)

    @classmethod
    def from_gguf(
        cls,
        data: torch.Tensor,
        *,
        quant_type: int | None = None,
        group_size: int,
        logical_dtype: torch.dtype = torch.bfloat16,
    ) -> "ConvRotInt8Tensor":
        """Decode packed GGUF storage directly into ConvRot INT8 storage.

        ``quant_type`` may be omitted when ``data`` exposes a ``quant_type``
        attribute, as piper-offload's GGUF tensor wrapper does. Triton conversion
        combines GGUF decoding, grouped rotation, and rowwise INT8 quantization
        without allocating a dense weight, including on ROCm.
        """
        from piper_kernels.weights.convrot.int8._gguf import convert  # noqa: PLC0415

        qdata, scale = convert(
            data,
            quant_type=quant_type,
            group_size=group_size,
            logical_dtype=logical_dtype,
        )
        return cls(qdata, scale, group_size, logical_dtype)

    def copy_from_gguf_(
        self,
        data: torch.Tensor,
        *,
        quant_type: int | None = None,
    ) -> "ConvRotInt8Tensor":
        """Refill this tensor from compatible packed GGUF storage in place."""
        self._require_matrix("copy_from_gguf_")
        require_untransposed(self, "copy_from_gguf_")
        from piper_kernels.weights.convrot.int8._gguf import convert  # noqa: PLC0415

        convert(
            data,
            quant_type=quant_type,
            group_size=self.group_size,
            logical_dtype=self.dtype,
            out=(self.qdata, self.scale),
        )
        return self

    @classmethod
    def from_trellis(
        cls,
        bitstreams: torch.Tensor,
        block_scales: torch.Tensor,
        scale: torch.Tensor,
        *,
        block_size: int,
        group_size: int,
        logical_dtype: torch.dtype = torch.bfloat16,
    ) -> "ConvRotInt8Tensor":
        """Decode trellis-coded storage directly into ConvRot INT8 storage.

        Each row is stored as ``in / T`` blocks of ``T = block_size``
        weights (16 to 256). ``bitstreams`` is uint8 ``[out, in / T, bytes]``, where
        the bytes per block set the rate: 3 to 6 bits per weight.
        ``block_scales`` is FP8 E4M3 ``[out, in / T]``, and ``scale`` is the FP32
        row scale, ``[out]`` or ``[out, 1]``. The streams already hold the weight
        rotated for ``group_size``, so decoding produces its INT8 values without
        rotating; contiguous ``scale`` storage is reused. Decoding is exact by
        construction, so every backend produces the same INT8 values. It requires a
        supported Triton accelerator and raises on other devices; it is validated on
        NVIDIA, and ROCm has compile coverage only.
        """
        from piper_kernels.weights.convrot.int8._trellis import convert  # noqa: PLC0415

        qdata, row_scale = convert(bitstreams, block_scales, scale, block_size=block_size)
        return cls(qdata, row_scale, group_size, logical_dtype)

    def copy_from_trellis_(
        self,
        bitstreams: torch.Tensor,
        block_scales: torch.Tensor,
        scale: torch.Tensor,
    ) -> "ConvRotInt8Tensor":
        """Refill this tensor in place from compatible trellis-coded storage.

        Arguments are as for :meth:`from_trellis`; the block size is this
        weight's input width over the blocks per row. The streams must encode
        this weight's shape in the basis of its group size, at any supported rate.
        """
        self._require_matrix("copy_from_trellis_")
        require_untransposed(self, "copy_from_trellis_")
        from piper_kernels.weights.convrot.int8._trellis import convert  # noqa: PLC0415

        convert(bitstreams, block_scales, scale, out=(self.qdata, self.scale))
        return self

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Recover the logical weight in the unrotated basis and requested dtype."""
        validate_storage(self.qdata, self.scale, self.group_size, self.dtype)
        if output_dtype is None:
            output_dtype = self.dtype
        result = dequantize_weight(self.qdata, self.scale, self.group_size, output_dtype)
        return result.t() if self.transposed else result

    def _require_matrix(self, operation: str) -> None:
        if self.ndim != 2:
            raise NotImplementedError(f"ConvRot INT8 {operation} requires a 2-D weight")

    def _validate_memory_format(self, memory_format: object) -> None:
        if memory_format not in (None, torch.preserve_format):
            require_untransposed(self, "to with a different memory format")
            if memory_format is not torch.contiguous_format:
                raise NotImplementedError("ConvRot INT8 only supports contiguous memory format")

    # Quantized updates require real coefficients; Tensor also accepts complex/symbolic scalars.
    def addmm_(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
        rounding_seed: int | None = None,
    ) -> "ConvRotInt8Tensor":
        """Update and requantize in place, optionally using stochastic rounding.

        ``alpha`` and ``beta`` must be finite real numbers; integers are accepted.
        Factors may share storage with ``qdata`` or ``scale``; the update uses
        their values from before the mutation.

        ``rounding_seed`` accepts the full unsigned 64-bit range. Supplying it
        makes terminal INT8 code selection reproducible for a fixed device and
        backend without consuming the process-global random-number generator.
        """
        self._require_matrix("addmm_")
        from piper_kernels.weights.convrot.int8 import _update  # noqa: PLC0415

        return run_addmm_(
            self,
            mat1,
            mat2,
            operation="ConvRot addmm_",
            apply=lambda left, right: _update.addmm_(
                self.qdata,
                self.scale,
                self.dtype,
                self.group_size,
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
    ) -> "ConvRotInt8Tensor":
        """Requantize ``beta * weight + alpha * other`` in place.

        ``alpha`` and ``beta`` must be finite real numbers; integers are accepted.
        ``beta`` is a Piper extension to Tensor.add_; its default preserves
        ordinary addition. Rounding follows the ``addmm_`` seed contract.
        """
        self._require_matrix("add_")
        from piper_kernels.weights.convrot.int8 import _update  # noqa: PLC0415

        return run_add_(
            self,
            other,
            operation="ConvRot add_",
            apply=lambda update: _update.add_(
                self.qdata,
                self.scale,
                self.dtype,
                self.group_size,
                update,
                beta=beta,
                alpha=alpha,
                rounding_seed=rounding_seed,
            ),
        )

    def _stable_hash_for_caching(self) -> str:
        """Return a metadata fingerprint for AOTAutograd's cross-process cache."""
        if self.act_per_tensor_scale is not None:
            # AOT's wrapper cache key omits aliases between inner tensors of different
            # weights. Reusing a graph traced with a shared input scale can silently
            # substitute that scale for an independent one. Keep Dynamo/Inductor
            # compilation and runtime graph reuse, but bypass this unsafe outer cache.
            from torch._functorch._aot_autograd.autograd_cache import (  # noqa: PLC0415
                BypassAOTAutogradCache,
            )

            raise BypassAOTAutogradCache(
                "ConvRot INT8 static input scales may alias across weights"
            )
        return repr(
            (
                type(self).__qualname__,
                tuple(self.shape),
                self.stride(),
                str(self.device),
                str(self.dtype),
                self.group_size,
                self.transposed,
                tuple(self.qdata.shape),
                self.qdata.stride(),
                tuple(self.scale.shape),
                self.scale.stride(),
            )
        )

    def _rebuild_with_logical_dtype(self, dtype: torch.dtype) -> Self:
        """Rebuild the semantic wrapper without converting quantized storage."""
        return type(self)(
            self.qdata,
            self.scale,
            self.group_size,
            dtype,
            self.act_per_tensor_scale,
            self.transposed,
        )

    def to(self, *args: object, **kwargs: object) -> Self:
        """Preserve explicit-copy semantics hidden by ``aten._to_copy``."""
        return to_quantized_weight(
            self,
            super().to,
            args,
            kwargs,
            format_name="ConvRot INT8",
        )


register_inference_mode_dispatch(ConvRotInt8Tensor)
register_view_ops(ConvRotInt8Tensor)
register_matrix_ops(ConvRotInt8Tensor)
register_conversion_ops(ConvRotInt8Tensor, format_name="ConvRot INT8")
register_update_ops(ConvRotInt8Tensor, format_name="ConvRot")


@ConvRotInt8Tensor.implements(torch.ops.aten.embedding.default)
def _embedding_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> torch.Tensor:
    """Dequantize only the looked-up rows, never the whole table.

    The remaining arguments (``padding_idx``, ``scale_grad_by_freq``, ``sparse``) affect
    only gradients, which frozen quantized weights do not have.
    """
    weight, indices = args[0], args[1]
    assert isinstance(weight, ConvRotInt8Tensor)
    weight._require_matrix("embedding")
    require_untransposed(weight, "embedding")
    rows = indices.reshape(-1)
    values = dequantize_weight(
        weight.qdata.index_select(0, rows),
        weight.scale.index_select(0, rows),
        weight.group_size,
        weight.dtype,
    )
    return values.view(*indices.shape, weight.shape[1])


@ConvRotInt8Tensor.implements(torch.ops.aten.linear.default)
@ConvRotInt8Tensor.implements_torch_function(torch.nn.functional.linear)
def _linear_dispatch(
    func: Callable[..., torch.Tensor],
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    from piper_kernels.linear.convrot.int8._functional import linear_dispatch  # noqa: PLC0415

    return linear_dispatch(func, types, args, kwargs)


@ConvRotInt8Tensor.implements(torch.ops.aten.conv3d.default)
@ConvRotInt8Tensor.implements(torch.ops.aten.conv3d.padding)
@ConvRotInt8Tensor.implements(torch.ops.aten.convolution.default)
@ConvRotInt8Tensor.implements(torch.ops.aten._convolution.default)
@ConvRotInt8Tensor.implements_torch_function(torch.nn.functional.conv3d)
def _conv3d_dispatch(
    func: Callable[..., torch.Tensor],
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    from piper_kernels.conv3d.convrot.int8._functional import conv3d_dispatch  # noqa: PLC0415

    return conv3d_dispatch(func, types, args, kwargs)
