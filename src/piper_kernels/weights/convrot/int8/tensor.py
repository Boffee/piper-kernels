"""Rotated INT8 W8A8 tensor subclass."""

from collections.abc import Callable
from typing import Any, ClassVar, Self, cast

import torch
from torch.types import Number
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import TorchAOBaseTensor

from piper_kernels.weights._dispatch import _explicit_to_copy_args
from piper_kernels.weights._matmul import register_matrix_ops
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
    currently uses dynamic activation scales and requires this field to be None.
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
        if qdata.ndim == 2 and act_per_tensor_scale is not None:
            raise NotImplementedError(
                "ConvRot INT8 static activation scaling currently requires a Conv3D weight"
            )
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
        if self.act_per_tensor_scale is not None:
            raise NotImplementedError(
                "ConvRot INT8 static activation scaling currently requires a Conv3D weight"
            )

    def _validate_memory_format(self, memory_format: object) -> None:
        if memory_format not in (None, torch.preserve_format):
            require_untransposed(self, "to with a different memory format")
            if memory_format is not torch.contiguous_format:
                raise NotImplementedError("ConvRot INT8 only supports contiguous memory format")

    # Quantized updates require concrete scalars; Tensor also types symbolic scalars.
    def addmm_(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta: int | float | complex = 1,
        alpha: int | float | complex = 1,
        rounding_seed: int | None = None,
    ) -> "ConvRotInt8Tensor":
        """Update and requantize in place, optionally using stochastic rounding.

        ``rounding_seed`` accepts the full unsigned 64-bit range. Supplying it
        makes terminal INT8 code selection reproducible for a fixed device and
        backend without consuming the process-global random-number generator.
        """
        self._require_matrix("addmm_")
        require_untransposed(self, "addmm_")
        if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
            raise TypeError("ConvRot addmm_ matrices must be tensors")
        from piper_kernels.weights.convrot.int8 import _update  # noqa: PLC0415

        _update.addmm_(
            self.qdata,
            self.scale,
            self.dtype,
            self.group_size,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
            rounding_seed=rounding_seed,
        )
        return self

    def add_(
        self,
        other: object,
        *,
        alpha: Number | complex | None = 1,
        rounding_seed: int | None = None,
    ) -> "ConvRotInt8Tensor":
        """Add a dense logical update and requantize in place."""
        self._require_matrix("add_")
        require_untransposed(self, "add_")
        if not isinstance(other, torch.Tensor):
            raise TypeError("ConvRot add_ update must be a tensor")
        if alpha is None:
            raise TypeError("ConvRot add_ alpha must be a real number, got None")
        from piper_kernels.weights.convrot.int8 import _update  # noqa: PLC0415

        _update.add_(
            self.qdata,
            self.scale,
            self.dtype,
            self.group_size,
            other,
            alpha=alpha,
            rounding_seed=rounding_seed,
        )
        return self

    def _stable_hash_for_caching(self) -> str:
        """Return a metadata fingerprint for AOTAutograd's cross-process cache."""
        return repr(
            (
                type(self).__qualname__,
                tuple(self.shape),
                self.stride(),
                str(self.device),
                str(self.dtype),
                self.group_size,
                self.transposed,
                self.act_per_tensor_scale is not None,
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
        self._validate_memory_format(kwargs.get("memory_format"))
        explicit_copy_args = _explicit_to_copy_args(args, kwargs)
        if explicit_copy_args is None:
            return cast(
                Self,
                super().to(*args, **kwargs),  # pyright: ignore[reportCallIssue, reportArgumentType]
            )
        to_args, to_kwargs = explicit_copy_args
        options = self._get_to_kwargs(*to_args, **to_kwargs)
        dtype = options["dtype"]
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"ConvRot INT8 logical dtype must be floating point, got {dtype}")
        device = options["device"]
        non_blocking = options["non_blocking"]
        copied = cast(
            Self,
            self._apply_fn_to_data(
                lambda value: value.to(
                    device=device,
                    non_blocking=non_blocking,
                    copy=True,
                )
            ),
        )
        if dtype is not copied.dtype:
            copied = copied._rebuild_with_logical_dtype(dtype)
        return copied


register_view_ops(ConvRotInt8Tensor)
register_matrix_ops(ConvRotInt8Tensor)


@ConvRotInt8Tensor.implements(torch.ops.aten._to_copy.default)
def _convrot_int8_to_copy(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ConvRotInt8Tensor:
    """Preserve quantized storage across logical dtype and device conversion."""
    tensor = args[0]
    assert isinstance(tensor, ConvRotInt8Tensor)
    arguments = dict(kwargs)
    dtype = arguments.pop("dtype", tensor.dtype)
    device = arguments.pop("device", tensor.device)
    non_blocking = arguments.pop("non_blocking", False)
    copy = arguments.pop("copy", False)
    memory_format = arguments.pop("memory_format", None)
    tensor._validate_memory_format(memory_format)
    layout = arguments.pop("layout", None)
    pin_memory = arguments.pop("pin_memory", None)
    if arguments:
        raise NotImplementedError(f"unsupported ConvRot INT8 conversion arguments: {arguments}")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"ConvRot INT8 logical dtype must be floating point, got {dtype}")
    metadata_only = (
        torch.device(device) == tensor.device
        and dtype is not tensor.dtype
        and not copy
        and memory_format in (None, torch.preserve_format)
        and layout in (None, tensor.layout)
        and pin_memory is None
    )
    moved = (
        tensor
        if metadata_only
        else cast(
            ConvRotInt8Tensor,
            tensor._apply_fn_to_data(
                lambda value: func(value, device=device, non_blocking=non_blocking)
            ),
        )
    )
    if dtype is not moved.dtype:
        moved = moved._rebuild_with_logical_dtype(dtype)
    return cast(
        ConvRotInt8Tensor,
        return_and_correct_aliasing(func, args, kwargs, moved),
    )


@ConvRotInt8Tensor.implements(torch.ops.aten.addmm_.default)
def _convrot_addmm_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ConvRotInt8Tensor:
    weight, mat1, mat2 = args
    if not isinstance(weight, ConvRotInt8Tensor):
        raise TypeError(f"ConvRot addmm_ weight must be ConvRotInt8Tensor, got {type(weight)}")
    return weight.addmm_(mat1, mat2, beta=kwargs.get("beta", 1), alpha=kwargs.get("alpha", 1))


@ConvRotInt8Tensor.implements(torch.ops.aten.add_.Tensor)
def _convrot_add_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ConvRotInt8Tensor:
    weight, update = args
    if not isinstance(weight, ConvRotInt8Tensor):
        raise TypeError(f"ConvRot add_ weight must be ConvRotInt8Tensor, got {type(weight)}")
    return weight.add_(update, alpha=kwargs.get("alpha", 1))


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
