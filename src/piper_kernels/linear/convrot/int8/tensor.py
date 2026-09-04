"""Rotated INT8 W8A8 tensor subclass."""

from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch.types import Number
from torchao.utils import TorchAOBaseTensor

from piper_kernels.linear._dispatch import bind_linear_arguments
from piper_kernels.linear._input_activations import InputActivation

from .._rotation import rotate_groups
from . import _update, dispatch
from .reference import quantize_weight, validate_storage


class ConvRotInt8Tensor(TorchAOBaseTensor):
    """INT8 rotated weight with per-output scale and logical floating dtype.

    Use :meth:`from_quantized` for existing quantized storage or :meth:`from_hp`
    to rotate and quantize a floating-point weight.
    """

    tensor_data_names: ClassVar[list[str]] = ["qdata", "scale"]
    tensor_attribute_names: ClassVar[list[str]] = ["group_size", "dtype"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ConvRotInt8Tensor":
        validate_storage(qdata, scale, group_size, dtype)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            qdata.shape,
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
    ) -> None:
        super().__init__()
        if self.dtype is not dtype:
            raise RuntimeError(f"ConvRot wrapper dtype mismatch: {self.dtype} != {dtype}")
        self.qdata = qdata
        self.scale = scale
        self.group_size = group_size

    @classmethod
    def from_quantized(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        *,
        group_size: int,
        logical_dtype: torch.dtype = torch.bfloat16,
    ) -> "ConvRotInt8Tensor":
        """Build a weight from quantized storage and canonicalize its layout.

        ``qdata`` contains the rotated INT8 weight and ``scale`` contains one
        float32 value per output row. A flat scale or an ``[out_features, 1]``
        scale is accepted. ``logical_dtype`` controls the wrapper's floating
        dtype and the dtype expected by ConvRot linear operations.
        """
        if qdata.ndim == 2:
            out_features = qdata.shape[0]
            valid_scale_shapes = ((out_features,), (out_features, 1))
            if tuple(scale.shape) not in valid_scale_shapes:
                raise ValueError(
                    "ConvRot INT8 from_quantized scale must have shape "
                    f"({out_features},) or ({out_features}, 1), got {tuple(scale.shape)}"
                )
        return cls(
            qdata.contiguous(),
            scale.reshape(-1, 1).contiguous(),
            group_size,
            logical_dtype,
        )

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        *,
        group_size: int,
    ) -> "ConvRotInt8Tensor":
        """Rotate and quantize a high-precision weight into ConvRot INT8 storage."""
        source = hp_tensor.detach()
        qdata, scale = quantize_weight(source, group_size)
        return cls(qdata.contiguous(), scale.contiguous(), group_size, source.dtype)

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
        attribute, as piper-offload's GGUF tensor wrapper does. CUDA conversion
        fuses GGUF decoding, grouped rotation, and rowwise INT8 quantization
        without allocating a dense weight.
        """
        from ._gguf import convert  # noqa: PLC0415

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
        from ._gguf import convert  # noqa: PLC0415

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
        rotated = self.qdata.to(output_dtype) * self.scale.to(output_dtype)
        return rotate_groups(rotated, self.group_size)

    def addmm_(
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
        if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
            raise TypeError("ConvRot addmm_ matrices must be tensors")
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
        if not isinstance(other, torch.Tensor):
            raise TypeError("ConvRot add_ update must be a tensor")
        if alpha is None:
            raise TypeError("ConvRot add_ alpha must be a real number, got None")
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
                tuple(self.qdata.shape),
                self.qdata.stride(),
                tuple(self.scale.shape),
                self.scale.stride(),
            )
        )


def convrot_int8_linear(
    input: torch.Tensor,  # noqa: A002
    weight: ConvRotInt8Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation_fn: InputActivation | None = None,
) -> torch.Tensor:
    """Apply an optional input activation followed by a ConvRot INT8 linear."""
    return dispatch.linear(
        input,
        weight.qdata,
        weight.scale,
        weight.dtype,
        weight.group_size,
        bias,
        activation_fn=activation_fn,
    )


@ConvRotInt8Tensor.implements(torch.ops.aten.linear.default)
@ConvRotInt8Tensor.implements_torch_function(torch.nn.functional.linear)
def _convrot_linear_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    linear_input, weight, bias = bind_linear_arguments(args, kwargs)
    if not isinstance(linear_input, torch.Tensor) or not isinstance(weight, ConvRotInt8Tensor):
        raise TypeError(
            "ConvRot linear dispatch requires a tensor input and ConvRotInt8Tensor weight"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(f"ConvRot linear bias must be a tensor or None, got {type(bias).__name__}")
    return dispatch.linear(
        linear_input,
        weight.qdata,
        weight.scale,
        weight.dtype,
        weight.group_size,
        bias,
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
    if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
        raise TypeError("ConvRot addmm_ matrices must be tensors")
    _update.addmm_(
        weight.qdata,
        weight.scale,
        weight.dtype,
        weight.group_size,
        mat1,
        mat2,
        beta=kwargs.get("beta", 1),
        alpha=kwargs.get("alpha", 1),
    )
    return weight


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
    if not isinstance(update, torch.Tensor):
        raise TypeError("ConvRot add_ update must be a tensor")
    _update.add_(
        weight.qdata,
        weight.scale,
        weight.dtype,
        weight.group_size,
        update,
        alpha=kwargs.get("alpha", 1),
    )
    return weight
