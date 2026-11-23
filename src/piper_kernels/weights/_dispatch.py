"""Shared dispatch helpers for quantized weight wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar, cast

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import TorchAOBaseTensor, _dispatch__torch_dispatch__

if TYPE_CHECKING:
    from torch._ops import OpOverload
    from torch.distributed.device_mesh import DeviceMesh

    from .convrot.int8.tensor import ConvRotInt8Tensor
    from .nvfp4.tensor import PiperNVFP4Tensor
type QuantizedWeight = ConvRotInt8Tensor | PiperNVFP4Tensor
_WeightT = TypeVar("_WeightT", bound="QuantizedWeight")


def require_untransposed(weight: QuantizedWeight, operation: str) -> None:
    """Reject operations that would interpret a transpose as canonical storage."""
    if weight.transposed:
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} does not support a transposed weight"
        )


def _torch_dispatch(
    cls: type[TorchAOBaseTensor],
    func: OpOverload,
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object:
    # object: an op returns whatever its schema declares, not only tensors.
    if func not in cls._ATEN_OP_TABLE[cls]:
        decomposed = func.decompose(*args, **kwargs)
        if decomposed is not NotImplemented:
            return decomposed
    elif func.is_view and torch.is_inference_mode_enabled() and not args[0].is_inference():
        with torch.inference_mode(False):
            return _dispatch__torch_dispatch__(cls, func, types, args, kwargs)
    return _dispatch__torch_dispatch__(cls, func, types, args, kwargs)


def _decompose_dispatch(
    func: OpOverload,
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object:
    return func.decompose(*args, **kwargs)


def register_inference_mode_dispatch(cls: type[TorchAOBaseTensor]) -> None:
    """Dispatch the same way under ``torch.inference_mode`` as under ``torch.no_grad``.

    Autograd dispatch runs CompositeImplicitAutograd decompositions, such as ``aten.to``
    returning ``self`` or lowering to ``_to_copy``, before an op reaches a wrapper.
    Inference mode skips autograd dispatch, so those ops arrive intact; running their
    decompositions here reaches the implementations every other grad mode reaches.
    Registered implementations, such as ``linear``, keep precedence.

    Inference mode still tracks views of a normal tensor, one created outside it, and
    that tracking requires the view to be a normal tensor too, as ATen's views are.
    A wrapper constructed in inference mode is an inference tensor, so registered view
    implementations of a normal weight run with inference mode disabled.
    """
    cls.__torch_dispatch__ = classmethod(_torch_dispatch)
    # Only inference mode reaches TorchAO's handler, which diverges from the
    # decomposition: it copies for an unindexed same device and rejects non_blocking.
    cls.implements(torch.ops.aten.to.dtype_layout)(_decompose_dispatch)


def unsupported_operation_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    _args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> torch.Tensor:
    """Reject operations without an implementation that preserves quantized weights."""
    raise NotImplementedError(f"Piper quantized weights do not support {func}")


def _explicit_to_copy_args(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[tuple[object, ...], dict[str, object]] | None:
    """Remove an explicit true ``copy`` before TorchAO parses ``Tensor.to``."""
    parsed_args = args
    parsed_kwargs = dict(kwargs)
    if "copy" in parsed_kwargs:
        copy = parsed_kwargs.pop("copy")
    else:
        copy_index = 2 if args and isinstance(args[0], (torch.Tensor, torch.dtype)) else 3
        if len(args) <= copy_index:
            return None
        copy = args[copy_index]
        parsed_args = (*args[:copy_index], *args[copy_index + 1 :])
    if copy is not True:
        return None
    return parsed_args, parsed_kwargs


def to_quantized_weight(
    tensor: _WeightT,
    super_to: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    format_name: str,
) -> _WeightT:
    """Preserve explicit-copy semantics hidden by ``aten._to_copy``."""
    tensor._validate_memory_format(kwargs.get("memory_format"))
    explicit_copy_args = _explicit_to_copy_args(args, kwargs)
    if explicit_copy_args is None:
        return cast(_WeightT, super_to(*args, **kwargs))
    to_args, to_kwargs = explicit_copy_args
    options = tensor._get_to_kwargs(*to_args, **to_kwargs)
    dtype = options["dtype"]
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{format_name} logical dtype must be floating point, got {dtype}")
    device = options["device"]
    non_blocking = options["non_blocking"]
    copied = cast(
        _WeightT,
        tensor._apply_fn_to_data(
            lambda value: value.to(
                device=device,
                non_blocking=non_blocking,
                copy=True,
            )
        ),
    )
    if dtype is not copied.dtype:
        copied = cast(_WeightT, copied._rebuild_with_logical_dtype(dtype))
    return copied


def register_conversion_ops(cls: type[TorchAOBaseTensor], *, format_name: str) -> None:
    """Preserve quantized storage across logical dtype and device conversion."""

    def _to_copy(
        func: Callable[..., torch.Tensor],
        _types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        tensor = cast("QuantizedWeight", args[0])
        assert isinstance(tensor, cls)
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
            raise NotImplementedError(
                f"unsupported {format_name} conversion arguments: {arguments}"
            )
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"{format_name} logical dtype must be floating point, got {dtype}")
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
                "QuantizedWeight",
                tensor._apply_fn_to_data(
                    lambda value: func(value, device=device, non_blocking=non_blocking)
                ),
            )
        )
        if dtype is not moved.dtype:
            moved = moved._rebuild_with_logical_dtype(dtype)
        return cast(
            torch.Tensor,
            return_and_correct_aliasing(func, args, kwargs, moved),
        )

    cls.implements(torch.ops.aten._to_copy.default)(_to_copy)


class QuantizedWeightMixin:
    """Shared Tensor subclass conversion, cache-key, and in-place update scaffolding."""

    _format_name: ClassVar[str]
    _update_format_name: ClassVar[str]
    _contiguous_memory_format_only: ClassVar[bool] = False

    def _validate_memory_format(self, memory_format: object) -> None:
        if memory_format not in (None, torch.preserve_format):
            require_untransposed(cast("QuantizedWeight", self), "to with a different memory format")
            if self._contiguous_memory_format_only and memory_format is not torch.contiguous_format:
                raise NotImplementedError(
                    f"{self._format_name} only supports contiguous memory format"
                )

    def _rebuild_with_logical_dtype(self, dtype: torch.dtype) -> Self:
        """Rebuild the concrete semantic wrapper while preserving quantized storage."""
        names, attributes = cast(TorchAOBaseTensor, self).__tensor_flatten__()
        tensors = {name: getattr(self, name) for name in names}
        updated = dict(attributes)
        updated["orig_dtype" if "orig_dtype" in updated else "dtype"] = dtype
        return cast(
            Self,
            cast(Any, type(self)).__tensor_unflatten__(tensors, updated, None, None),
        )

    def _stable_cache_key(self, *extra: object) -> str:
        """Build an AOTAutograd cache fingerprint from common metadata and format fields."""
        weight = cast("QuantizedWeight", self)
        return repr(
            (
                type(self).__qualname__,
                tuple(weight.shape),
                weight.stride(),
                str(weight.device),
                str(weight.dtype),
                tuple(weight.qdata.shape),
                weight.qdata.stride(),
                tuple(weight.scale.shape),
                weight.scale.stride(),
                *extra,
            )
        )

    def _update_target(self, _operation: str) -> tuple[Any, tuple[object, ...]]:
        raise NotImplementedError

    def addmm_(
        self,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
        rounding_seed: int | None = None,
    ) -> Self:
        """Update and requantize in place; ``alpha``/``beta`` finite real, optional uint64 seed."""
        from piper_kernels.weights._update import run_addmm_  # noqa: PLC0415

        target, prefix = self._update_target("addmm_")
        return cast(
            Self,
            run_addmm_(
                cast("QuantizedWeight", self),
                mat1,
                mat2,
                operation=f"{self._update_format_name} addmm_",
                apply=lambda left, right: target.addmm_(
                    *prefix,
                    left,
                    right,
                    beta=beta,
                    alpha=alpha,
                    rounding_seed=rounding_seed,
                ),
            ),
        )

    def add_(
        self,
        other: object,
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
        rounding_seed: int | None = None,
    ) -> Self:
        """Requantize ``beta * weight + alpha * other``; finite real scalars, uint64 seed."""
        from piper_kernels.weights._update import run_add_  # noqa: PLC0415

        target, prefix = self._update_target("add_")
        return cast(
            Self,
            run_add_(
                cast("QuantizedWeight", self),
                other,
                operation=f"{self._update_format_name} add_",
                apply=lambda update: target.add_(
                    *prefix,
                    update,
                    beta=beta,
                    alpha=alpha,
                    rounding_seed=rounding_seed,
                ),
            ),
        )

    def to(self, *args: object, **kwargs: object) -> Self:
        """Preserve explicit-copy semantics hidden by ``aten._to_copy``."""
        return cast(
            Self,
            to_quantized_weight(
                cast("QuantizedWeight", self),
                cast(Any, super()).to,
                args,
                kwargs,
                format_name=self._format_name,
            ),
        )

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
    ) -> tuple[tuple[torch.Tensor, ...], object]:
        """Expose sharded storage tensors and metadata for PyTorch FSDP2."""
        from piper_kernels.weights._fsdp import fsdp_pre_all_gather  # noqa: PLC0415

        return fsdp_pre_all_gather(cast("QuantizedWeight", self), mesh)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: object,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
        """Reconstruct or refresh an unsharded weight from FSDP2 all-gather outputs."""
        from piper_kernels.weights._fsdp import fsdp_post_all_gather  # noqa: PLC0415

        return fsdp_post_all_gather(
            cast("QuantizedWeight", self),
            all_gather_outputs,
            metadata,
            param_dtype,
            out=out,
        )


def register_quantized_weight_ops(cls: type[TorchAOBaseTensor]) -> None:
    """Register all standard dispatch handlers for a concrete quantized weight subclass."""
    from piper_kernels.weights._fsdp import register_fsdp_ops  # noqa: PLC0415
    from piper_kernels.weights._matmul import register_matrix_ops  # noqa: PLC0415
    from piper_kernels.weights._update import register_update_ops  # noqa: PLC0415
    from piper_kernels.weights._views import register_view_ops  # noqa: PLC0415

    format_name = cast(Any, cls)._format_name
    update_format_name = cast(Any, cls)._update_format_name
    register_inference_mode_dispatch(cls)
    register_view_ops(cls)
    register_fsdp_ops(cls)
    register_matrix_ops(cls)
    register_conversion_ops(cls, format_name=format_name)
    register_update_ops(cls, format_name=update_format_name)
