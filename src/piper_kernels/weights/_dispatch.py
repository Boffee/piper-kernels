"""Shared base class, dispatch helpers, and zero-copy storage boundary for quantized weights."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

if TYPE_CHECKING:
    from torch._ops import OpOverload

_IGNORED_COPY_METADATA = frozenset({"dtype", "transposed"})


def require_untransposed(weight: QuantizedWeight, operation: str) -> None:
    """Reject operations that would interpret a transpose as canonical storage."""
    if weight.transposed:
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} does not support a transposed weight"
        )


def _decompose_dispatch(
    func: OpOverload,
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object:
    return func.decompose(*args, **kwargs)


def _alias_or_detach_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        return_and_correct_aliasing(
            func,
            args,
            kwargs,
            cast("QuantizedWeight", args[0])._apply_fn_to_data(
                lambda value: func(value, *args[1:], **kwargs)
            ),
        ),
    )


def _compatible_shallow_copy_dispatch(
    _func: Callable[..., object],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> bool:
    return type(args[0]) is type(args[1]) and args[0].shape == args[1].shape


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
    """Remove an explicit true ``copy`` before parsing ``Tensor.to`` arguments."""
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


def _to_copy_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Preserve quantized storage across logical dtype and device conversion."""
    tensor = cast(QuantizedWeight, args[0])
    format_name = tensor._format_name
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
        raise NotImplementedError(f"unsupported {format_name} conversion arguments: {arguments}")
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
            QuantizedWeight,
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


def _copy_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    dst = args[0]
    src = args[1]
    non_blocking = bool(args[2] if len(args) > 2 else kwargs.get("non_blocking", False))
    if type(dst) is not type(src):
        raise NotImplementedError(f"cannot copy {type(src).__name__} into {type(dst).__name__}")
    dst_weight = cast(QuantizedWeight, dst)
    src_weight = cast(QuantizedWeight, src)
    dst_tensors = dst_weight._copy_storage_tensors("copy_")
    src_tensors = src_weight._copy_storage_tensors("copy_")
    if tuple(dst_weight.shape) != tuple(src_weight.shape):
        raise RuntimeError(
            f"shape mismatch in quantized weight copy_: {tuple(dst_weight.shape)} vs "
            f"{tuple(src_weight.shape)}"
        )

    _, dst_attrs = dst_weight.__tensor_flatten__()
    _, src_attrs = src_weight.__tensor_flatten__()
    dst_meta = {k: v for k, v in dst_attrs.items() if k not in _IGNORED_COPY_METADATA}
    src_meta = {k: v for k, v in src_attrs.items() if k not in _IGNORED_COPY_METADATA}
    if dst_meta != src_meta:
        raise NotImplementedError(
            f"quantized metadata mismatch in {type(dst_weight).__name__} copy_: "
            f"{dst_meta} vs {src_meta}"
        )
    if set(dst_tensors) != set(src_tensors):
        raise ValueError(
            f"quantized weight storage tensors do not match: {sorted(dst_tensors)} "
            f"vs {sorted(src_tensors)}"
        )
    for name, dst_val in dst_tensors.items():
        src_val = src_tensors[name]
        fp8_scale_reshape = (
            name == "scale"
            and dst_val.dtype is torch.float8_e4m3fn
            and dst_val.numel() == src_val.numel()
        )
        if dst_val.dtype is not src_val.dtype or (
            tuple(dst_val.shape) != tuple(src_val.shape) and not fp8_scale_reshape
        ):
            raise ValueError(
                f"quantized weight storage tensor {name!r} mismatch in "
                f"{type(dst_weight).__name__} copy_: "
                f"{dst_val.dtype} {tuple(dst_val.shape)} vs {src_val.dtype} {tuple(src_val.shape)}"
            )

    for name, dst_val in dst_tensors.items():
        src_val = src_tensors[name]
        if dst_val is not src_val:
            src_copy = (
                src_val.view(dst_val.shape)
                if tuple(src_val.shape) != tuple(dst_val.shape)
                else src_val
            )
            dst_val.copy_(src_copy, non_blocking=non_blocking)
    return cast(torch.Tensor, dst_weight)


def _is_pinned_dispatch(
    func: Callable[..., object],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    weight = cast(QuantizedWeight, args[0])
    names, _ = weight.__tensor_flatten__()
    return all(bool(func(getattr(weight, name), *args[1:], **kwargs)) for name in names)


def _pin_memory_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast(QuantizedWeight, args[0])
    return cast(
        torch.Tensor,
        weight._apply_fn_to_data(lambda value: func(value, *args[1:], **kwargs)),
    )


class QuantizedWeight(torch.Tensor):
    """Base tensor subclass for Piper quantized weights and zero-copy storage transport."""

    tensor_data_names: ClassVar[list[str]]
    tensor_attribute_names: ClassVar[list[str]]
    optional_tensor_data_names: ClassVar[list[str]] = []
    optional_tensor_attribute_names: ClassVar[list[str]] = []

    _format_name: ClassVar[str]
    _contiguous_memory_format_only: ClassVar[bool] = False
    _supports_flat_1d_view: ClassVar[bool] = False
    _validates_storage_in_new: ClassVar[bool] = False
    _ATEN_OP_TABLE: ClassVar[dict[Any, Callable[..., Any]]] = {}
    _TORCH_FN_TABLE: ClassVar[dict[Any, Callable[..., Any]]] = {}

    transposed: bool = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._ATEN_OP_TABLE = {}
        cls._TORCH_FN_TABLE = {}
        for parent in reversed(cls.__bases__):
            if isinstance(parent, type) and issubclass(parent, QuantizedWeight):
                cls._ATEN_OP_TABLE.update(parent._ATEN_OP_TABLE)
                cls._TORCH_FN_TABLE.update(parent._TORCH_FN_TABLE)

    @classmethod
    def implements(
        cls,
        aten_ops: object | Sequence[object],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an ATen operator handler for this quantized weight subclass."""
        ops = aten_ops if isinstance(aten_ops, (list, tuple)) else [aten_ops]

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            for op in ops:
                cls._ATEN_OP_TABLE[op] = fn
            return fn

        return decorator

    @classmethod
    def implements_torch_function(
        cls,
        torch_fns: object | Sequence[object],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a ``__torch_function__`` handler for this quantized weight subclass."""
        fns = torch_fns if isinstance(torch_fns, (list, tuple)) else [torch_fns]

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            for fn_key in fns:
                cls._TORCH_FN_TABLE[fn_key] = fn
            return fn

        return decorator

    @classmethod
    def __torch_function__(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        # object: dispatched torch functions can return tensors, tuples, bools, or None.
        resolved_kwargs = {} if kwargs is None else kwargs
        handler = cls._TORCH_FN_TABLE.get(func)
        if handler is not None:
            return handler(func, types, args, resolved_kwargs)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **resolved_kwargs)

    @classmethod
    def __torch_dispatch__(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        """Dispatch registered ATen ops and preserve inference-mode decomposition/view rules."""
        # object: an op returns whatever its schema declares, not only tensors.
        resolved_kwargs = {} if kwargs is None else kwargs
        handler = cls._ATEN_OP_TABLE.get(func)
        if handler is None:
            decomposed = func.decompose(*args, **resolved_kwargs)
            if decomposed is not NotImplemented:
                return decomposed
            arg_types = tuple(type(arg) for arg in args)
            kwarg_types = {key: type(value) for key, value in resolved_kwargs.items()}
            raise NotImplementedError(
                f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: "
                f"{func=}, {types=}, {arg_types=}, {kwarg_types=}"
            )
        if (
            func.is_view
            and torch.is_inference_mode_enabled()
            and args
            and isinstance(args[0], torch.Tensor)
            and not args[0].is_inference()
        ):
            with torch.inference_mode(False):
                return handler(func, types, args, resolved_kwargs)
        return handler(func, types, args, resolved_kwargs)

    def __tensor_flatten__(self) -> tuple[list[str], dict[str, Any]]:
        """Flatten active storage tensors and non-tensor metadata for PyTorch subclasses."""
        if not hasattr(self, "tensor_data_names") or not hasattr(self, "tensor_attribute_names"):
            raise NotImplementedError(
                f"{type(self).__name__} must define tensor_data_names and tensor_attribute_names"
            )
        tensor_names = list(self.tensor_data_names)
        for name in self.optional_tensor_data_names:
            if getattr(self, name) is not None:
                tensor_names.append(name)
        attributes = {name: getattr(self, name) for name in self.tensor_attribute_names}
        for name in self.optional_tensor_attribute_names:
            attributes[name] = getattr(self, name)
        return tensor_names, attributes

    @classmethod
    def __tensor_unflatten__(
        cls,
        tensor_data_dict: Mapping[str, torch.Tensor],
        tensor_attributes: Mapping[str, Any],
        outer_size: object = None,
        outer_stride: object = None,
    ) -> Self:
        """Reconstruct a quantized weight from flattened storage tensors and metadata."""
        del outer_size, outer_stride
        if not hasattr(cls, "tensor_data_names") or not hasattr(cls, "tensor_attribute_names"):
            raise NotImplementedError(
                f"{cls.__name__} must define tensor_data_names and tensor_attribute_names"
            )
        required_tensors = [tensor_data_dict[name] for name in cls.tensor_data_names]
        optional_tensors = {
            name: tensor_data_dict.get(name, None) for name in cls.optional_tensor_data_names
        }
        required_attributes = [tensor_attributes[name] for name in cls.tensor_attribute_names]
        optional_attributes = {
            name: tensor_attributes[name] for name in cls.optional_tensor_attribute_names
        }
        return cls(
            *required_tensors,
            *required_attributes,
            **optional_tensors,
            **optional_attributes,
        )

    def _apply_fn_to_data(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        names, attributes = self.__tensor_flatten__()
        tensors = {name: fn(getattr(self, name)) for name in names}
        return type(self).__tensor_unflatten__(tensors, attributes, None, None)

    def _get_to_kwargs(self, *args: object, **kwargs: object) -> dict[str, Any]:
        filtered_args = tuple(arg for arg in args if not isinstance(arg, torch.layout))
        filtered_kwargs = dict(kwargs)
        filtered_kwargs.pop("layout", None)
        # PyTorch's internal C++ _parse_to helper is omitted from type stubs.
        device, dtype, non_blocking, _ = cast(Any, torch._C)._nn._parse_to(
            *filtered_args, **filtered_kwargs
        )
        return {
            "device": self.device if device is None else device,
            "dtype": self.dtype if dtype is None else dtype,
            "non_blocking": non_blocking,
        }

    def __repr__(self, *, tensor_contents: object = None) -> str:
        del tensor_contents
        if not hasattr(self, "tensor_data_names") or not hasattr(self, "tensor_attribute_names"):
            return f"{type(self).__name__}(shape={tuple(self.shape)}, dtype={self.dtype})"
        fields = [
            f"{name}={getattr(self, name)}"
            for name in (
                *self.tensor_data_names,
                *self.tensor_attribute_names,
                *self.optional_tensor_data_names,
                *self.optional_tensor_attribute_names,
            )
        ]
        return f"{type(self).__name__}({', '.join(fields)})"

    def __setstate__(self, state: object) -> None:
        # PyTorch's internal _set_obj_state helper is omitted from type stubs.
        cast(Any, torch)._utils._set_obj_state(self, state)
        cls = type(self)
        for name in (*self.optional_tensor_data_names, *self.optional_tensor_attribute_names):
            if name not in self.__dict__:
                setattr(self, name, getattr(cls, name, None))

    def _validate_storage_for_export(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement quantized storage export"
        )

    def as_storage(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Return canonical zero-copy storage tensors and plain metadata for transport."""
        require_untransposed(self, "as_storage")
        self._validate_storage_for_export()
        names, attributes = self.__tensor_flatten__()
        tensors = {name: getattr(self, name) for name in names}
        metadata = {key: value for key, value in attributes.items() if key != "transposed"}
        return tensors, metadata

    @classmethod
    def _parse_storage_tensors(
        cls,
        tensors: object,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor | None]]:
        if not isinstance(tensors, Mapping):
            raise TypeError(f"{cls.__name__} from_storage tensors must be a mapping")
        required_names = cls.tensor_data_names
        optional_names = cls.optional_tensor_data_names
        allowed_names = set(required_names) | set(optional_names)
        missing = [name for name in required_names if name not in tensors]
        if missing:
            raise ValueError(f"{cls.__name__} from_storage missing required tensors: {missing}")
        extra = sorted(set(tensors) - allowed_names)
        if extra:
            raise ValueError(f"{cls.__name__} from_storage received unexpected tensors: {extra}")

        required: list[torch.Tensor] = []
        for name in required_names:
            value = tensors[name]
            if not isinstance(value, torch.Tensor) or isinstance(value, QuantizedWeight):
                raise TypeError(
                    f"{cls.__name__} from_storage tensor {name!r} must be a plain torch.Tensor"
                )
            required.append(value)

        optional: dict[str, torch.Tensor | None] = {}
        for name in optional_names:
            if name not in tensors:
                continue
            value = tensors[name]
            if value is not None and (
                not isinstance(value, torch.Tensor) or isinstance(value, QuantizedWeight)
            ):
                raise TypeError(
                    f"{cls.__name__} from_storage optional tensor {name!r} must be a plain "
                    "torch.Tensor or None"
                )
            optional[name] = value
        return required, optional

    @classmethod
    def _parse_storage_metadata(
        cls,
        metadata: object,
    ) -> tuple[list[Any], dict[str, Any]]:
        if not isinstance(metadata, Mapping):
            raise TypeError(f"{cls.__name__} from_storage metadata must be a mapping")
        if "transposed" in metadata:
            raise ValueError(
                f"{cls.__name__} from_storage does not accept transposed weight metadata"
            )
        required_names = cls.tensor_attribute_names
        optional_names = [
            name for name in cls.optional_tensor_attribute_names if name != "transposed"
        ]
        allowed_names = set(required_names) | set(optional_names)
        missing = [name for name in required_names if name not in metadata]
        if missing:
            raise ValueError(f"{cls.__name__} from_storage missing required metadata: {missing}")
        extra = sorted(set(metadata) - allowed_names)
        if extra:
            raise ValueError(f"{cls.__name__} from_storage received unexpected metadata: {extra}")
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{cls.__name__} from_storage metadata {key!r} must not contain tensors"
                )
        return (
            [metadata[name] for name in required_names],
            {name: metadata[name] for name in optional_names if name in metadata},
        )

    @classmethod
    def _from_storage_components(
        cls,
        required_tensors: list[torch.Tensor],
        optional_tensors: dict[str, torch.Tensor | None],
        required_attrs: list[Any],
        optional_attrs: dict[str, Any],
    ) -> Self:
        weight = cls(
            *required_tensors,
            *required_attrs,
            **optional_tensors,
            **optional_attrs,
        )
        if not cls._validates_storage_in_new:
            weight._validate_storage_for_export()
        return weight

    @classmethod
    def from_storage(
        cls,
        tensors: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any],
    ) -> Self:
        """Reconstruct an untransposed quantized weight zero-copy from storage buffers."""
        if cls is QuantizedWeight or not hasattr(cls, "tensor_data_names"):
            raise TypeError("QuantizedWeight.from_storage must be called on a concrete subclass")
        required_tensors, optional_tensors = cls._parse_storage_tensors(tensors)
        required_attrs, optional_attrs = cls._parse_storage_metadata(metadata)
        return cls._from_storage_components(
            required_tensors, optional_tensors, required_attrs, optional_attrs
        )

    def _validate_memory_format(self, memory_format: object) -> None:
        if memory_format not in (None, torch.preserve_format):
            require_untransposed(self, "to with a different memory format")
            if self._contiguous_memory_format_only and memory_format is not torch.contiguous_format:
                raise NotImplementedError(
                    f"{self._format_name} only supports contiguous memory format"
                )

    def _rebuild_with_logical_dtype(self, dtype: torch.dtype) -> Self:
        """Rebuild the concrete semantic wrapper while preserving quantized storage."""
        names, attributes = self.__tensor_flatten__()
        tensors = {name: getattr(self, name) for name in names}
        updated = dict(attributes)
        updated["dtype"] = dtype
        return type(self).__tensor_unflatten__(tensors, updated, None, None)

    def _stable_cache_key(self, *extra: object) -> str:
        """Build an AOTAutograd cache fingerprint from flattened storage and metadata."""
        names, attributes = self.__tensor_flatten__()
        tensors = tuple(
            (
                name,
                tuple(tensor.shape),
                tensor.stride(),
                str(tensor.dtype),
                str(tensor.device),
            )
            for name in names
            for tensor in (getattr(self, name),)
        )
        metadata = tuple(
            (key, str(value) if isinstance(value, torch.dtype) else value)
            for key, value in attributes.items()
        )
        return repr(
            (
                type(self).__qualname__,
                tuple(self.shape),
                self.stride(),
                str(self.device),
                str(self.dtype),
                self.transposed,
                tensors,
                metadata,
                *extra,
            )
        )

    def _stable_hash_for_caching(self) -> str:
        """Return a metadata fingerprint for AOTAutograd's persistent cache."""
        return self._stable_cache_key()

    def _copy_storage_tensors(self, operation: str) -> dict[str, torch.Tensor]:
        require_untransposed(self, operation)
        self._validate_storage_for_export()
        names, _ = self.__tensor_flatten__()
        return {name: getattr(self, name) for name in names}

    def _transpose(self) -> Self:
        raise NotImplementedError

    def to(self, *args: object, **kwargs: object) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Preserve explicit-copy semantics hidden by ``aten._to_copy``."""
        self._validate_memory_format(kwargs.get("memory_format"))
        explicit_copy_args = _explicit_to_copy_args(args, kwargs)
        if explicit_copy_args is None:
            return cast(Self, cast(Any, super()).to(*args, **kwargs))
        to_args, to_kwargs = explicit_copy_args
        options = self._get_to_kwargs(*to_args, **to_kwargs)
        dtype = options["dtype"]
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                f"{self._format_name} logical dtype must be floating point, got {dtype}"
            )
        copied = self._apply_fn_to_data(
            lambda value: value.to(
                device=options["device"], non_blocking=options["non_blocking"], copy=True
            )
        )
        return copied if dtype is copied.dtype else copied._rebuild_with_logical_dtype(dtype)


QuantizedWeight.implements(torch.ops.aten.to.dtype_layout)(_decompose_dispatch)
QuantizedWeight.implements([torch.ops.aten.detach.default, torch.ops.aten.alias.default])(
    _alias_or_detach_dispatch
)
QuantizedWeight.implements(torch.ops.aten._to_copy.default)(_to_copy_dispatch)
QuantizedWeight.implements(torch.ops.aten.copy_.default)(_copy_dispatch)
QuantizedWeight.implements(torch.ops.aten.is_pinned.default)(_is_pinned_dispatch)
QuantizedWeight.implements(torch.ops.aten._pin_memory.default)(_pin_memory_dispatch)
QuantizedWeight.implements_torch_function(torch._has_compatible_shallow_copy_type)(
    _compatible_shallow_copy_dispatch
)


__all__ = [
    "QuantizedWeight",
    "require_untransposed",
    "unsupported_operation_dispatch",
]
