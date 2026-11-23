"""PyTorch FSDP2 (``fully_shard``) storage and collective hooks for quantized weights."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, cast

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import TorchAOBaseTensor

from piper_kernels.weights._dispatch import require_untransposed

if TYPE_CHECKING:
    from torch._ops import OpOverload
    from torch.distributed.device_mesh import DeviceMesh

    from piper_kernels.weights._dispatch import QuantizedWeight
    from piper_kernels.weights.convrot.int8.tensor import ConvRotInt8Tensor
    from piper_kernels.weights.nvfp4.tensor import PiperNVFP4Tensor

_SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_IGNORED_COPY_METADATA = frozenset({"dtype", "orig_dtype", "transposed", "use_triton_kernel"})


class _FlatQuantizedWeightView(torch.Tensor):
    """Opaque 1-D storage handle returned for ``weight.view(-1)`` during FSDP2 setup."""

    def __new__(cls, source: torch.Tensor) -> _FlatQuantizedWeightView:
        # Avoid holding a reference to source so state-dict swaps can reclaim stale shards.
        return cast(
            _FlatQuantizedWeightView,
            torch.Tensor._make_wrapper_subclass(
                cls,
                (source.numel(),),
                dtype=source.dtype,
                device=source.device,
                requires_grad=False,
            ),
        )

    def __repr__(self, *, tensor_contents: object = None) -> str:
        _ = tensor_contents
        return (
            f"{type(self).__name__}(shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, device={self.device})"
        )

    @classmethod
    def __torch_dispatch__(
        cls,
        func: OpOverload,
        _types: tuple[type, ...],
        _args: tuple[object, ...],
        _kwargs: dict[str, object] | None = None,
    ) -> object:
        raise NotImplementedError(
            f"1-D flat views of Piper quantized weights do not support {func}"
        )


@dataclass(frozen=True)
class _FSDPAllGatherMetadata:
    """Host metadata carried from ``fsdp_pre_all_gather`` to ``fsdp_post_all_gather``."""

    cls: type[TorchAOBaseTensor]
    local_rows: int
    features: int
    world_size: int
    attributes: dict[str, object]
    optional_tensors: dict[str, torch.Tensor]
    reswizzle_nvfp4_scales: bool
    single_rank_scale: torch.Tensor | None = None


class _NVFP4Layout(Protocol):
    BLOCK_SIZE: int
    SCALE_ROW_TILE: int

    def qdata_shape(self, rows: int, features: int) -> tuple[int, int]: ...

    def scale_shape(self, rows: int, features: int) -> tuple[int, int]: ...


def _nvfp4_layout() -> _NVFP4Layout:
    from piper_kernels.weights.nvfp4 import _layout  # noqa: PLC0415

    return cast(_NVFP4Layout, _layout)


def _is_int8_weight(weight: object) -> TypeGuard[ConvRotInt8Tensor]:
    from piper_kernels.weights.convrot.int8.tensor import ConvRotInt8Tensor  # noqa: PLC0415

    return isinstance(weight, ConvRotInt8Tensor)


def _is_nvfp4_weight(weight: object) -> TypeGuard[PiperNVFP4Tensor]:
    if _is_int8_weight(weight):
        return False
    from piper_kernels.weights.nvfp4.tensor import PiperNVFP4Tensor  # noqa: PLC0415

    return isinstance(weight, PiperNVFP4Tensor)


def _canonical_scale_shape(weight: QuantizedWeight, rows: int, features: int) -> tuple[int, int]:
    if _is_int8_weight(weight):
        return (rows, 1)
    if not _is_nvfp4_weight(weight):
        raise TypeError(f"unsupported quantized weight type {type(weight).__name__}")
    return (
        _nvfp4_layout().scale_shape(rows, features)
        if weight.is_swizzled_scales
        else (rows, features // weight.block_size)
    )


def _validate_optional_scalar(
    scale: torch.Tensor | None,
    device: torch.device,
    *,
    label: str,
) -> None:
    if scale is None:
        return
    if scale.ndim != 0 or scale.dtype is not torch.float32 or scale.device != device:
        raise NotImplementedError(
            f"{label} requires scalar FP32 global scales on the weight device"
        )


def _require_fsdp_weight(weight: QuantizedWeight, operation: str) -> torch.Tensor:
    """Validate 2-D untransposed storage and return canonical 2-D scale storage."""
    if weight.ndim != 2:
        raise NotImplementedError(f"{type(weight).__name__} {operation} requires a 2-D weight")
    require_untransposed(weight, operation)
    if not weight.qdata.is_contiguous() or not weight.scale.is_contiguous():
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} requires contiguous storage"
        )
    if _is_int8_weight(weight):
        from piper_kernels.weights.convrot.int8._quantization import (  # noqa: PLC0415
            validate_activation_scale,
            validate_storage,
        )

        validate_storage(weight.qdata, weight.scale, weight.group_size, weight.dtype)
        validate_activation_scale(weight.act_per_tensor_scale, weight.device)
        return weight.scale
    if not _is_nvfp4_weight(weight):
        raise TypeError(f"unsupported quantized weight type {type(weight).__name__}")

    layout = _nvfp4_layout()
    rows, features = weight.shape
    if weight.block_size != layout.BLOCK_SIZE or features % layout.BLOCK_SIZE:
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} requires complete 16-value quantization blocks"
        )
    if weight.qdata.dtype is not torch.uint8 or tuple(weight.qdata.shape) != layout.qdata_shape(
        rows, features
    ):
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} requires canonical packed UINT8 data"
        )
    if weight.scale.dtype is not torch.float8_e4m3fn or weight.scale.device != weight.device:
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} requires FP8 E4M3 scales on the weight device"
        )
    _validate_optional_scalar(
        weight.per_tensor_scale, weight.device, label=f"{type(weight).__name__} {operation}"
    )
    _validate_optional_scalar(
        weight.act_per_tensor_scale, weight.device, label=f"{type(weight).__name__} {operation}"
    )
    expected_2d = _canonical_scale_shape(weight, rows, features)
    if tuple(weight.scale.shape) not in (expected_2d, (expected_2d[0] * expected_2d[1],)):
        raise NotImplementedError(
            f"{type(weight).__name__} {operation} requires canonical or flat block-scale storage"
        )
    return weight.scale.view(expected_2d)


def _optional_tensors(weight: QuantizedWeight, names: list[str]) -> dict[str, torch.Tensor]:
    return {name: getattr(weight, name) for name in names if name not in ("qdata", "scale")}


def _validate_optional_tensor_keys(
    target: QuantizedWeight,
    expected: dict[str, torch.Tensor],
    *,
    target_names: list[str] | None = None,
) -> None:
    names = target.__tensor_flatten__()[0] if target_names is None else target_names
    target_optional = {name for name in names if name not in ("qdata", "scale")}
    if target_optional != set(expected):
        raise ValueError(
            f"quantized weight optional tensors do not match: {sorted(target_optional)} "
            f"vs {sorted(expected)}"
        )


def _copy_optional_tensors(
    target: QuantizedWeight,
    expected: dict[str, torch.Tensor],
    *,
    non_blocking: bool = False,
) -> None:
    for name, src in expected.items():
        dst = getattr(target, name)
        if dst is not src:
            dst.copy_(src, non_blocking=non_blocking)


def fsdp_pre_all_gather(
    weight: QuantizedWeight,
    mesh: DeviceMesh,
) -> tuple[tuple[torch.Tensor, ...], _FSDPAllGatherMetadata]:
    """Return all-gather payloads and metadata for PyTorch FSDP2."""
    canonical_scale = _require_fsdp_weight(weight, "fsdp_pre_all_gather")
    local_rows, features = weight.shape
    world_size = mesh.size()
    names, attributes = weight.__tensor_flatten__()
    reswizzle = False
    scale_to_gather = canonical_scale
    if not _is_int8_weight(weight) and _is_nvfp4_weight(weight) and world_size > 1:
        layout = _nvfp4_layout()
        reswizzle = weight.is_swizzled_scales and local_rows % layout.SCALE_ROW_TILE != 0
        if reswizzle:
            from torchao.prototype.mx_formats.utils import from_blocked  # noqa: PLC0415

            scale_to_gather = from_blocked(
                canonical_scale.view(torch.uint8),
                local_rows,
                features // layout.BLOCK_SIZE,
            ).contiguous()

    # PyTorch FSDP2 forwards only all_gather_inputs[0] when world_size == 1.
    single_rank_scale = scale_to_gather if world_size == 1 else None
    metadata = _FSDPAllGatherMetadata(
        cls=type(weight),
        local_rows=local_rows,
        features=features,
        world_size=world_size,
        attributes=attributes,
        optional_tensors=_optional_tensors(weight, names),
        reswizzle_nvfp4_scales=reswizzle,
        single_rank_scale=single_rank_scale,
    )
    payloads: tuple[torch.Tensor, ...] = (
        (weight.qdata,) if world_size == 1 else (weight.qdata, scale_to_gather)
    )
    return payloads, metadata


def _unpack_all_gather_outputs(
    all_gather_outputs: tuple[torch.Tensor, ...],
    metadata: _FSDPAllGatherMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if metadata.world_size == 1:
        if metadata.single_rank_scale is None or metadata.reswizzle_nvfp4_scales:
            raise ValueError("invalid single-rank FSDP2 all-gather metadata")
        if len(all_gather_outputs) != 1:
            raise ValueError(
                f"expected 1 all-gather output for world_size=1, got {len(all_gather_outputs)}"
            )
        return all_gather_outputs[0], metadata.single_rank_scale
    if metadata.single_rank_scale is not None:
        raise ValueError("unexpected single-rank scale in multi-rank FSDP2 metadata")
    if len(all_gather_outputs) != 2:
        raise ValueError(f"expected 2 all-gather outputs, got {len(all_gather_outputs)}")
    return all_gather_outputs[0], all_gather_outputs[1]


def _prepare_refresh_scale(
    target_weight: QuantizedWeight,
    gathered_scale: torch.Tensor,
    metadata: _FSDPAllGatherMetadata,
    device: torch.device,
    expected_target_scale_shape: tuple[int, int],
) -> torch.Tensor | None:
    if metadata.reswizzle_nvfp4_scales or metadata.single_rank_scale is not None:
        expected_dtype = (
            torch.float8_e4m3fn if metadata.reswizzle_nvfp4_scales else gathered_scale.dtype
        )
        if (
            target_weight.scale.dtype is not expected_dtype
            or target_weight.scale.device != device
            or not target_weight.scale.is_contiguous()
            or tuple(target_weight.scale.shape) != expected_target_scale_shape
        ):
            kind = "reswizzle" if metadata.reswizzle_nvfp4_scales else "single-rank"
            raise ValueError(
                f"unsharded {kind} target scale mismatch: expected "
                f"shape={expected_target_scale_shape}, dtype={expected_dtype}, "
                f"device={device}, contiguous=True; got "
                f"shape={tuple(target_weight.scale.shape)}, "
                f"dtype={target_weight.scale.dtype}, device={target_weight.scale.device}, "
                f"contiguous={target_weight.scale.is_contiguous()}"
            )
        if metadata.reswizzle_nvfp4_scales:
            from torchao.prototype.mx_formats.utils import to_blocked  # noqa: PLC0415

            return to_blocked(gathered_scale).view(expected_target_scale_shape)
        return gathered_scale

    if target_weight.scale.untyped_storage().data_ptr() != (
        gathered_scale.untyped_storage().data_ptr()
    ):
        raise RuntimeError("expected out.scale to alias FSDP2 all-gather output")
    return None


def _refresh_unsharded_param(
    out: torch.Tensor,
    qdata_full: torch.Tensor,
    gathered_scale: torch.Tensor,
    metadata: _FSDPAllGatherMetadata,
    param_dtype: torch.dtype,
    full_rows: int,
    expected_target_scale_shape: tuple[int, int],
) -> None:
    from torch.distributed.tensor import DTensor  # noqa: PLC0415

    target = out._local_tensor if isinstance(out, DTensor) else out
    if type(target) is not metadata.cls:
        raise TypeError(
            f"expected unsharded parameter of type {metadata.cls.__name__}, "
            f"got {type(target).__name__}"
        )
    target_weight = cast("QuantizedWeight", target)
    if tuple(target_weight.shape) != (full_rows, metadata.features):
        raise ValueError(
            f"unsharded parameter shape mismatch: expected {(full_rows, metadata.features)}, "
            f"got {tuple(target_weight.shape)}"
        )
    if target_weight.dtype is not param_dtype:
        raise ValueError(
            f"unsharded parameter dtype mismatch: expected {param_dtype}, got {target_weight.dtype}"
        )
    if target_weight.device != qdata_full.device:
        raise ValueError(
            f"unsharded parameter device mismatch: expected {qdata_full.device}, "
            f"got {target_weight.device}"
        )
    if target_weight.qdata.untyped_storage().data_ptr() != (
        qdata_full.untyped_storage().data_ptr()
    ):
        raise RuntimeError("expected out.qdata to alias FSDP2 all-gather output")

    _validate_optional_tensor_keys(target_weight, metadata.optional_tensors)
    for name in metadata.optional_tensors:
        _validate_optional_scalar(
            getattr(target_weight, name),
            qdata_full.device,
            label=f"{type(target_weight).__name__} fsdp_post_all_gather out ({name})",
        )

    prepared_scale = _prepare_refresh_scale(
        target_weight,
        gathered_scale,
        metadata,
        qdata_full.device,
        expected_target_scale_shape,
    )

    _copy_optional_tensors(target_weight, metadata.optional_tensors)
    if prepared_scale is not None:
        if target_weight.scale.dtype is torch.float8_e4m3fn:
            target_weight.scale.view(torch.uint8).copy_(prepared_scale.view(torch.uint8))
        else:
            target_weight.scale.copy_(prepared_scale)


def _validate_all_gather_outputs(
    weight: QuantizedWeight,
    qdata_full: torch.Tensor,
    gathered_scale: torch.Tensor,
    metadata: _FSDPAllGatherMetadata,
    full_rows: int,
) -> tuple[int, int]:
    if _is_int8_weight(weight):
        if metadata.reswizzle_nvfp4_scales:
            raise ValueError("ConvRotInt8Tensor does not support NVFP4 scale reswizzling")
        expected_qdata_shape: tuple[int, int] = (full_rows, metadata.features)
        expected_qdata_dtype = torch.int8
        expected_scale_shape: tuple[int, int] = (full_rows, 1)
        expected_scale_dtype = torch.float32
        expected_target_scale_shape: tuple[int, int] = expected_scale_shape
    elif _is_nvfp4_weight(weight):
        layout = _nvfp4_layout()
        expected_qdata_shape = layout.qdata_shape(full_rows, metadata.features)
        expected_qdata_dtype = torch.uint8
        expected_target_scale_shape = _canonical_scale_shape(weight, full_rows, metadata.features)
        if metadata.reswizzle_nvfp4_scales:
            if not weight.is_swizzled_scales:
                raise ValueError(
                    "reswizzle_nvfp4_scales requires a weight with swizzled NVFP4 scales"
                )
            expected_scale_shape = (full_rows, metadata.features // layout.BLOCK_SIZE)
            expected_scale_dtype = torch.uint8
        else:
            expected_scale_shape = expected_target_scale_shape
            expected_scale_dtype = torch.float8_e4m3fn
    else:
        raise TypeError(f"unsupported quantized weight type {type(weight).__name__}")

    if (
        tuple(qdata_full.shape) != expected_qdata_shape
        or qdata_full.dtype is not expected_qdata_dtype
        or not qdata_full.is_contiguous()
    ):
        raise ValueError(
            "all-gathered qdata mismatch: expected "
            f"shape={expected_qdata_shape}, dtype={expected_qdata_dtype}, contiguous=True; "
            f"got shape={tuple(qdata_full.shape)}, dtype={qdata_full.dtype}, "
            f"contiguous={qdata_full.is_contiguous()}"
        )
    if (
        tuple(gathered_scale.shape) != expected_scale_shape
        or gathered_scale.dtype is not expected_scale_dtype
        or gathered_scale.device != qdata_full.device
        or not gathered_scale.is_contiguous()
    ):
        raise ValueError(
            "all-gathered scale mismatch: expected "
            f"shape={expected_scale_shape}, dtype={expected_scale_dtype}, "
            f"device={qdata_full.device}, contiguous=True; "
            f"got shape={tuple(gathered_scale.shape)}, dtype={gathered_scale.dtype}, "
            f"device={gathered_scale.device}, contiguous={gathered_scale.is_contiguous()}"
        )
    for name, tensor in metadata.optional_tensors.items():
        _validate_optional_scalar(
            tensor,
            qdata_full.device,
            label=f"{type(weight).__name__} fsdp_post_all_gather ({name})",
        )
    return expected_target_scale_shape


def fsdp_post_all_gather(
    weight: QuantizedWeight,
    all_gather_outputs: tuple[torch.Tensor, ...],
    metadata: object,
    param_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
    """Reconstruct or refresh an unsharded quantized weight from FSDP2 all-gather outputs."""
    if not isinstance(metadata, _FSDPAllGatherMetadata):
        raise TypeError(f"expected _FSDPAllGatherMetadata, got {type(metadata).__name__}")
    if type(weight) is not metadata.cls:
        raise TypeError(
            f"weight type {type(weight).__name__} does not match metadata {metadata.cls.__name__}"
        )
    if param_dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise TypeError(f"quantized weight logical dtype must be floating point, got {param_dtype}")

    qdata_full, gathered_scale = _unpack_all_gather_outputs(all_gather_outputs, metadata)
    full_rows = metadata.local_rows * metadata.world_size
    expected_target_scale_shape = _validate_all_gather_outputs(
        weight, qdata_full, gathered_scale, metadata, full_rows
    )

    if out is not None:
        _refresh_unsharded_param(
            out,
            qdata_full,
            gathered_scale,
            metadata,
            param_dtype,
            full_rows,
            expected_target_scale_shape,
        )
        return None

    if metadata.reswizzle_nvfp4_scales:
        from torchao.prototype.mx_formats.utils import to_blocked  # noqa: PLC0415

        scale_full = (
            to_blocked(gathered_scale).view(expected_target_scale_shape).view(torch.float8_e4m3fn)
        )
        extra_inner_tensors: tuple[torch.Tensor, ...] = (scale_full,)
    elif metadata.single_rank_scale is not None:
        if gathered_scale.dtype is torch.float8_e4m3fn:
            scale_full = gathered_scale.view(torch.uint8).clone().view(torch.float8_e4m3fn)
        else:
            scale_full = gathered_scale.clone()
        extra_inner_tensors = (scale_full,)
    else:
        scale_full = gathered_scale
        extra_inner_tensors = ()

    updated_attrs = dict(metadata.attributes)
    updated_attrs["orig_dtype" if "orig_dtype" in updated_attrs else "dtype"] = param_dtype
    tensors = {"qdata": qdata_full, "scale": scale_full, **metadata.optional_tensors}
    unsharded = metadata.cls.__tensor_unflatten__(tensors, updated_attrs, None, None)
    return cast(torch.Tensor, unsharded), extra_inner_tensors


def _split_dispatch(
    func: Callable[..., object],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[torch.Tensor]:
    weight = cast("QuantizedWeight", args[0])
    split_size = args[1] if len(args) > 1 else kwargs["split_size"]
    dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
    canonical_scale = _require_fsdp_weight(weight, "split")
    if type(dim) is not int:
        raise TypeError(f"split dim must be an integer, got {dim!r}")
    if dim not in (-2, -1, 0, 1):
        raise IndexError(f"split dimension {dim} out of range for a 2-D quantized weight")
    if dim not in (-2, 0):
        raise NotImplementedError(f"{type(weight).__name__} split only supports dim=0")
    if type(split_size) is not int:
        raise NotImplementedError(f"{type(weight).__name__} split requires an integer split_size")
    if split_size <= 0:
        raise ValueError(f"split_size must be positive, got {split_size}")

    rows = weight.shape[0]
    if rows == 0 or (split_size < rows and rows % split_size != 0):
        raise NotImplementedError(
            f"{type(weight).__name__} split requires even dim=0 chunks for {rows} rows "
            f"(got split_size={split_size})"
        )

    if _is_int8_weight(weight):
        swizzled = False
        scale_row_tile = 0
    else:
        swizzled = _is_nvfp4_weight(weight) and weight.is_swizzled_scales
        scale_row_tile = _nvfp4_layout().SCALE_ROW_TILE if swizzled else 0
    if swizzled and split_size < rows and split_size % scale_row_tile != 0:
        from piper_kernels.weights.sharding import shard_quantized_weight  # noqa: PLC0415

        return [
            cast(
                torch.Tensor,
                shard_quantized_weight(weight, dim=0, start=i * split_size, length=split_size),
            )
            for i in range(rows // split_size)
        ]

    qdata_chunks = weight.qdata.split(split_size, dim=0)
    if split_size >= rows:
        scale_split = canonical_scale.shape[0]
    elif swizzled:
        scale_split = (split_size // scale_row_tile) * 32
    else:
        scale_split = split_size
    scale_chunks = canonical_scale.split(scale_split, dim=0)

    names, metadata = weight.__tensor_flatten__()
    optional = _optional_tensors(weight, names)
    chunks = [
        cast(
            torch.Tensor,
            type(weight).__tensor_unflatten__(
                {"qdata": qdata_chunk, "scale": scale_chunk, **optional},
                metadata,
                None,
                None,
            ),
        )
        for qdata_chunk, scale_chunk in zip(qdata_chunks, scale_chunks, strict=True)
    ]
    return cast(list[torch.Tensor], return_and_correct_aliasing(func, args, kwargs, chunks))


def _new_zeros_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast("QuantizedWeight", args[0])
    size = tuple(args[1] if len(args) > 1 else kwargs["size"])
    dtype = kwargs.get("dtype")
    layout = kwargs.get("layout")
    device = kwargs.get("device")
    pin_memory = kwargs.get("pin_memory")
    if pin_memory not in (None, False, True):
        raise TypeError(f"pin_memory must be a bool or None, got {pin_memory!r}")

    _require_fsdp_weight(weight, "new_zeros")
    logical_dtype = weight.dtype if dtype is None else dtype
    if logical_dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise TypeError(
            f"{type(weight).__name__} logical dtype must be floating point, got {logical_dtype}"
        )
    if layout not in (None, torch.strided):
        raise NotImplementedError(f"{type(weight).__name__} only supports strided layout")
    if (
        len(size) != 2
        or any(type(dim) is not int or dim < 0 for dim in size)
        or (size[1] != weight.shape[1])
    ):
        raise NotImplementedError(
            f"{type(weight).__name__} new_zeros requires a 2-D size with features={weight.shape[1]}"
        )

    rows, features = size
    target_device = weight.device if device is None else torch.device(device)
    pin = bool(pin_memory) if pin_memory is not None else False

    if _is_int8_weight(weight):
        int8_weight = True
        qdata_dims: tuple[int, int] = (rows, features)
    elif _is_nvfp4_weight(weight):
        int8_weight = False
        qdata_dims = _nvfp4_layout().qdata_shape(rows, features)
    else:
        raise TypeError(f"unsupported quantized weight type {type(weight).__name__}")
    qdata = torch.zeros(
        qdata_dims,
        dtype=weight.qdata.dtype,
        device=target_device,
        pin_memory=pin,
    )
    scale_dims = _canonical_scale_shape(weight, rows, features)
    if int8_weight:
        scale = torch.zeros(
            scale_dims,
            dtype=torch.float32,
            device=target_device,
            pin_memory=pin,
        )
    else:
        # Allocate uint8 zeros so CPU and meta devices avoid direct FP8 zero-fill ops.
        scale = torch.zeros(
            scale_dims,
            dtype=torch.uint8,
            device=target_device,
            pin_memory=pin,
        ).view(torch.float8_e4m3fn)

    names, metadata = weight.__tensor_flatten__()
    tensors: dict[str, torch.Tensor] = {"qdata": qdata, "scale": scale}
    for name in names:
        if name not in tensors:
            tensors[name] = torch.zeros(
                (), dtype=torch.float32, device=target_device, pin_memory=pin
            )

    updated = dict(metadata)
    updated["orig_dtype" if "orig_dtype" in updated else "dtype"] = logical_dtype
    return cast(torch.Tensor, type(weight).__tensor_unflatten__(tensors, updated, None, None))


def _copy_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    dst = cast("QuantizedWeight", args[0])
    src = args[1]
    non_blocking = bool(args[2] if len(args) > 2 else kwargs.get("non_blocking", False))
    if type(dst) is not type(src):
        raise NotImplementedError(f"cannot copy {type(src).__name__} into {type(dst).__name__}")
    src_weight = cast("QuantizedWeight", src)
    dst_scale = _require_fsdp_weight(dst, "copy_")
    src_scale = _require_fsdp_weight(src_weight, "copy_")
    if tuple(dst.shape) != tuple(src_weight.shape):
        raise RuntimeError(
            f"shape mismatch in quantized weight copy_: {tuple(dst.shape)} vs "
            f"{tuple(src_weight.shape)}"
        )

    dst_names, dst_attrs = dst.__tensor_flatten__()
    src_names, src_attrs = src_weight.__tensor_flatten__()
    dst_meta = {k: v for k, v in dst_attrs.items() if k not in _IGNORED_COPY_METADATA}
    src_meta = {k: v for k, v in src_attrs.items() if k not in _IGNORED_COPY_METADATA}
    if dst_meta != src_meta:
        raise NotImplementedError(
            f"quantized metadata mismatch in {type(dst).__name__} copy_: {dst_meta} vs {src_meta}"
        )
    src_optional = _optional_tensors(src_weight, src_names)
    _validate_optional_tensor_keys(dst, src_optional, target_names=dst_names)

    dst.qdata.copy_(src_weight.qdata, non_blocking=non_blocking)
    if _is_int8_weight(dst):
        dst_scale.copy_(src_scale, non_blocking=non_blocking)
    else:
        dst_scale.view(torch.uint8).copy_(src_scale.view(torch.uint8), non_blocking=non_blocking)
    _copy_optional_tensors(dst, src_optional, non_blocking=non_blocking)
    return dst


def _is_pinned_dispatch(
    func: Callable[..., object],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    weight = cast("QuantizedWeight", args[0])
    names, _ = weight.__tensor_flatten__()
    return all(bool(func(getattr(weight, name), *args[1:], **kwargs)) for name in names)


def _pin_memory_dispatch(
    func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast("QuantizedWeight", args[0])
    return cast(
        torch.Tensor,
        weight._apply_fn_to_data(lambda value: func(value, *args[1:], **kwargs)),
    )


def register_fsdp_ops(cls: type[TorchAOBaseTensor]) -> None:
    """Register FSDP2 parameter-sharding and CPU-offload ATen handlers."""
    aten = torch.ops.aten
    cls.implements(aten.split.Tensor)(_split_dispatch)
    cls.implements(aten.new_zeros.default)(_new_zeros_dispatch)
    cls.implements(aten.copy_.default)(_copy_dispatch)
    cls.implements(aten.is_pinned.default)(_is_pinned_dispatch)
    cls.implements(aten._pin_memory.default)(_pin_memory_dispatch)


__all__ = [
    "_FlatQuantizedWeightView",
    "fsdp_post_all_gather",
    "fsdp_pre_all_gather",
    "register_fsdp_ops",
]
