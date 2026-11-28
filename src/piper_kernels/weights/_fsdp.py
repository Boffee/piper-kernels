"""PyTorch FSDP2 (``fully_shard``) storage and collective hooks for quantized weights."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard, cast

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from piper_kernels.weights._dispatch import QuantizedWeight, require_untransposed

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from piper_kernels.weights.convrot.int8.tensor import ConvRotInt8Tensor
    from piper_kernels.weights.nvfp4.tensor import PiperNVFP4Tensor


_SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class FSDPWeightMixin:
    """Opt-in FSDP2 protocol hooks for quantized weights with 2-D row-sharded storage."""

    _supports_flat_1d_view: ClassVar[bool] = True

    qdata: torch.Tensor
    scale: torch.Tensor

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
    ) -> tuple[tuple[torch.Tensor, ...], object]:
        """Expose sharded storage tensors and metadata for PyTorch FSDP2."""
        return fsdp_pre_all_gather(cast("_FSDPWeight", self), mesh)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: object,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
        """Reconstruct or refresh an unsharded weight from FSDP2 all-gather outputs."""
        return fsdp_post_all_gather(
            cast("_FSDPWeight", self),
            all_gather_outputs,
            metadata,
            param_dtype,
            out=out,
        )


if TYPE_CHECKING:
    type _FSDPWeight = ConvRotInt8Tensor | PiperNVFP4Tensor


@dataclass(frozen=True)
class _FSDPAllGatherMetadata:
    """Host metadata carried from ``fsdp_pre_all_gather`` to ``fsdp_post_all_gather``."""

    cls: type[QuantizedWeight]
    local_rows: int
    features: int
    world_size: int
    attributes: dict[str, object]
    optional_tensors: dict[str, torch.Tensor]
    reswizzle_nvfp4_scales: bool
    single_rank_scale: torch.Tensor | None = None


def _is_int8_weight(weight: object) -> TypeGuard[ConvRotInt8Tensor]:
    from piper_kernels.weights.convrot.int8.tensor import ConvRotInt8Tensor  # noqa: PLC0415

    return isinstance(weight, ConvRotInt8Tensor)


def _is_nvfp4_weight(weight: object) -> TypeGuard[PiperNVFP4Tensor]:
    if _is_int8_weight(weight):
        return False
    from piper_kernels.weights.nvfp4.tensor import PiperNVFP4Tensor  # noqa: PLC0415

    return isinstance(weight, PiperNVFP4Tensor)


def _canonical_scale_shape(weight: _FSDPWeight, rows: int, features: int) -> tuple[int, int]:
    if _is_int8_weight(weight):
        return (rows, 1)
    if not _is_nvfp4_weight(weight):
        raise TypeError(f"unsupported quantized weight type {type(weight).__name__}")
    from piper_kernels.weights.nvfp4._layout import canonical_scale_shape  # noqa: PLC0415

    return cast(
        tuple[int, int],
        canonical_scale_shape(
            rows,
            features,
            is_swizzled_scales=weight.is_swizzled_scales,
            block_size=weight.block_size,
        ),
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


def _require_fsdp_weight(weight: _FSDPWeight, operation: str) -> torch.Tensor:
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

    from piper_kernels.weights.nvfp4._layout import validate_nvfp4_weight  # noqa: PLC0415

    return validate_nvfp4_weight(
        weight,
        allowed_ndims=(2,),
        error_type=NotImplementedError,
        prefix=f"{type(weight).__name__} {operation}",
    )


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
    weight: _FSDPWeight,
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
        from piper_kernels.weights.nvfp4._layout import (  # noqa: PLC0415
            BLOCK_SIZE,
            SCALE_ROW_TILE,
            from_blocked,
        )

        reswizzle = weight.is_swizzled_scales and local_rows % SCALE_ROW_TILE != 0
        if reswizzle:
            scale_to_gather = from_blocked(
                canonical_scale.view(torch.uint8),
                local_rows,
                features // BLOCK_SIZE,
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
    target_weight: _FSDPWeight,
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
            from piper_kernels.weights.nvfp4._layout import to_blocked  # noqa: PLC0415

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
    target_weight = cast("_FSDPWeight", target)
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
    weight: _FSDPWeight,
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
        from piper_kernels.weights.nvfp4._layout import BLOCK_SIZE, qdata_shape  # noqa: PLC0415

        expected_qdata_shape = cast(tuple[int, int], qdata_shape(full_rows, metadata.features))
        expected_qdata_dtype = torch.uint8
        expected_target_scale_shape = _canonical_scale_shape(weight, full_rows, metadata.features)
        if metadata.reswizzle_nvfp4_scales:
            if not weight.is_swizzled_scales:
                raise ValueError(
                    "reswizzle_nvfp4_scales requires a weight with swizzled NVFP4 scales"
                )
            expected_scale_shape = (full_rows, metadata.features // BLOCK_SIZE)
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
    weight: _FSDPWeight,
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
        from piper_kernels.weights.nvfp4._layout import to_blocked  # noqa: PLC0415

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
    updated_attrs["dtype"] = param_dtype
    tensors = {"qdata": qdata_full, "scale": scale_full, **metadata.optional_tensors}
    unsharded = metadata.cls.__tensor_unflatten__(tensors, updated_attrs, None, None)
    return cast(torch.Tensor, unsharded), extra_inner_tensors


def _nvfp4_split_scale_row_tile(weight: _FSDPWeight) -> int:
    if _is_int8_weight(weight) or not (_is_nvfp4_weight(weight) and weight.is_swizzled_scales):
        return 0
    from piper_kernels.weights.nvfp4._layout import SCALE_ROW_TILE  # noqa: PLC0415

    return SCALE_ROW_TILE


def _split_dispatch(
    func: Callable[..., object],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[torch.Tensor]:
    weight = cast("_FSDPWeight", args[0])
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

    scale_row_tile = _nvfp4_split_scale_row_tile(weight)
    swizzled = scale_row_tile > 0
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
    weight = cast("_FSDPWeight", args[0])
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
        from piper_kernels.weights.nvfp4._layout import qdata_shape  # noqa: PLC0415

        int8_weight = False
        qdata_dims = cast(tuple[int, int], qdata_shape(rows, features))
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
    updated["dtype"] = logical_dtype
    return cast(torch.Tensor, type(weight).__tensor_unflatten__(tensors, updated, None, None))


def register_fsdp_ops(cls: type[QuantizedWeight]) -> None:
    """Register FSDP2 parameter-sharding and unshard ATen handlers."""
    aten = torch.ops.aten
    cls.implements(aten.split.Tensor)(_split_dispatch)
    cls.implements(aten.new_zeros.default)(_new_zeros_dispatch)


__all__ = [
    "FSDPWeightMixin",
    "fsdp_post_all_gather",
    "fsdp_pre_all_gather",
    "register_fsdp_ops",
]
