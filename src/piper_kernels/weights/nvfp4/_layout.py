"""Canonical layout, swizzle, and allocation helpers for hardware-ready NVFP4 storage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from piper_kernels.weights.nvfp4.tensor import PiperNVFP4Tensor

BLOCK_SIZE = 16
QDATA_BLOCK_SIZE = BLOCK_SIZE // 2
SCALE_ROW_TILE = 128
SCALE_COLUMN_TILE = 64
E4M3_MAX = 448.0
E2M1_MAX = 6.0
# Keep reciprocal(global) / smallest_nonzero_E4M3 finite when computing weight scales.
MIN_WEIGHT_GLOBAL_SCALE = torch.finfo(torch.float32).tiny / (2.0**-9)
SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
SUPPORTED_ACT_QUANT = ("static", "dynamic")


def swap_packed_pairs(qdata: torch.Tensor) -> torch.Tensor:
    """Exchange the two E2M1 values stored in every packed byte."""
    return ((qdata & 0x0F) << 4) | (qdata >> 4)


def per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert a per-tensor amax into the matching FP32 NVFP4 global decoding scale."""
    return amax.to(torch.float32) / (E4M3_MAX * E2M1_MAX)


def tensor_size_fp4x2_to_hp(
    qdata_shape: Sequence[int | torch.SymInt],
    *,
    transposed: bool,
) -> tuple[int | torch.SymInt, ...]:
    """Compute the logical high-precision shape corresponding to packed FP4x2 storage."""
    if not transposed:
        return (*qdata_shape[:-1], cast(Any, qdata_shape[-1]) * 2)
    return (*qdata_shape[:-2], cast(Any, qdata_shape[-2]) * 2, qdata_shape[-1])


def qdata_shape(
    rows: int | torch.SymInt,
    features: int | torch.SymInt,
) -> tuple[int | torch.SymInt, int | torch.SymInt]:
    """Return the UINT8 shape holding two logical FP4 values per byte."""
    features_value = cast(Any, features)
    return cast(
        tuple[int | torch.SymInt, int | torch.SymInt],
        (rows, features_value // 2),
    )


def scale_shape(
    rows: int | torch.SymInt,
    features: int | torch.SymInt,
) -> tuple[int | torch.SymInt, int | torch.SymInt]:
    """Return the physical shape of the swizzled FP8 block scales."""
    rows_value = cast(Any, rows)
    features_value = cast(Any, features)
    return cast(
        tuple[int | torch.SymInt, int | torch.SymInt],
        (
            (rows_value + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * 32,
            (features_value + SCALE_COLUMN_TILE - 1) // SCALE_COLUMN_TILE * 16,
        ),
    )


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Swizzle 2-D row-major block scales into hardware ``SWIZZLE_32_4_4`` layout."""
    is_fp8 = input_matrix.dtype is torch.float8_e4m3fn
    # Route FP8 zero-padding through uint8 so CPU and meta devices avoid unsupported FP8 fills.
    work = input_matrix.view(torch.uint8) if is_fp8 else input_matrix
    rows, cols = work.shape
    n_row_blocks = (rows + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * SCALE_ROW_TILE
    padded_cols = n_col_blocks * 4
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=work.device, dtype=work.dtype)
        padded[:rows, :cols] = work
    else:
        padded = work
    blocks = padded.view(n_row_blocks, SCALE_ROW_TILE, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = (
        blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(n_row_blocks * 32, n_col_blocks * 16)
    )
    return rearranged.view(torch.float8_e4m3fn) if is_fp8 else rearranged


def from_blocked(
    blocked_tensor: torch.Tensor,
    original_rows: int,
    original_cols: int,
) -> torch.Tensor:
    """Inverse of :func:`to_blocked`: recover 2-D row-major scales ``[rows, cols]``."""
    is_fp8 = blocked_tensor.dtype is torch.float8_e4m3fn
    work = blocked_tensor.view(torch.uint8) if is_fp8 else blocked_tensor
    n_row_blocks = (original_rows + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE
    n_col_blocks = (original_cols + 3) // 4
    rearranged = work.reshape(n_row_blocks * n_col_blocks, 32, 16)
    temp = rearranged.reshape(n_row_blocks * n_col_blocks, 32, 4, 4).transpose(1, 2)
    blocks = temp.reshape(n_row_blocks, n_col_blocks, SCALE_ROW_TILE, 4)
    padded = blocks.permute(0, 2, 1, 3).reshape(n_row_blocks * SCALE_ROW_TILE, n_col_blocks * 4)
    result = padded[:original_rows, :original_cols]
    return result.view(torch.float8_e4m3fn) if is_fp8 else result


def has_scale_padding(rows: int, features: int) -> bool:
    """Return whether unused physical scale lanes must be initialized."""
    return rows % SCALE_ROW_TILE != 0 or features % SCALE_COLUMN_TILE != 0


def canonical_scale_shape(
    rows: int | torch.SymInt,
    features: int | torch.SymInt,
    *,
    is_swizzled_scales: bool,
    block_size: int = BLOCK_SIZE,
) -> tuple[int | torch.SymInt, int | torch.SymInt]:
    """Return the canonical 2-D physical shape for NVFP4 FP8 block scales."""
    if is_swizzled_scales:
        return scale_shape(rows, features)
    return cast(
        tuple[int | torch.SymInt, int | torch.SymInt],
        (rows, cast(Any, features) // block_size),
    )


def validate_nvfp4_options(
    *,
    block_size: int,
    dtype: torch.dtype,
    is_swizzled_scales: bool,
    act_quant: object,
    high_first: bool,
) -> None:
    """Validate NVFP4 metadata options shared across construction and storage checks."""
    if dtype not in SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(f"NVFP4 logical dtype must be float16, bfloat16, or float32, got {dtype}")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError(f"NVFP4 block_size must be a positive integer, got {block_size!r}")
    if type(is_swizzled_scales) is not bool:
        raise TypeError(
            f"NVFP4 is_swizzled_scales must be bool, got {type(is_swizzled_scales).__name__}"
        )
    if act_quant is not None and (
        type(act_quant) is not str or act_quant not in SUPPORTED_ACT_QUANT
    ):
        raise ValueError(f"NVFP4 act_quant must be None, 'static', or 'dynamic', got {act_quant!r}")
    if type(high_first) is not bool:
        raise TypeError(f"NVFP4 high_first must be bool, got {type(high_first).__name__}")


def validate_nvfp4_global_scales(
    per_tensor_scale: torch.Tensor | None,
    act_per_tensor_scale: torch.Tensor | None,
    *,
    device: torch.device,
    ndim: int,
    experts: int,
    error_type: type[Exception] = ValueError,
    prefix: str = "NVFP4",
) -> None:
    """Validate optional NVFP4 weight and activation global scales."""
    for name, scale, valid_shapes in (
        ("per_tensor_scale", per_tensor_scale, ((),) if ndim == 2 else ((), (experts, 1, 1))),
        ("act_per_tensor_scale", act_per_tensor_scale, ((),)),
    ):
        if scale is None:
            continue
        if (
            scale.dtype is not torch.float32
            or scale.layout is not torch.strided
            or scale.device != device
        ):
            raise error_type(
                f"{prefix} {name} must be a strided float32 tensor on the weight device"
            )
        if tuple(scale.shape) not in valid_shapes:
            raise error_type(
                f"{prefix} {name} shape {tuple(scale.shape)} is incompatible with {ndim}-D qdata"
            )


def validate_nvfp4_storage(  # noqa: PLR0912, PLR0913
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int,
    dtype: torch.dtype,
    per_tensor_scale: torch.Tensor | None = None,
    act_per_tensor_scale: torch.Tensor | None = None,
    is_swizzled_scales: bool = False,
    act_quant: object = None,
    high_first: bool = False,
    require_contiguous: bool = True,
    allowed_ndims: tuple[int, ...] = (2, 3),
    allow_empty: bool = True,
    allow_meta: bool = True,
    logical_shape: tuple[int, ...] | None = None,
    error_type: type[Exception] = ValueError,
    prefix: str = "NVFP4",
) -> torch.Tensor:
    """Validate NVFP4 storage tensors and return canonical 2-D/3-D block-scale storage."""
    validate_nvfp4_options(
        block_size=block_size,
        dtype=dtype,
        is_swizzled_scales=is_swizzled_scales,
        act_quant=act_quant,
        high_first=high_first,
    )
    if block_size != BLOCK_SIZE:
        raise error_type(f"{prefix} requires block_size={BLOCK_SIZE}, got {block_size}")
    if not allow_meta and qdata.device.type == "meta":
        raise error_type(f"{prefix} cannot update a meta tensor without values")
    if qdata.ndim not in allowed_ndims or qdata.dtype is not torch.uint8:
        ranks = " or ".join(f"{ndim}-D" for ndim in allowed_ndims)
        raise error_type(
            f"{prefix} qdata must be a {ranks} uint8 tensor, got {qdata.dtype} {tuple(qdata.shape)}"
        )
    if qdata.layout is not torch.strided or scale.layout is not torch.strided:
        raise error_type(f"{prefix} weight storage must use strided layout")
    if scale.dtype is not torch.float8_e4m3fn or scale.device != qdata.device:
        raise error_type(
            f"{prefix} scale must be float8_e4m3fn on device {qdata.device}, "
            f"got {scale.dtype} on {scale.device}"
        )

    if logical_shape is not None:
        rows, features = logical_shape[-2], logical_shape[-1]
        expected_qdata = (*logical_shape[:-2], *qdata_shape(rows, features))
        if len(logical_shape) != qdata.ndim or tuple(qdata.shape) != expected_qdata:
            raise error_type(
                f"{prefix} requires packed uint8 qdata with shape {expected_qdata}, "
                f"got {qdata.dtype} {tuple(qdata.shape)}"
            )
    else:
        rows, features = qdata.shape[-2], qdata.shape[-1] * 2
    if not allow_empty and (rows == 0 or features == 0):
        raise error_type(f"{prefix} requires nonempty block-{BLOCK_SIZE} storage")
    if features % BLOCK_SIZE:
        raise error_type(
            f"{prefix} input features {features} must be divisible by block_size {BLOCK_SIZE}"
        )

    expected_2d = canonical_scale_shape(rows, features, is_swizzled_scales=is_swizzled_scales)
    if qdata.ndim == 2:
        flat_shape = (cast(Any, expected_2d[0]) * cast(Any, expected_2d[1]),)
        if tuple(scale.shape) not in (expected_2d, flat_shape):
            raise error_type(
                f"{prefix} scale shape {tuple(scale.shape)} does not match expected "
                f"{expected_2d} (or flat {flat_shape})"
            )
        canonical_scale = scale if tuple(scale.shape) == expected_2d else scale.view(expected_2d)
    else:
        expected_3d = (qdata.shape[0], *expected_2d)
        if tuple(scale.shape) != expected_3d:
            raise error_type(
                f"{prefix} 3-D scale shape {tuple(scale.shape)} does not match expected "
                f"{expected_3d}"
            )
        canonical_scale = scale

    validate_nvfp4_global_scales(
        per_tensor_scale,
        act_per_tensor_scale,
        device=qdata.device,
        ndim=qdata.ndim,
        experts=qdata.shape[0] if qdata.ndim == 3 else 0,
        error_type=error_type,
        prefix=prefix,
    )
    if require_contiguous and any(
        not tensor.is_contiguous()
        for tensor in (qdata, scale, per_tensor_scale, act_per_tensor_scale)
        if tensor is not None
    ):
        raise error_type(
            f"{prefix} storage tensors must be contiguous; use from_quantized to canonicalize "
            "storage"
        )
    return canonical_scale


def validate_nvfp4_weight(
    weight: PiperNVFP4Tensor,
    *,
    require_contiguous: bool = True,
    allowed_ndims: tuple[int, ...] = (2, 3),
    allow_empty: bool = True,
    allow_meta: bool = True,
    error_type: type[Exception] = ValueError,
    prefix: str = "NVFP4",
) -> torch.Tensor:
    """Validate an existing NVFP4 wrapper's storage and return canonical 2-D/3-D scales."""
    return validate_nvfp4_storage(
        weight.qdata,
        weight.scale,
        block_size=weight.block_size,
        dtype=weight.dtype,
        per_tensor_scale=weight.per_tensor_scale,
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=weight.is_swizzled_scales,
        act_quant=weight.act_quant,
        high_first=weight.high_first,
        require_contiguous=require_contiguous,
        allowed_ndims=allowed_ndims,
        allow_empty=allow_empty,
        allow_meta=allow_meta,
        logical_shape=tuple(weight.shape),
        error_type=error_type,
        prefix=prefix,
    )


__all__ = [
    "BLOCK_SIZE",
    "E2M1_MAX",
    "E4M3_MAX",
    "MIN_WEIGHT_GLOBAL_SCALE",
    "QDATA_BLOCK_SIZE",
    "SCALE_COLUMN_TILE",
    "SCALE_ROW_TILE",
    "SUPPORTED_ACT_QUANT",
    "SUPPORTED_LOGICAL_DTYPES",
    "canonical_scale_shape",
    "from_blocked",
    "has_scale_padding",
    "per_tensor_amax_to_scale",
    "qdata_shape",
    "scale_shape",
    "swap_packed_pairs",
    "tensor_size_fp4x2_to_hp",
    "to_blocked",
    "validate_nvfp4_global_scales",
    "validate_nvfp4_options",
    "validate_nvfp4_storage",
    "validate_nvfp4_weight",
]
