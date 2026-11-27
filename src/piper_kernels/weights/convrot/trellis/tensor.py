"""Trellis-coded ConvRot linear weight subclass."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Self

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor

from piper_kernels._triton import runtime
from piper_kernels.trellis._layout import (
    TrellisLayout,
    stored_bytes_bounds,
    validate_block_size,
)
from piper_kernels.weights._dispatch import (
    QuantizedWeight,
    register_conversion_ops,
    unsupported_operation_dispatch,
)
from piper_kernels.weights._matmul import register_matrix_ops
from piper_kernels.weights._views import register_view_ops, require_untransposed
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.convrot.int8._quantization import dequantize_weight

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from piper_kernels.trellis import TrellisHessian, TrellisStorage

_SUPPORTED_LOGICAL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _resolve_block_size(
    block_size: int | None,
    blocks_per_row: int,
    inferred_width: int | None,
    *,
    prefix: str,
) -> int:
    if block_size is not None:
        return block_size
    if inferred_width is None or blocks_per_row == 0 or inferred_width % blocks_per_row:
        raise ValueError(f"{prefix} output storage is incompatible")
    return inferred_width // blocks_per_row


def _validate_uniform_storage(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    block_size: int | None,
    inferred_width: int | None,
    *,
    prefix: str,
) -> tuple[int, int, int, TrellisLayout | None]:
    if bitstreams.dtype is not torch.uint8 or bitstreams.ndim != 3:
        raise ValueError(
            f"{prefix} bitstreams must be a 3-D uint8 tensor "
            f"[out, blocks, bytes], got {bitstreams.dtype} {tuple(bitstreams.shape)}"
        )
    rows, blocks_per_row, bytes_per_block = bitstreams.shape
    resolved_block_size = _resolve_block_size(
        block_size, blocks_per_row, inferred_width, prefix=prefix
    )
    if type(bytes_per_block) is int:
        layout: TrellisLayout | None = TrellisLayout.from_storage(
            resolved_block_size, bytes_per_block
        )
    else:
        min_stored, max_stored = stored_bytes_bounds(resolved_block_size)
        if not min_stored <= bytes_per_block <= max_stored:
            raise ValueError(
                f"trellis blocks of {resolved_block_size} weights take {min_stored} to "
                f"{max_stored} stored bytes, got {bytes_per_block}"
            )
        layout = None
    if block_scales.dtype is not torch.float8_e4m3fn or tuple(block_scales.shape) != (
        rows,
        blocks_per_row,
    ):
        raise ValueError(
            f"{prefix} block_scales must be float8_e4m3fn with shape "
            f"{(rows, blocks_per_row)}, got {block_scales.dtype} {tuple(block_scales.shape)}"
        )
    return rows, blocks_per_row, resolved_block_size, layout


def _validate_variable_row_storage(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    block_size: int | None,
    inferred_width: int | None,
    *,
    prefix: str,
) -> tuple[int, int, int]:
    if block_size is not None:
        validate_block_size(block_size)
    if bitstreams.dtype is not torch.uint8 or bitstreams.ndim != 1:
        raise ValueError(
            f"{prefix} variable-row bitstreams must be a 1-D uint8 tensor "
            f"[total_bytes], got {bitstreams.dtype} {tuple(bitstreams.shape)}"
        )
    if block_scales.dtype is not torch.float8_e4m3fn or block_scales.ndim != 2:
        raise ValueError(
            f"{prefix} block_scales must be a 2-D float8_e4m3fn tensor "
            f"[out, blocks], got {block_scales.dtype} {tuple(block_scales.shape)}"
        )
    rows, blocks_per_row = block_scales.shape
    resolved_block_size = validate_block_size(
        _resolve_block_size(block_size, blocks_per_row, inferred_width, prefix=prefix)
    )
    if row_widths.dtype is not torch.uint8 or tuple(row_widths.shape) != (rows,):
        raise ValueError(
            f"{prefix} row_widths must be uint8 with shape {(rows,)}, "
            f"got {row_widths.dtype} {tuple(row_widths.shape)}"
        )
    return rows, blocks_per_row, resolved_block_size


def validate_trellis_tensors(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    block_size: int | None,
    *,
    row_widths: torch.Tensor | None = None,
    inferred_width: int | None = None,
    require_contiguous: bool = True,
    allow_1d_scale: bool = False,
    prefix: str = "ConvRot Trellis",
) -> tuple[int, int, int, TrellisLayout | None, torch.Tensor]:
    """Validate trellis tensor shapes, dtypes, layout, and devices."""
    tensors = (
        (bitstreams, block_scales, scale)
        if row_widths is None
        else (bitstreams, block_scales, scale, row_widths)
    )
    if any(tensor.layout is not torch.strided for tensor in tensors):
        raise ValueError(f"{prefix} weight storage must use strided layout")

    device = bitstreams.device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError(
            f"{prefix} weight tensors must share a device, "
            f"got {tuple(tensor.device for tensor in tensors)}"
        )

    if row_widths is None:
        rows, blocks_per_row, resolved_block_size, layout = _validate_uniform_storage(
            bitstreams, block_scales, block_size, inferred_width, prefix=prefix
        )
    else:
        rows, blocks_per_row, resolved_block_size = _validate_variable_row_storage(
            bitstreams, block_scales, row_widths, block_size, inferred_width, prefix=prefix
        )
        layout = None

    expected_shapes = ((rows,), (rows, 1)) if allow_1d_scale else ((rows, 1),)
    if tuple(scale.shape) not in expected_shapes:
        shape_desc = f"({rows},) or ({rows}, 1)" if allow_1d_scale else f"{(rows, 1)}"
        raise ValueError(f"{prefix} scale must have shape {shape_desc}, got {tuple(scale.shape)}")
    if scale.dtype is not torch.float32:
        raise ValueError(f"{prefix} scale must be float32, got {scale.dtype}")

    if require_contiguous and any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(
            f"{prefix} storage tensors must be contiguous; "
            "use from_quantized to canonicalize storage"
        )
    return rows, blocks_per_row, resolved_block_size, layout, scale.view(rows, 1)


def validate_storage(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    group_size: int,
    dtype: torch.dtype,
    *,
    row_widths: torch.Tensor | None = None,
    require_contiguous: bool = True,
) -> tuple[int, int, TrellisLayout | None]:
    """Validate ConvRot Trellis storage tensors and return ``(rows, in_features, layout)``."""
    validate_group_size(group_size)
    if dtype not in _SUPPORTED_LOGICAL_DTYPES:
        raise ValueError(
            f"ConvRot Trellis logical dtype must be float16, bfloat16, or float32, got {dtype}"
        )
    rows, blocks_per_row, resolved_block_size, layout, _ = validate_trellis_tensors(
        bitstreams,
        block_scales,
        scale,
        block_size,
        row_widths=row_widths,
        require_contiguous=require_contiguous,
        allow_1d_scale=False,
        prefix="ConvRot Trellis",
    )
    in_features = blocks_per_row * resolved_block_size
    if in_features % group_size:
        raise ValueError(
            f"ConvRot Trellis in_features {in_features} is not divisible by group size {group_size}"
        )
    return rows, in_features, layout


class ConvRotTrellisTensor(QuantizedWeight):  # pyright: ignore[reportIncompatibleMethodOverride]
    """Trellis-coded ConvRot 2-D linear weight for direct ``F.linear`` execution.

    Stores the same packed tensors encoded by :class:`~piper_kernels.trellis.TrellisStorage`:
    ``bitstreams`` (uint8 ``[out, blocks, bytes]`` for uniform rates, or 1-D uint8
    ``[total_bytes]`` with ``row_widths`` uint8 ``[out]`` for variable-row rates),
    ``block_scales`` (float8_e4m3fn ``[out, blocks]``), and ``scale`` (float32
    ``[out, 1]``).

    Caller preconditions match trellis decoding: E4M3 block scales must be non-NaN
    (not ``0x7F`` or ``0xFF``), and variable-row ``row_widths`` (including 2 trailing
    wrap guard bytes) must lie in ``[ceil(2 * block_size / 8) + 2, floor(6 * block_size / 8) + 2]``
    and fit within ``bitstreams``. Small-batch inference (``M <= 4``) and ``dequantize()``
    use continuous block-scaled FP32 accumulation without per-weight INT8 rounding,
    while larger batches (``M >= 5``) decode prescaled INT8 weights for W8A8 GEMM.
    Wrapping and inference never issue host readbacks or synchronizations.
    """

    _format_name: ClassVar[str] = "ConvRot Trellis"
    _update_format_name: ClassVar[str] = "ConvRot Trellis"
    _contiguous_memory_format_only: ClassVar[bool] = True

    tensor_data_names: ClassVar[list[str]] = ["bitstreams", "block_scales", "scale"]
    tensor_attribute_names: ClassVar[list[str]] = ["block_size", "group_size", "dtype"]
    optional_tensor_data_names: ClassVar[list[str]] = ["row_widths"]
    row_widths: torch.Tensor | None = None
    optional_tensor_attribute_names: ClassVar[list[str]] = ["transposed"]
    transposed: bool = False

    def __new__(
        cls,
        bitstreams: torch.Tensor,
        block_scales: torch.Tensor,
        scale: torch.Tensor,
        block_size: int,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
        row_widths: torch.Tensor | None = None,
        transposed: bool = False,
    ) -> ConvRotTrellisTensor:
        if type(transposed) is not bool:
            raise TypeError("ConvRot Trellis transposed must be bool")
        rows, in_features, _ = validate_storage(
            bitstreams,
            block_scales,
            scale,
            block_size,
            group_size,
            dtype,
            row_widths=row_widths,
            require_contiguous=True,
        )
        return torch.Tensor._make_wrapper_subclass(
            cls,
            (in_features, rows) if transposed else (rows, in_features),
            strides=(1, max(1, in_features)) if transposed else None,
            device=bitstreams.device,
            dtype=dtype,
            requires_grad=False,
        )

    def __init__(
        self,
        bitstreams: torch.Tensor,
        block_scales: torch.Tensor,
        scale: torch.Tensor,
        block_size: int,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
        row_widths: torch.Tensor | None = None,
        transposed: bool = False,
    ) -> None:
        super().__init__()
        if self.dtype is not dtype:
            raise RuntimeError(f"ConvRot Trellis wrapper dtype mismatch: {self.dtype} != {dtype}")
        self.bitstreams = bitstreams
        self.block_scales = block_scales
        self.scale = scale
        self.block_size = block_size
        self.group_size = group_size
        self.row_widths = row_widths
        self.transposed = transposed

    def _transpose(self) -> Self:
        """Reverse the logical axes while retaining canonical rowwise storage."""
        return type(self)(
            self.bitstreams,
            self.block_scales,
            self.scale,
            self.block_size,
            self.group_size,
            self.dtype,
            self.row_widths,
            not self.transposed,
        )

    @classmethod
    def from_quantized(
        cls,
        bitstreams: torch.Tensor,
        block_scales: torch.Tensor,
        scale: torch.Tensor,
        *,
        block_size: int,
        group_size: int,
        logical_dtype: torch.dtype = torch.bfloat16,
        row_widths: torch.Tensor | None = None,
    ) -> ConvRotTrellisTensor:
        """Wrap trellis-coded checkpoint storage without decoding to dense INT8.

        Accepts ``scale`` with shape ``[out]`` or ``[out, 1]``. Contiguous storage is
        reused zero-copy; non-contiguous storage is canonicalized.
        """
        if scale.layout is not torch.strided:
            raise ValueError("ConvRot Trellis weight storage must use strided layout")
        canonical_scale = scale.view(-1, 1) if scale.ndim == 1 else scale
        validate_storage(
            bitstreams,
            block_scales,
            canonical_scale,
            block_size,
            group_size,
            logical_dtype,
            row_widths=row_widths,
            require_contiguous=False,
        )
        return cls(
            bitstreams.contiguous(),
            block_scales.contiguous(),
            canonical_scale.contiguous(),
            block_size,
            group_size,
            logical_dtype,
            None if row_widths is None else row_widths.contiguous(),
        )

    @classmethod
    def from_trellis_storage(
        cls,
        storage: TrellisStorage,
        *,
        group_size: int,
        logical_dtype: torch.dtype = torch.bfloat16,
    ) -> ConvRotTrellisTensor:
        """Wrap an encoded :class:`~piper_kernels.trellis.TrellisStorage` zero-copy."""
        return cls.from_quantized(
            storage.bitstreams,
            storage.block_scales,
            storage.scale,
            block_size=storage.block_size,
            group_size=group_size,
            logical_dtype=logical_dtype,
        )

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        *,
        group_size: int,
        bits_per_weight: float | None = None,
        bytes_per_block: int | None = None,
        hessian: torch.Tensor | TrellisHessian | None = None,
        hessian_damping: float | None = None,
    ) -> ConvRotTrellisTensor:
        """Rotate and trellis-encode a high-precision 2-D weight into a ``ConvRotTrellisTensor``.

        Delegates encoding to :meth:`~piper_kernels.trellis.TrellisStorage.from_hp`, which
        requires an NVIDIA SM80+ GPU and finite input weights.
        """
        from piper_kernels.trellis import TrellisStorage  # noqa: PLC0415

        storage = TrellisStorage.from_hp(
            hp_tensor,
            group_size=group_size,
            bits_per_weight=bits_per_weight,
            bytes_per_block=bytes_per_block,
            hessian=hessian,
            hessian_damping=hessian_damping,
        )
        return cls.from_trellis_storage(
            storage,
            group_size=group_size,
            logical_dtype=hp_tensor.dtype,
        )

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Recover the logical weight in the unrotated basis and requested dtype."""
        if output_dtype is None:
            output_dtype = self.dtype
        if output_dtype not in _SUPPORTED_LOGICAL_DTYPES:
            raise ValueError(
                "ConvRot Trellis dequantization output dtype must be float16, bfloat16, or "
                f"float32, got {output_dtype}"
            )
        rows, in_features, layout = validate_storage(
            self.bitstreams,
            self.block_scales,
            self.scale,
            self.block_size,
            self.group_size,
            self.dtype,
            row_widths=self.row_widths,
            require_contiguous=True,
        )
        if (
            self.device.type == "meta"
            or rows == 0
            or in_features == 0
            or isinstance(self.bitstreams, FakeTensor)
            or detect_fake_mode() is not None
        ):
            return self.bitstreams.new_empty(self.shape, dtype=output_dtype)

        if runtime.supports_device(self.device):
            from piper_kernels.trellis.triton import (  # noqa: PLC0415
                decode_fp32_out,
                decode_variable_rows_fp32_out,
            )

            values = torch.empty((rows, in_features), device=self.device, dtype=torch.float32)
            if self.row_widths is None:
                assert layout is not None
                decode_fp32_out(
                    self.bitstreams.view(-1, layout.bytes_per_block),
                    self.block_scales.view(torch.uint8).view(-1),
                    layout,
                    values,
                )
            else:
                decode_variable_rows_fp32_out(
                    self.bitstreams,
                    self.block_scales.view(torch.uint8),
                    self.row_widths,
                    self.block_size,
                    values,
                )
        elif self.row_widths is None:
            from piper_kernels.trellis._reference import decode_reference  # noqa: PLC0415

            assert layout is not None
            values = decode_reference(
                self.bitstreams, self.block_scales, layout, round_int8=False
            ).reshape(rows, in_features)
        else:
            from piper_kernels.trellis._reference import (  # noqa: PLC0415
                decode_variable_rows_reference,
            )

            values = decode_variable_rows_reference(
                self.bitstreams,
                self.block_scales,
                self.row_widths,
                self.block_size,
                round_int8=False,
            )

        with torch.amp.autocast(values.device.type, enabled=False):
            result = dequantize_weight(values, self.scale, self.group_size, output_dtype)
        return result.t() if self.transposed else result

    def _stable_hash_for_caching(self) -> str:
        """Return a metadata fingerprint for AOTAutograd's cross-process cache."""
        return repr(
            (
                type(self).__qualname__,
                "piper_kernels::convrot_int8_linear",
                tuple(self.shape),
                self.stride(),
                str(self.device),
                str(self.dtype),
                tuple(self.bitstreams.shape),
                self.bitstreams.stride(),
                tuple(self.block_scales.shape),
                self.block_scales.stride(),
                tuple(self.scale.shape),
                self.scale.stride(),
                None if self.row_widths is None else tuple(self.row_widths.shape),
                None if self.row_widths is None else self.row_widths.stride(),
                self.block_size,
                self.group_size,
                self.transposed,
            )
        )

    def _validate_storage_for_export(self) -> None:
        require_untransposed(self, "storage export")
        validate_storage(
            self.bitstreams,
            self.block_scales,
            self.scale,
            self.block_size,
            self.group_size,
            self.dtype,
            row_widths=self.row_widths,
            require_contiguous=True,
        )

    def _copy_storage_tensors(self, operation: str) -> dict[str, torch.Tensor]:
        require_untransposed(self, operation)
        validate_storage(
            self.bitstreams,
            self.block_scales,
            self.scale,
            self.block_size,
            self.group_size,
            self.dtype,
            row_widths=self.row_widths,
            require_contiguous=True,
        )
        names, _ = self.__tensor_flatten__()
        return {name: getattr(self, name) for name in names}

    def _update_target(self, operation: str) -> tuple[Any, tuple[object, ...]]:
        raise NotImplementedError(f"ConvRot Trellis does not support in-place {operation}")

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
    ) -> tuple[tuple[torch.Tensor, ...], object]:
        del mesh
        raise NotImplementedError("ConvRot Trellis does not support FSDP2")

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: object,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
        del all_gather_outputs, metadata, param_dtype, out
        raise NotImplementedError("ConvRot Trellis does not support FSDP2")


register_view_ops(ConvRotTrellisTensor)
register_matrix_ops(ConvRotTrellisTensor)
register_conversion_ops(ConvRotTrellisTensor, format_name=ConvRotTrellisTensor._format_name)
ConvRotTrellisTensor.implements(
    [
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.addmm_.default,
        torch.ops.aten.embedding.default,
    ]
)(unsupported_operation_dispatch)


@ConvRotTrellisTensor.implements(torch.ops.aten.linear.default)
@ConvRotTrellisTensor.implements_torch_function(torch.nn.functional.linear)
def _linear_dispatch(
    func: Callable[..., torch.Tensor],
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    from piper_kernels.linear.convrot.int8._functional import linear_dispatch  # noqa: PLC0415

    return linear_dispatch(func, types, args, kwargs)


__all__ = ["ConvRotTrellisTensor", "validate_storage"]
