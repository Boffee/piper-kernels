"""Chunk-major quantized KV cache storage specification, pool container, and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, NoReturn

import torch
from torch._guards import detect_fake_mode
from torch.autograd import forward_ad

from piper_kernels._triton.targets import AcceleratorTarget

from . import _backend
from ._kv_overlap import _slot_spans_overlap, _SymInt, _validate_slot_strided

KV_CHUNK_TOKENS = 1024
_KV_TILE_TOKENS = 64
_KV_TILES_PER_CHUNK = KV_CHUNK_TOKENS // _KV_TILE_TOKENS

_SUPPORTED_HEAD_DIMS = (64, 128)
_SUPPORTED_KEY_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_KV_STORAGE_DTYPES = (torch.int8, torch.uint8)
_SUPPORTED_LSE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_KEY_SCALE_COUNTS = (_KV_TILES_PER_CHUNK, KV_CHUNK_TOKENS)
_POOL_FIELD_NAMES = (
    "key_int8",
    "key_scale",
    "key_center",
    "value_int8",
    "value_scale",
    "value_mean",
)
_SM_TARGET_PATTERN = re.compile(r"sm_?(\d{2,3})")
_GFX_TARGET_PATTERN = re.compile(r"gfx[0-9a-f]+(?::[a-z0-9_+-]+)*")
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class _KVChunkConfig:
    key_scales: int
    grouped_qk: bool
    pack_wmma: bool
    prefill_backend: Literal["amd", "nvidia"] | None = None
    target: AcceleratorTarget | None = None


def _require_tensor(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _require_tensors(
    pairs: tuple[tuple[str, object], ...],
    *,
    allow_none: bool = False,
) -> None:
    for name, value in pairs:
        if allow_none and value is None:
            continue
        if not isinstance(value, torch.Tensor):
            suffix = " or None" if allow_none else ""
            raise TypeError(f"{name} must be a torch.Tensor{suffix}, got {type(value).__name__}")


def _describe_invalid_value(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    return type(value).__name__


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {_describe_invalid_value(value)}")
    return value


def _require_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
    allow_symint: bool = False,
) -> _SymInt:
    allowed = (int, torch.SymInt) if allow_symint else (int,)
    if isinstance(value, bool) or not isinstance(value, allowed):
        raise TypeError(f"{name} must be an integer, got {_describe_invalid_value(value)}")
    if isinstance(value, torch.SymInt) or torch.compiler.is_compiling():
        torch._check(value >= minimum, lambda: f"{name} must be >= {minimum}")
        if maximum is not None:
            torch._check(value <= maximum, lambda: f"{name} must be <= {maximum}")
        return value
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _is_fake_or_compiling(tensors: tuple[torch.Tensor, ...]) -> bool:
    return torch.compiler.is_compiling() or detect_fake_mode(tensors) is not None


def _resolve_target(target: AcceleratorTarget | torch.device | str | None) -> AcceleratorTarget:
    if target is None:
        return AcceleratorTarget(backend="cpu")
    if isinstance(target, AcceleratorTarget):
        return target
    if isinstance(target, torch.device):
        return AcceleratorTarget.from_device(target)
    if isinstance(target, str):
        normalized = target.strip().lower()
        if normalized.startswith("sm"):
            match = _SM_TARGET_PATTERN.fullmatch(normalized)
            if match is None:
                raise ValueError(f"Invalid CUDA SM target specifier {target!r}")
            return AcceleratorTarget(backend="cuda", architecture=f"sm{match.group(1)}")
        if normalized.startswith("gfx"):
            if _GFX_TARGET_PATTERN.fullmatch(normalized) is None:
                raise ValueError(f"Invalid HIP GFX target specifier {target!r}")
            return AcceleratorTarget(backend="hip", architecture=normalized)
        return AcceleratorTarget.from_device(torch.device(target))
    raise TypeError(f"device must be torch.device, str, or None, got {target!r}")


def _resolve_kv_chunk_config(
    target: AcceleratorTarget | torch.device | str | None = None,
) -> _KVChunkConfig:
    resolved = _resolve_target(target)
    key_scales, grouped_qk, pack_wmma = _backend.kv_chunk_layout(resolved)
    return _KVChunkConfig(
        key_scales=key_scales,
        grouped_qk=grouped_qk,
        pack_wmma=pack_wmma,
        prefill_backend=_backend.prefill_backend(resolved),
        target=resolved,
    )


def kv_chunk_storage_specs(
    num_slots: int,
    kv_heads: int,
    head_dim: int,
    *,
    device: torch.device | str | None = None,
    kv_dtype: torch.dtype = torch.int8,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return buffer shapes and dtypes required to allocate a ``PiperKVChunkPool``.

    Args:
        num_slots: Number of 1024-token KV chunk slots (``>= 0``).
        kv_heads: Number of KV heads per slot (``>= 1``).
        head_dim: Feature width per head (``64`` or ``128``).
        device: Target PyTorch device, device string (e.g. ``"cuda:0"``, ``"cpu"``),
            target architecture string (e.g. ``"sm80"``, ``"sm120"``, ``"gfx1201"``),
            or ``None`` (defaults to ``"cpu"`` portable layout). Pass the target GPU
            device when allocating host or device staging buffers for that target so
            target-dependent ``key_scale`` shapes (``1024`` on SM8x vs ``16`` on
            SM12x / RDNA4 / portable) match the executing GPU backend.
        kv_dtype: Storage dtype for ``key_int8`` and ``value_int8`` (``torch.int8``
            or ``torch.uint8`` for shared byte arenas).

    Returns:
        Mapping from the 6 ``PiperKVChunkPool`` field names (``key_int8``,
        ``key_scale``, ``key_center``, ``value_int8``, ``value_scale``, ``value_mean``)
        to ``(shape, dtype)`` tuples.
    """
    _require_int(num_slots, name="num_slots", minimum=0)
    _require_int(kv_heads, name="kv_heads", minimum=1)
    _require_int(head_dim, name="head_dim", minimum=1)
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        supported = ", ".join(map(str, _SUPPORTED_HEAD_DIMS))
        raise ValueError(f"head_dim must be one of {supported}, got {head_dim}")
    if kv_dtype not in _SUPPORTED_KV_STORAGE_DTYPES:
        raise ValueError(f"kv_dtype must be torch.int8 or torch.uint8, got {kv_dtype}")
    if device is not None and not isinstance(device, (torch.device, str)):
        raise TypeError(f"device must be torch.device, str, or None, got {device!r}")

    config = _resolve_kv_chunk_config(device)
    return {
        "key_int8": ((num_slots, kv_heads, KV_CHUNK_TOKENS, head_dim), kv_dtype),
        "key_scale": ((num_slots, kv_heads, config.key_scales), torch.float32),
        "key_center": ((num_slots, kv_heads, head_dim), torch.float32),
        "value_int8": (
            (num_slots, kv_heads, _KV_TILES_PER_CHUNK, head_dim, _KV_TILE_TOKENS),
            kv_dtype,
        ),
        "value_scale": ((num_slots, kv_heads, KV_CHUNK_TOKENS), torch.float32),
        "value_mean": ((num_slots, kv_heads, head_dim), torch.float32),
    }


def _validate_pool_dtypes(
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
) -> None:
    if key_int8.dtype not in _SUPPORTED_KV_STORAGE_DTYPES:
        raise ValueError(f"key_int8 must use torch.int8 or torch.uint8, got {key_int8.dtype}")
    if value_int8.dtype is not key_int8.dtype:
        raise ValueError(
            f"value_int8 dtype ({value_int8.dtype}) must match key_int8 dtype ({key_int8.dtype})"
        )
    for name, tensor in (
        ("key_scale", key_scale),
        ("key_center", key_center),
        ("value_scale", value_scale),
        ("value_mean", value_mean),
    ):
        if tensor.dtype is not torch.float32:
            raise ValueError(f"{name} must use float32, got {tensor.dtype}")


def _validate_pool_shapes_and_dtypes(
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
) -> tuple[_SymInt, _SymInt, _SymInt, _SymInt]:
    _validate_pool_dtypes(key_int8, key_scale, key_center, value_int8, value_scale, value_mean)
    if key_int8.ndim != 4:
        raise ValueError(
            f"key_int8 must have 4 dims [num_slots, kv_heads, {KV_CHUNK_TOKENS}, head_dim], "
            f"got {tuple(key_int8.shape)}"
        )
    num_slots, kv_heads, chunk_tokens, head_dim = key_int8.shape
    if num_slots < 0:
        raise ValueError(f"num_slots must be >= 0, got {num_slots}")
    if kv_heads < 1:
        raise ValueError(f"kv_heads must be >= 1, got {kv_heads}")
    if chunk_tokens != KV_CHUNK_TOKENS:
        raise ValueError(f"key_int8 chunk length must be {KV_CHUNK_TOKENS}, got {chunk_tokens}")
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        supported = ", ".join(map(str, _SUPPORTED_HEAD_DIMS))
        raise ValueError(f"head_dim must be one of {supported}, got {head_dim}")

    if key_scale.ndim != 3:
        raise ValueError(
            f"key_scale must have 3 dims [num_slots, kv_heads, key_scales], "
            f"got {tuple(key_scale.shape)}"
        )
    key_scales = key_scale.shape[2]
    if key_scales not in _KEY_SCALE_COUNTS:
        raise ValueError(
            f"key_scale must have shape ({num_slots}, {kv_heads}, 16|1024), "
            f"got {tuple(key_scale.shape)}"
        )
    expected_shapes = (
        ("key_scale", key_scale, (num_slots, kv_heads, key_scales)),
        ("key_center", key_center, (num_slots, kv_heads, head_dim)),
        (
            "value_int8",
            value_int8,
            (num_slots, kv_heads, _KV_TILES_PER_CHUNK, head_dim, _KV_TILE_TOKENS),
        ),
        ("value_scale", value_scale, (num_slots, kv_heads, KV_CHUNK_TOKENS)),
        ("value_mean", value_mean, (num_slots, kv_heads, head_dim)),
    )
    for name, tensor, expected in expected_shapes:
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
    return num_slots, kv_heads, head_dim, key_scales


def _validate_pool_tensors(
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    *,
    check_disjoint: bool = True,
) -> tuple[_SymInt, _SymInt, _SymInt, _SymInt]:
    tensors = (
        ("key_int8", key_int8),
        ("key_scale", key_scale),
        ("key_center", key_center),
        ("value_int8", value_int8),
        ("value_scale", value_scale),
        ("value_mean", value_mean),
    )
    _require_tensors(tensors)
    device = key_int8.device
    for name, tensor in tensors[1:]:
        if tensor.device != device:
            raise ValueError(
                f"pool buffers must share device {device}, got {name} on {tensor.device}"
            )

    num_slots, kv_heads, head_dim, key_scales = _validate_pool_shapes_and_dtypes(
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
    )
    for name, tensor in tensors:
        _validate_slot_strided(tensor, name=name)
    if check_disjoint and not torch.compiler.is_compiling():
        for i, (name_a, tensor_a) in enumerate(tensors):
            for name_b, tensor_b in tensors[i + 1 :]:
                if _slot_spans_overlap(tensor_a, tensor_b):
                    raise ValueError(
                        f"pool buffers {name_a} and {name_b} must not overlap in storage"
                    )
    return num_slots, kv_heads, head_dim, key_scales


@dataclass(frozen=True, slots=True)
class PiperKVChunkPool:
    """Global chunk-major quantized KV pool shared across layers for one head layout.

    Each slot holds one finalized 1024-token chunk per KV head:
    - ``key_int8``: ``[num_slots, kv_heads, 1024, head_dim]`` (``int8`` or ``uint8``).
    - ``key_scale``: ``[num_slots, kv_heads, 16 | 1024]`` (``float32``).
    - ``key_center``: ``[num_slots, kv_heads, head_dim]`` (``float32``).
    - ``value_int8``: ``[num_slots, kv_heads, 16, head_dim, 64]`` (``int8`` or ``uint8``,
      target-specific opaque storage ordering along the trailing 64-token axis).
    - ``value_scale``: ``[num_slots, kv_heads, 1024]`` (``float32``).
    - ``value_mean``: ``[num_slots, kv_heads, head_dim]`` (``float32``).

    All six tensors must share the same device and be dense row-major within each slot with
    16-byte-aligned base pointers, storage offsets, and multi-slot ``stride(0)`` byte steps
    (``stride(0) >= per_slot_elements``), and pairwise disjoint per-slot byte spans.
    """

    key_int8: torch.Tensor
    key_scale: torch.Tensor
    key_center: torch.Tensor
    value_int8: torch.Tensor
    value_scale: torch.Tensor
    value_mean: torch.Tensor

    def __post_init__(self) -> None:
        _validate_pool_tensors(
            self.key_int8,
            self.key_scale,
            self.key_center,
            self.value_int8,
            self.value_scale,
            self.value_mean,
        )

    @property
    def num_slots(self) -> _SymInt:
        """Number of chunk slots addressed along dimension 0."""
        return self.key_int8.shape[0]

    @property
    def kv_heads(self) -> _SymInt:
        """Number of KV heads stored per chunk slot."""
        return self.key_int8.shape[1]

    @property
    def head_dim(self) -> _SymInt:
        """Head feature width (64 or 128)."""
        return self.key_int8.shape[3]

    @property
    def key_scales(self) -> _SymInt:
        """Number of FP32 QK scales per chunk per head (16 or 1024)."""
        return self.key_scale.shape[2]

    @property
    def kv_dtype(self) -> torch.dtype:
        """Element storage dtype of ``key_int8`` and ``value_int8`` (``int8`` or ``uint8``)."""
        return self.key_int8.dtype

    @property
    def device(self) -> torch.device:
        """PyTorch device owning the pool buffers."""
        return self.key_int8.device

    @classmethod
    def from_tensors(
        cls,
        key_int8: torch.Tensor,
        key_scale: torch.Tensor,
        key_center: torch.Tensor,
        value_int8: torch.Tensor,
        value_scale: torch.Tensor,
        value_mean: torch.Tensor,
    ) -> PiperKVChunkPool:
        """Wrap and validate pre-allocated persistent KV chunk buffers without copying.

        Accepts contiguous or strided-``stride(0)`` views (such as interleaved
        ``[num_slots, 2, kv_heads, 1024, head_dim]`` slices and dynamic ``uint8`` arenas).
        Performs pure metadata validation with zero tensor operations or device probes.
        """
        return cls(
            key_int8=key_int8,
            key_scale=key_scale,
            key_center=key_center,
            value_int8=value_int8,
            value_scale=value_scale,
            value_mean=value_mean,
        )

    @classmethod
    def empty(
        cls,
        num_slots: int,
        kv_heads: int,
        head_dim: int,
        *,
        device: torch.device | str | None = None,
        kv_dtype: torch.dtype = torch.int8,
    ) -> PiperKVChunkPool:
        """Allocate uninitialized persistent KV chunk buffers for ``device``.

        Args:
            num_slots: Number of 1024-token chunk slots (``>= 0``).
            kv_heads: Number of KV heads per slot (``>= 1``).
            head_dim: Head feature width (``64`` or ``128``).
            device: Target PyTorch device or device string (defaults to ``"cpu"``).
            kv_dtype: Element dtype for ``key_int8`` and ``value_int8`` (``int8`` or ``uint8``).

        Returns:
            A validated ``PiperKVChunkPool`` with freshly allocated storage on ``device``.
        """
        if device is not None and not isinstance(device, (torch.device, str)):
            raise TypeError(f"device must be torch.device, str, or None, got {device!r}")
        resolved_device = torch.device("cpu" if device is None else device)
        specs = kv_chunk_storage_specs(
            num_slots,
            kv_heads,
            head_dim,
            device=resolved_device,
            kv_dtype=kv_dtype,
        )
        return cls(
            key_int8=torch.empty(
                specs["key_int8"][0], device=resolved_device, dtype=specs["key_int8"][1]
            ),
            key_scale=torch.empty(
                specs["key_scale"][0], device=resolved_device, dtype=specs["key_scale"][1]
            ),
            key_center=torch.empty(
                specs["key_center"][0], device=resolved_device, dtype=specs["key_center"][1]
            ),
            value_int8=torch.empty(
                specs["value_int8"][0], device=resolved_device, dtype=specs["value_int8"][1]
            ),
            value_scale=torch.empty(
                specs["value_scale"][0], device=resolved_device, dtype=specs["value_scale"][1]
            ),
            value_mean=torch.empty(
                specs["value_mean"][0], device=resolved_device, dtype=specs["value_mean"][1]
            ),
        )


def _reject_autograd(
    *tensors: torch.Tensor | None,
    operator: str = "freeze_kv_chunks_",
) -> None:
    has_reverse_grad = torch.is_grad_enabled() and any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in tensors
    )
    has_forward_grad = any(
        isinstance(tensor, torch.Tensor) and forward_ad.unpack_dual(tensor).tangent is not None
        for tensor in tensors
    )
    if has_reverse_grad or has_forward_grad:
        raise RuntimeError(f"{operator} is inference-only and does not support autograd")


def _validate_strided_tensor(
    tensor: object,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype | tuple[torch.dtype, ...],
    dtype_label: str,
    ndim: int | None = None,
    ndim_desc: str | None = None,
    last_dim_contiguous: bool = False,
) -> torch.Tensor:
    checked = _require_tensor(tensor, name=name)
    if checked.device != device:
        raise ValueError(f"{name} must be on {device}, got {checked.device}")
    valid_dtype = checked.dtype in dtype if isinstance(dtype, tuple) else checked.dtype is dtype
    if not valid_dtype:
        raise ValueError(f"{name} must use {dtype_label}, got {checked.dtype}")
    if checked.layout is not torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if checked.is_neg() or checked.is_conj():
        raise ValueError(f"{name} must not use negated or conjugated views")
    if ndim is not None and checked.ndim != ndim:
        desc = f"{ndim} dims" if ndim_desc is None else ndim_desc
        raise ValueError(f"{name} must have {desc}, got {tuple(checked.shape)}")
    if last_dim_contiguous and checked.stride(-1) != 1:
        raise ValueError(f"{name} head dimension must be contiguous")
    return checked


def _validate_int32_1d(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    length: _SymInt,
) -> None:
    _validate_strided_tensor(
        tensor,
        name=name,
        device=device,
        dtype=torch.int32,
        dtype_label="int32",
    )
    if tuple(tensor.shape) != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _resolve_checked_config(device: torch.device, key_scales: _SymInt) -> _KVChunkConfig:
    if device.type == "cuda":
        config = _resolve_kv_chunk_config(device)
        if key_scales != config.key_scales:
            raise ValueError(
                f"pool key_scales ({key_scales}) does not match device requirement "
                f"({config.key_scales})"
            )
        return config
    scales_int = int(key_scales)
    return _KVChunkConfig(
        key_scales=scales_int,
        grouped_qk=scales_int == _KV_TILES_PER_CHUNK,
        pack_wmma=False,
    )


def _reject_neg_or_conj_views(*_args: object, **_kwargs: object) -> NoReturn:
    raise ValueError("operators on PiperKVChunkPool must not use negated or conjugated views")
