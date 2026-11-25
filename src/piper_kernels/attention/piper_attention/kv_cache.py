"""Global chunk-major quantized KV cache storage and freeze operators."""

from __future__ import annotations

import math
import re
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor

from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget

from . import _backend

KV_CHUNK_TOKENS = 1024
_KV_TILE_TOKENS = 64
_KV_TILES_PER_CHUNK = KV_CHUNK_TOKENS // _KV_TILE_TOKENS

_SUPPORTED_HEAD_DIMS = (64, 128)
_SUPPORTED_KEY_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_KV_STORAGE_DTYPES = (torch.int8, torch.uint8)
_KEY_SCALE_COUNTS = (_KV_TILES_PER_CHUNK, KV_CHUNK_TOKENS)
_ALIGNMENT_BYTES = 16
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
_CONCRETE_ADDRESS_SPACE = object()

type _SymInt = int | torch.SymInt


@dataclass(frozen=True, slots=True)
class _KVChunkConfig:
    key_scales: int
    grouped_qk: bool
    pack_wmma: bool


@dataclass(frozen=True, slots=True)
class _SlotBufferSpan:
    storage_key: object
    start_byte: _SymInt
    step_byte: _SymInt
    slot_bytes: _SymInt
    num_slots: _SymInt


_VALIDATED_CONFIG: ContextVar[_KVChunkConfig | None] = ContextVar(
    "piper_kv_freeze_validated_config", default=None
)


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
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
    key_scales, grouped_qk, pack_wmma = _backend.kv_chunk_layout(_resolve_target(target))
    return _KVChunkConfig(key_scales=key_scales, grouped_qk=grouped_qk, pack_wmma=pack_wmma)


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


def _storage_identity(tensor: torch.Tensor) -> tuple[object, int]:
    storage = tensor.untyped_storage()
    if (
        torch.compiler.is_compiling()
        or isinstance(tensor, FakeTensor)
        or tensor.device.type == "meta"
    ):
        return storage, 0
    ptr = storage.data_ptr()
    if ptr == 0:
        return storage, 0
    return _CONCRETE_ADDRESS_SPACE, ptr


def _validate_alignment(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.numel() == 0:
        return
    elem_size = tensor.element_size()
    if (tensor.storage_offset() * elem_size) % _ALIGNMENT_BYTES != 0:
        raise ValueError(
            f"{name} storage_offset byte address must be {_ALIGNMENT_BYTES}-byte aligned"
        )
    if (
        not torch.compiler.is_compiling()
        and not isinstance(tensor, FakeTensor)
        and tensor.device.type != "meta"
    ):
        ptr = tensor.data_ptr()
        if ptr != 0 and ptr % _ALIGNMENT_BYTES != 0:
            raise ValueError(f"{name} data pointer must be {_ALIGNMENT_BYTES}-byte aligned")


def _validate_slot_strided(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.layout is not torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if not torch.compiler.is_compiling() and (tensor.is_neg() or tensor.is_conj()):
        raise ValueError(f"{name} must not use negated or conjugated views")

    expected_stride: _SymInt = 1
    for dim in range(tensor.ndim - 1, 0, -1):
        size = tensor.shape[dim]
        stride = tensor.stride(dim)
        if size != 1 and stride != expected_stride:
            raise ValueError(
                f"{name} must be dense row-major within each slot, "
                f"expected stride({dim}) == {expected_stride}, got {stride}"
            )
        expected_stride *= size

    num_slots = tensor.shape[0]
    stride0 = tensor.stride(0)
    if num_slots > 1:
        if stride0 < expected_stride:
            raise ValueError(
                f"{name} stride(0) ({stride0}) must be >= per-slot elements ({expected_stride})"
            )
        if (stride0 * tensor.element_size()) % _ALIGNMENT_BYTES != 0:
            raise ValueError(f"{name} stride(0) byte step must be {_ALIGNMENT_BYTES}-byte aligned")
    _validate_alignment(tensor, name=name)


def _slot_buffer_span(tensor: torch.Tensor, base_ptr: int) -> _SlotBufferSpan:
    storage_key, _ = _storage_identity(tensor)
    elem_size = tensor.element_size()
    start_byte = base_ptr + tensor.storage_offset() * elem_size

    inner: _SymInt = 1
    for dim in range(1, tensor.ndim):
        inner *= tensor.shape[dim]
    return _SlotBufferSpan(
        storage_key=storage_key,
        start_byte=start_byte,
        step_byte=tensor.stride(0) * elem_size,
        slot_bytes=inner * elem_size,
        num_slots=tensor.shape[0],
    )


def _general_byte_span(tensor: torch.Tensor, base_ptr: int) -> tuple[_SymInt, _SymInt]:
    elem_size = tensor.element_size()
    span_elems: _SymInt = 0
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        span_elems += (size - 1) * stride
    start = base_ptr + tensor.storage_offset() * elem_size
    end = start + (span_elems + 1) * elem_size
    return start, end


_MAX_STRIDED_OVERLAP_CHECKS = 1024


def _span_extent_bytes(block_bytes: int, outer_dims: tuple[tuple[int, int], ...]) -> int:
    return block_bytes + sum((count - 1) * step for count, step in outer_dims)


def _decompose_byte_spans(
    tensor: torch.Tensor, base_ptr: int
) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
    shape = tensor.shape
    strides = tensor.stride()
    storage_offset = tensor.storage_offset()
    if any(isinstance(v, torch.SymInt) or v < 0 for v in (*shape, *strides, storage_offset)):
        return None
    concrete_dims = sorted(
        (int(st), int(s)) for s, st in zip(shape, strides, strict=True) if s > 1 and st > 0
    )
    block_elems = 1
    split_idx = 0
    for stride, size in concrete_dims:
        if stride != block_elems:
            break
        block_elems *= size
        split_idx += 1
    elem_size = tensor.element_size()
    start_byte = base_ptr + int(storage_offset) * elem_size
    outer_dims = tuple((size, stride * elem_size) for stride, size in concrete_dims[split_idx:])
    return start_byte, block_elems * elem_size, outer_dims


def _interval_avoids_periodic_slots(
    start: _SymInt,
    end: _SymInt,
    slot_start: _SymInt,
    slot_step: _SymInt,
    slot_len: _SymInt,
    num_slots: _SymInt,
) -> bool:
    if num_slots <= 0 or end <= slot_start:
        return True
    if start >= slot_start + (num_slots - 1) * slot_step + slot_len:
        return True
    if slot_step <= slot_len or (end - start) > (slot_step - slot_len):
        return False
    offset = (start - slot_start) % slot_step
    return bool(offset >= slot_len and offset + (end - start) <= slot_step)


def _min_step_in_mod_range(step: int, mod: int, lo: int, hi: int) -> int | None:
    """Return smallest integer t >= 0 with ``lo <= (t * step) % mod <= hi``, or ``None``."""
    if lo == 0:
        return 0
    step %= mod
    if step == 0:
        return None
    direct = (lo + step - 1) // step
    if direct * step <= hi:
        return direct
    wraps = _min_step_in_mod_range(mod % step, step, step - (hi % step), step - (lo % step))
    if wraps is None:
        return None
    return (wraps * mod + lo + step - 1) // step


def _finite_periodic_spans_disjoint(
    a_start: int,
    a_step: int,
    a_len: int,
    a_slots: int,
    b_start: int,
    b_step: int,
    b_len: int,
    b_slots: int,
) -> bool:
    if a_step <= 0 or b_step <= 0:
        return False
    # Overlap requires lo <= i * a_step - j * b_step <= hi
    # for some 0 <= i < a_slots and 0 <= j < b_slots.
    lo = b_start - a_start - a_len + 1
    hi = b_start - a_start + b_len - 1
    j_min = max(0, (-hi + b_step - 1) // b_step)
    j_max = min(b_slots - 1, ((a_slots - 1) * a_step - lo) // b_step)
    if j_min > j_max:
        return True
    width = hi - lo
    if width >= a_step - 1:
        return False
    count = j_max - j_min + 1
    r0 = (-lo - j_min * b_step) % a_step
    if r0 <= width:
        return False
    first_t = _min_step_in_mod_range((-b_step) % a_step, a_step, a_step - r0, a_step - r0 + width)
    return first_t is None or first_t >= count


def _periodic_spans_disjoint(
    a_start: _SymInt,
    a_step: _SymInt,
    a_len: _SymInt,
    a_slots: _SymInt,
    b_start: _SymInt,
    b_step: _SymInt,
    b_len: _SymInt,
    b_slots: _SymInt,
) -> bool:
    if a_slots <= 0 or b_slots <= 0:
        return True
    a_end = a_start + (a_slots - 1) * a_step + a_len
    b_end = b_start + (b_slots - 1) * b_step + b_len
    if (
        a_end <= b_start
        or b_end <= a_start
        or _interval_avoids_periodic_slots(b_start, b_end, a_start, a_step, a_len, a_slots)
        or _interval_avoids_periodic_slots(a_start, a_end, b_start, b_step, b_len, b_slots)
    ):
        return True

    # Interleaved periodic views on the same arena avoid each other across all slot
    # counts whenever [rem, rem + b_len) falls strictly in the gap [a_len, gcd_step).
    if a_slots <= 1 or b_slots <= 1:
        return False
    if b_start < a_start:
        a_start, a_step, a_len, a_slots, b_start, b_step, b_len, b_slots = (
            b_start,
            b_step,
            b_len,
            b_slots,
            a_start,
            a_step,
            a_len,
            a_slots,
        )
    gcd_step: _SymInt
    if a_step == b_step and a_step > 0:
        gcd_step = a_step
    elif isinstance(a_step, int) and isinstance(b_step, int):
        gcd_step = math.gcd(a_step, b_step)
    else:
        return False
    if gcd_step <= 0:
        return False
    diff = b_start - a_start
    rem = diff if diff < gcd_step else diff % gcd_step
    return bool(rem >= a_len and (gcd_step - rem) >= b_len) or (
        isinstance(a_start, int)
        and isinstance(b_start, int)
        and isinstance(a_step, int)
        and isinstance(a_len, int)
        and isinstance(a_slots, int)
        and isinstance(b_step, int)
        and isinstance(b_len, int)
        and isinstance(b_slots, int)
        and _finite_periodic_spans_disjoint(
            a_start, a_step, a_len, a_slots, b_start, b_step, b_len, b_slots
        )
    )


def _slot_spans_overlap(first: torch.Tensor, second: torch.Tensor) -> bool:
    if first.device != second.device:
        return False
    a_key, a_base = _storage_identity(first)
    b_key, b_base = _storage_identity(second)
    if a_key != b_key or first.numel() == 0 or second.numel() == 0:
        return False
    a = _slot_buffer_span(first, a_base)
    b = _slot_buffer_span(second, b_base)
    if a.slot_bytes == 0 or b.slot_bytes == 0:
        return False
    return not _periodic_spans_disjoint(
        a.start_byte,
        a.step_byte,
        a.slot_bytes,
        a.num_slots,
        b.start_byte,
        b.step_byte,
        b.slot_bytes,
        b.num_slots,
    )


def _strided_subtensor_avoids_periodic_slots(
    start_byte: int,
    block_bytes: int,
    outer_dims: tuple[tuple[int, int], ...],
    pool_span: _SlotBufferSpan,
    budget: int,
) -> bool:
    extent = _span_extent_bytes(block_bytes, outer_dims)
    if _interval_avoids_periodic_slots(
        start_byte,
        start_byte + extent,
        pool_span.start_byte,
        pool_span.step_byte,
        pool_span.slot_bytes,
        pool_span.num_slots,
    ):
        return True
    if not outer_dims:
        return False
    if len(outer_dims) == 1:
        count, step = outer_dims[0]
        return _periodic_spans_disjoint(
            start_byte,
            step,
            block_bytes,
            count,
            pool_span.start_byte,
            pool_span.step_byte,
            pool_span.slot_bytes,
            pool_span.num_slots,
        )
    outer_count, outer_step = outer_dims[-1]
    rest = outer_dims[:-1]
    rest_extent = _span_extent_bytes(block_bytes, rest)
    if outer_step > rest_extent and _periodic_spans_disjoint(
        start_byte,
        outer_step,
        rest_extent,
        outer_count,
        pool_span.start_byte,
        pool_span.step_byte,
        pool_span.slot_bytes,
        pool_span.num_slots,
    ):
        return True
    split_idx = min(range(len(outer_dims)), key=lambda idx: outer_dims[idx][0])
    split_count, split_step = outer_dims[split_idx]
    if split_count > budget:
        return False
    sub_dims = outer_dims[:split_idx] + outer_dims[split_idx + 1 :]
    sub_budget = budget // split_count
    return all(
        _strided_subtensor_avoids_periodic_slots(
            start_byte + step_idx * split_step,
            block_bytes,
            sub_dims,
            pool_span,
            sub_budget,
        )
        for step_idx in range(split_count)
    )


def _input_overlaps_pool_tensor(read_tensor: torch.Tensor, pool_tensor: torch.Tensor) -> bool:
    if read_tensor.device != pool_tensor.device:
        return False
    read_key, read_base = _storage_identity(read_tensor)
    pool_key, pool_base = _storage_identity(pool_tensor)
    if read_key != pool_key or read_tensor.numel() == 0 or pool_tensor.numel() == 0:
        return False
    pool_span = _slot_buffer_span(pool_tensor, pool_base)
    decomposed = _decompose_byte_spans(read_tensor, read_base)
    if decomposed is None:
        read_start, read_end = _general_byte_span(read_tensor, read_base)
        return not _interval_avoids_periodic_slots(
            read_start,
            read_end,
            pool_span.start_byte,
            pool_span.step_byte,
            pool_span.slot_bytes,
            pool_span.num_slots,
        )
    start_byte, block_bytes, outer_dims = decomposed
    return not _strided_subtensor_avoids_periodic_slots(
        start_byte,
        block_bytes,
        outer_dims,
        pool_span,
        _MAX_STRIDED_OVERLAP_CHECKS,
    )


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
) -> tuple[_SymInt, _SymInt, _SymInt, _SymInt]:
    tensors = (
        ("key_int8", key_int8),
        ("key_scale", key_scale),
        ("key_center", key_center),
        ("value_int8", value_int8),
        ("value_scale", value_scale),
        ("value_mean", value_mean),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
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
    if not torch.compiler.is_compiling():
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


def _reject_autograd(*tensors: torch.Tensor) -> None:
    if not torch.is_grad_enabled():
        return
    if any(isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in tensors):
        raise RuntimeError("freeze_kv_chunks_ is inference-only and does not support autograd")


def _validate_freeze_kv_input(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    kv_heads: _SymInt,
    head_dim: _SymInt,
    num_new: _SymInt,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype not in _SUPPORTED_KEY_DTYPES:
        raise ValueError(f"{name} must use float16 or bfloat16, got {tensor.dtype}")
    if tensor.layout is not torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if tensor.is_neg() or tensor.is_conj():
        raise ValueError(f"{name} must not use negated or conjugated views")
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape [num_new, kv_heads, {KV_CHUNK_TOKENS}, head_dim], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} head dimension must be contiguous")
    expected_shape = (num_new, kv_heads, KV_CHUNK_TOKENS, head_dim)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")


def _validate_slot_ids(
    slot_ids: torch.Tensor,
    *,
    device: torch.device,
    num_new: _SymInt,
) -> None:
    if slot_ids.device != device:
        raise ValueError(f"slot_ids must be on {device}, got {slot_ids.device}")
    if slot_ids.dtype is not torch.int32:
        raise ValueError(f"slot_ids must use int32, got {slot_ids.dtype}")
    if slot_ids.layout is not torch.strided:
        raise ValueError("slot_ids must use strided layout")
    if slot_ids.is_neg() or slot_ids.is_conj():
        raise ValueError("slot_ids must not use negated or conjugated views")
    if tuple(slot_ids.shape) != (num_new,):
        raise ValueError(f"slot_ids must have shape ({num_new},), got {tuple(slot_ids.shape)}")
    if not slot_ids.is_contiguous():
        raise ValueError("slot_ids must be contiguous")


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


def _validate_freeze_before_mutating(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    check_target: bool,
) -> _KVChunkConfig | None:
    _, kv_heads, head_dim, key_scales = _validate_pool_tensors(
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
    )
    pool_tensors = (key_int8, key_scale, key_center, value_int8, value_scale, value_mean)
    if not torch.is_inference_mode_enabled():
        for pool_name, pool_tensor in zip(_POOL_FIELD_NAMES, pool_tensors, strict=True):
            if pool_tensor.is_inference():
                raise RuntimeError(
                    f"Inplace update to inference tensor {pool_name} outside InferenceMode "
                    "is not allowed."
                )

    for name, tensor in (("key", key), ("value", value), ("slot_ids", slot_ids)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if key.ndim != 4:
        raise ValueError(
            f"key must have shape [num_new, kv_heads, {KV_CHUNK_TOKENS}, head_dim], "
            f"got {tuple(key.shape)}"
        )

    device = key_int8.device
    num_new = key.shape[0]
    _validate_freeze_kv_input(
        key, name="key", device=device, kv_heads=kv_heads, head_dim=head_dim, num_new=num_new
    )
    _validate_freeze_kv_input(
        value, name="value", device=device, kv_heads=kv_heads, head_dim=head_dim, num_new=num_new
    )
    if value.dtype is not key.dtype:
        raise ValueError(f"key ({key.dtype}) and value ({value.dtype}) must share a dtype")
    _validate_slot_ids(slot_ids, device=device, num_new=num_new)

    for read_name, read_tensor in (("key", key), ("value", value), ("slot_ids", slot_ids)):
        for pool_name, pool_tensor in zip(_POOL_FIELD_NAMES, pool_tensors, strict=True):
            if _input_overlaps_pool_tensor(read_tensor, pool_tensor):
                raise ValueError(f"{read_name} must not overlap {pool_name} in storage")

    return _resolve_checked_config(device, key_scales) if check_target else None


@torch.library.custom_op(
    "piper_kernels::piper_kv_freeze_",
    mutates_args=(
        "key_int8",
        "key_scale",
        "key_center",
        "value_int8",
        "value_scale",
        "value_mean",
    ),
)
def _piper_kv_freeze_op(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
) -> None:
    tensors = (
        key,
        value,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        slot_ids,
    )
    _reject_autograd(*tensors)
    config = _VALIDATED_CONFIG.get()
    if config is not None:
        _VALIDATED_CONFIG.set(None)
    else:
        config = _validate_freeze_before_mutating(
            *tensors,
            check_target=not _is_fake_or_compiling(tensors),
        )
    if key.device.type == "meta" or key.shape[0] == 0 or key_int8.shape[0] == 0:
        return

    assert config is not None
    if runtime.supports_device(key.device):
        from . import _quantization  # noqa: PLC0415 - GPU-only Triton dependency

        _quantization.freeze_kv_chunks_triton(
            key,
            value,
            slot_ids,
            key_int8.view(torch.int8),
            key_scale,
            key_center,
            value_int8.view(torch.int8),
            value_scale,
            value_mean,
            grouped_qk=config.grouped_qk,
            pack_wmma=config.pack_wmma,
        )
        return

    from .reference import freeze_kv_chunks_reference  # noqa: PLC0415 - lazy CPU fallback

    freeze_kv_chunks_reference(
        key,
        value,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        slot_ids,
        pack_wmma=config.pack_wmma,
    )


@_piper_kv_freeze_op.register_fake
def _piper_kv_freeze_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
) -> None:
    tensors = (
        key,
        value,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        slot_ids,
    )
    _reject_autograd(*tensors)
    _validate_freeze_before_mutating(*tensors, check_target=False)


@torch.library.impl("piper_kernels::piper_kv_freeze_", "Autograd")
def _piper_kv_freeze_autograd(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
) -> None:
    tensors = (
        key,
        value,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        slot_ids,
    )
    _reject_autograd(*tensors)
    # Validate eagerly above ADInplaceOrView so rejected calls leave pool versions intact.
    config = (
        None
        if _is_fake_or_compiling(tensors)
        else _validate_freeze_before_mutating(*tensors, check_target=True)
    )
    token = _VALIDATED_CONFIG.set(config)
    try:
        with torch._C._AutoDispatchBelowAutograd():
            torch.ops.piper_kernels.piper_kv_freeze_.default(*tensors)
    finally:
        _VALIDATED_CONFIG.reset(token)


def freeze_kv_chunks_(
    key: torch.Tensor,
    value: torch.Tensor,
    pool: PiperKVChunkPool,
    slot_ids: torch.Tensor,
) -> None:
    """Quantize and scatter finalized 1024-token KV chunks into persistent ``pool`` slots.

    Args:
        key: Finalized unrotated keys with shape ``[num_new, kv_heads, 1024, head_dim]``
            in ``float16`` or ``bfloat16`` on ``pool.device`` (``stride(-1) == 1``;
            outer dims may be strided; must not alias ``pool`` storage).
        value: Finalized values matching ``key`` in shape, dtype, device, and layout.
        pool: Destination ``PiperKVChunkPool``. Mutated in place only at valid ``slot_ids``;
            if any pool tensor was created in ``torch.inference_mode()``, callers must
            invoke inside ``torch.inference_mode()``.
        slot_ids: Contiguous 1-D ``int32`` destination indices of shape ``[num_new]``
            on ``pool.device``. Entries ``< 0`` or ``>= pool.num_slots`` are skipped
            with zero host readbacks; valid entries in a single call must be distinct.

    Backends & graphs:
        Executes native Triton K1024/V1024 quantization kernels on supported GPU targets
        and the portable PyTorch reference on CPU or fallback devices. Supports
        ``torch.compile(..., fullgraph=True, dynamic=True)`` and CUDA Graph capture;
        inference-only (rejects tensors requiring gradients when grad is enabled).
    """
    if not isinstance(pool, PiperKVChunkPool):
        raise TypeError(f"pool must be a PiperKVChunkPool, got {type(pool).__name__}")
    for arg_name, arg_tensor in (("key", key), ("value", value), ("slot_ids", slot_ids)):
        if not isinstance(arg_tensor, torch.Tensor):
            raise TypeError(f"{arg_name} must be a torch.Tensor, got {type(arg_tensor).__name__}")
    tensors = (
        key,
        value,
        pool.key_int8,
        pool.key_scale,
        pool.key_center,
        pool.value_int8,
        pool.value_scale,
        pool.value_mean,
        slot_ids,
    )
    # InferenceMode skips the Autograd key; validate concrete calls before ADInplaceOrView.
    if not _is_fake_or_compiling(tensors) and torch.is_inference_mode_enabled():
        config = _validate_freeze_before_mutating(*tensors, check_target=True)
        token = _VALIDATED_CONFIG.set(config)
        try:
            torch.ops.piper_kernels.piper_kv_freeze_.default(*tensors)
        finally:
            _VALIDATED_CONFIG.reset(token)
        return
    torch.ops.piper_kernels.piper_kv_freeze_.default(*tensors)


__all__ = [
    "KV_CHUNK_TOKENS",
    "PiperKVChunkPool",
    "freeze_kv_chunks_",
    "kv_chunk_storage_specs",
]
