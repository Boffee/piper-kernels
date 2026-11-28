"""Storage identity, alignment, and periodic slot-overlap proofs for KV cache pools."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor

_ALIGNMENT_BYTES = 16
_CONCRETE_ADDRESS_SPACE = object()
_MAX_STRIDED_OVERLAP_CHECKS = 1024

type _SymInt = int | torch.SymInt


@dataclass(frozen=True, slots=True)
class _SlotBufferSpan:
    storage_key: object
    start_byte: _SymInt
    step_byte: _SymInt
    slot_bytes: _SymInt
    num_slots: _SymInt


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
