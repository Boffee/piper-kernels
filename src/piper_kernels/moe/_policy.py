"""Triton-free target schedule routers for quantized MoE operators."""

from __future__ import annotations

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from . import _schedule
from ._amd import policy as _amd_policy
from ._nvidia import policy as _nvidia_policy


def select_nvfp4_m1_schedule(target: AcceleratorTarget) -> _schedule.Nvfp4M1Schedule:
    """Resolve the NVFP4 ``rows == 1`` decode schedule for ``target``."""
    return _nvidia_policy.select_nvfp4_m1_schedule(target)


def select_nvfp4_batched_schedule(
    target: AcceleratorTarget,
    *,
    rows: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> _schedule.Nvfp4BatchedSchedule:
    """Resolve the NVFP4 batched decode schedule for ``target``."""
    return _nvidia_policy.select_nvfp4_batched_schedule(
        target, rows=rows, active_dtype=active_dtype, logical_dtype=logical_dtype
    )


def select_nvfp4_prefill_schedule(
    target: AcceleratorTarget,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    active_dtype: torch.dtype,
    logical_dtype: torch.dtype,
) -> _schedule.Nvfp4PrefillSchedule:
    """Resolve the NVFP4 grouped prefill schedule for ``target``."""
    return _nvidia_policy.select_nvfp4_prefill_schedule(
        target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        active_dtype=active_dtype,
        logical_dtype=logical_dtype,
    )


def select_nvfp4_w4a4_prefill_schedule(
    target: AcceleratorTarget,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
) -> _schedule.Nvfp4W4A4PrefillSchedule:
    """Resolve the NVFP4 W4A4 grouped prefill schedule for ``target``."""
    return _nvidia_policy.select_nvfp4_w4a4_prefill_schedule(
        target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
    )


def select_int8_launch_schedule(target: AcceleratorTarget) -> _schedule.Int8LaunchSchedule:
    """Resolve the INT8 decode launch schedule for ``target``."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.select_int8_launch_schedule(target)
    if target.is_amd_hip:
        return _amd_policy.select_int8_launch_schedule(target)
    return _schedule.default_int8_launch_schedule()


def select_int8_prefill_schedule(
    target: AcceleratorTarget,
    *,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
) -> _schedule.Int8PrefillSchedule:
    """Resolve the INT8 grouped prefill schedule for ``target``."""
    return _schedule.build_int8_prefill_schedule(
        select_int8_launch_schedule(target),
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
    )


def select_trellis_schedule(
    target: AcceleratorTarget,
    *,
    batched: bool,
    num_pairs: int,
    expected_slots: int,
) -> _schedule.TrellisMoeSchedule:
    """Resolve the Trellis GEMV schedule for ``target``."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.select_trellis_schedule(
            target, batched=batched, num_pairs=num_pairs, expected_slots=expected_slots
        )
    if target.is_amd_hip:
        return _amd_policy.select_trellis_schedule(
            target, batched=batched, num_pairs=num_pairs, expected_slots=expected_slots
        )
    return _schedule.default_trellis_schedule(batched=batched)


def supports_trellis_int8_prefill(target: AcceleratorTarget) -> bool:
    """Return whether ``target`` permits routing Trellis prefill through an INT8 backend."""
    if target.is_nvidia_cuda:
        return _nvidia_policy.supports_trellis_int8_prefill(target)
    return True
