"""Shared Triton-free Split-K sizing and SM-count lookup helpers."""

from __future__ import annotations

import torch

_SM_COUNT_CACHE: dict[int, int] = {}


def device_sm_count(device: torch.device | int) -> int:
    """Return cached multiprocessor count for a CUDA/ROCm device, or ``0`` if unavailable."""
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return 0
        if device.index is not None:
            device_index = int(device.index)
        elif not torch.cuda.is_available():
            return 0
        else:
            device_index = torch.cuda.current_device()
    else:
        device_index = int(device)
    cached = _SM_COUNT_CACHE.get(device_index)
    if cached is not None:
        return cached
    if not torch.cuda.is_available():
        return 0
    count = getattr(torch.cuda.get_device_properties(device_index), "multi_processor_count", None)
    if type(count) is not int or count <= 0:
        return 0
    _SM_COUNT_CACHE[device_index] = count
    return count


def select_split_k(
    num_sms: int,
    *,
    output_tiles: int,
    k_tiles: int,
    weight_bytes: int,
    min_weight_bytes: int,
    max_split_k: int,
    cta_multiplier: int = 1,
    min_k_tiles: int = 8,
    min_tiles_per_split: int = 4,
    round_up: bool = True,
) -> int:
    """Choose a bounded Split-K factor from output parallelism, K depth, and SM count."""
    if num_sms <= 0 or output_tiles < 1 or k_tiles < min_k_tiles or weight_bytes < min_weight_bytes:
        return 1
    target_ctas = max(1, cta_multiplier * num_sms)
    if output_tiles >= target_ctas:
        return 1
    desired = (
        (target_ctas + output_tiles - 1) // output_tiles
        if round_up
        else target_ctas // output_tiles
    )
    max_for_k = max(1, k_tiles // min_tiles_per_split)
    upper = min(desired, max_for_k, max_split_k)
    return int(max(1, upper))


__all__ = ["device_sm_count", "select_split_k"]
