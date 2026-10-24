"""Fixed-budget forced routes shared across attention batches and heads.

IDs/counts are device data. Valid counts, unique in-range active IDs, and enough
space in every affected head's keep budget are caller preconditions. Nothing in
this module inspects tensor contents for validation.
"""

import torch


def validate_force_tiles(
    ids: torch.Tensor | None,
    counts: torch.Tensor | None,
    *,
    query_blocks: int,
    device: torch.device,
) -> None:
    """Check only storage metadata for one padded ID list per query block."""
    if ids is None and counts is None:
        return
    if ids is None or counts is None:
        raise ValueError("force_tile_ids and force_tile_counts must be supplied together")
    if ids.ndim != 2 or ids.shape[0] != query_blocks or counts.shape != (query_blocks,):
        raise ValueError(
            "forced tiles require IDs [query_blocks,capacity] and counts [query_blocks]"
        )
    if any(t.dtype is not torch.int32 or t.device != device for t in (ids, counts)):
        raise ValueError("forced tile IDs and counts must be INT32 on the attention device")
    if any(t.layout is not torch.strided or not t.is_contiguous() for t in (ids, counts)):
        raise ValueError("forced tile IDs and counts must use contiguous strided storage")


def portable_force_tile_bitmask(
    ids: torch.Tensor, counts: torch.Tensor, sparse_key_blocks: int
) -> torch.Tensor:
    """Pack unique active IDs into 32-bit words without reading them on the host."""
    active = torch.arange(ids.shape[1], device=ids.device)[None, :] < counts[:, None]
    safe_ids = torch.where(active, ids, 0).long()
    bits = (1 << (safe_ids % 32)) * active
    words = torch.zeros(
        (ids.shape[0], (sparse_key_blocks + 31) // 32), dtype=torch.int64, device=ids.device
    )
    words.scatter_add_(1, safe_ids // 32, bits)
    return words.to(torch.int32)
