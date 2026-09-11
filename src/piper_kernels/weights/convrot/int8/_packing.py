"""Chunk planning for rowwise ConvRot INT8 packing."""


def fused_preparation_chunks(in_features: int) -> tuple[int, int] | None:
    """Return ``(chunk_count, chunk_size)`` for fused row rotation and packing.

    Preserve the inexpensive single-chunk path for small rows. Above it, choose
    the supported layout with the least padded work, preferring fewer chunks on
    ties.
    """
    block_size = max(128, 1 << (in_features - 1).bit_length())
    if block_size <= 4096:
        return 1, block_size

    candidates: list[tuple[int, int]] = []
    for chunk_count in (1, 2, 3):
        chunk_width = (in_features + chunk_count - 1) // chunk_count
        chunk_size = max(128, 1 << (chunk_width - 1).bit_length())
        if chunk_size > 16384:
            continue
        if chunk_count == 1 and chunk_size > 8192:
            continue
        if chunk_size * (chunk_count - 1) >= in_features:
            continue
        candidates.append((chunk_count, chunk_size))

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (candidate[0] * candidate[1], candidate[0]),
    )
