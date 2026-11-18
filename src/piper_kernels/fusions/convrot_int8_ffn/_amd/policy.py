"""Select the portion of a long SwiGLU FFN that can use packed weights."""

from piper_kernels._triton.targets import AcceleratorTarget

PACKED_CHUNK_ROWS = 4096


def packed_prefix_rows(
    target: AcceleratorTarget,
    rows: int,
    input_features: int,
    intermediate_features: int,
    output_features: int,
    chunk_rows: int,
) -> int:
    """Return complete BF16 output chunks ending before the borrowed weight storage."""
    widths = (input_features, intermediate_features, output_features)
    if (
        not target.is_architecture("gfx1201")
        or chunk_rows != PACKED_CHUNK_ROWS
        or min(widths) < 4096
        # Wider rows use a separate rotation and quantization schedule.
        or max(input_features, intermediate_features) > 16384
        or any(width % 256 for width in widths)
        # Reductions at 8 KiB multiples already have a separate wide-K GEMM schedule.
        or input_features % 8192 == 0
        or intermediate_features % 8192 == 0
    ):
        return 0
    source_bytes = input_features * intermediate_features
    down_bytes = intermediate_features * output_features
    # The pack kernel indexes each weight and the GEMM indexes each chunk in INT32.
    if (
        max(source_bytes, down_bytes, chunk_rows * max(2 * intermediate_features, output_features))
        >= 2**31
    ):
        return 0
    packed_bytes = 2 * source_bytes + down_bytes
    output_row_bytes = 2 * output_features
    borrowed_rows = (packed_bytes + output_row_bytes - 1) // output_row_bytes
    prefix = (rows - borrowed_rows) // chunk_rows * chunk_rows
    # Amortize the three per-call packing launches over at least eight chunks.
    return prefix if prefix >= 8 * chunk_rows else 0
