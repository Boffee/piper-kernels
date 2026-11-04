"""FP32 reductions shared by projection fusions and attention preparation."""

import triton
import triton.language as tl


def mean_reduction_tile(num_chunks: int) -> tuple[int, int]:
    """Bound a four-warp FP32 reduction to 8192 values per program.

    Narrow the feature tile as the chunk axis grows, rather than spilling a
    sequence-sized tile to device memory. At most 4096 chunk rows keep loop
    addressing from exhausting registers at the narrowest feature tiles. Longer
    axes are accumulated in batches by ``store_mean_from_partials``. Callers
    provide at least one partial chunk.
    """
    block_chunks = min(triton.next_power_of_2(num_chunks), 4096)
    return block_chunks, min(64, 8192 // block_chunks)


@triton.jit
def store_mean_from_partials(
    partial_ptr,
    mean_ptr,
    sequence_length,
    num_chunks,
    features: tl.constexpr,
    block_chunks: tl.constexpr,
    block_d: tl.constexpr,
):
    """Merge FP32 tile sums, dividing by the caller's logical sequence length.

    Partials are contiguous [group, chunk, feature]. The first two program
    indices select the group and feature block; the output is [group, feature].
    Chunk batches keep the per-program working set bounded for long sequences.
    """
    group = tl.program_id(0)
    offsets_c = tl.arange(0, block_chunks)
    offsets_d = tl.program_id(1) * block_d + tl.arange(0, block_d)
    offsets = (group * num_chunks + offsets_c[:, None]) * features + offsets_d[None, :]
    sums = tl.full((block_d,), 0.0, tl.float32)
    for start in range(0, num_chunks, block_chunks):
        partials = tl.load(
            partial_ptr + offsets + start * features,
            mask=(start + offsets_c[:, None] < num_chunks) & (offsets_d[None, :] < features),
            other=0.0,
        )
        sums += tl.sum(partials, axis=0)
    tl.store(
        mean_ptr + group * features + offsets_d,
        sums / sequence_length,
        mask=offsets_d < features,
    )
