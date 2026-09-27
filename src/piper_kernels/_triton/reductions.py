"""FP32 reductions shared by projection fusions and attention preparation."""

import triton
import triton.language as tl


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
    """
    group = tl.program_id(0)
    offsets_c = tl.arange(0, block_chunks)
    offsets_d = tl.program_id(1) * block_d + tl.arange(0, block_d)
    offsets = (group * num_chunks + offsets_c[:, None]) * features + offsets_d[None, :]
    partials = tl.load(
        partial_ptr + offsets,
        mask=(offsets_c[:, None] < num_chunks) & (offsets_d[None, :] < features),
        other=0.0,
    )
    tl.store(
        mean_ptr + group * features + offsets_d,
        tl.sum(partials, axis=0) / sequence_length,
        mask=offsets_d < features,
    )
