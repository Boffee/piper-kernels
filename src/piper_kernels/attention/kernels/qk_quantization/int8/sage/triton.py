"""Triton kernels for Sage-style INT8 Q/K quantization."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.kernels._grid import select_block_grid

from ._rotation import SIGNED_HADAMARD_MASK

_LOG2_E = tl.constexpr(1.4426950408889634)
_SCALE_EPSILON = tl.constexpr(1e-7)
_QUERY_BLOCK = 32
_KEY_BLOCK = 64
_SIGNED_HADAMARD_WORD_0 = tl.constexpr(SIGNED_HADAMARD_MASK[0])
_SIGNED_HADAMARD_WORD_1 = tl.constexpr(SIGNED_HADAMARD_MASK[1])
_SIGNED_HADAMARD_WORD_2 = tl.constexpr(SIGNED_HADAMARD_MASK[2])
_SIGNED_HADAMARD_WORD_3 = tl.constexpr(SIGNED_HADAMARD_MASK[3])


@dataclass(frozen=True, slots=True)
class PreparedInt8QueryKey:
    """Attention-independent Sage-style INT8 Q/K operands.

    ``query_scale`` includes the attention softmax scale and the conversion
    from natural exponentials to the base-2 recurrence used by the attention
    kernels. ``key`` is centered globally (by ``key_mean``) or per tile/chunk
    before the signed Hadamard transform and quantization.
    """

    query: torch.Tensor
    key: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor


@triton.jit
def _hadamard_stage(
    values,
    head_dim: tl.constexpr,
    butterfly_distance: tl.constexpr,
):
    rows: tl.constexpr = values.shape[0]  # pyright: ignore[reportAssignmentType]
    outer: tl.constexpr = head_dim // (2 * butterfly_distance)
    grouped = tl.reshape(values, (rows, outer, 2, butterfly_distance))
    pairs = tl.permute(grouped, (0, 1, 3, 2))
    low, high = tl.split(pairs)
    transformed = tl.join(low + high, low - high)
    return tl.reshape(tl.permute(transformed, (0, 1, 3, 2)), (rows, head_dim))


@triton.jit
def rotate_signed_hadamard_heads(values, head_dim: tl.constexpr):
    """Apply a fixed signed, normalized Hadamard to D64 or D128 heads."""
    offsets = tl.arange(0, head_dim)
    word_group = offsets // 32
    words = tl.where(
        word_group == 0,
        _SIGNED_HADAMARD_WORD_0,
        tl.where(
            word_group == 1,
            _SIGNED_HADAMARD_WORD_1,
            tl.where(
                word_group == 2,
                _SIGNED_HADAMARD_WORD_2,
                _SIGNED_HADAMARD_WORD_3,
            ),
        ),
    ).to(tl.uint32)
    signs = tl.where(((words >> (offsets % 32)) & 1) != 0, 1.0, -1.0)
    values *= signs[None, :]

    values = _hadamard_stage(values, head_dim, tl.constexpr(1))
    values = _hadamard_stage(values, head_dim, tl.constexpr(2))
    values = _hadamard_stage(values, head_dim, tl.constexpr(4))
    values = _hadamard_stage(values, head_dim, tl.constexpr(8))
    values = _hadamard_stage(values, head_dim, tl.constexpr(16))
    values = _hadamard_stage(values, head_dim, tl.constexpr(32))
    if head_dim == 128:
        values = _hadamard_stage(values, head_dim, tl.constexpr(64))
        return values * 0.08838834764831845
    return values * 0.125


@triton.jit
def round_to_int8(values):
    """Round symmetrically and clamp to SageAttention's signed INT8 range."""
    rounded = values + 0.5 * tl.where(values >= 0, 1.0, -1.0)
    return tl.maximum(-127.0, tl.minimum(127.0, rounded)).to(tl.int8)


@triton.jit
def quantize_query_tile(
    values,
    group_valid,
    softmax_scale,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Return grouped INT8 Q and base-2 recurrence scales for one tile."""
    smoothed = rotate_signed_hadamard_heads(
        tl.reshape(values, (block_m * heads_per_program, head_dim)),
        head_dim,
    )
    smoothed = tl.permute(
        tl.reshape(smoothed, (block_m, heads_per_program, head_dim)),
        (1, 0, 2),
    )
    grouped = tl.reshape(
        smoothed,
        (
            heads_per_program,
            block_m // scale_rows,
            scale_rows,
            head_dim,
        ),
    )
    maximum = tl.max(tl.max(tl.abs(grouped), axis=3), axis=2)
    raw_scale = maximum / 127.0 + _SCALE_EPSILON  # pyright: ignore[reportOperatorIssue]
    quantized = round_to_int8(grouped / tl.where(group_valid, raw_scale, 1.0)[:, :, None, None])
    stored_scale = tl.where(
        group_valid,
        raw_scale * (softmax_scale * _LOG2_E),
        0.0,
    )
    return tl.reshape(quantized, (heads_per_program, block_m, head_dim)), stored_scale


@triton.jit
def quantize_query_per_thread_tile(
    values,
    row_valid,
    softmax_scale: tl.constexpr,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
):
    """Return per-thread INT8 Q and per-row base-2 recurrence scales for one tile.

    As in the per-thread Q kernel, rows 32b + 8i + t (i = 0..3) share a scale.
    Every row stores its group's scale; rows outside ``row_valid`` store zero.
    """
    smoothed = rotate_signed_hadamard_heads(
        tl.reshape(values, (block_m * heads_per_program, head_dim)),
        head_dim,
    )
    smoothed = tl.permute(
        tl.reshape(smoothed, (block_m, heads_per_program, head_dim)),
        (1, 0, 2),
    )
    grouped = tl.reshape(smoothed, (heads_per_program, block_m // 32, 4, 8, head_dim))
    maximum = tl.max(tl.max(tl.abs(grouped), axis=4), axis=2)
    raw_scale = maximum / 127.0 + _SCALE_EPSILON  # pyright: ignore[reportOperatorIssue]
    quantized = round_to_int8(grouped / raw_scale[:, :, None, :, None])
    row_scale = tl.reshape(
        tl.broadcast_to(raw_scale[:, :, None, :], (heads_per_program, block_m // 32, 4, 8)),
        (heads_per_program, block_m),
    )
    stored_scale = tl.where(row_valid, row_scale * (softmax_scale * _LOG2_E), 0.0)
    return tl.reshape(quantized, (heads_per_program, block_m, head_dim)), stored_scale


@triton.jit
def encode_residual_int8_tile(values):
    """Encode an FP32 [heads, tiles, D] tile as two residual INT8 terms + scales."""
    max_0 = tl.max(tl.abs(values), axis=2)
    scale_0 = max_0 / 127.0 + _SCALE_EPSILON  # pyright: ignore[reportOperatorIssue]
    quantized_0 = round_to_int8(values / scale_0[:, :, None])
    residual = values - quantized_0.to(tl.float32) * scale_0[:, :, None]
    max_1 = tl.max(tl.abs(residual), axis=2)
    scale_1 = max_1 / 127.0 + _SCALE_EPSILON  # pyright: ignore[reportOperatorIssue]
    quantized_1 = round_to_int8(residual / scale_1[:, :, None])
    return quantized_0, quantized_1, scale_0, scale_1


@triton.jit
def store_key_center_tile(
    key_center_ptr,
    key_center_scale_ptr,
    tile_offset,
    center_0,
    center_1,
    center_scale_0,
    center_scale_1,
    feature_offsets,
    head_dim: tl.constexpr,
):
    """Store one tile's 2-term INT8 rotated center and FP32 residual scales."""
    center_base = tile_offset * 2
    tl.store(
        key_center_ptr + center_base * head_dim + feature_offsets,
        tl.reshape(center_0, (head_dim,)),
    )
    tl.store(
        key_center_ptr + (center_base + 1) * head_dim + feature_offsets,
        tl.reshape(center_1, (head_dim,)),
    )
    tl.store(key_center_scale_ptr + center_base, tl.reshape(center_scale_0, ()))
    tl.store(key_center_scale_ptr + center_base + 1, tl.reshape(center_scale_1, ()))


@triton.jit
def _maximum_and_minimum(maximum_a, minimum_a, maximum_b, minimum_b):
    """Combine ``tl.reduce`` partials with the same semantics as ``tl.max`` and ``tl.min``."""
    return tl.maximum(maximum_a, maximum_b), tl.minimum(minimum_a, minimum_b)


@triton.jit
def quantize_tile_centered_key_tile(
    values,
    valid,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Tile-center and INT8-encode K in one Hadamard pass plus 2-term rotated midpoint.

    ``values`` must already be zeroed outside ``valid`` so the Hadamard
    butterfly never mixes uninitialized/padded elements into valid features.
    """
    smoothed = rotate_signed_hadamard_heads(
        tl.reshape(values, (block_m * heads_per_program, head_dim)),
        head_dim,
    )
    num_tiles: tl.constexpr = block_m // scale_rows
    grouped = tl.reshape(
        tl.permute(
            tl.reshape(smoothed, (block_m, heads_per_program, head_dim)),
            (1, 0, 2),
        ),
        (heads_per_program, num_tiles, scale_rows, head_dim),
    )
    grouped_valid = tl.reshape(valid, (num_tiles, scale_rows))
    valid_mask = grouped_valid[None, :, :, None]
    has_valid = tl.sum(grouped_valid.to(tl.int32), axis=1)[None, :, None] > 0  # pyright: ignore[reportArgumentType]
    # One two-operand reduction rather than separate tl.max/tl.min: with one warp per
    # D64 tile, two reductions keep both masked copies live and spill registers.
    rotated_max, rotated_min = tl.reduce(
        (tl.where(valid_mask, grouped, -float("inf")), tl.where(valid_mask, grouped, float("inf"))),
        2,
        _maximum_and_minimum,
    )
    rotated_midpoint = 0.5 * (rotated_max + rotated_min)  # pyright: ignore[reportOperatorIssue]
    rotated_center = tl.where(has_valid, rotated_midpoint, 0.0)
    centered = tl.where(valid_mask, grouped - rotated_center[:, :, None, :], 0.0)
    maximum = tl.max(tl.max(tl.abs(centered), axis=3), axis=2)
    key_scale = maximum / 127.0 + _SCALE_EPSILON  # pyright: ignore[reportOperatorIssue]
    quantized = round_to_int8(centered / key_scale[:, :, None, None])
    center_0, center_1, center_scale_0, center_scale_1 = encode_residual_int8_tile(rotated_center)
    return (
        tl.reshape(quantized, (heads_per_program, block_m, head_dim)),
        key_scale,
        center_0,
        center_1,
        center_scale_0,
        center_scale_1,
    )


@triton.jit
def _max_propagate_nan(a, b):
    """Max reducer that propagates NaNs across lanes."""
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


# Per-row scale strides follow the sequence length; specializing on their
# divisibility would compile a new variant per length class for no benefit.
@triton.jit(do_not_specialize=["query_length", "stride_sb", "stride_sh"])
def quantize_query_per_thread_kernel(
    query_ptr,
    output_ptr,
    scale_ptr,
    query_length,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    head_dim: tl.constexpr,
    per_row: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    """Quantize one per-thread Q scale group (per-row or 4 rows per thread, stride 8)."""
    # A head or sequence stride can fit INT32 while its row offset does not.
    scale_group = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    query_block = scale_group // 8
    thread = scale_group % 8
    offsets_n = query_block * 32 + tl.arange(0, 4) * 8 + thread
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < query_length
    values = tl.load(
        query_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_n[:, None] * stride_qn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = rotate_signed_hadamard_heads(values, head_dim)
    if per_row:
        row_max = tl.reduce(tl.abs(values), 1, _max_propagate_nan)
        raw_scale = row_max / 127.0  # pyright: ignore[reportOperatorIssue]
        safe_div = tl.where(raw_scale == 0.0, 1.0, raw_scale)
        quantized = round_to_int8(values / safe_div[:, None])
        stored_scale = raw_scale * (softmax_scale * _LOG2_E)
    else:
        maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
        scale = maximum / 127.0 + _SCALE_EPSILON  # pyright: ignore[reportOperatorIssue]
        quantized = round_to_int8(values / scale)
        stored_scale = scale * (softmax_scale * _LOG2_E)
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_n[:, None] * stride_on
        + offsets_d[None, :],
        quantized,
        mask=mask,
    )
    tl.store(
        scale_ptr + batch * stride_sb + head * stride_sh + offsets_n,
        stored_scale,
        mask=offsets_n < query_length,
    )


@triton.jit(do_not_specialize=["query_length", "heads_inner"])
def quantize_query_per_warp_kernel(
    query_ptr,
    output_ptr,
    scale_ptr,
    query_length,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    heads_inner,
    head_dim: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    scale_group = tl.where(heads_inner != 0, pid1, pid0).to(tl.int64)
    head = tl.where(heads_inner != 0, pid0, pid1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    offsets_n = scale_group * 32 + tl.arange(0, 32)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < query_length
    values = tl.load(
        query_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_n[:, None] * stride_qn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    quantized, stored_scale = quantize_query_tile(
        tl.reshape(values, (32, 1, head_dim)),
        tl.full((1, 1), True, dtype=tl.int1),
        softmax_scale,
        tl.constexpr(1),
        head_dim,
        tl.constexpr(32),
        tl.constexpr(32),
    )
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_n[:, None] * stride_on
        + offsets_d[None, :],
        tl.reshape(quantized, (32, head_dim)),
        mask=mask,
    )
    tl.store(
        scale_ptr + batch * stride_sb + head * stride_sh + scale_group,
        tl.reshape(stored_scale, ()),
    )


@triton.jit(
    do_not_specialize=["key_length", "heads", "tile_count", "storage_key_length", "heads_inner"]
)
def quantize_key_tile_centered_per_block_kernel(
    key_ptr,
    output_ptr,
    scale_ptr,
    key_center_ptr,
    key_center_scale_ptr,
    block_lengths_ptr,
    key_length,
    storage_key_length,
    tile_count,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    heads,
    heads_inner,
    mask_block_lengths: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Quantize K with per-block (K64) tile centering and 2-term rotated midpoints."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    scale_group = tl.where(heads_inner != 0, pid1, pid0).to(tl.int64)
    head = tl.where(heads_inner != 0, pid0, pid1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    rows = tl.arange(0, block_n)
    offsets_n = scale_group * block_n + rows
    offsets_d = tl.arange(0, head_dim)
    if mask_block_lengths:
        block_length = tl.load(block_lengths_ptr + scale_group)
        valid_rows = (rows < block_length) & (offsets_n < key_length)
    else:
        valid_rows = offsets_n < key_length
    values = tl.load(
        key_ptr
        + batch * stride_kb
        + head * stride_kh
        + offsets_n[:, None] * stride_kn
        + offsets_d[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    ).to(tl.float32)
    quantized, key_scale, center_0, center_1, center_scale_0, center_scale_1 = (
        quantize_tile_centered_key_tile(
            tl.reshape(values, (block_n, 1, head_dim)),
            tl.reshape(valid_rows, (1, block_n)),
            tl.constexpr(1),
            head_dim,
            block_n,
            block_n,
        )
    )
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_n[:, None] * stride_on
        + offsets_d[None, :],
        tl.reshape(quantized, (block_n, head_dim)),
        mask=offsets_n[:, None] < storage_key_length,
    )
    tl.store(
        scale_ptr + batch * stride_sb + head * stride_sh + scale_group,
        tl.reshape(key_scale, ()),
    )
    store_key_center_tile(
        key_center_ptr,
        key_center_scale_ptr,
        (batch * heads + head) * tile_count + scale_group,
        center_0,
        center_1,
        center_scale_0,
        center_scale_1,
        offsets_d,
        head_dim,
    )


def prepare_tile_centered_key(
    key: torch.Tensor,
    *,
    storage_key_length: int,
    block_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K with per-K64 tile centering and emit 2-term rotated midpoints."""
    batch, heads, key_length, head_dim = key.shape
    if storage_key_length < key_length:
        raise ValueError("K storage length cannot be smaller than the logical key length")
    tile_count = int(triton.cdiv(storage_key_length, _KEY_BLOCK))
    key_int8 = torch.empty(
        (batch, heads, storage_key_length, head_dim),
        device=key.device,
        dtype=torch.int8,
    )
    key_scale = torch.empty((batch, heads, tile_count), device=key.device, dtype=torch.float32)
    key_center = torch.empty(
        (batch, heads, tile_count, 2, head_dim),
        device=key.device,
        dtype=torch.int8,
    )
    key_center_scale = torch.empty(
        (batch, heads, tile_count, 2),
        device=key.device,
        dtype=torch.float32,
    )
    lengths_or_placeholder = block_lengths if block_lengths is not None else key_scale
    grid, heads_inner = select_block_grid(key, tile_count)
    with device_context(key.device):
        quantize_key_tile_centered_per_block_kernel[grid](
            key,
            key_int8,
            key_scale,
            key_center,
            key_center_scale,
            lengths_or_placeholder,
            key_length,
            storage_key_length,
            tile_count,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key_int8.stride(0),
            key_int8.stride(1),
            key_int8.stride(2),
            key_scale.stride(0),
            key_scale.stride(1),
            heads,
            heads_inner,
            mask_block_lengths=block_lengths is not None,
            head_dim=head_dim,
            block_n=_KEY_BLOCK,
            num_warps=2 if head_dim == 128 else 1,
        )
    return key_int8, key_scale, key_center, key_center_scale


def prepare_query(
    query: torch.Tensor,
    softmax_scale: float,
    *,
    grouped: bool,
    per_row: bool = False,
    storage_query_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate and quantize Q with the selected SageAttention granularity.

    ``storage_query_length`` may pad the INT8 destination beyond the logical
    input length. Padded rows and scale groups are zero so an aligned attention
    launch cannot let padding change real-row quantization.
    """
    if grouped and per_row:
        raise ValueError("per_row cannot be combined with grouped=True")
    batch, heads, query_length, head_dim = query.shape
    storage_length = query_length if storage_query_length is None else storage_query_length
    if storage_length < query_length:
        raise ValueError("Q storage length cannot be smaller than the logical query length")
    query_int8 = torch.empty(
        (batch, heads, storage_length, head_dim),
        device=query.device,
        dtype=torch.int8,
    )
    if storage_length != query_length:
        query_int8[:, :, query_length:].zero_()
    with device_context(query.device):
        if grouped:
            logical_scale_groups = int(triton.cdiv(query_length, _QUERY_BLOCK))
            storage_scale_groups = int(triton.cdiv(storage_length, _QUERY_BLOCK))
            query_scale = torch.empty(
                (batch, heads, storage_scale_groups),
                device=query.device,
                dtype=torch.float32,
            )
            if storage_scale_groups != logical_scale_groups:
                query_scale[:, :, logical_scale_groups:].zero_()
            grid, heads_inner = select_block_grid(query, logical_scale_groups)
            quantize_query_per_warp_kernel[grid](
                query,
                query_int8,
                query_scale,
                query_length,
                softmax_scale,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query_int8.stride(0),
                query_int8.stride(1),
                query_int8.stride(2),
                query_scale.stride(0),
                query_scale.stride(1),
                heads_inner,
                head_dim=head_dim,
                num_warps=4,
            )
        else:
            query_scale = torch.empty(
                (batch, heads, storage_length),
                device=query.device,
                dtype=torch.float32,
            )
            if storage_length != query_length:
                query_scale[:, :, query_length:].zero_()
            quantize_query_per_thread_kernel[
                (triton.cdiv(query_length, _QUERY_BLOCK) * 8, heads, batch)
            ](
                query,
                query_int8,
                query_scale,
                query_length,
                softmax_scale,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query_int8.stride(0),
                query_int8.stride(1),
                query_int8.stride(2),
                query_scale.stride(0),
                query_scale.stride(1),
                head_dim=head_dim,
                per_row=per_row,
                num_warps=4,
            )
        return query_int8, query_scale
