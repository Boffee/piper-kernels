"""Triton kernels for Sage-style INT8 Q/K quantization."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

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
    kernels. ``key`` is centered by ``key_mean`` before the signed Hadamard
    transform and quantization.
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

    values = _hadamard_stage(values, head_dim, 1)
    values = _hadamard_stage(values, head_dim, 2)
    values = _hadamard_stage(values, head_dim, 4)
    values = _hadamard_stage(values, head_dim, 8)
    values = _hadamard_stage(values, head_dim, 16)
    values = _hadamard_stage(values, head_dim, 32)
    if head_dim == 128:
        values = _hadamard_stage(values, head_dim, 64)
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
    softmax_scale: tl.constexpr,
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
    raw_scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = round_to_int8(grouped / tl.where(group_valid, raw_scale, 1.0)[:, :, None, None])
    stored_scale = tl.where(
        group_valid,
        raw_scale * (softmax_scale * _LOG2_E),
        0.0,
    )
    return tl.reshape(quantized, (heads_per_program, block_m, head_dim)), stored_scale


@triton.jit
def quantize_key_tile(
    values,
    heads_per_program: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    scale_rows: tl.constexpr,
):
    """Encode one tile of values after any caller-required K centering."""
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
    key_scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = round_to_int8(grouped / key_scale[:, :, None, None])
    return tl.reshape(quantized, (heads_per_program, block_m, head_dim)), key_scale


@triton.jit
def quantize_query_per_thread_group(
    query_ptr,
    output_ptr,
    scale_ptr,
    scale_group,
    head,
    batch,
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
):
    """Quantize one per-thread Q scale group for standalone or fused launchers."""
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
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = round_to_int8(values / scale)
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
        scale * (softmax_scale * _LOG2_E),
        mask=offsets_n < query_length,
    )


@triton.jit
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
):
    """Standalone launcher for the shared per-thread Q component."""
    quantize_query_per_thread_group(
        query_ptr,
        output_ptr,
        scale_ptr,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
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
        head_dim,
    )


@triton.jit
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
    head_dim: tl.constexpr,
):
    scale_group = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
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
    values = rotate_signed_hadamard_heads(values, head_dim)
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = round_to_int8(values / scale)
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
        scale_ptr + batch * stride_sb + head * stride_sh + scale_group,
        scale * (softmax_scale * _LOG2_E),
    )


@triton.jit
def quantize_key_per_thread_group(
    key_ptr,
    mean_ptr,
    output_ptr,
    scale_ptr,
    scale_group,
    head,
    batch,
    key_length,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Quantize one per-thread K scale group for standalone or fused launchers."""
    key_block = scale_group // 4
    thread = scale_group % 4
    group_offsets = tl.arange(0, 16)
    offsets_n = key_block * 64 + (group_offsets // 2) * 8 + (group_offsets % 2) + thread * 2
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length
    mean = tl.load(mean_ptr + (batch * heads + head) * head_dim + offsets_d)
    values = tl.load(
        key_ptr
        + batch * stride_kb
        + head * stride_kh
        + offsets_n[:, None] * stride_kn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = tl.where(mask, values - mean[None, :], 0.0)
    values = rotate_signed_hadamard_heads(values, head_dim)
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = round_to_int8(values / scale)
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
        scale,
        mask=offsets_n < key_length,
    )


@triton.jit
def quantize_key_per_thread_kernel(
    key_ptr,
    mean_ptr,
    output_ptr,
    scale_ptr,
    key_length,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Standalone launcher for the shared per-thread K component."""
    quantize_key_per_thread_group(
        key_ptr,
        mean_ptr,
        output_ptr,
        scale_ptr,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        key_length,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_ob,
        stride_oh,
        stride_on,
        stride_sb,
        stride_sh,
        heads,
        head_dim,
    )


@triton.jit
def quantize_key_per_block_kernel(
    key_ptr,
    mean_ptr,
    output_ptr,
    scale_ptr,
    key_length,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    scale_group = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = scale_group * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length
    mean = tl.load(mean_ptr + (batch * heads + head) * head_dim + offsets_d)
    values = tl.load(
        key_ptr
        + batch * stride_kb
        + head * stride_kh
        + offsets_n[:, None] * stride_kn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = tl.where(mask, values - mean[None, :], 0.0)
    values = rotate_signed_hadamard_heads(values, head_dim)
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = round_to_int8(values / scale)
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
        scale_ptr + batch * stride_sb + head * stride_sh + scale_group,
        scale,
    )


def prepare_query(
    query: torch.Tensor,
    softmax_scale: float,
    *,
    grouped: bool,
    storage_query_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate and quantize Q with the selected SageAttention granularity.

    ``storage_query_length`` may pad the INT8 destination beyond the logical
    input length. Padded rows and scale groups are zero so an aligned attention
    launch cannot let padding change real-row quantization.
    """
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
        quantize_query_per_warp_kernel[(logical_scale_groups, heads, batch)](
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
            num_warps=4,
        )
    return query_int8, query_scale


def prepare_key(
    key: torch.Tensor,
    key_mean: torch.Tensor,
    *,
    grouped: bool,
    storage_key_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate and quantize centered K with the selected SageAttention granularity."""
    batch, heads, key_length, head_dim = key.shape
    key_shape = (batch, heads, storage_key_length, head_dim)
    key_int8 = (
        torch.zeros(key_shape, device=key.device, dtype=torch.int8)
        if storage_key_length != key_length
        else torch.empty(key_shape, device=key.device, dtype=torch.int8)
    )
    if grouped:
        scale_groups = int(triton.cdiv(key_length, _KEY_BLOCK))
        key_scale = torch.empty(
            (batch, heads, scale_groups),
            device=key.device,
            dtype=torch.float32,
        )
        quantize_key_per_block_kernel[(scale_groups, heads, batch)](
            key,
            key_mean,
            key_int8,
            key_scale,
            key_length,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key_int8.stride(0),
            key_int8.stride(1),
            key_int8.stride(2),
            key_scale.stride(0),
            key_scale.stride(1),
            heads=heads,
            head_dim=head_dim,
            block_n=_KEY_BLOCK,
            num_warps=4,
        )
    else:
        key_scale = torch.empty(key.shape[:3], device=key.device, dtype=torch.float32)
        quantize_key_per_thread_kernel[(triton.cdiv(key_length, _KEY_BLOCK) * 4, heads, batch)](
            key,
            key_mean,
            key_int8,
            key_scale,
            key_length,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key_int8.stride(0),
            key_int8.stride(1),
            key_int8.stride(2),
            key_scale.stride(0),
            key_scale.stride(1),
            heads=heads,
            head_dim=head_dim,
            num_warps=4,
        )
    return key_int8, key_scale


def prepare_query_key(
    query: torch.Tensor,
    key: torch.Tensor,
    key_mean: torch.Tensor,
    softmax_scale: float,
    *,
    grouped: bool,
    storage_key_length: int,
    storage_query_length: int | None = None,
) -> PreparedInt8QueryKey:
    """Prepare the shared INT8 Q/K contract consumed by quantized attention.

    Attention-specific V encoding, routing summaries, descriptors, and output
    storage deliberately remain outside this function. A fused projection
    frontend can produce the same contract without materializing BF16 Q/K.
    """
    query_int8, query_scale = prepare_query(
        query,
        softmax_scale,
        grouped=grouped,
        storage_query_length=storage_query_length,
    )
    key_int8, key_scale = prepare_key(
        key,
        key_mean,
        grouped=grouped,
        storage_key_length=storage_key_length,
    )
    return PreparedInt8QueryKey(
        query=query_int8,
        key=key_int8,
        query_scale=query_scale,
        key_scale=key_scale,
    )
