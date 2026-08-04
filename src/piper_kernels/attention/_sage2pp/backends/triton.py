"""Pure-Triton SageAttention2++ backend for consumer Ada and Blackwell GPUs."""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels.attention._convrot_triton import rotate_rows_in_registers

_LOG2_E = tl.constexpr(1.4426950408889634)
_P_FP8_RANGE = tl.constexpr(448.0)
_V_FP8_RANGE = tl.constexpr(2.25)
_SCALE_EPSILON = tl.constexpr(1e-7)


@triton.jit
def _round_to_int8(values, quantization_range: tl.constexpr):
    rounded = values + 0.5 * tl.where(values >= 0, 1.0, -1.0)
    return tl.maximum(-quantization_range, tl.minimum(quantization_range, rounded)).to(tl.int8)


@triton.jit
def _kv_statistics_partial_kernel(
    key_ptr,
    value_ptr,
    key_sum_ptr,
    value_max_ptr,
    key_length,
    num_partials,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    partial = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_n = partial * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length

    key = tl.load(
        key_ptr
        + batch * stride_kb
        + head * stride_kh
        + offsets_n[:, None] * stride_kn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    output_offsets = (batch_head * num_partials + partial) * head_dim + offsets_d
    tl.store(key_sum_ptr + output_offsets, tl.sum(key, axis=0))
    tl.store(value_max_ptr + output_offsets, tl.max(tl.abs(value), axis=0))


@triton.jit
def _finish_kv_statistics_kernel(
    key_sum_ptr,
    value_max_ptr,
    key_mean_ptr,
    value_scale_ptr,
    key_length,
    num_partials,
    head_dim: tl.constexpr,
    partial_block: tl.constexpr,
    block_d: tl.constexpr,
):
    block_d_id = tl.program_id(0)
    batch_head = tl.program_id(1)
    offsets_p = tl.arange(0, partial_block)
    offsets_d = block_d_id * block_d + tl.arange(0, block_d)
    mask = (offsets_p[:, None] < num_partials) & (offsets_d[None, :] < head_dim)
    pointers = (
        key_sum_ptr
        + (batch_head * num_partials + offsets_p[:, None]) * head_dim
        + offsets_d[None, :]
    )
    key_sums = tl.load(pointers, mask=mask, other=0.0)
    value_maxima = tl.load(
        value_max_ptr
        + (batch_head * num_partials + offsets_p[:, None]) * head_dim
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    )
    output_offsets = batch_head * head_dim + offsets_d
    tl.store(
        key_mean_ptr + output_offsets,
        tl.sum(key_sums, axis=0) / key_length,
        mask=offsets_d < head_dim,
    )
    value_scale = tl.max(value_maxima, axis=0) / _V_FP8_RANGE + _SCALE_EPSILON
    # Fold the probability range into the scale consumed by the attention loop.
    tl.store(
        value_scale_ptr + output_offsets,
        value_scale / _P_FP8_RANGE,
        mask=offsets_d < head_dim,
    )


@triton.jit
def _quantize_query_kernel(
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
    quantization_range: tl.constexpr,
    rotation_group: tl.constexpr,
):
    scale_group = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
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
    values = rotate_rows_in_registers(values, offsets_d, 4, rotation_group)
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / quantization_range + _SCALE_EPSILON
    quantized = _round_to_int8(values / scale, quantization_range)
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
def _quantize_query_per_warp_kernel(
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
    quantization_range: tl.constexpr,
    rotation_group: tl.constexpr,
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
    values = rotate_rows_in_registers(values, offsets_d, 32, rotation_group)
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / quantization_range + _SCALE_EPSILON
    quantized = _round_to_int8(values / scale, quantization_range)
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
def _quantize_key_kernel(
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
    quantization_range: tl.constexpr,
    rotation_group: tl.constexpr,
):
    scale_group = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
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
    values -= mean[None, :]
    values = rotate_rows_in_registers(values, offsets_d, 16, rotation_group)
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / quantization_range + _SCALE_EPSILON
    quantized = _round_to_int8(values / scale, quantization_range)
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
def _quantize_key_per_block_kernel(
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
    quantization_range: tl.constexpr,
    rotation_group: tl.constexpr,
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
    values -= mean[None, :]
    values = rotate_rows_in_registers(values, offsets_d, block_n, rotation_group)
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / quantization_range + _SCALE_EPSILON
    quantized = _round_to_int8(values / scale, quantization_range)
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


@triton.jit
def _store_value_tile(
    output_ptr,
    values,
    batch,
    head,
    offsets_n,
    offsets_d,
    mask,
    stride_ob,
    stride_oh,
    stride_on,
    output_transposed: tl.constexpr,
):
    if output_transposed:
        pointers = (
            output_ptr
            + batch * stride_ob
            + head * stride_oh
            + offsets_d[None, :] * stride_on
            + offsets_n[:, None]
        )
    else:
        pointers = (
            output_ptr
            + batch * stride_ob
            + head * stride_oh
            + offsets_n[:, None] * stride_on
            + offsets_d[None, :]
        )
    tl.store(pointers, values, mask=mask)


@triton.jit
def _quantize_value_kernel(
    value_ptr,
    value_scale_ptr,
    output_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_on,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    output_transposed: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask = offsets_n[:, None] < key_length
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(value_scale_ptr + (batch * heads + head) * head_dim + offsets_d) * _P_FP8_RANGE
    quantized = (value / scale[None, :]).to(tl.float8e4nv)
    _store_value_tile(
        output_ptr,
        quantized,
        batch,
        head,
        offsets_n,
        offsets_d,
        mask,
        stride_ob,
        stride_oh,
        stride_on,
        output_transposed,
    )


@triton.jit
def _load_value_tile(
    value_ptr,
    batch,
    head,
    current_n,
    offsets_d,
    key_length,
    value_transposed: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    if value_transposed:
        pointers = (
            value_ptr
            + ((batch * heads + head) * head_dim + offsets_d[None, :]) * key_length
            + current_n[:, None]
        )
    else:
        pointers = (
            value_ptr
            + ((batch * heads + head) * key_length + current_n[:, None]) * head_dim
            + offsets_d[None, :]
        )
    return tl.load(
        pointers,
        mask=current_n[:, None] < key_length,
        other=0.0,
    )


@triton.jit
def _load_attention_key_tile(
    key_ptr,
    batch_head,
    start_n,
    current_n,
    offsets_d,
    key_length,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
):
    if use_tensor_descriptors:
        return key_ptr.load([batch_head, start_n, 0]).reshape((block_n, head_dim)).T
    else:
        return tl.load(
            key_ptr
            + (batch_head * key_length + current_n[None, :]) * head_dim
            + offsets_d[:, None],
            mask=current_n[None, :] < key_length,
            other=0,
        )


@triton.jit
def _load_attention_value_tile(
    value_ptr,
    batch,
    head,
    batch_head,
    start_n,
    current_n,
    offsets_d,
    key_length,
    value_transposed: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    if use_tensor_descriptors:
        return value_ptr.load([batch_head, 0, start_n]).reshape((head_dim, block_n)).T
    else:
        return _load_value_tile(
            value_ptr,
            batch,
            head,
            current_n,
            offsets_d,
            key_length,
            value_transposed,
            heads,
            head_dim,
        )


@triton.jit
def _sage_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    pv_accumulator_fp32: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    value_transposed: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
    use_tensor_descriptors: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)

    query = tl.load(
        query_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=offsets_m[:, None] < query_length,
        other=0,
    )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr + (batch * heads + head) * tl.cdiv(query_length, 32) + offsets_m // 32,
            mask=offsets_m < query_length,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + (batch * heads + head) * query_length + offsets_m,
            mask=offsets_m < query_length,
            other=0.0,
        )
    accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, (query_block + 1) * block_m)

    for start_n in tl.range(0, end_n, block_n, disable_licm=True):
        current_n = start_n + offsets_n
        batch_head = batch * heads + head
        key = _load_attention_key_tile(
            key_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores = tl.dot(query, key)
        if grouped_qk:
            key_scale = tl.load(
                key_scale_ptr
                + (batch * heads + head) * tl.cdiv(key_length, block_n)
                + start_n // block_n
            )
            score_scale = query_scale * key_scale
            scores = integer_scores.to(tl.float32) * score_scale[:, None]
        else:
            key_scale = tl.load(
                key_scale_ptr + (batch * heads + head) * key_length + current_n,
                mask=current_n < key_length,
                other=0.0,
            )
            scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
        valid_keys = current_n[None, :] < key_length
        if is_causal:
            valid_keys &= current_n[None, :] <= offsets_m[:, None]
        scores = tl.where(valid_keys, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_weight = tl.exp2(running_max - next_max)
        probabilities = tl.exp2(scores - next_max[:, None])
        accumulator *= old_weight[:, None]
        denominator = denominator * old_weight + tl.sum(probabilities, axis=1)

        probability_fp8 = (probabilities * _P_FP8_RANGE).to(tl.float8e4nv)
        value = _load_attention_value_tile(
            value_ptr,
            batch,
            head,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            value_transposed,
            use_tensor_descriptors,
            heads,
            head_dim,
            block_n,
        )
        if pv_accumulator_fp32:
            accumulator += tl.dot(
                probability_fp8,
                value,
                out_dtype=tl.float32,
            )
        else:
            partial_fp16 = tl.dot(
                probability_fp8,
                value,
                acc=tl.zeros((block_m, head_dim), dtype=tl.float16),
                out_dtype=tl.float16,
            )
            accumulator += partial_fp16.to(tl.float32)
        running_max = next_max

    output = accumulator / denominator[:, None]
    value_scale = tl.load(value_scale_ptr + (batch * heads + head) * head_dim + offsets_d)
    output *= value_scale[None, :]
    tl.store(
        output_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        output,
        mask=offsets_m[:, None] < query_length,
    )


def _select_query_block(
    query: torch.Tensor,
    batch: int,
    heads: int,
    query_length: int,
) -> int:
    """Choose the largest tile that launches at least one CTA per SM."""
    num_sms = torch.cuda.get_device_properties(query.device).multi_processor_count
    parallelism = batch * heads
    for block_m in (128, 64):
        if triton.cdiv(query_length, block_m) * parallelism >= num_sms:
            return block_m
    return 32


def _make_attention_tensor_descriptors(
    key: torch.Tensor,
    value: torch.Tensor,
    batch: int,
    heads: int,
    key_length: int,
    head_dim: int,
) -> tuple[TensorDescriptor, TensorDescriptor]:
    """Describe flattened-BH K and feature-major V for descriptor loads."""
    key_descriptor = TensorDescriptor(
        base=key,
        shape=[batch * heads, key_length, head_dim],
        strides=[key_length * head_dim, head_dim, 1],
        block_shape=[1, 64, head_dim],
    )
    value_descriptor = TensorDescriptor(
        base=value,
        shape=[batch * heads, head_dim, key_length],
        strides=[head_dim * key_length, key_length, 1],
        block_shape=[1, head_dim, 64],
    )
    return key_descriptor, value_descriptor


def _should_use_attention_tensor_descriptors(
    query: torch.Tensor,
    block_m: int,
    head_dim: int,
    key_length: int,
    value_transposed: bool,
) -> bool:
    """Use the descriptor schedule only for its measured SM120 sweet spot."""
    device_major = torch.cuda.get_device_capability(query.device)[0]
    return (
        device_major == 12
        and block_m == 128
        and head_dim == 128
        and key_length % 16 == 0
        and value_transposed
    )


def _make_attention_arguments(
    key: torch.Tensor,
    value: torch.Tensor,
    batch: int,
    heads: int,
    key_length: int,
    head_dim: int,
    value_transposed: bool,
    use_tensor_descriptors: bool,
) -> tuple[torch.Tensor | TensorDescriptor, torch.Tensor | TensorDescriptor]:
    if not use_tensor_descriptors:
        return key, value
    if not value_transposed:
        raise ValueError("tensor descriptors require transposed value storage")
    return _make_attention_tensor_descriptors(
        key,
        value,
        batch,
        heads,
        key_length,
        head_dim,
    )


@torch.library.custom_op("piper_kernels::sage_attention_2pp", mutates_args=())
def triton_sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Run preprocessing and the fused SageAttention2++ 8+8 kernel."""
    return _run_sage_attention(
        query,
        key,
        value,
        scale,
        is_causal,
        qk_quantization_range=127,
    )


def _run_sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    qk_quantization_range: int,
    grouped_qk: bool | None = None,
    rotation_group: int | None = None,
    value_transposed: bool = True,
    use_tensor_descriptors: bool | None = None,
) -> torch.Tensor:
    """Run Sage2++ with an experimental Q/K quantization range.

    Values are currently stored in INT8 tensors and consumed by an INT8
    ``tl.dot`` even when ``qk_quantization_range`` is 7.  That range is useful
    for evaluating INT4 quantization quality, but it does not select native
    packed-INT4 MMA instructions.
    """
    batch, heads, query_length, head_dim = query.shape
    compiled_rotation_group = 0 if rotation_group is None else rotation_group
    key_length = key.shape[2]
    statistics_block = 256
    num_partials = (key_length + statistics_block - 1) // statistics_block
    partial_shape = (batch, heads, num_partials, head_dim)
    key_sum_partial = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    value_max_partial = torch.empty_like(key_sum_partial)
    statistics_grid = (num_partials, batch * heads)
    _kv_statistics_partial_kernel[statistics_grid](
        key,
        value,
        key_sum_partial,
        value_max_partial,
        key_length,
        num_partials,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=statistics_block,
        num_warps=4,
    )

    key_mean = torch.empty((batch, heads, head_dim), device=query.device, dtype=torch.float32)
    value_scale = torch.empty_like(key_mean)
    partial_block = triton.next_power_of_2(num_partials)
    statistics_finish_grid = (triton.cdiv(head_dim, 32), batch * heads)
    _finish_kv_statistics_kernel[statistics_finish_grid](
        key_sum_partial,
        value_max_partial,
        key_mean,
        value_scale,
        key_length,
        num_partials,
        head_dim=head_dim,
        partial_block=partial_block,
        block_d=32,
        num_warps=4,
    )

    # Contiguous intermediates let the hot kernel specialize its indexing while
    # these preprocessing kernels continue to accept arbitrary input strides.
    query_int8 = torch.empty(query.shape, device=query.device, dtype=torch.int8)
    key_int8 = torch.empty(key.shape, device=query.device, dtype=torch.int8)
    value_fp8_shape = (
        (batch, heads, head_dim, key_length) if value_transposed else value.shape
    )
    value_fp8 = torch.empty(
        value_fp8_shape,
        device=query.device,
        dtype=torch.float8_e4m3fn,
    )
    if grouped_qk is None:
        grouped_qk = torch.cuda.get_device_capability(query.device)[0] == 12
    if grouped_qk:
        query_scale_groups = (query_length + 31) // 32
        key_scale_groups = (key_length + 63) // 64
        query_scale = torch.empty(
            (batch, heads, query_scale_groups),
            device=query.device,
            dtype=torch.float32,
        )
        key_scale = torch.empty(
            (batch, heads, key_scale_groups),
            device=query.device,
            dtype=torch.float32,
        )
        query_grid = (query_scale_groups, heads, batch)
        _quantize_query_per_warp_kernel[query_grid](
            query,
            query_int8,
            query_scale,
            query_length,
            scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query_int8.stride(0),
            query_int8.stride(1),
            query_int8.stride(2),
            query_scale.stride(0),
            query_scale.stride(1),
            head_dim=head_dim,
            quantization_range=qk_quantization_range,
            rotation_group=compiled_rotation_group,
            num_warps=4,
        )
        key_grid = (key_scale_groups, heads, batch)
        _quantize_key_per_block_kernel[key_grid](
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
            block_n=64,
            quantization_range=qk_quantization_range,
            rotation_group=compiled_rotation_group,
            num_warps=4,
        )
    else:
        query_scale = torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        key_scale = torch.empty(key.shape[:3], device=query.device, dtype=torch.float32)
        query_grid = (triton.cdiv(query_length, 32) * 8, heads, batch)
        _quantize_query_kernel[query_grid](
            query,
            query_int8,
            query_scale,
            query_length,
            scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query_int8.stride(0),
            query_int8.stride(1),
            query_int8.stride(2),
            query_scale.stride(0),
            query_scale.stride(1),
            head_dim=head_dim,
            quantization_range=qk_quantization_range,
            rotation_group=compiled_rotation_group,
            num_warps=4,
        )
        key_grid = (triton.cdiv(key_length, 64) * 4, heads, batch)
        _quantize_key_kernel[key_grid](
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
            quantization_range=qk_quantization_range,
            rotation_group=compiled_rotation_group,
            num_warps=4,
        )
    value_grid = (triton.cdiv(key_length, 64), heads, batch)
    _quantize_value_kernel[value_grid](
        value,
        value_scale,
        value_fp8,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_fp8.stride(0),
        value_fp8.stride(1),
        value_fp8.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=64,
        output_transposed=value_transposed,
        num_warps=4,
    )

    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    # The 128-row causal variant creates severe accumulator spills in the
    # current NVIDIA lowering; 64 rows avoids that spill-heavy code shape.
    block_m = 64 if is_causal else _select_query_block(query, batch, heads, query_length)
    if use_tensor_descriptors is None:
        use_tensor_descriptors = _should_use_attention_tensor_descriptors(
            query,
            block_m,
            head_dim,
            key_length,
            value_transposed,
        )
    attention_grid = (triton.cdiv(query_length, block_m), heads, batch)
    key_argument, value_argument = _make_attention_arguments(
        key_int8,
        value_fp8,
        batch,
        heads,
        key_length,
        head_dim,
        value_transposed,
        use_tensor_descriptors,
    )
    _sage_attention_kernel[attention_grid](
        query_int8,
        key_argument,
        value_argument,
        query_scale,
        key_scale,
        value_scale,
        output,
        query_length,
        key_length,
        is_causal=is_causal,
        grouped_qk=grouped_qk,
        pv_accumulator_fp32=False,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=64,
        value_transposed=value_transposed,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=3,
        num_warps=4,
    )
    return output


@triton_sage_attention.register_fake
def _triton_sage_attention_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
) -> torch.Tensor:
    return torch.empty_like(query)
