"""Pure-Triton backend for the canonical SageAttention2++ 8+8 algorithm.

SageAttention2++ originates from the SageAttention project. This independently
maintained backend targets consumer Ada and Blackwell GPUs without CUDA source
or inline PTX. See the repository NOTICE for upstream attribution.
"""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.scheduling import select_query_block

# Canonical Sage2++ shifts the online-softmax frame by log2(448), so
# probabilities are born in FP8's usable range instead of scaled afterward.
_P_FP8_LOG2_RANGE = tl.constexpr(8.807354922057604)
_V_FP8_RANGE = tl.constexpr(2.25)
_SCALE_EPSILON = tl.constexpr(1e-7)


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
    tl.store(
        value_scale_ptr + output_offsets,
        value_scale,
        mask=offsets_d < head_dim,
    )


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
    scale = tl.load(value_scale_ptr + (batch * heads + head) * head_dim + offsets_d)
    quantized = (value / scale[None, :]).to(tl.float8e4nv)
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_d[None, :] * stride_on
        + offsets_n[:, None],
        quantized,
        mask=mask,
    )


@triton.jit
def _dispatch_kv_quantization_kernel(
    key_ptr,
    value_ptr,
    key_mean_ptr,
    value_scale_ptr,
    key_output_ptr,
    value_output_ptr,
    key_scale_ptr,
    key_length,
    key_groups,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_kob,
    stride_koh,
    stride_kon,
    stride_vob,
    stride_voh,
    stride_von,
    stride_ksb,
    stride_ksh,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Dispatch uniform K and V quantization roles from one grid."""
    role = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_d = tl.arange(0, head_dim)

    if role < key_groups:
        key_offsets_n = role * 64 + tl.arange(0, 64)
        key_mask = key_offsets_n[:, None] < key_length
        key_mean = tl.load(key_mean_ptr + batch_head * head_dim + offsets_d)
        key_values = tl.load(
            key_ptr
            + batch * stride_kb
            + head * stride_kh
            + key_offsets_n[:, None] * stride_kn
            + offsets_d[None, :],
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        key_values = tl.where(key_mask, key_values - key_mean[None, :], 0.0)
        key_maximum = tl.max(tl.max(tl.abs(key_values), axis=1), axis=0)
        key_scale = key_maximum / 127.0 + _SCALE_EPSILON
        key_quantized = qk_quantization.round_to_int8(key_values / key_scale)
        tl.store(
            key_output_ptr
            + batch * stride_kob
            + head * stride_koh
            + key_offsets_n[:, None] * stride_kon
            + offsets_d[None, :],
            key_quantized,
            mask=key_mask,
        )
        tl.store(
            key_scale_ptr + batch * stride_ksb + head * stride_ksh + role,
            key_scale,
        )
    else:
        value_scale_group = role - key_groups
        value_offsets_n = value_scale_group * 64 + tl.arange(0, 64)
        value_mask = value_offsets_n[:, None] < key_length
        value_values = tl.load(
            value_ptr
            + batch * stride_vb
            + head * stride_vh
            + value_offsets_n[:, None] * stride_vn
            + offsets_d[None, :],
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        value_scale = tl.load(value_scale_ptr + batch_head * head_dim + offsets_d)
        value_quantized = (value_values / value_scale[None, :]).to(tl.float8e4nv)
        tl.store(
            value_output_ptr
            + batch * stride_vob
            + head * stride_voh
            + offsets_d[None, :] * stride_von
            + value_offsets_n[:, None],
            value_quantized,
            mask=value_mask,
        )


@triton.jit
def _load_value_tile(
    value_ptr,
    batch,
    head,
    current_n,
    offsets_d,
    key_length,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    pointers = (
        value_ptr
        + ((batch * heads + head) * head_dim + offsets_d[None, :]) * key_length
        + current_n[:, None]
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
            heads,
            head_dim,
        )


@triton.jit
def _causal_attention_tile(
    query,
    query_scale,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    accumulator,
    denominator,
    running_max,
    batch,
    head,
    batch_head,
    start_n,
    offsets_m,
    offsets_n,
    offsets_d,
    key_length,
    diagonal_or_tail: tl.constexpr,
    qk_per_warp: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
):
    """Advance causal online-softmax state by one prefix or boundary tile."""
    current_n = start_n + offsets_n
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
    if qk_per_warp:
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
    if diagonal_or_tail:
        valid_keys = (current_n[None, :] < key_length) & (
            current_n[None, :] <= offsets_m[:, None]
        )
        scores = tl.where(valid_keys, scores, -float("inf"))

    block_max = tl.max(scores, axis=1) - _P_FP8_LOG2_RANGE
    next_max = tl.maximum(running_max, block_max)
    old_weight = tl.exp2(running_max - next_max)
    probabilities = tl.exp2(scores - next_max[:, None])
    accumulator *= old_weight[:, None]
    denominator = denominator * old_weight + tl.sum(probabilities, axis=1)

    probability_fp8 = probabilities.to(tl.float8e4nv)
    value = _load_attention_value_tile(
        value_ptr,
        batch,
        head,
        batch_head,
        start_n,
        current_n,
        offsets_d,
        key_length,
        use_tensor_descriptors,
        heads,
        head_dim,
        block_n,
    )
    partial_fp16 = tl.dot(
        probability_fp8,
        value,
        acc=tl.zeros((block_m, head_dim), dtype=tl.float16),
        out_dtype=tl.float16,
    )
    accumulator += partial_fp16.to(tl.float32)
    return accumulator, denominator, next_max


@triton.jit
def _sage_attention_2pp_kernel(  # noqa: PLR0915 - keep the noncausal loop monolithic
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
    qk_per_warp: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
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
    if qk_per_warp:
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

    batch_head = batch * heads + head
    if is_causal:
        # Omit masks only for complete key tiles strictly before this query
        # block. Ragged tails and overlapping 64-key scale groups stay masked.
        full_key_end = key_length // block_n * block_n
        causal_prefix_end = query_block * block_m // block_n * block_n
        prefix_end = tl.minimum(causal_prefix_end, full_key_end)
        for start_n in tl.range(0, prefix_end, block_n, disable_licm=True):
            accumulator, denominator, running_max = _causal_attention_tile(
                query,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                accumulator,
                denominator,
                running_max,
                batch,
                head,
                batch_head,
                start_n,
                offsets_m,
                offsets_n,
                offsets_d,
                key_length,
                diagonal_or_tail=False,
                qk_per_warp=qk_per_warp,
                heads=heads,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
            )
        for start_n in tl.range(prefix_end, end_n, block_n, disable_licm=True):
            accumulator, denominator, running_max = _causal_attention_tile(
                query,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                accumulator,
                denominator,
                running_max,
                batch,
                head,
                batch_head,
                start_n,
                offsets_m,
                offsets_n,
                offsets_d,
                key_length,
                diagonal_or_tail=True,
                qk_per_warp=qk_per_warp,
                heads=heads,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
            )
    else:
        # Keep the noncausal loop monolithic to preserve its register allocation.
        for start_n in tl.range(0, end_n, block_n, disable_licm=True):
            current_n = start_n + offsets_n
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
            if qk_per_warp:
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
            scores = tl.where(valid_keys, scores, -float("inf"))

            block_max = tl.max(scores, axis=1) - _P_FP8_LOG2_RANGE
            next_max = tl.maximum(running_max, block_max)
            old_weight = tl.exp2(running_max - next_max)
            probabilities = tl.exp2(scores - next_max[:, None])
            accumulator *= old_weight[:, None]
            denominator = denominator * old_weight + tl.sum(probabilities, axis=1)

            probability_fp8 = probabilities.to(tl.float8e4nv)
            value = _load_attention_value_tile(
                value_ptr,
                batch,
                head,
                batch_head,
                start_n,
                current_n,
                offsets_d,
                key_length,
                use_tensor_descriptors,
                heads,
                head_dim,
                block_n,
            )
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
) -> bool:
    """Use the descriptor schedule only for its measured SM120 sweet spot."""
    device_major = torch.cuda.get_device_capability(query.device)[0]
    return (
        device_major == 12
        and block_m == 128
        and head_dim == 128
        and key_length % 16 == 0
    )


def _make_attention_arguments(
    key: torch.Tensor,
    value: torch.Tensor,
    batch: int,
    heads: int,
    key_length: int,
    head_dim: int,
    use_tensor_descriptors: bool,
) -> tuple[torch.Tensor | TensorDescriptor, torch.Tensor | TensorDescriptor]:
    if not use_tensor_descriptors:
        return key, value
    return _make_attention_tensor_descriptors(
        key,
        value,
        batch,
        heads,
        key_length,
        head_dim,
    )


@torch.library.custom_op("piper_kernels::sage_attention_2pp", mutates_args=())
def triton_sage_attention_2pp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Run preprocessing and the fused SageAttention2++ 8+8 kernel."""
    return _run_sage_attention_2pp(query, key, value, scale, is_causal)


def _run_sage_attention_2pp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    use_tensor_descriptors: bool | None = None,
) -> torch.Tensor:
    """Run canonical SageAttention2++ with architecture-specific scheduling."""
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    device_major = torch.cuda.get_device_capability(query.device)[0]
    block_m = select_query_block(query, batch, heads, query_length)
    # For causal SM120 kernels, 64 rows win through 4K; at longer lengths the
    # K/V reuse of 128 rows outweighs the larger accumulator footprint.
    if is_causal and (device_major != 12 or query_length <= 4096):
        block_m = min(block_m, 64)
    padded_key_length = int(triton.cdiv(key_length, 64)) * 64
    if use_tensor_descriptors is None:
        use_tensor_descriptors = _should_use_attention_tensor_descriptors(
            query,
            block_m,
            head_dim,
            padded_key_length,
        )
    storage_key_length = (
        padded_key_length
        if use_tensor_descriptors and key_length % 16 != 0
        else key_length
    )
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
    key_int8_shape = (batch, heads, storage_key_length, head_dim)
    key_int8 = (
        torch.zeros(key_int8_shape, device=query.device, dtype=torch.int8)
        if storage_key_length != key_length
        else torch.empty(key_int8_shape, device=query.device, dtype=torch.int8)
    )
    value_fp8_shape = (batch, heads, head_dim, storage_key_length)
    value_fp8 = (
        torch.zeros(value_fp8_shape, device=query.device, dtype=torch.float8_e4m3fn)
        if storage_key_length != key_length
        else torch.empty(
            value_fp8_shape,
            device=query.device,
            dtype=torch.float8_e4m3fn,
        )
    )
    qk_per_warp = device_major == 12
    if qk_per_warp:
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
        qk_quantization.quantize_query_per_warp_kernel[query_grid](
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
            num_warps=4,
        )
        _dispatch_kv_quantization_kernel[(key_scale_groups * 2, batch * heads)](
            key,
            value,
            key_mean,
            value_scale,
            key_int8,
            value_fp8,
            key_scale,
            key_length,
            key_scale_groups,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            key_int8.stride(0),
            key_int8.stride(1),
            key_int8.stride(2),
            value_fp8.stride(0),
            value_fp8.stride(1),
            value_fp8.stride(2),
            key_scale.stride(0),
            key_scale.stride(1),
            heads=heads,
            head_dim=head_dim,
            num_warps=4,
        )
    else:
        query_scale = torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        key_scale = torch.empty(key.shape[:3], device=query.device, dtype=torch.float32)
        query_grid = (triton.cdiv(query_length, 32) * 8, heads, batch)
        qk_quantization.quantize_query_per_thread_kernel[query_grid](
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
            num_warps=4,
        )
        key_grid = (triton.cdiv(key_length, 64) * 4, heads, batch)
        qk_quantization.quantize_key_per_thread_kernel[key_grid](
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
    if not qk_per_warp:
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
            num_warps=4,
        )

    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    key_argument, value_argument = _make_attention_arguments(
        key_int8,
        value_fp8,
        batch,
        heads,
        storage_key_length,
        head_dim,
        use_tensor_descriptors,
    )
    _sage_attention_2pp_kernel[(triton.cdiv(query_length, block_m), heads, batch)](
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
        qk_per_warp=qk_per_warp,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=64,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=3,
        num_warps=4,
    )
    return output


@triton_sage_attention_2pp.register_fake
def _triton_sage_attention_2pp_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
