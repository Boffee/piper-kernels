"""Pure-Triton backend for the canonical SageAttention2++ 8+8 algorithm.

SageAttention2++ originates from the SageAttention project. This independently
maintained backend targets consumer Ada and Blackwell GPUs without CUDA source
extensions. See the repository NOTICE for upstream attribution.
"""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

from . import _policy

# Triton 3.7 requires globals referenced inside JIT functions to be constexpr.
_BLOCK_N = tl.constexpr(64)
_QUERY_GROUP_SIZE = tl.constexpr(32)
_INT8_MAX = tl.constexpr(127.0)
# Canonical SageAttention2++ shifts the online-softmax frame by log2(448), so
# probabilities are born in FP8's usable range instead of scaled afterward.
_P_FP8_LOG2_MAX = tl.constexpr(8.807354922057604)
_V_FP8_MAX = tl.constexpr(2.25)
_SCALE_EPSILON = tl.constexpr(1e-7)
_LOG2_E = tl.constexpr(1.4426950408889634)


@triton.jit
def _ptx_float32_to_e4m3x4(values):
    """Use the native packed SM89+ conversion used by canonical SageAttention."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
            cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
            mov.b32 $0, {lo, hi};
        }
        """,
        constraints="=r,f,f,f,f",
        args=[values],
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )


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
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
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
    value_scale = tl.max(value_maxima, axis=0) / _V_FP8_MAX + _SCALE_EPSILON
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
    stride_od,
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
    quantized = _ptx_float32_to_e4m3x4(value / scale[None, :])
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_d[None, :] * stride_od
        + offsets_n[:, None],
        quantized,
        mask=mask,
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
def _quantize_query_tile(
    query,
    softmax_scale,
    block_m: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Quantize the query rows owned exclusively by this attention CTA."""
    tl.static_assert(
        block_m % _QUERY_GROUP_SIZE == 0,
        "fused Q quantization requires complete query groups",
    )
    groups: tl.constexpr = block_m // _QUERY_GROUP_SIZE
    grouped_query = query.to(tl.float32).reshape((groups, _QUERY_GROUP_SIZE, head_dim))
    maximum = tl.max(tl.max(tl.abs(grouped_query), axis=2), axis=1)
    quantization_scale = maximum / _INT8_MAX + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(
        grouped_query / quantization_scale[:, None, None]
    ).reshape((block_m, head_dim))
    score_scale = quantization_scale * (softmax_scale * _LOG2_E)
    return quantized, score_scale


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
    grouped_qk: tl.constexpr,
    fuse_query_quantization: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
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
    if grouped_qk:
        groups: tl.constexpr = block_m // _QUERY_GROUP_SIZE
        key_scale = tl.load(
            key_scale_ptr
            + (batch * heads + head) * tl.cdiv(key_length, block_n)
            + start_n // block_n
        )
        raw_scores = integer_scores.to(tl.float32).reshape((groups, _QUERY_GROUP_SIZE, block_n))
        if fuse_query_quantization:
            score_scale = (query_scale * key_scale)[:, None]
        else:
            score_scale = (query_scale * key_scale).reshape((groups, _QUERY_GROUP_SIZE))
        scores = (raw_scores * score_scale[:, :, None]).reshape((block_m, block_n))
    else:
        key_scale = tl.load(
            key_scale_ptr + (batch * heads + head) * key_length + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
        scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
    if diagonal_or_tail:
        valid_keys = (current_n[None, :] < key_length) & (current_n[None, :] <= offsets_m[:, None])
        scores = tl.where(valid_keys, scores, -float("inf"))

    # A grouped Q scale and this tile's K scale form one positive multiplier
    # for each score row, so complete causal-prefix tiles can reduce first.
    if grouped_qk and not diagonal_or_tail:
        raw_block_max = tl.max(raw_scores, axis=2)
        block_max = tl.fma(
            raw_block_max,
            score_scale,
            -_P_FP8_LOG2_MAX,
        ).reshape((block_m,))
    else:
        block_max = tl.max(scores, axis=1) - _P_FP8_LOG2_MAX
    next_max = tl.maximum(running_max, block_max)
    old_weight = tl.exp2(running_max - next_max)
    if grouped_qk and not diagonal_or_tail:
        probability_log2 = tl.fma(
            raw_scores,
            score_scale[:, :, None],
            -next_max.reshape((groups, _QUERY_GROUP_SIZE, 1)),
        ).reshape((block_m, block_n))
        probabilities = tl.exp2(probability_log2)
    else:
        probabilities = tl.exp2(scores - next_max[:, None])
    accumulator *= old_weight[:, None]
    denominator = denominator * old_weight + tl.sum(probabilities, axis=1)

    if use_packed_probability_conversion:
        probability_fp8 = _ptx_float32_to_e4m3x4(probabilities)
    else:
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
def _sage_attention_2pp_kernel(  # noqa: PLR0912, PLR0915 - keep noncausal loop monolithic
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qn,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    fuse_query_quantization: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    reverse_causal_blocks: tl.constexpr,
    loop_num_stages: tl.constexpr,
    loop_licm: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
):
    tl.static_assert(
        not fuse_query_quantization or grouped_qk,
        "fused Q quantization requires grouped Q/K scales",
    )
    tl.static_assert(block_n == _BLOCK_N, "SageAttention2++ requires 64-key tiles")
    query_block = tl.program_id(0)
    if is_causal and reverse_causal_blocks:
        query_block = tl.num_programs(0) - 1 - query_block
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)

    query_values = tl.load(
        query_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qn
        + offsets_d[None, :],
        mask=offsets_m[:, None] < query_length,
        other=0,
    )
    if grouped_qk:
        groups: tl.constexpr = block_m // _QUERY_GROUP_SIZE
    if fuse_query_quantization:
        query, query_scale = _quantize_query_tile(  # pyright: ignore[reportGeneralTypeIssues]
            query_values,
            softmax_scale,
            block_m,
            head_dim,
        )
    elif grouped_qk:
        query = query_values
        query_scale = tl.load(
            query_scale_ptr
            + (batch * heads + head) * tl.cdiv(query_length, _QUERY_GROUP_SIZE)
            + offsets_m // _QUERY_GROUP_SIZE,
            mask=offsets_m < query_length,
            other=0.0,
        )
    else:
        query = query_values
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
        for start_n in tl.range(
            0,
            prefix_end,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=not loop_licm,
        ):
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
                grouped_qk=grouped_qk,
                fuse_query_quantization=fuse_query_quantization,
                use_packed_probability_conversion=use_packed_probability_conversion,
                heads=heads,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
            )
        for start_n in tl.range(
            prefix_end,
            end_n,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=not loop_licm,
        ):
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
                grouped_qk=grouped_qk,
                fuse_query_quantization=fuse_query_quantization,
                use_packed_probability_conversion=use_packed_probability_conversion,
                heads=heads,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
            )
    else:
        # Keep the noncausal loop monolithic to preserve its register allocation.
        for start_n in tl.range(
            0,
            end_n,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=not loop_licm,
        ):
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
            if grouped_qk:
                key_scale = tl.load(
                    key_scale_ptr
                    + (batch * heads + head) * tl.cdiv(key_length, block_n)
                    + start_n // block_n
                )
                raw_scores = integer_scores.to(tl.float32).reshape(
                    (groups, _QUERY_GROUP_SIZE, block_n)
                )
                if fuse_query_quantization:
                    score_scale = (query_scale * key_scale)[:, None]
                else:
                    score_scale = (query_scale * key_scale).reshape((groups, _QUERY_GROUP_SIZE))
            else:
                key_scale = tl.load(
                    key_scale_ptr + (batch * heads + head) * key_length + current_n,
                    mask=current_n < key_length,
                    other=0.0,
                )
                scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
            valid_keys = current_n[None, :] < key_length
            if grouped_qk:
                valid_keys = tl.broadcast_to(valid_keys, (block_m, block_n))
                raw_scores = tl.where(
                    valid_keys.reshape((groups, _QUERY_GROUP_SIZE, block_n)),
                    raw_scores,
                    -float("inf"),
                )
                raw_block_max = tl.max(raw_scores, axis=2)
                block_max = tl.fma(
                    raw_block_max,
                    score_scale,
                    -_P_FP8_LOG2_MAX,
                ).reshape((block_m,))
            else:
                scores = tl.where(valid_keys, scores, -float("inf"))
                block_max = tl.max(scores, axis=1) - _P_FP8_LOG2_MAX
            next_max = tl.maximum(running_max, block_max)
            old_weight = tl.exp2(running_max - next_max)
            if grouped_qk:
                probability_log2 = tl.fma(
                    raw_scores,
                    score_scale[:, :, None],
                    -next_max.reshape((groups, _QUERY_GROUP_SIZE, 1)),
                ).reshape((block_m, block_n))
                probabilities = tl.exp2(probability_log2)
            else:
                probabilities = tl.exp2(scores - next_max[:, None])
            accumulator *= old_weight[:, None]
            denominator = denominator * old_weight + tl.sum(probabilities, axis=1)

            if use_packed_probability_conversion:
                probability_fp8 = _ptx_float32_to_e4m3x4(probabilities)
            else:
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
) -> tuple[TensorDescriptor, TensorDescriptor]:
    """Describe flattened-BH K and feature-major V for descriptor loads."""
    batch, heads, key_length, head_dim = key.shape
    key_descriptor = TensorDescriptor(
        base=key,
        shape=[batch * heads, key_length, head_dim],
        strides=[key_length * head_dim, head_dim, 1],
        block_shape=[1, int(_BLOCK_N), head_dim],
    )
    value_descriptor = TensorDescriptor(
        base=value,
        shape=[batch * heads, head_dim, key_length],
        strides=[head_dim * key_length, key_length, 1],
        block_shape=[1, head_dim, int(_BLOCK_N)],
    )
    return key_descriptor, value_descriptor


def _default_sage_attention_2pp_execution_plan(
    query: torch.Tensor,
    is_causal: bool,
    *,
    target: AcceleratorTarget | None = None,
) -> _policy.SageAttention2ppExecutionPlan:
    """Resolve the production plan used by both benchmark metadata and launch."""
    head_dim = query.shape[3]
    target = AcceleratorTarget.from_device(query.device) if target is None else target
    return _policy.select_execution_plan(
        target,
        candidate_block_m=128,
        head_dim=head_dim,
        is_causal=is_causal,
    )


@dataclass(frozen=True, slots=True)
class _PreparedSageAttention2pp:
    """Quantized tensors and metadata required by the attention launch."""

    query: torch.Tensor
    key: torch.Tensor | TensorDescriptor
    value: torch.Tensor | TensorDescriptor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    value_scale: torch.Tensor
    output: torch.Tensor
    softmax_scale: float
    key_length: int
    is_causal: bool
    plan: _policy.SageAttention2ppExecutionPlan


def _compute_kv_statistics(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-head K mean and per-feature V scale."""
    batch, heads, key_length, head_dim = key.shape
    statistics_block = 256
    num_partials = int(triton.cdiv(key_length, statistics_block))
    partial_shape = (batch, heads, num_partials, head_dim)
    key_sum_partial = torch.empty(partial_shape, device=key.device, dtype=torch.float32)
    value_max_partial = torch.empty_like(key_sum_partial)
    _kv_statistics_partial_kernel[(num_partials, heads, batch)](
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

    key_mean = torch.empty((batch, heads, head_dim), device=key.device, dtype=torch.float32)
    value_scale = torch.empty_like(key_mean)
    _finish_kv_statistics_kernel[(triton.cdiv(head_dim, 32), batch * heads)](
        key_sum_partial,
        value_max_partial,
        key_mean,
        value_scale,
        key_length,
        num_partials,
        head_dim=head_dim,
        partial_block=triton.next_power_of_2(num_partials),
        block_d=32,
        num_warps=4,
    )
    return key_mean, value_scale


def _quantize_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    key_mean: torch.Tensor,
    value_scale: torch.Tensor,
    storage_key_length: int,
    plan: _policy.SageAttention2ppExecutionPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K and V into the attention kernel's storage layouts."""
    batch, heads, key_length, head_dim = key.shape
    value_shape = (batch, heads, head_dim, storage_key_length)
    value_fp8 = (
        torch.zeros(value_shape, device=value.device, dtype=torch.float8_e4m3fn)
        if storage_key_length != key_length
        else torch.empty(value_shape, device=value.device, dtype=torch.float8_e4m3fn)
    )

    num_key_blocks = int(triton.cdiv(key_length, _BLOCK_N))
    key_int8, key_scale = qk_quantization.prepare_key(
        key,
        key_mean,
        grouped=plan.grouped_qk,
        storage_key_length=storage_key_length,
    )
    _quantize_value_kernel[(num_key_blocks, heads, batch)](
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
        block_n=_BLOCK_N,
        num_warps=4,
    )
    return key_int8, value_fp8, key_scale


def _prepare_query(
    query: torch.Tensor,
    value_scale: torch.Tensor,
    softmax_scale: float,
    plan: _policy.SageAttention2ppExecutionPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Q and its scale argument for the selected attention specialization."""
    if plan.fuse_query_quantization:
        # The placeholder scale argument is compiled away in this specialization.
        return query, value_scale
    return qk_quantization.prepare_query(
        query,
        softmax_scale,
        grouped=plan.grouped_qk,
    )


def _prepare_sage_attention_2pp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    execution_plan: _policy.SageAttention2ppExecutionPlan,
) -> _PreparedSageAttention2pp:
    """Quantize inputs and construct the selected attention specialization."""
    key_length = key.shape[2]
    plan = execution_plan
    if plan.reverse_causal_blocks and not is_causal:
        raise ValueError("reverse block order requires causal attention")
    # The feature-major FP8 V descriptor requires its row stride to be
    # 16-byte aligned; pad only storage whose sequence length violates it.
    storage_key_length = (
        int(triton.cdiv(key_length, _BLOCK_N)) * int(_BLOCK_N)
        if plan.use_tensor_descriptors and key_length % 16 != 0
        else key_length
    )
    key_mean, value_scale = _compute_kv_statistics(key, value)
    query_argument, query_scale = _prepare_query(query, value_scale, scale, plan)
    key_int8, value_fp8, key_scale = _quantize_key_value(
        key,
        value,
        key_mean,
        value_scale,
        storage_key_length,
        plan,
    )
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    key_argument: torch.Tensor | TensorDescriptor = key_int8
    value_argument: torch.Tensor | TensorDescriptor = value_fp8
    if plan.use_tensor_descriptors:
        key_argument, value_argument = _make_attention_tensor_descriptors(
            key_int8,
            value_fp8,
        )
    return _PreparedSageAttention2pp(
        query=query_argument,
        key=key_argument,
        value=value_argument,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        output=output,
        softmax_scale=scale,
        key_length=key_length,
        is_causal=is_causal,
        plan=plan,
    )


def _launch_sage_attention_2pp(prepared: _PreparedSageAttention2pp) -> torch.Tensor:
    """Launch only the fused attention recurrence on prepared quantized inputs."""
    batch, heads, query_length, head_dim = prepared.output.shape
    plan = prepared.plan
    _sage_attention_2pp_kernel[(triton.cdiv(query_length, plan.block_m), heads, batch)](
        prepared.query,
        prepared.key,
        prepared.value,
        prepared.query_scale,
        prepared.key_scale,
        prepared.value_scale,
        prepared.output,
        query_length,
        prepared.key_length,
        prepared.softmax_scale,
        prepared.query.stride(0),
        prepared.query.stride(1),
        prepared.query.stride(2),
        is_causal=prepared.is_causal,
        grouped_qk=plan.grouped_qk,
        fuse_query_quantization=plan.fuse_query_quantization,
        use_packed_probability_conversion=plan.use_packed_probability_conversion,
        reverse_causal_blocks=plan.reverse_causal_blocks,
        loop_num_stages=plan.loop_num_stages,
        loop_licm=plan.loop_licm,
        heads=heads,
        head_dim=head_dim,
        block_m=plan.block_m,
        block_n=_BLOCK_N,
        use_tensor_descriptors=plan.use_tensor_descriptors,
        num_stages=plan.num_stages,
        num_warps=plan.num_warps,
    )
    return prepared.output


def _run_sage_attention_2pp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    execution_plan: _policy.SageAttention2ppExecutionPlan | None = None,
) -> torch.Tensor:
    """Run SageAttention2++ preprocessing and its fused recurrence."""
    plan = (
        execution_plan
        if execution_plan is not None
        else _default_sage_attention_2pp_execution_plan(
            query,
            is_causal,
        )
    )
    prepared = _prepare_sage_attention_2pp(
        query,
        key,
        value,
        scale,
        is_causal,
        execution_plan=plan,
    )
    return _launch_sage_attention_2pp(prepared)


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


@triton_sage_attention_2pp.register_fake
def _triton_sage_attention_2pp_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
