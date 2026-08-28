"""Pure-Triton backend for Piper Attention.

The kernel keeps Sage-style INT8 QK and FP32 online softmax, but replaces the
FP8 PV path with one signed-INT8 scale per V row. Each row scale is folded into
the nonnegative probability operand, producing UINT8-by-INT8 tensor-core dots.
Native mixed-sign MMA is used on supported NVIDIA targets; other targets use
the portable PyTorch reference.
"""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from dataclasses import dataclass
from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels._triton.mixed_int8 import (
    install_uint8_int8_dot_hook,
    uint8_int8_dot,
)
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)

from . import _policy, _quantization

_BLOCK_N = 64
_P_UINT8_RANGE = tl.constexpr(255.0)
_P_UINT8_LOG2_RANGE = tl.constexpr(7.994353436858858)
# Pad the analytical maximum to the next effective FP32 constant so the
# lowered expression remains an upper bound after integer-to-FP32 rounding.
_VALUE_LOG_BOUND_CORRECTION = tl.constexpr(0.086085)


@triton.jit
def _ptx_float32_to_uint8x4(values):
    """Truncate and saturate four probability codes with packed SM72+ PTX."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .s32 a, b, c, d;
            .reg .b32 lo;
            cvt.rzi.s32.f32 a, $1;
            cvt.rzi.s32.f32 b, $2;
            cvt.rzi.s32.f32 c, $3;
            cvt.rzi.s32.f32 d, $4;
            cvt.pack.sat.u8.s32.b32 lo, d, c, 0;
            cvt.pack.sat.u8.s32.b32 $0, b, a, lo;
        }
        """,
        constraints="=r,f,f,f,f",
        args=[values],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _conservative_value_log_scale_bound(value_scale_multiplier):
    """Bound log2(scale) from the positive FP32 multiplier's IEEE-754 bits."""
    multiplier_bits = value_scale_multiplier.to(tl.int32, bitcast=True)
    return multiplier_bits.to(tl.float32) * (1.0 / 8388608.0) - (
        127.0 + _P_UINT8_LOG2_RANGE - _VALUE_LOG_BOUND_CORRECTION
    )


@triton.jit
def _quantize_value_per_key_kernel(
    value_ptr,
    value_mean_ptr,
    scale_multiplier_ptr,
    log_scale_ptr,
    output_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_od,
    stride_ok,
    is_causal: tl.constexpr,
    store_log_scale: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Standalone launcher for the shared per-key V component."""
    _quantization.quantize_value_per_key_block(
        value_ptr,
        value_mean_ptr,
        output_ptr,
        scale_multiplier_ptr,
        log_scale_ptr,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        key_length,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        stride_od,
        stride_ok,
        is_causal,
        store_log_scale,
        heads,
        head_dim,
        block_n,
    )


@triton.jit
def _load_key_tile(
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
def _load_value_tile(
    value_ptr,
    batch_head,
    start_n,
    current_n,
    offsets_d,
    key_length,
    feature_start: tl.constexpr,
    feature_block: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
):
    if use_tensor_descriptors:
        return (
            value_ptr.load([batch_head, feature_start, start_n]).reshape((feature_block, block_n)).T
        )
    else:
        return tl.load(
            value_ptr
            + (batch_head * head_dim + feature_start + offsets_d[None, :]) * key_length
            + current_n[:, None],
            mask=current_n[:, None] < key_length,
            other=0,
        )


@triton.jit
def _attention_tile(  # noqa: PLR0915
    query,
    query_scale,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    numerator,
    denominator,
    running_max,
    batch_head,
    start_n,
    offsets_m,
    offsets_n,
    offsets_d,
    valid_queries,
    key_length,
    mask_keys: tl.constexpr,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    derive_value_log_bound: tl.constexpr,
):
    """Advance online-softmax state by one key tile.

    FP32 numerators remain in UINT8 probability-code units for every attention
    mode. The common scale is removed once in the epilogue, before non-causal
    value-mean restoration.
    """
    current_n = start_n + offsets_n
    key = _load_key_tile(
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
    integer_scores = tl.dot(query, key, out_dtype=tl.int32)
    if grouped_qk:
        key_scale = tl.load(
            key_scale_ptr + batch_head * tl.cdiv(key_length, block_n) + start_n // block_n
        )
        scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
    else:
        key_scale = tl.load(
            key_scale_ptr + batch_head * key_length + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
        scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]

    if mask_keys:
        valid_keys = current_n[None, :] < key_length
        if is_causal:
            valid_keys &= current_n[None, :] <= offsets_m[:, None]
        scores = tl.where(valid_queries[:, None] & valid_keys, scores, -float("inf"))
    else:
        valid_keys = tl.full((block_m, block_n), True, dtype=tl.int1)

    if derive_value_log_bound:
        value_scale_multiplier = tl.load(
            value_scale_multiplier_ptr + batch_head * key_length + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
        value_log_scale = _conservative_value_log_scale_bound(value_scale_multiplier)
    else:
        value_log_scale = tl.load(
            value_log_scale_ptr + batch_head * key_length + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
    shifted_scores = scores + value_log_scale[None, :]
    block_max = tl.max(shifted_scores, axis=1)
    next_max = tl.maximum(running_max, block_max)
    old_weight = tl.where(
        valid_queries,
        tl.exp2(running_max - next_max),
        0.0,
    )
    current_weight = tl.where(
        valid_queries,
        tl.exp2(block_max - next_max),
        0.0,
    )
    probabilities = tl.where(
        valid_queries[:, None] & valid_keys,
        tl.exp2(scores - block_max[:, None]),
        0.0,
    )
    denominator = denominator * old_weight + tl.sum(probabilities, axis=1) * current_weight
    if not derive_value_log_bound:
        value_scale_multiplier = tl.load(
            value_scale_multiplier_ptr + batch_head * key_length + current_n,
            mask=current_n < key_length,
            other=0.0,
        )
    probability_values = probabilities * value_scale_multiplier[None, :] + 0.5
    if use_packed_probability_conversion:
        probability_uint8 = _ptx_float32_to_uint8x4(probability_values)
    else:
        probability_codes = tl.minimum(
            _P_UINT8_RANGE,
            probability_values,
        ).to(tl.int32)
        probability_uint8 = probability_codes.to(tl.uint8)

    if split_pv_head_dim:
        accumulator_low, accumulator_high = numerator
        half_head_dim: tl.constexpr = head_dim // 2
        offsets_vd = tl.arange(0, half_head_dim)
        value_low = _load_value_tile(
            value_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_vd,
            key_length,
            0,
            half_head_dim,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        value_high = _load_value_tile(
            value_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_vd,
            key_length,
            half_head_dim,
            half_head_dim,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        partial_low = uint8_int8_dot(probability_uint8, value_low)
        partial_high = uint8_int8_dot(probability_uint8, value_high)
        accumulator_low = (
            accumulator_low * old_weight[:, None]
            + partial_low.to(tl.float32) * current_weight[:, None]
        )
        accumulator_high = (
            accumulator_high * old_weight[:, None]
            + partial_high.to(tl.float32) * current_weight[:, None]
        )
        numerator = (accumulator_low, accumulator_high)
    else:
        accumulator = numerator
        value_tile = _load_value_tile(
            value_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            0,
            head_dim,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        partial = uint8_int8_dot(probability_uint8, value_tile)
        accumulator = (
            accumulator * old_weight[:, None] + partial.to(tl.float32) * current_weight[:, None]
        )
        numerator = accumulator
    return numerator, denominator, next_max


@triton.jit
def _piper_attention_kernel(  # noqa: PLR0912, PLR0915
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    value_mean_ptr,
    output_ptr,
    query_length,
    key_length,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    unmasked_query_tiles: tl.constexpr,
    unmasked_key_tiles: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_query_tensor_descriptor: tl.constexpr,
    optimize_causal_traversal: tl.constexpr,
    loop_num_stages: tl.constexpr,
    loop_licm: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    derive_value_log_bound: tl.constexpr,
):
    """Fused UINT8-P/INT8-V online attention."""
    if unmasked_query_tiles:
        query_block = tl.program_id(0)
        if is_causal and optimize_causal_traversal:
            query_block = tl.num_programs(0) - 1 - query_block
    else:
        # A masked launch contains only the single ragged query tail. Derive
        # its block at runtime so the exact query length is not a JIT key.
        query_block = query_length // block_m
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    if unmasked_query_tiles:
        valid_queries = tl.full((block_m,), True, dtype=tl.int1)
    else:
        valid_queries = offsets_m < query_length

    if use_query_tensor_descriptor:
        query = query_ptr.load([batch_head, query_block * block_m, 0]).reshape((block_m, head_dim))
    else:
        query = tl.load(
            query_ptr
            + ((batch_head * query_length + offsets_m[:, None]) * head_dim)
            + offsets_d[None, :],
            mask=valid_queries[:, None],
            other=0,
        )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr + batch_head * tl.cdiv(query_length, 32) + offsets_m // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + batch_head * query_length + offsets_m,
            mask=valid_queries,
            other=0.0,
        )

    if split_pv_head_dim:
        half_head_dim: tl.constexpr = head_dim // 2
        offsets_vd = tl.arange(0, half_head_dim)
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    else:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, (query_block + 1) * block_m)

    if is_causal and optimize_causal_traversal:
        numerator = (accumulator_low, accumulator_high) if split_pv_head_dim else accumulator
        # Only complete K tiles strictly before the first query row are
        # mask-free. Keep ragged tails and diagonal overlap in the boundary.
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
            numerator, denominator, running_max = _attention_tile(
                query,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                numerator,
                denominator,
                running_max,
                batch_head,
                start_n,
                offsets_m,
                offsets_n,
                offsets_d,
                valid_queries,
                key_length,
                mask_keys=False,
                is_causal=is_causal,
                grouped_qk=grouped_qk,
                split_pv_head_dim=split_pv_head_dim,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
                use_packed_probability_conversion=use_packed_probability_conversion,
                derive_value_log_bound=derive_value_log_bound,
            )
        for start_n in tl.range(
            prefix_end,
            end_n,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=not loop_licm,
        ):
            numerator, denominator, running_max = _attention_tile(
                query,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                numerator,
                denominator,
                running_max,
                batch_head,
                start_n,
                offsets_m,
                offsets_n,
                offsets_d,
                valid_queries,
                key_length,
                mask_keys=True,
                is_causal=is_causal,
                grouped_qk=grouped_qk,
                split_pv_head_dim=split_pv_head_dim,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
                use_packed_probability_conversion=use_packed_probability_conversion,
                derive_value_log_bound=derive_value_log_bound,
            )
        if split_pv_head_dim:
            accumulator_low, accumulator_high = numerator
        else:
            accumulator = numerator
    else:
        numerator = (accumulator_low, accumulator_high) if split_pv_head_dim else accumulator
        for start_n in tl.range(
            0,
            end_n,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=not loop_licm,
        ):
            numerator, denominator, running_max = _attention_tile(
                query,
                query_scale,
                key_ptr,
                value_ptr,
                key_scale_ptr,
                value_scale_multiplier_ptr,
                value_log_scale_ptr,
                numerator,
                denominator,
                running_max,
                batch_head,
                start_n,
                offsets_m,
                offsets_n,
                offsets_d,
                valid_queries,
                key_length,
                mask_keys=not unmasked_key_tiles,
                is_causal=is_causal,
                grouped_qk=grouped_qk,
                split_pv_head_dim=split_pv_head_dim,
                head_dim=head_dim,
                block_m=block_m,
                block_n=block_n,
                use_tensor_descriptors=use_tensor_descriptors,
                use_packed_probability_conversion=use_packed_probability_conversion,
                derive_value_log_bound=derive_value_log_bound,
            )
        if split_pv_head_dim:
            accumulator_low, accumulator_high = numerator
        else:
            accumulator = numerator
    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    denominator_code_units = denominator_safe * _P_UINT8_RANGE
    if split_pv_head_dim:
        output_low = accumulator_low / denominator_code_units
        output_high = accumulator_high / denominator_code_units
        if not is_causal:
            value_mean_base = value_mean_ptr + batch_head * head_dim
            output_low += tl.load(value_mean_base + offsets_vd)[None, :]
            output_high += tl.load(value_mean_base + half_head_dim + offsets_vd)[None, :]
        output_base = output_ptr + (batch_head * query_length + offsets_m[:, None]) * head_dim
        tl.store(
            output_base + offsets_vd[None, :],
            output_low,
            mask=valid_queries[:, None],
        )
        tl.store(
            output_base + half_head_dim + offsets_vd[None, :],
            output_high,
            mask=valid_queries[:, None],
        )
    else:
        output = accumulator / denominator_code_units
        if not is_causal:
            output += tl.load(value_mean_ptr + batch_head * head_dim + offsets_d)[None, :]
        tl.store(
            output_ptr
            + (batch_head * query_length + offsets_m[:, None]) * head_dim
            + offsets_d[None, :],
            output,
            mask=valid_queries[:, None],
        )


def _make_key_value_descriptors(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    split_pv_head_dim: bool,
) -> tuple[TensorDescriptor, TensorDescriptor]:
    batch, heads, storage_key_length, head_dim = key.shape
    key_descriptor = TensorDescriptor(
        base=key,
        shape=[batch * heads, storage_key_length, head_dim],
        strides=[storage_key_length * head_dim, head_dim, 1],
        block_shape=[1, _BLOCK_N, head_dim],
    )
    value_descriptor = TensorDescriptor(
        base=value,
        shape=[batch * heads, head_dim, storage_key_length],
        strides=[head_dim * storage_key_length, storage_key_length, 1],
        block_shape=[
            1,
            head_dim // 2 if split_pv_head_dim else head_dim,
            _BLOCK_N,
        ],
    )
    return key_descriptor, value_descriptor


def _make_query_descriptor(
    query: torch.Tensor,
    block_m: int,
) -> TensorDescriptor:
    """Describe flattened-BH Q for complete query-block loads."""
    batch, heads, query_length, head_dim = query.shape
    return TensorDescriptor(
        base=query,
        shape=[batch * heads, query_length, head_dim],
        strides=[query_length * head_dim, head_dim, 1],
        block_shape=[1, block_m, head_dim],
    )


def _default_piper_attention_execution_plan(
    query: torch.Tensor,
    is_causal: bool,
    *,
    target: AcceleratorTarget | None = None,
) -> _policy.PiperAttentionExecutionPlan:
    """Resolve production policy for preparation, benchmarks, and tuning."""
    head_dim = query.shape[3]
    target = AcceleratorTarget.from_device(query.device) if target is None else target
    return _policy.select_execution_plan(
        target,
        head_dim=head_dim,
        is_causal=is_causal,
    )


@dataclass(frozen=True, slots=True)
class _PreparedPiperAttention:
    query: torch.Tensor
    query_descriptor: TensorDescriptor | None
    key: torch.Tensor | TensorDescriptor
    value: torch.Tensor | TensorDescriptor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_log_scale: torch.Tensor
    value_mean: torch.Tensor
    output: torch.Tensor
    key_length: int
    is_causal: bool
    plan: _policy.PiperAttentionExecutionPlan


def _prepare_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    execution_plan: _policy.PiperAttentionExecutionPlan,
) -> _PreparedPiperAttention:
    """Quantize Q/K/V and construct the selected launch specialization."""
    batch, heads, _query_length, head_dim = query.shape
    key_length = key.shape[2]
    plan = execution_plan
    if plan.split_pv_head_dim and head_dim != 128:
        raise ValueError("split-PV Piper Attention requires head_dim=128")
    if plan.optimize_causal_traversal and not is_causal:
        raise ValueError("optimized causal traversal requires causal attention")
    with torch.cuda.device(query.device):
        install_uint8_int8_dot_hook()
    padded_key_length = int(triton.cdiv(key_length, _BLOCK_N)) * _BLOCK_N
    storage_key_length = padded_key_length if plan.use_tensor_descriptors else key_length

    # A sequence-wide V mean is valid only for non-causal attention. Per-row
    # INT8 rounding would otherwise let future V rows perturb earlier outputs.
    key_mean, value_mean = _quantization.compute_kv_means(
        key,
        value,
        is_causal=is_causal,
    )
    prepared_qk = qk_quantization.prepare_query_key(
        query,
        key,
        key_mean,
        scale,
        grouped=plan.grouped_qk,
        storage_key_length=storage_key_length,
    )

    value_shape = (batch, heads, head_dim, storage_key_length)
    value_int8 = (
        torch.zeros(value_shape, device=value.device, dtype=torch.int8)
        if storage_key_length != key_length
        else torch.empty(value_shape, device=value.device, dtype=torch.int8)
    )
    value_scale_multiplier = torch.empty(
        (batch, heads, key_length),
        device=value.device,
        dtype=torch.float32,
    )
    value_log_scale = torch.empty(
        (1,) if plan.derive_value_log_bound else (batch, heads, key_length),
        device=value.device,
        dtype=torch.float16,
    )
    _quantize_value_per_key_kernel[(triton.cdiv(key_length, _BLOCK_N), heads, batch)](
        value,
        value_mean,
        value_scale_multiplier,
        value_log_scale,
        value_int8,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_int8.stride(0),
        value_int8.stride(1),
        value_int8.stride(2),
        value_int8.stride(3),
        is_causal=is_causal,
        store_log_scale=not plan.derive_value_log_bound,
        heads=heads,
        head_dim=head_dim,
        block_n=_BLOCK_N,
        num_warps=4,
    )

    key_argument: torch.Tensor | TensorDescriptor = prepared_qk.key
    value_argument: torch.Tensor | TensorDescriptor = value_int8
    if plan.use_tensor_descriptors:
        key_argument, value_argument = _make_key_value_descriptors(
            prepared_qk.key,
            value_int8,
            split_pv_head_dim=plan.split_pv_head_dim,
        )
    query_descriptor = (
        _make_query_descriptor(
            prepared_qk.query,
            plan.block_m,
        )
        if plan.use_tensor_descriptors and plan.block_m == 128
        else None
    )
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    return _PreparedPiperAttention(
        query=prepared_qk.query,
        query_descriptor=query_descriptor,
        key=key_argument,
        value=value_argument,
        query_scale=prepared_qk.query_scale,
        key_scale=prepared_qk.key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_log_scale=value_log_scale,
        value_mean=value_mean,
        output=output,
        key_length=key_length,
        is_causal=is_causal,
        plan=plan,
    )


def _launch_piper_attention(prepared: _PreparedPiperAttention) -> torch.Tensor:
    """Launch only the fused attention recurrence on prepared integer inputs."""
    batch, heads, query_length, head_dim = prepared.output.shape
    plan = prepared.plan
    attention_kernel = cast(Any, _piper_attention_kernel)
    launch_options = {
        "num_warps": plan.num_warps,
        "num_stages": plan.num_stages,
    }

    def launch(query_blocks: int, unmasked_queries: bool) -> None:
        use_query_tensor_descriptor = unmasked_queries and prepared.query_descriptor is not None
        query_argument = (
            prepared.query_descriptor if use_query_tensor_descriptor else prepared.query
        )
        attention_kernel[(query_blocks, heads, batch)](
            query_argument,
            prepared.key,
            prepared.value,
            prepared.query_scale,
            prepared.key_scale,
            prepared.value_scale_multiplier,
            prepared.value_log_scale,
            prepared.value_mean,
            prepared.output,
            query_length,
            prepared.key_length,
            is_causal=prepared.is_causal,
            grouped_qk=plan.grouped_qk,
            split_pv_head_dim=plan.split_pv_head_dim,
            unmasked_query_tiles=unmasked_queries,
            unmasked_key_tiles=(
                unmasked_queries and not prepared.is_causal and prepared.key_length % _BLOCK_N == 0
            ),
            heads=heads,
            head_dim=head_dim,
            block_m=plan.block_m,
            block_n=_BLOCK_N,
            use_tensor_descriptors=plan.use_tensor_descriptors,
            use_query_tensor_descriptor=use_query_tensor_descriptor,
            optimize_causal_traversal=plan.optimize_causal_traversal,
            loop_num_stages=plan.loop_num_stages,
            loop_licm=plan.loop_licm,
            use_packed_probability_conversion=plan.use_packed_probability_conversion,
            derive_value_log_bound=plan.derive_value_log_bound,
            **launch_options,
        )

    full_query_blocks = query_length // plan.block_m
    has_partial_query_block = query_length % plan.block_m != 0
    if plan.optimize_causal_traversal and has_partial_query_block:
        launch(1, False)
    if full_query_blocks:
        launch(full_query_blocks, True)
    if not plan.optimize_causal_traversal and has_partial_query_block:
        launch(1, False)
    return prepared.output


def _run_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    execution_plan: _policy.PiperAttentionExecutionPlan | None = None,
) -> torch.Tensor:
    """Run Piper Attention preprocessing and its fused recurrence."""
    plan = (
        execution_plan
        if execution_plan is not None
        else _default_piper_attention_execution_plan(
            query,
            is_causal,
        )
    )
    prepared = _prepare_piper_attention(
        query,
        key,
        value,
        scale,
        is_causal,
        execution_plan=plan,
    )
    return _launch_piper_attention(prepared)


@torch.library.custom_op("piper_kernels::piper_attention", mutates_args=())
def triton_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Run Piper Attention preprocessing and its fused integer-PV kernel."""
    return _run_piper_attention(
        query,
        key,
        value,
        scale,
        is_causal,
    )


@triton_piper_attention.register_fake
def _triton_piper_attention_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
