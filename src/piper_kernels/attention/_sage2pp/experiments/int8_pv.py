"""Direct signed-INT8 PV baselines for SageAttention2++.

These experiments deliberately omit ConvRot. They keep the production INT8 QK
and FP32 online softmax, then compare Sage2++'s FP8 PV path with:

* a fastest-case fixed-scale control with one V scale per head/output channel;
* a quality-oriented path with block-local P normalization and V scales.

Both map nonnegative ``P`` to signed-INT8 codes ``[0, 127]`` and use symmetric
signed-INT8 V. Their tensor-core dot accumulates in INT32 before the FP32 online
recurrence. The fixed-scale control factors all PV scales out of the loop; the
block-scaled path pays one partial-output rescale per tile to retain quality.
"""

# ruff: noqa: ANN001, ANN202, PLR0912, PLR0913, PLR0915, PLR0917

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.uint4_pv_convrot import _prepare_qk

_INT8_RANGE = tl.constexpr(127.0)
_FP8_FOLDED_RANGE = tl.constexpr(1008.0)
_INT8_FOLDED_RANGE = tl.constexpr(16129.0)


@triton.jit
def _quantize_value_block_int8_kernel(
    value_ptr,
    scale_ptr,
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
    output_transposed: tl.constexpr,
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
    value_scale = tl.max(tl.abs(value), axis=0) / _INT8_RANGE + 1e-7
    quantized = _sage_backend._round_to_int8(value / value_scale[None, :], _INT8_RANGE)
    scale_block = (batch * heads + head) * tl.cdiv(key_length, block_n) + key_block
    tl.store(scale_ptr + scale_block * head_dim + offsets_d, value_scale)
    _sage_backend._store_value_tile(
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
def _quantize_value_int8_kernel(
    value_ptr,
    folded_fp8_scale_ptr,
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
    output_transposed: tl.constexpr,
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

    # The production statistics kernel stores max(abs(V)) / (448 * 2.25).
    # Recover max(abs(V)) / 127 without launching another reduction.
    folded_scale = tl.load(folded_fp8_scale_ptr + (batch * heads + head) * head_dim + offsets_d)
    value_scale = folded_scale * (_FP8_FOLDED_RANGE / 127.0)
    quantized = _sage_backend._round_to_int8(value / value_scale[None, :], _INT8_RANGE)
    _sage_backend._store_value_tile(
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
def _int8_pv_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    folded_fp8_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    block_scaled_pv: tl.constexpr,
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
    valid_queries = offsets_m < query_length

    query = tl.load(
        query_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr + (batch * heads + head) * tl.cdiv(query_length, 32) + offsets_m // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + (batch * heads + head) * query_length + offsets_m,
            mask=valid_queries,
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
        key = _sage_backend._load_attention_key_tile(
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
            if block_n == 64:
                key_scale = tl.load(
                    key_scale_ptr + (batch * heads + head) * tl.cdiv(key_length, 64) + start_n // 64
                )
                scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
            else:
                key_scale = tl.load(
                    key_scale_ptr
                    + (batch * heads + head) * tl.cdiv(key_length, 64)
                    + current_n // 64,
                    mask=current_n < key_length,
                    other=0.0,
                )
                scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
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
        scores = tl.where(valid_queries[:, None] & valid_keys, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_weight = tl.where(valid_queries, tl.exp2(running_max - next_max), 0.0)
        if block_scaled_pv:
            current_weight = tl.where(valid_queries, tl.exp2(block_max - next_max), 0.0)
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - block_max[:, None]),
                0.0,
            )
        else:
            current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - next_max[:, None]),
                0.0,
            )
        accumulator *= old_weight[:, None]
        denominator = denominator * old_weight + tl.sum(probabilities, axis=1) * current_weight

        # Probabilities are already in [0, 1].  Adding one half before the
        # float-to-integer truncation implements round-to-nearest without the
        # sign branch and clamps needed by the general quantizer.
        probability_int8 = (probabilities * _INT8_RANGE + 0.5).to(tl.int8)
        value = _sage_backend._load_attention_value_tile(
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
        partial_int32 = tl.dot(probability_int8, value, out_dtype=tl.int32)
        if block_scaled_pv:
            value_scale_block = (batch * heads + head) * tl.cdiv(
                key_length, block_n
            ) + start_n // block_n
            value_scale = tl.load(folded_fp8_scale_ptr + value_scale_block * head_dim + offsets_d)
            accumulator += (
                partial_int32.to(tl.float32)
                * current_weight[:, None]
                * (value_scale[None, :] / _INT8_RANGE)
            )
        else:
            accumulator += partial_int32.to(tl.float32)
        running_max = next_max

    output = accumulator / tl.maximum(denominator, 1e-30)[:, None]
    if not block_scaled_pv:
        folded_fp8_scale = tl.load(
            folded_fp8_scale_ptr + (batch * heads + head) * head_dim + offsets_d
        )
        output *= folded_fp8_scale[None, :] * (_FP8_FOLDED_RANGE / _INT8_FOLDED_RANGE)
    tl.store(
        output_ptr
        + ((batch * heads + head) * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        output,
        mask=valid_queries[:, None],
    )


def _prepare_int8_pv_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    value_transposed: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Prepare Q/K/V once for hot-kernel benchmarking."""
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    statistics_block = 256
    num_partials = (key_length + statistics_block - 1) // statistics_block
    partial_shape = (batch, heads, num_partials, head_dim)
    key_sum_partial = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    value_max_partial = torch.empty_like(key_sum_partial)
    _sage_backend._kv_statistics_partial_kernel[(num_partials, batch * heads)](
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
    folded_fp8_scale = torch.empty_like(key_mean)
    _sage_backend._finish_kv_statistics_kernel[(triton.cdiv(head_dim, 32), batch * heads)](
        key_sum_partial,
        value_max_partial,
        key_mean,
        folded_fp8_scale,
        key_length,
        num_partials,
        head_dim=head_dim,
        partial_block=triton.next_power_of_2(num_partials),
        block_d=32,
        num_warps=4,
    )

    query_int8 = torch.empty(query.shape, device=query.device, dtype=torch.int8)
    key_int8 = torch.empty(key.shape, device=query.device, dtype=torch.int8)
    if grouped_qk:
        query_scale = torch.empty(
            (batch, heads, (query_length + 31) // 32),
            device=query.device,
            dtype=torch.float32,
        )
        key_scale = torch.empty(
            (batch, heads, (key_length + 63) // 64),
            device=query.device,
            dtype=torch.float32,
        )
        _sage_backend._quantize_query_per_warp_kernel[
            (triton.cdiv(query_length, 32), heads, batch)
        ](
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
            quantization_range=127,
            rotation_group=0,
            num_warps=4,
        )
        _sage_backend._quantize_key_per_block_kernel[(triton.cdiv(key_length, 64), heads, batch)](
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
            quantization_range=127,
            rotation_group=0,
            num_warps=4,
        )
    else:
        query_scale = torch.empty(query.shape[:3], device=query.device, dtype=torch.float32)
        key_scale = torch.empty(key.shape[:3], device=query.device, dtype=torch.float32)
        _sage_backend._quantize_query_kernel[(triton.cdiv(query_length, 32) * 8, heads, batch)](
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
            quantization_range=127,
            rotation_group=0,
            num_warps=4,
        )
        _sage_backend._quantize_key_kernel[(triton.cdiv(key_length, 64) * 4, heads, batch)](
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
            quantization_range=127,
            rotation_group=0,
            num_warps=4,
        )

    value_int8_shape = (
        (batch, heads, head_dim, key_length) if value_transposed else value.shape
    )
    value_int8 = torch.empty(value_int8_shape, device=query.device, dtype=torch.int8)
    _quantize_value_int8_kernel[(triton.cdiv(key_length, 64), heads, batch)](
        value,
        folded_fp8_scale,
        value_int8,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_int8.stride(0),
        value_int8.stride(1),
        value_int8.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=64,
        output_transposed=value_transposed,
        num_warps=4,
    )
    return query_int8, key_int8, value_int8, query_scale, key_scale, folded_fp8_scale


def _launch_int8_pv_attention(
    prepared: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    query_length: int,
    key_length: int,
    is_causal: bool,
    *,
    grouped_qk: bool,
    block_m: int,
    num_warps: int,
    num_stages: int,
    block_n: int = 64,
    block_scaled_pv: bool = False,
    value_transposed: bool = True,
    use_tensor_descriptors: bool = False,
) -> torch.Tensor:
    query, key, value, query_scale, key_scale, value_scale = prepared
    batch, heads, _, head_dim = query.shape
    key_argument, value_argument = _sage_backend._make_attention_arguments(
        key,
        value,
        batch,
        heads,
        key_length,
        head_dim,
        value_transposed,
        use_tensor_descriptors,
    )
    _int8_pv_attention_kernel[(triton.cdiv(query_length, block_m), heads, batch)](
        query,
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
        block_scaled_pv=block_scaled_pv,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        value_transposed=value_transposed,
        use_tensor_descriptors=use_tensor_descriptors,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def triton_sage_attention_int8_pv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
) -> torch.Tensor:
    """Run the end-to-end fixed-scale signed-INT8 PV baseline."""
    if grouped_qk is None:
        grouped_qk = torch.cuda.get_device_capability(query.device)[0] == 12
    prepared = _prepare_int8_pv_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
        value_transposed=True,
    )
    batch, heads, query_length, _ = query.shape
    key_length = key.shape[2]
    output = torch.empty_like(query)
    block_m = (
        64 if is_causal else _sage_backend._select_query_block(query, batch, heads, query_length)
    )
    use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
        query,
        block_m,
        query.shape[-1],
        key_length,
        True,
    )
    return _launch_int8_pv_attention(
        prepared,
        output,
        query_length,
        key_length,
        is_causal,
        grouped_qk=grouped_qk,
        block_m=block_m,
        num_warps=4,
        num_stages=3,
        value_transposed=True,
        use_tensor_descriptors=use_tensor_descriptors,
    )


def _prepare_block_int8_pv_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    value_transposed: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Prepare canonical Q/K plus 64-token block-scaled signed-INT8 V."""
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        scale,
        grouped_qk,
    )
    batch, heads, key_length, head_dim = value.shape
    value_int8_shape = (
        (batch, heads, head_dim, key_length) if value_transposed else value.shape
    )
    value_int8 = torch.empty(value_int8_shape, device=value.device, dtype=torch.int8)
    value_scale = torch.empty(
        (batch, heads, (key_length + 63) // 64, head_dim),
        device=value.device,
        dtype=torch.float32,
    )
    _quantize_value_block_int8_kernel[(triton.cdiv(key_length, 64), heads, batch)](
        value,
        value_scale,
        value_int8,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_int8.stride(0),
        value_int8.stride(1),
        value_int8.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=64,
        output_transposed=value_transposed,
        num_warps=4,
    )
    return query_int8, key_int8, value_int8, query_scale, key_scale, value_scale


def triton_sage_attention_int8_pv_block_scaled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    grouped_qk: bool | None = None,
) -> torch.Tensor:
    """Run signed-INT8 PV with block-local P normalization and V scales."""
    if grouped_qk is None:
        grouped_qk = torch.cuda.get_device_capability(query.device)[0] == 12
    prepared = _prepare_block_int8_pv_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=grouped_qk,
        value_transposed=True,
    )
    batch, heads, query_length, _ = query.shape
    key_length = key.shape[2]
    output = torch.empty_like(query)
    block_m = (
        64 if is_causal else _sage_backend._select_query_block(query, batch, heads, query_length)
    )
    use_tensor_descriptors = _sage_backend._should_use_attention_tensor_descriptors(
        query,
        block_m,
        query.shape[-1],
        key_length,
        True,
    )
    return _launch_int8_pv_attention(
        prepared,
        output,
        query_length,
        key_length,
        is_causal,
        grouped_qk=grouped_qk,
        block_m=block_m,
        num_warps=4,
        num_stages=3,
        block_scaled_pv=True,
        value_transposed=True,
        use_tensor_descriptors=use_tensor_descriptors,
    )
