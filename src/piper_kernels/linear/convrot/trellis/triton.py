"""Triton hybrid execution for ConvRot Trellis linear.

Small batches (``M <= 8``, and Triton GPUs without an optimized ConvRot INT8
W8A8 backend) execute a fused in-register ``Trellis -> A16`` Split-K GEMV
kernel without materializing decoded INT8 weights in global memory. Larger
batches with an INT8 backend rotate/quantize inputs once and stream bounded
scratch-decoded N chunks through ``ConvRotInt8`` W8A8 execution.
"""

# Triton's JIT launcher and constexpr options are not fully represented by its typing surface.
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.runtime import device_context
from piper_kernels.linear._splitk import device_sm_count, select_split_k
from piper_kernels.linear.convrot.int8 import _backend as convrot_int8_backend
from piper_kernels.trellis._layout import TrellisLayout, validate_block_size
from piper_kernels.trellis.triton import (
    _bit_window_params,
    _decode_states_to_q_fp32,
    _load_cyclic_states,
    compute_row_offsets,
    decode_int8_out,
    decode_variable_rows_int8_out,
)
from piper_kernels.weights.convrot.triton import rotate_input

_GEMV_MAX_ROWS = 8
_SCRATCH_DECODE_MAX_BYTES = 32 * 1024 * 1024
_REDUCE_BLOCK_N = 256
_SPLITK_MIN_WEIGHT_BYTES = 2 * 1024 * 1024
_SPLITK_MAX = 8
_TRITON_FLOAT_POINTER_TYPES = {
    torch.float16: "*fp16",
    torch.bfloat16: "*bf16",
    torch.float32: "*fp32",
}


@triton.jit
def _gemv_accumulate_k_blocks(
    x_rot_ptr,
    bitstreams_ptr,
    block_scales_ptr,
    row_index,
    row_mask,
    ms,
    steps,
    elem_mask,
    x_mask,
    byte,
    shift,
    head_mask,
    has_tail,
    tail_bits_u32,
    tail_shift,
    pid_k,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    bytes_per_block,
    stride_xm,
    split_k,
    row_base_bytes,
    width64,
    u_b0_mask,
    u_b1_mask,
    u_m0,
    u_m1,
    u_m2,
    block_m: tl.constexpr,
    tile_steps: tl.constexpr,
    variable_rows: tl.constexpr,
):
    """Run the unified uniform/variable K-block accumulation loop for one CTA split."""
    blocks_per_split = (blocks_per_row + split_k - 1) // split_k
    b_start = pid_k * blocks_per_split
    b_end = tl.minimum(b_start + blocks_per_split, blocks_per_row)
    row_blk_base = row_index.to(tl.int64) * blocks_per_row
    bpb64 = bytes_per_block.to(tl.int64)

    acc = tl.zeros((block_m, tile_steps), dtype=tl.float32)
    for b in range(b_start, b_end):
        blk_idx = row_blk_base + b
        if variable_rows:
            base = row_base_bytes + tl.full((), b, tl.int64) * width64
            # Bound loads to bitstreams even if caller-owned row_widths are malformed.
            remaining_bytes = tl.maximum(bitstreams_bytes - base, 0)
            stream_bytes = tl.minimum(width64, remaining_bytes).to(tl.int32)
            stream_bytes_2d = stream_bytes[:, None]
            block_ptr_1d = bitstreams_ptr + base
            iter_b0_mask = row_mask & (stream_bytes > 0)
            iter_b1_mask = row_mask & (stream_bytes > 1)
            iter_m0 = elem_mask & (byte < stream_bytes_2d)
            iter_m1 = elem_mask & ((byte + 1) < stream_bytes_2d)
            iter_m2 = elem_mask & ((byte + 2) < stream_bytes_2d)
        else:
            block_ptr_1d = bitstreams_ptr + blk_idx * bpb64
            iter_b0_mask = u_b0_mask
            iter_b1_mask = u_b1_mask
            iter_m0 = u_m0
            iter_m1 = u_m1
            iter_m2 = u_m2
        states = _load_cyclic_states(
            block_ptr_1d,
            block_ptr_1d[:, None],
            byte,
            shift,
            head_mask,
            has_tail,
            tail_bits_u32,
            tail_shift,
            iter_b0_mask,
            iter_b1_mask,
            iter_m0,
            iter_m1,
            iter_m2,
        )
        scale_codes = tl.load(block_scales_ptr + blk_idx, mask=row_mask, other=0)
        q = _decode_states_to_q_fp32(states, scale_codes)
        x_offs = ms[:, None].to(tl.int64) * stride_xm + (b * block_size + steps)[None, :]
        x_val = tl.load(x_rot_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)
        acc += x_val * q
    return acc


@triton.jit(
    do_not_specialize=[
        "rows",
        "out_features",
        "blocks_per_row",
        "bitstreams_bytes",
        "block_size",
        "block_bits",
        "step_bits",
        "num_wide_steps",
        "bytes_per_block",
        "stride_xm",
        "stride_om",
        "split_k",
    ]
)
def trellis_gemv_splitk_kernel(
    x_rot_ptr,
    bitstreams_ptr,
    block_scales_ptr,
    row_scale_ptr,
    bias_ptr,
    row_widths_ptr,
    row_offsets_ptr,
    output_ptr,
    rows,
    out_features,
    blocks_per_row,
    bitstreams_bytes,
    block_size,
    block_bits,
    step_bits,
    num_wide_steps,
    bytes_per_block,
    stride_xm,
    stride_om,
    split_k,
    tile_steps: tl.constexpr,
    block_m: tl.constexpr,
    is_splitk: tl.constexpr,
    variable_rows: tl.constexpr,
    exact_pow2: tl.constexpr,
    has_bias: tl.constexpr,
):
    """Fused in-register Trellis -> A16 Split-K GEMV over one output row per CTA."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)

    row_index = pid_n + tl.arange(0, 1)
    row_mask = row_index < out_features
    ms = pid_m * block_m + tl.arange(0, block_m)
    m_mask = ms < rows
    steps = tl.arange(0, tile_steps)
    clamped_steps = steps if exact_pow2 else tl.minimum(steps, block_size - 1)
    elem_mask = (
        row_mask[:, None] if exact_pow2 else (row_mask[:, None] & (steps < block_size)[None, :])
    )
    x_mask = m_mask[:, None] if exact_pow2 else (m_mask[:, None] & (steps < block_size)[None, :])
    denom = tile_steps if exact_pow2 else block_size

    if variable_rows:
        width = tl.load(row_widths_ptr + row_index, mask=row_mask, other=0).to(tl.int32)
        row_offset = tl.load(row_offsets_ptr + row_index, mask=row_mask, other=0).to(tl.int64)
        row_base_bytes = row_offset * blocks_per_row
        row_bits = width * 8
        row_step_bits = row_bits // denom
        row_wide_steps = row_bits - row_step_bits * denom
        start_bits = (
            row_step_bits[:, None] * clamped_steps[None, :]
            + row_wide_steps[:, None] * (clamped_steps[None, :] + 1) // denom
        )
        byte, shift, head_mask, has_tail, tail_bits_u32, tail_shift = _bit_window_params(
            start_bits, row_bits[:, None]
        )
        width64 = width.to(tl.int64)
        u_b0_mask = elem_mask
        u_b1_mask = elem_mask
        u_m0 = elem_mask
        u_m1 = elem_mask
        u_m2 = elem_mask
    else:
        offsets = step_bits * clamped_steps + (clamped_steps + 1) * num_wide_steps // denom
        byte, shift, head_mask, has_tail, tail_bits_u32, tail_shift = _bit_window_params(
            offsets[None, :], block_bits
        )
        row_base_bytes = tl.full((1,), 0, tl.int64)
        width64 = tl.full((1,), 0, tl.int64)
        u_b0_mask = row_mask & (bytes_per_block > 0)
        u_b1_mask = row_mask & (bytes_per_block > 1)
        u_m0 = elem_mask & (byte < bytes_per_block)
        u_m1 = elem_mask & ((byte + 1) < bytes_per_block)
        u_m2 = elem_mask & ((byte + 2) < bytes_per_block)

    acc = _gemv_accumulate_k_blocks(
        x_rot_ptr,
        bitstreams_ptr,
        block_scales_ptr,
        row_index,
        row_mask,
        ms,
        steps,
        elem_mask,
        x_mask,
        byte,
        shift,
        head_mask,
        has_tail,
        tail_bits_u32,
        tail_shift,
        pid_k,
        blocks_per_row,
        bitstreams_bytes,
        block_size,
        bytes_per_block,
        stride_xm,
        split_k,
        row_base_bytes,
        width64,
        u_b0_mask,
        u_b1_mask,
        u_m0,
        u_m1,
        u_m2,
        block_m=block_m,
        tile_steps=tile_steps,
        variable_rows=variable_rows,
    )

    sums = tl.sum(acc, axis=1)
    n_valid = m_mask & (pid_n < out_features)
    if is_splitk:
        partial_offsets = (pid_k.to(tl.int64) * rows + ms.to(tl.int64)) * out_features + pid_n
        tl.store(output_ptr + partial_offsets, sums, mask=n_valid)
    else:
        row_scale = tl.load(row_scale_ptr + pid_n, mask=pid_n < out_features, other=0.0).to(
            tl.float32
        )
        result = sums * row_scale
        if has_bias:
            result += tl.load(bias_ptr + pid_n, mask=pid_n < out_features, other=0.0).to(tl.float32)
        tl.store(
            output_ptr + ms.to(tl.int64) * stride_om + pid_n,
            result.to(output_ptr.dtype.element_ty),
            mask=n_valid,
        )


@triton.jit(do_not_specialize=["rows", "out_features", "split_k"])
def trellis_splitk_reduce_kernel(
    partial_ptr,
    row_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features,
    split_k,
    block_n: tl.constexpr,
    has_bias: tl.constexpr,
):
    """Reduce Split-K partial sums in fixed K order and apply row scale + bias epilogue."""
    pid_n = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    acc = tl.zeros((block_n,), dtype=tl.float32)
    split_stride = tl.full((), rows, tl.int64) * out_features
    ptr = partial_ptr + row * out_features + n_offs_i64
    for _ in range(split_k):
        acc += tl.load(ptr, mask=n_valid, other=0.0)
        ptr += split_stride

    acc *= tl.load(row_scale_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
    if has_bias:
        acc += tl.load(bias_ptr + n_offs_i64, mask=n_valid, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * out_features + n_offs_i64,
        acc.to(output_ptr.dtype.element_ty),
        mask=n_valid,
    )


def _select_gemv_config(
    num_sms: int,
    rows: int,
    out_features: int,
    blocks_per_row: int,
    weight_bytes: int,
) -> tuple[int, int, int]:
    """Select ``(block_m, num_warps, split_k)`` for ``trellis_gemv_splitk_kernel``."""
    block_m = 1 if rows == 1 else (2 if rows == 2 else (4 if rows <= 4 else _GEMV_MAX_ROWS))
    num_warps = 1
    if rows > _GEMV_MAX_ROWS:
        # Large-M fallback (e.g. SM75 without W8A8) already fills SMs on M x N.
        return (block_m, num_warps, 1)
    output_tiles = out_features * triton.cdiv(rows, block_m)
    split_k = select_split_k(
        num_sms,
        output_tiles=output_tiles,
        k_tiles=blocks_per_row,
        weight_bytes=weight_bytes,
        min_weight_bytes=_SPLITK_MIN_WEIGHT_BYTES,
        max_split_k=_SPLITK_MAX,
        cta_multiplier=4,
        min_k_tiles=4,
        min_tiles_per_split=2,
    )
    return (block_m, num_warps, split_k)


def _empty_linear_output(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    leading_shape = input.shape[:-1]
    if math.prod(leading_shape) == 0 or out_features == 0:
        return input.new_empty((*leading_shape, out_features))
    if in_features == 0:
        if bias is None:
            return input.new_zeros((*leading_shape, out_features))
        return bias.to(input.dtype).expand(*leading_shape, out_features).clone()
    return None


def linear_a16(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Execute ConvRot Trellis linear via fused in-register A16 Split-K GEMV."""
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    empty = _empty_linear_output(input, out_features, in_features, bias)
    if empty is not None:
        return empty
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)

    input_2d = input.reshape(rows, in_features).contiguous()
    output = torch.empty((rows, out_features), device=input.device, dtype=input.dtype)
    rotated = torch.empty((rows, in_features), device=input.device, dtype=torch.float32)
    rotate_input(input_2d, rotated, group_size, num_warps=4)

    variable_rows = row_widths is not None
    if variable_rows:
        block_bits = step_bits = num_wide_steps = bytes_per_block = 0
        bitstreams_ptr = bitstreams
    else:
        layout = TrellisLayout.from_storage(block_size, bitstreams.shape[2])
        block_bits = layout.block_bits
        step_bits = layout.step_bits
        num_wide_steps = layout.num_wide_steps
        bytes_per_block = layout.bytes_per_block
        bitstreams_ptr = bitstreams.view(-1)

    block_m, num_warps, split_k = _select_gemv_config(
        device_sm_count(input.device),
        rows,
        out_features,
        blocks_per_row,
        bitstreams.numel(),
    )
    is_splitk = split_k > 1
    tile_steps = triton.next_power_of_2(block_size)
    exact_pow2 = block_size == tile_steps
    has_bias = bias is not None
    row_scale = scale.reshape(-1)
    target = (
        torch.empty((split_k, rows, out_features), device=input.device, dtype=torch.float32)
        if is_splitk
        else output
    )
    grid = (out_features, split_k, triton.cdiv(rows, block_m))
    with device_context(input.device):
        row_offsets = compute_row_offsets(row_widths) if row_widths is not None else None
        trellis_gemv_splitk_kernel[grid](
            rotated,
            bitstreams_ptr,
            block_scales.view(torch.uint8),
            row_scale,
            bias,
            row_widths,
            row_offsets,
            target,
            rows,
            out_features,
            blocks_per_row,
            bitstreams.numel(),
            block_size,
            block_bits,
            step_bits,
            num_wide_steps,
            bytes_per_block,
            rotated.stride(0),
            output.stride(0),
            split_k,
            tile_steps=tile_steps,
            block_m=block_m,
            is_splitk=is_splitk,
            variable_rows=variable_rows,
            exact_pow2=exact_pow2,
            has_bias=has_bias,
            num_warps=num_warps,
        )
        if is_splitk:
            reduce_grid = (triton.cdiv(out_features, _REDUCE_BLOCK_N), rows)
            trellis_splitk_reduce_kernel[reduce_grid](
                target,
                row_scale,
                bias,
                output,
                rows,
                out_features,
                split_k,
                block_n=_REDUCE_BLOCK_N,
                has_bias=has_bias,
                num_warps=4,
            )
    return output.reshape(*leading_shape, out_features)


def _linear_w8a8(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
    *,
    int8_backend: convrot_int8_backend.LinearBackend,
) -> torch.Tensor:
    """Execute prefill ConvRot Trellis linear via bounded N-chunked scratch decode + W8A8."""
    out_features, blocks_per_row = block_scales.shape
    in_features = blocks_per_row * block_size
    empty = _empty_linear_output(input, out_features, in_features, bias)
    if empty is not None:
        return empty
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)

    input_2d = input.reshape(rows, in_features).contiguous()
    output = torch.empty((rows, out_features), device=input.device, dtype=input.dtype)
    input_qdata, input_scale = int8_backend.prepare_input(input_2d, group_size)

    raw_chunk_rows = max(1, _SCRATCH_DECODE_MAX_BYTES // in_features)
    if raw_chunk_rows < out_features and raw_chunk_rows >= 128:
        chunk_rows = (raw_chunk_rows // 128) * 128
    else:
        chunk_rows = min(out_features, raw_chunk_rows)
    q_scratch = torch.empty((chunk_rows, in_features), device=input.device, dtype=torch.int8)
    block_scales_u8 = block_scales.view(torch.uint8)

    if row_widths is None:
        layout = TrellisLayout.from_storage(block_size, bitstreams.shape[2])
        row_offsets = None
    else:
        layout = None
        with device_context(input.device):
            row_offsets = compute_row_offsets(row_widths)

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        q_view = q_scratch[: end - start]
        if row_widths is None:
            assert layout is not None
            decode_int8_out(
                bitstreams[start:end].view(-1, layout.bytes_per_block),
                block_scales_u8[start:end].view(-1),
                layout,
                q_view,
            )
        else:
            assert row_offsets is not None
            decode_variable_rows_int8_out(
                bitstreams,
                block_scales_u8[start:end],
                row_widths[start:end],
                block_size,
                q_view,
                row_offsets=row_offsets[start:end],
            )
        chunk_bias = None if bias is None else bias[start:end]
        int8_backend.linear_prepared(
            input_qdata,
            input_scale,
            q_view,
            scale[start:end],
            chunk_bias,
            input.dtype,
            out=output[:, start:end],
        )
    return output.reshape(*leading_shape, out_features)


def linear(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    scale: torch.Tensor,
    row_widths: torch.Tensor | None,
    bias: torch.Tensor | None,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch ConvRot Trellis linear between fused A16 GEMV and W8A8 prefill."""
    rows = math.prod(input.shape[:-1])
    int8_backend = (
        convrot_int8_backend.select_linear_backend(input) if rows > _GEMV_MAX_ROWS else None
    )
    if int8_backend is not None:
        return _linear_w8a8(
            input,
            bitstreams,
            block_scales,
            scale,
            row_widths,
            bias,
            block_size,
            group_size,
            int8_backend=int8_backend,
        )
    return linear_a16(
        input,
        bitstreams,
        block_scales,
        scale,
        row_widths,
        bias,
        block_size,
        group_size,
    )


def compile_trellis_kernels(
    target: GPUTarget,
    *,
    rows: int,
    out_features: int,
    input_features: int,
    block_size: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype | None = None,
    variable_rows: bool = False,
    bytes_per_block: int | None = None,
    num_sms: int = 64,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile the production Trellis A16 GEMV (and Split-K reduce) kernels for a target."""
    if rows <= 0 or out_features <= 0 or input_features <= 0 or num_sms <= 0:
        raise ValueError("Workload dimensions and SM count must be positive integers")
    validate_block_size(block_size)
    if input_features % block_size:
        raise ValueError(
            f"input_features {input_features} must be divisible by block_size {block_size}"
        )
    if input_dtype not in _TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported Trellis input dtype: {input_dtype}")
    if bias_dtype is not None and bias_dtype not in _TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported Trellis bias dtype: {bias_dtype}")

    out_ty = _TRITON_FLOAT_POINTER_TYPES[input_dtype]
    has_bias = bias_dtype is not None
    bias_signature = (
        {} if bias_dtype is None else {"bias_ptr": _TRITON_FLOAT_POINTER_TYPES[bias_dtype]}
    )
    bias_constexpr: dict[str, object] = {"bias_ptr": None} if bias_dtype is None else {}
    blocks_per_row = input_features // block_size
    effective_bytes = (
        (9 * block_size) // 16
        if bytes_per_block is None
        else TrellisLayout.from_storage(block_size, bytes_per_block).bytes_per_block
    )
    weight_bytes = out_features * blocks_per_row * effective_bytes

    block_m, num_warps, split_k = _select_gemv_config(
        num_sms,
        rows,
        out_features,
        blocks_per_row,
        weight_bytes,
    )
    is_splitk = split_k > 1
    tile_steps = triton.next_power_of_2(block_size)
    exact_pow2 = block_size == tile_steps

    gemv_signature: dict[str, str] = {
        "x_rot_ptr": "*fp32",
        "bitstreams_ptr": "*u8",
        "block_scales_ptr": "*u8",
        "row_scale_ptr": "*fp32",
        **bias_signature,
        "output_ptr": "*fp32" if is_splitk else out_ty,
        "rows": "i32",
        "out_features": "i32",
        "blocks_per_row": "i32",
        "bitstreams_bytes": "i64",
        "block_size": "i32",
        "block_bits": "i32",
        "step_bits": "i32",
        "num_wide_steps": "i32",
        "bytes_per_block": "i32",
        "stride_xm": "i32",
        "stride_om": "i32",
        "split_k": "i32",
    }
    gemv_constexprs: dict[str, object] = {
        **bias_constexpr,
        "tile_steps": tile_steps,
        "block_m": block_m,
        "is_splitk": is_splitk,
        "variable_rows": variable_rows,
        "exact_pow2": exact_pow2,
        "has_bias": has_bias,
    }
    if variable_rows:
        gemv_signature["row_widths_ptr"] = "*u8"
        gemv_signature["row_offsets_ptr"] = "*i64"
    else:
        gemv_constexprs["row_widths_ptr"] = None
        gemv_constexprs["row_offsets_ptr"] = None

    compiled: dict[str, triton.compiler.CompiledKernel] = {
        "gemv": triton.compile(
            ASTSource(trellis_gemv_splitk_kernel, gemv_signature, constexprs=gemv_constexprs),
            target=target,
            options={"num_warps": num_warps},
        )
    }
    if is_splitk:
        compiled["reduce"] = triton.compile(
            ASTSource(
                trellis_splitk_reduce_kernel,
                {
                    "partial_ptr": "*fp32",
                    "row_scale_ptr": "*fp32",
                    **bias_signature,
                    "output_ptr": out_ty,
                    "rows": "i32",
                    "out_features": "i32",
                    "split_k": "i32",
                },
                constexprs={
                    **bias_constexpr,
                    "block_n": _REDUCE_BLOCK_N,
                    "has_bias": has_bias,
                },
            ),
            target=target,
            options={"num_warps": 4},
        )
    return compiled


__all__ = [
    "compile_trellis_kernels",
    "linear",
    "linear_a16",
    "trellis_gemv_splitk_kernel",
    "trellis_splitk_reduce_kernel",
]
