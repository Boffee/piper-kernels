"""NVFP4 W4A16 MoE Triton kernels and host dispatcher."""

# Triton's JIT launcher, constexpr arguments, and wide kernel signatures are not
# fully represented by Python typing rules.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton.primitives import splitk_tile_bounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._splitk import device_sm_count
from piper_kernels.linear.nvfp4._w4a16_triton import (
    _accumulate_scaled_row,
    _accumulate_scaled_row_blocks,
    _load_nvfp4_tile,
    _load_scaled_nvfp4_pairs,
    _split_global_scale,
)
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

from . import _routing_triton, _schedule
from ._routing_triton import _resolve_slot, _swizzle_grouped_pid, moe_m1_activate_rotate_kernel

_NVFP4_BLOCK_SIZE_TL = tl.constexpr(nvfp4_layout.BLOCK_SIZE)
_NVFP4_QDATA_BLOCK_SIZE_TL = tl.constexpr(nvfp4_layout.QDATA_BLOCK_SIZE)


@triton.jit(do_not_specialize=["split_k"])
def nvfp4_moe_m1_gemv_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    topk_ids_ptr,
    expert_to_slot_ptr,
    output_ptr,
    out_features,
    packed_k,
    scale_column_blocks,
    expert_scale_stride,
    num_experts,
    num_slots,
    top_k,
    split_k,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    has_slot_map: tl.constexpr,
    has_global_scale: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr = False,
):
    """Single-token slot-indirect NVFP4 W4A16 GEMV across ``(N * top_k, split_k)``."""
    n_tiles = tl.cdiv(out_features, block_n)
    pid_0 = tl.program_id(0)
    pid_n = pid_0 % n_tiles
    k_idx = pid_0 // n_tiles
    pid_k = tl.program_id(1)
    slot = _resolve_slot(
        topk_ids_ptr,
        expert_to_slot_ptr,
        k_idx,
        num_experts,
        num_slots,
        has_slot_map,
    )
    if slot < 0:
        return

    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    input_features = packed_k * 2
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * packed_k
    slot_scale_ptr = weight_scale_ptr + slot * expert_scale_stride
    slot_input_ptr = input_ptr + (k_idx.to(tl.int64) * input_features if per_expert_input else 0)
    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        pre_scale, post_scale = _split_global_scale(global_scale)
    else:
        pre_scale = 1.0
        post_scale = 1.0

    k_tile_start, k_tile_end = splitk_tile_bounds(tl.cdiv(packed_k, block_k // 2), pid_k, split_k)

    block_scale_k: tl.constexpr = block_k // _NVFP4_BLOCK_SIZE_TL
    acc = tl.zeros((block_n, block_scale_k), dtype=tl.float32)
    for k_tile in range(k_tile_start, k_tile_end):
        k_base = k_tile * block_k
        w_even, w_odd, scales = _load_nvfp4_tile(
            slot_weight_ptr,
            slot_scale_ptr,
            n_offs_i64,
            n_valid,
            k_base,
            packed_k,
            scale_k,
            scale_column_blocks,
            block_k,
            high_first,
            use_sm120_e2m1x2,
        )
        k_elem_offs = k_base + tl.arange(0, block_k)
        k_elem_valid = k_elem_offs < input_features
        x_vec = tl.load(slot_input_ptr + k_elem_offs, mask=k_elem_valid, other=0.0).to(tl.float32)
        if has_global_scale:
            x_vec = x_vec * pre_scale
        acc += _accumulate_scaled_row_blocks(x_vec, w_even, w_odd, scales, block_n, block_k)

    acc = tl.sum(acc, axis=1)
    if has_global_scale and not is_splitk:
        acc *= post_scale
    row_idx = pid_k.to(tl.int64) * top_k + k_idx if is_splitk else k_idx.to(tl.int64)
    tl.store(output_ptr + row_idx * out_features + n_offs_i64, acc, mask=n_valid)


@triton.jit
def _accumulate_nvfp4_batched_nodot(
    acc,
    input_ptr,
    tile_pair_base,
    k_elem_offs,
    k_elem_valid,
    k_stride,
    pre_scale,
    w_even,
    w_odd,
    scales,
    m_offs,
    tile_len,
    top_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    pre_scale_in_gemv: tl.constexpr,
    per_expert_input: tl.constexpr,
):
    # Triton's typing does not declare static_range iterable.
    for row_idx in tl.static_range(block_m):  # pyright: ignore[reportGeneralTypeIssues]
        if row_idx < tile_len:
            pair_id = tl.load(tile_pair_base + row_idx)
            src_row = (pair_id if per_expert_input else pair_id // top_k).to(tl.int64)
            x_vec = tl.load(
                input_ptr + src_row * k_stride + k_elem_offs,
                mask=k_elem_valid,
                other=0.0,
            ).to(tl.float32)
            if pre_scale_in_gemv:
                x_vec = x_vec * pre_scale
            row_acc = _accumulate_scaled_row(x_vec, w_even, w_odd, scales, block_n, block_k)
            acc += tl.where((m_offs == row_idx)[:, None], row_acc[None, :], 0.0)
    return acc


@triton.jit
def _accumulate_nvfp4_batched_dot(
    acc,
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    n_offs_i64,
    n_valid,
    k_base,
    packed_k,
    scale_k,
    scale_column_blocks,
    src_row_offsets,
    init_valid,
    pre_scale,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    pre_scale_in_gemv: tl.constexpr,
    cast_fp32_to_bf16: tl.constexpr,
    force_fp32_dot: tl.constexpr = False,
    half_scales: tl.constexpr = False,
    use_sm120_e2m1x2: tl.constexpr = False,
    single_dot: tl.constexpr = False,
):
    dot_dtype = (
        tl.float32
        if force_fp32_dot
        else (tl.bfloat16 if cast_fp32_to_bf16 else input_ptr.dtype.element_ty)
    )
    # Every finite (E2M1, E4M3FN) product fits in FP16, halving live weight registers on SM120.
    pair_dtype = tl.float16 if (use_sm120_e2m1x2 and force_fp32_dot) else dot_dtype
    w_e, w_o = _load_scaled_nvfp4_pairs(
        weight_ptr,
        weight_scale_ptr,
        n_offs_i64,
        n_valid,
        k_base,
        packed_k,
        scale_k,
        scale_column_blocks,
        block_n,
        block_k,
        high_first,
        tl.constexpr(pair_dtype),
        use_sm120_e2m1x2,
        half_scales and not force_fp32_dot,
    )
    input_features = packed_k * 2
    k_elem_offs = k_base + tl.arange(0, block_k)
    k_elem_valid = k_elem_offs < input_features
    x_tile = tl.load(
        input_ptr + src_row_offsets[:, None] + k_elem_offs[None, :],
        mask=init_valid[:, None] & k_elem_valid[None, :],
        other=0.0,
    )
    if pre_scale_in_gemv:
        x_tile = (x_tile.to(tl.float32) * pre_scale).to(dot_dtype)
    elif cast_fp32_to_bf16 or force_fp32_dot:
        x_tile = x_tile.to(dot_dtype)
    if single_dot:
        w_full = tl.reshape(tl.join(w_e, w_o), (block_n, block_k))
        if force_fp32_dot:
            return tl.dot(x_tile, tl.trans(w_full.to(tl.float32)), acc, input_precision="ieee")
        else:
            return tl.dot(x_tile, tl.trans(w_full), acc)
    else:
        x_even, x_odd = tl.split(tl.reshape(x_tile, (block_m, block_k // 2, 2)))
        if force_fp32_dot:
            acc = tl.dot(x_even, tl.trans(w_e.to(tl.float32)), acc, input_precision="ieee")
            return tl.dot(x_odd, tl.trans(w_o.to(tl.float32)), acc, input_precision="ieee")
        else:
            acc = tl.dot(x_even, tl.trans(w_e), acc)
            return tl.dot(x_odd, tl.trans(w_o), acc)


@triton.jit(do_not_specialize=["split_k"])
def nvfp4_moe_batched_gemv_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    tile_slots_ptr,
    tile_pair_ids_ptr,
    output_ptr,
    out_features,
    packed_k,
    scale_column_blocks,
    expert_scale_stride,
    num_pairs,
    top_k,
    split_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    high_first: tl.constexpr,
    has_global_scale: tl.constexpr,
    per_expert_input: tl.constexpr,
    is_splitk: tl.constexpr,
    use_dot: tl.constexpr,
    cast_fp32_to_bf16: tl.constexpr = False,
    use_sm120_e2m1x2: tl.constexpr = False,
    single_dot: tl.constexpr = False,
):
    """Slot-batched NVFP4 W4A16 projection across ``(N, bucket_tile, split_k)``."""
    pid_n = tl.program_id(0)
    tile_idx = tl.program_id(1)
    pid_k = tl.program_id(2)
    slot = tl.load(tile_slots_ptr + tile_idx).to(tl.int64)
    if slot < 0:
        return

    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features
    m_offs = tl.arange(0, block_m)
    tile_pair_base = tile_pair_ids_ptr + tile_idx.to(tl.int64) * block_m
    init_pair_ids = tl.load(tile_pair_base + m_offs)
    init_valid = init_pair_ids >= 0
    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    input_features = packed_k * 2
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    k_stride = tl.full((), input_features, tl.int64)
    if use_dot:
        src_pairs = tl.where(
            init_valid, init_pair_ids if per_expert_input else init_pair_ids // top_k, 0
        ).to(tl.int64)
        src_row_offsets = src_pairs * k_stride
    else:
        tile_len = tl.sum(init_valid.to(tl.int32))

    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * packed_k
    slot_scale_ptr = weight_scale_ptr + slot * expert_scale_stride

    # Keep FP16 tl.dot unscaled (cannot overflow FP32, would underflow FP16);
    # BF16 pre-scales against FP32 overflow.
    pre_scale_in_gemv: tl.constexpr = tl.constexpr(
        has_global_scale and not (use_dot and input_ptr.dtype.element_ty == tl.float16)
    )
    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        pre_scale, post_scale = _split_global_scale(global_scale)
        epilogue_scale = post_scale if pre_scale_in_gemv else global_scale
    else:
        pre_scale = 1.0
        epilogue_scale = 1.0

    k_tile_start, k_tile_end = splitk_tile_bounds(tl.cdiv(packed_k, block_k // 2), pid_k, split_k)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_tile in range(k_tile_start, k_tile_end):
        k_base = k_tile * block_k
        if use_dot:
            acc = _accumulate_nvfp4_batched_dot(
                acc,
                input_ptr,
                slot_weight_ptr,
                slot_scale_ptr,
                n_offs_i64,
                n_valid,
                k_base,
                packed_k,
                scale_k,
                scale_column_blocks,
                src_row_offsets,  # pyright: ignore[reportPossiblyUnboundVariable]
                init_valid,
                pre_scale,
                block_m,
                block_n,
                block_k,
                high_first,
                pre_scale_in_gemv,
                cast_fp32_to_bf16,
                use_sm120_e2m1x2=use_sm120_e2m1x2,
                single_dot=single_dot,
            )
        else:
            w_even, w_odd, scales = _load_nvfp4_tile(
                slot_weight_ptr,
                slot_scale_ptr,
                n_offs_i64,
                n_valid,
                k_base,
                packed_k,
                scale_k,
                scale_column_blocks,
                block_k,
                high_first,
                use_sm120_e2m1x2,
            )
            k_elem_offs = k_base + tl.arange(0, block_k)
            k_elem_valid = k_elem_offs < input_features
            acc = _accumulate_nvfp4_batched_nodot(
                acc,
                input_ptr,
                tile_pair_base,
                k_elem_offs,
                k_elem_valid,
                k_stride,
                pre_scale,
                w_even,
                w_odd,
                scales,
                m_offs,
                tile_len,  # pyright: ignore[reportPossiblyUnboundVariable]
                top_k,
                block_m,
                block_n,
                block_k,
                pre_scale_in_gemv,
                per_expert_input,
            )

    if has_global_scale and not is_splitk:
        acc *= epilogue_scale
    pair_ids = tl.load(tile_pair_base + m_offs).to(tl.int64)
    row_valid = pair_ids >= 0
    dst_rows = tl.where(
        row_valid, pid_k.to(tl.int64) * num_pairs + pair_ids if is_splitk else pair_ids, 0
    )
    tl.store(
        output_ptr + dst_rows[:, None] * out_features + n_offs_i64[None, :],
        acc,
        mask=row_valid[:, None] & n_valid[None, :],
    )


@triton.jit
def nvfp4_moe_grouped_gemm_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    tile_slots_ptr,
    tile_m_start_ptr,
    tile_m_len_ptr,
    output_ptr,
    max_tiles,
    out_features,
    packed_k,
    scale_column_blocks,
    expert_scale_stride,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    high_first: tl.constexpr,
    has_global_scale: tl.constexpr,
    cast_fp32_to_bf16: tl.constexpr,
    force_fp32_dot: tl.constexpr,
    half_scales: tl.constexpr,
    use_sm120_e2m1x2: tl.constexpr = False,
    single_dot: tl.constexpr = False,
):
    """Grouped NVFP4 W4A16 GEMM over contiguous slot-compacted tiles."""
    num_pid_n = tl.cdiv(out_features, block_n)
    tile_idx, pid_n = _swizzle_grouped_pid(tl.program_id(0), max_tiles, num_pid_n, group_m)
    slot = tl.load(tile_slots_ptr + tile_idx).to(tl.int64)
    if slot < 0:
        return
    m_start = tl.load(tile_m_start_ptr + tile_idx).to(tl.int64)
    tile_len = tl.load(tile_m_len_ptr + tile_idx)

    m_offs = tl.arange(0, block_m)
    m_offs_i64 = m_offs.to(tl.int64)
    m_valid = m_offs < tile_len
    n_offs = pid_n * block_n + tl.arange(0, block_n)
    n_offs_i64 = n_offs.to(tl.int64)
    n_valid = n_offs < out_features

    # Pass packed_k so Triton keeps 16-byte divisibility for vectorized weight loads.
    in_features = packed_k * 2
    k_stride = in_features.to(tl.int64)
    scale_k = packed_k // _NVFP4_QDATA_BLOCK_SIZE_TL
    slot_input_ptr = input_ptr + m_start * k_stride
    slot_weight_ptr = weight_ptr + slot * out_features.to(tl.int64) * packed_k
    slot_scale_ptr = weight_scale_ptr + slot * expert_scale_stride
    src_row_offsets = m_offs_i64 * k_stride

    # Keep native FP16 tl.dot unscaled to avoid underflow; pre-scale BF16/FP32 against overflow.
    is_fp16_dot: tl.constexpr = tl.constexpr(
        (not force_fp32_dot)
        and (not cast_fp32_to_bf16)
        and (input_ptr.dtype.element_ty == tl.float16)
    )
    pre_scale_in_gemm: tl.constexpr = tl.constexpr(has_global_scale and not is_fp16_dot)
    if has_global_scale:
        global_scale = tl.load(weight_global_scale_ptr + slot).to(tl.float32)
        pre_scale, post_scale = _split_global_scale(global_scale)
        epilogue_scale = post_scale if pre_scale_in_gemm else global_scale
    else:
        pre_scale = 1.0
        epilogue_scale = 1.0

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    num_k_tiles = tl.cdiv(packed_k, block_k // 2)
    for k_tile in range(num_k_tiles):
        k_base = k_tile * block_k
        acc = _accumulate_nvfp4_batched_dot(
            acc,
            slot_input_ptr,
            slot_weight_ptr,
            slot_scale_ptr,
            n_offs_i64,
            n_valid,
            k_base,
            packed_k,
            scale_k,
            scale_column_blocks,
            src_row_offsets,
            m_valid,
            pre_scale,
            block_m,
            block_n,
            block_k,
            high_first,
            pre_scale_in_gemm,
            cast_fp32_to_bf16,
            force_fp32_dot,
            half_scales,
            use_sm120_e2m1x2,
            single_dot,
        )

    if has_global_scale:
        acc *= epilogue_scale
    tl.store(
        output_ptr + (m_start + m_offs_i64)[:, None] * out_features + n_offs_i64[None, :],
        acc,
        mask=m_valid[:, None] & n_valid[None, :],
    )


def launch_nvfp4_proj(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None,
    routing: _routing_triton.A16ChunkRouting,
    output: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    scale_cols: int,
    scale_stride: int,
    num_slots: int,
    split_k: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    high_first: bool,
    has_global_scale: bool,
    per_expert_input: bool,
    is_splitk: bool,
    use_dot: bool,
    cast_fp32_to_bf16: bool,
    use_sm120_e2m1x2: bool = False,
    single_dot: bool = False,
) -> None:
    scale_u8 = scale.view(torch.uint8)
    packed_k = in_features // 2
    n_tiles = triton.cdiv(out_features, block_n)
    with device_context(x.device):
        if routing.tile_slots is not None:
            assert routing.tile_pair_ids is not None
            nvfp4_moe_batched_gemv_kernel[(n_tiles, routing.tile_slots.shape[0], split_k)](
                x,
                qdata,
                scale_u8,
                global_scale,
                routing.tile_slots,
                routing.tile_pair_ids,
                output,
                out_features,
                packed_k,
                scale_cols,
                scale_stride,
                routing.num_pairs,
                routing.top_k,
                split_k,
                block_m=_schedule.BATCH_BLOCK_M,
                block_n=block_n,
                block_k=block_k,
                high_first=high_first,
                has_global_scale=has_global_scale,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                use_dot=use_dot,
                cast_fp32_to_bf16=cast_fp32_to_bf16,
                use_sm120_e2m1x2=use_sm120_e2m1x2,
                single_dot=single_dot,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            nvfp4_moe_m1_gemv_kernel[(n_tiles * routing.top_k, split_k)](
                x,
                qdata,
                scale_u8,
                global_scale,
                routing.topk_ids_flat,
                routing.expert_to_slot,
                output,
                out_features,
                packed_k,
                scale_cols,
                scale_stride,
                routing.num_experts,
                num_slots,
                routing.top_k,
                split_k,
                block_n=block_n,
                block_k=block_k,
                high_first=high_first,
                has_slot_map=routing.has_slot_map,
                has_global_scale=has_global_scale,
                per_expert_input=per_expert_input,
                is_splitk=is_splitk,
                use_sm120_e2m1x2=use_sm120_e2m1x2,
                num_warps=num_warps,
            )


def launch_nvfp4_grouped_gemm(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None,
    tile_slots: torch.Tensor,
    tile_m_start: torch.Tensor,
    tile_m_len: torch.Tensor,
    output: torch.Tensor,
    *,
    max_tiles: int,
    out_features: int,
    in_features: int,
    scale_cols: int,
    scale_stride: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    high_first: bool,
    has_global_scale: bool,
    cast_fp32_to_bf16: bool,
    force_fp32_dot: bool,
    half_scales: bool,
    use_sm120_e2m1x2: bool = False,
    single_dot: bool = False,
) -> None:
    grid = (max_tiles * triton.cdiv(out_features, block_n),)
    with device_context(x.device):
        nvfp4_moe_grouped_gemm_kernel[grid](
            x,
            qdata,
            scale.view(torch.uint8),
            global_scale,
            tile_slots,
            tile_m_start,
            tile_m_len,
            output,
            max_tiles,
            out_features,
            in_features // 2,
            scale_cols,
            scale_stride,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=_schedule.PREFILL_GROUP_M,
            high_first=high_first,
            has_global_scale=has_global_scale,
            cast_fp32_to_bf16=cast_fp32_to_bf16,
            force_fp32_dot=force_fp32_dot,
            half_scales=half_scales,
            use_sm120_e2m1x2=use_sm120_e2m1x2,
            single_dot=single_dot,
            num_warps=num_warps,
            num_stages=1,
        )


def run_nvfp4_prefill_chunk(
    active_x_chunk: torch.Tensor,
    topk_ids_chunk: torch.Tensor,
    topk_weights_chunk: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    out_chunk: torch.Tensor,
    *,
    target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
    up_scale_cols: int,
    up_scale_stride: int,
    down_scale_cols: int,
    down_scale_stride: int,
) -> None:
    device = active_x_chunk.device
    chunk_rows = active_x_chunk.shape[0]
    top_k = topk_ids_chunk.shape[1]
    num_pairs = chunk_rows * top_k
    num_slots, hidden_size, packed_inter = down_qdata.shape
    intermediate_size = 2 * packed_inter
    up_gate_out = up_gate_qdata.shape[1]
    prefill_schedule = _routing_triton.select_nvfp4_prefill_schedule(
        target,
        chunk_rows=chunk_rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        active_dtype=active_x_chunk.dtype,
        logical_dtype=out_chunk.dtype,
    )
    cast_bf16 = prefill_schedule.cast_fp32_to_bf16
    force_fp32_dot = prefill_schedule.force_fp32_dot
    half_scales = prefill_schedule.half_scales
    block_m = prefill_schedule.block_m
    block_n = prefill_schedule.block_n
    block_k = prefill_schedule.block_k
    num_warps = prefill_schedule.num_warps
    single_dot = prefill_schedule.single_dot
    use_sm120_e2m1x2 = prefill_schedule.use_sm120_e2m1x2
    tile_slots, tile_m_start, tile_m_len, compact_to_pair, pair_to_compact, max_tiles, _ = (
        _routing_triton.launch_moe_prefill_bucket(
            topk_ids_chunk.reshape(-1), expert_to_slot, num_slots, block_m
        )
    )
    x_compact = torch.empty((num_pairs, hidden_size), device=device, dtype=active_x_chunk.dtype)
    mid = torch.empty((num_pairs, up_gate_out), device=device, dtype=torch.float32)

    _routing_triton.launch_moe_prefill_gather(
        active_x_chunk,
        None,
        compact_to_pair,
        x_compact,
        None,
        hidden_size,
        top_k,
        has_scale=False,
    )
    launch_nvfp4_grouped_gemm(
        x_compact,
        up_gate_qdata,
        up_gate_scale,
        up_gate_global_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        mid,
        max_tiles=max_tiles,
        out_features=up_gate_out,
        in_features=hidden_size,
        scale_cols=up_scale_cols,
        scale_stride=up_scale_stride,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        high_first=high_first,
        has_global_scale=up_gate_global_scale is not None,
        cast_fp32_to_bf16=cast_bf16,
        force_fp32_dot=force_fp32_dot,
        half_scales=half_scales,
        use_sm120_e2m1x2=use_sm120_e2m1x2,
        single_dot=single_dot,
    )
    del x_compact
    act_dtype = torch.float32 if group_size > 0 else out_chunk.dtype
    act = torch.empty((num_pairs, intermediate_size), device=device, dtype=act_dtype)
    with device_context(device):
        moe_m1_activate_rotate_kernel[
            (triton.cdiv(intermediate_size, _schedule.ACTIVATE_BLOCK_SIZE) * num_pairs,)
        ](
            mid,
            act,
            out_chunk,
            compact_to_pair,
            None,
            intermediate_size,
            num_pairs,
            num_pairs,
            block_size=_schedule.ACTIVATE_BLOCK_SIZE,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            accelerator_backend=target.backend,
            has_slot_map=False,
            num_warps=4,
        )
    del mid
    down_out = torch.empty((num_pairs, hidden_size), device=device, dtype=torch.float32)
    launch_nvfp4_grouped_gemm(
        act,
        down_qdata,
        down_scale,
        down_global_scale,
        tile_slots,
        tile_m_start,
        tile_m_len,
        down_out,
        max_tiles=max_tiles,
        out_features=hidden_size,
        in_features=intermediate_size,
        scale_cols=down_scale_cols,
        scale_stride=down_scale_stride,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        high_first=high_first,
        has_global_scale=down_global_scale is not None,
        cast_fp32_to_bf16=cast_bf16,
        force_fp32_dot=force_fp32_dot,
        half_scales=half_scales,
        use_sm120_e2m1x2=use_sm120_e2m1x2,
        single_dot=single_dot,
    )
    del act
    _routing_triton.launch_moe_prefill_weighted_reduce(
        down_out,
        pair_to_compact,
        topk_weights_chunk.reshape(-1),
        out_chunk,
        hidden_size,
        top_k,
        accumulate=accumulate,
    )


def dispatch_nvfp4_moe(
    input_2d: torch.Tensor,
    ids_2d: torch.Tensor,
    weights_2d: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None,
    down_global_scale: torch.Tensor | None,
    expert_to_slot: torch.Tensor | None,
    prepared_data: torch.Tensor | None,
    out_2d: torch.Tensor,
    *,
    resolved_target: AcceleratorTarget,
    accumulate: bool,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    active_slots_hint: int,
) -> None:
    device = input_2d.device
    rows, hidden_size = input_2d.shape
    num_slots, _, packed_inter = down_qdata.shape
    intermediate_size = 2 * packed_inter
    up_gate_out = up_gate_qdata.shape[1]
    top_k = ids_2d.shape[1]
    up_scale_cols = int(triton.cdiv(hidden_size, nvfp4_layout.SCALE_COLUMN_TILE))
    up_scale_stride = up_gate_scale.shape[1] * up_gate_scale.shape[2]
    down_scale_cols = int(triton.cdiv(intermediate_size, nvfp4_layout.SCALE_COLUMN_TILE))
    down_scale_stride = down_scale.shape[1] * down_scale.shape[2]

    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=False,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=input_2d.element_size(),
        group_size=group_size,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill":
        for active_chunk, ids_chunk, weights_chunk, out_chunk in _routing_triton.iter_a16_chunks(
            input_2d, ids_2d, weights_2d, out_2d, prepared_data, group_size, chunk_rows
        ):
            run_nvfp4_prefill_chunk(
                active_chunk,
                ids_chunk,
                weights_chunk,
                up_gate_qdata,
                up_gate_scale,
                down_qdata,
                down_scale,
                up_gate_global_scale,
                down_global_scale,
                expert_to_slot,
                out_chunk,
                target=resolved_target,
                accumulate=accumulate,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                high_first=high_first,
                active_slots_hint=active_slots_hint,
                up_scale_cols=up_scale_cols,
                up_scale_stride=up_scale_stride,
                down_scale_cols=down_scale_cols,
                down_scale_stride=down_scale_stride,
            )
        return

    batched = regime == "batched"
    split_k_up = split_k_down = 1
    m1_schedule = _routing_triton.select_nvfp4_m1_schedule(resolved_target)
    active_schedule: _schedule.Nvfp4M1Schedule | _schedule.Nvfp4BatchedSchedule = m1_schedule
    post_scale_only = True
    num_sms = device_sm_count(device)
    if not batched:
        active_slots = _schedule.expected_bucket_tiles(
            top_k, num_slots, active_slots_hint, batched=False
        )
        split_k_up = _schedule.select_nvfp4_split_k(
            num_sms,
            active_slots,
            up_gate_out,
            hidden_size,
            block_n=m1_schedule.block_n,
            block_k=m1_schedule.block_k,
            cta_multiplier=m1_schedule.splitk_cta_multiplier,
        )
        split_k_down = _schedule.select_nvfp4_split_k(
            num_sms,
            active_slots,
            hidden_size,
            intermediate_size,
            block_n=m1_schedule.block_n,
            block_k=m1_schedule.block_k,
            cta_multiplier=m1_schedule.splitk_cta_multiplier,
        )

    for active_chunk, ids_chunk, weights_chunk, out_chunk in _routing_triton.iter_a16_chunks(
        input_2d, ids_2d, weights_2d, out_2d, prepared_data, group_size, chunk_rows
    ):
        if batched:
            chunk_n = active_chunk.shape[0]
            batched_schedule = _routing_triton.select_nvfp4_batched_schedule(
                resolved_target,
                rows=chunk_n,
                active_dtype=active_chunk.dtype,
                logical_dtype=out_chunk.dtype,
            )
            active_schedule = batched_schedule
            post_scale_only = batched_schedule.post_scale_only
            tiles = _schedule.expected_bucket_tiles(
                chunk_n * top_k, num_slots, active_slots_hint, batched=True
            )
            split_k_up = _schedule.select_nvfp4_split_k(
                num_sms,
                tiles,
                up_gate_out,
                hidden_size,
                block_n=batched_schedule.block_n,
                block_k=batched_schedule.block_k,
            )
            split_k_down = _schedule.select_nvfp4_split_k(
                num_sms,
                tiles,
                hidden_size,
                intermediate_size,
                block_n=batched_schedule.block_n,
                block_k=batched_schedule.block_k,
            )

        def _launch_nvfp4_step(
            per_expert_input: bool,
            x: torch.Tensor,
            routing: _routing_triton.A16ChunkRouting,
            output: torch.Tensor,
            split_k: int,
            *,
            sched: _schedule.Nvfp4M1Schedule | _schedule.Nvfp4BatchedSchedule = active_schedule,
        ) -> None:
            if not per_expert_input:
                q, s, gs = up_gate_qdata, up_gate_scale, up_gate_global_scale
                n_out, n_in, sc, ss = up_gate_out, hidden_size, up_scale_cols, up_scale_stride
            else:
                q, s, gs = down_qdata, down_scale, down_global_scale
                n_out, n_in, sc, ss = (
                    hidden_size,
                    intermediate_size,
                    down_scale_cols,
                    down_scale_stride,
                )
            is_batched_sched = isinstance(sched, _schedule.Nvfp4BatchedSchedule)
            launch_nvfp4_proj(
                x,
                q,
                s,
                gs,
                routing,
                output,
                out_features=n_out,
                in_features=n_in,
                scale_cols=sc,
                scale_stride=ss,
                num_slots=num_slots,
                split_k=split_k,
                block_n=sched.block_n,
                block_k=sched.block_k,
                num_warps=sched.num_warps,
                high_first=high_first,
                has_global_scale=gs is not None,
                per_expert_input=per_expert_input,
                is_splitk=per_expert_input or split_k > 1,
                use_dot=sched.use_dot if is_batched_sched else False,
                cast_fp32_to_bf16=sched.cast_fp32_to_bf16 if is_batched_sched else False,
                use_sm120_e2m1x2=sched.use_sm120_e2m1x2,
                single_dot=sched.single_dot if is_batched_sched else False,
            )

        _routing_triton.run_a16_chunk(
            active_chunk,
            ids_chunk,
            weights_chunk,
            out_chunk,
            num_slots=num_slots,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            up_gate_out=up_gate_out,
            expert_to_slot=expert_to_slot,
            target=resolved_target,
            batched=batched,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            split_k_up=split_k_up,
            split_k_down=split_k_down,
            up_row_scale=None,
            down_row_scale=None,
            up_global_scale=up_gate_global_scale,
            down_global_scale=down_global_scale,
            post_scale_only=post_scale_only,
            launch_proj=_launch_nvfp4_step,
        )
