"""Fused multi-wave attention LSE merge Triton kernel and host launchers."""

# Triton JIT kernels do not use Python runtime type annotations or narrow signatures.
# ruff: noqa: ANN001, ANN202, PLR0124, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context

from ._decode_schedule import REDUCE_NUM_WARPS

MAX_MERGE_WAVES = 8
MERGE_STRIDE_FIELDS: tuple[str, ...] = tuple(
    field
    for wave_idx in range(MAX_MERGE_WAVES)
    for field in (
        f"so0_{wave_idx}",
        f"so1_{wave_idx}",
        f"so2_{wave_idx}",
        f"so3_{wave_idx}",
        f"sod_{wave_idx}",
        f"sl0_{wave_idx}",
        f"sl1_{wave_idx}",
        f"sl2_{wave_idx}",
        f"sl3_{wave_idx}",
    )
)


@triton.jit
def _load_wave_lse(lse_ptr, row, idx, sl, is_contiguous: tl.constexpr):
    if is_contiguous:
        return tl.load(lse_ptr + row).to(tl.float32)
    else:
        i0, i1, i2, i3 = idx
        sl0, sl1, sl2, sl3 = sl
        return tl.load(lse_ptr + i0 * sl0 + i1 * sl1 + i2 * sl2 + i3 * sl3).to(tl.float32)


@triton.jit
def _load_wave_out(out_ptr, row, offs_d, head_dim, idx, so, d_mask, is_contiguous: tl.constexpr):
    if is_contiguous:
        return tl.load(out_ptr + row * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    else:
        i0, i1, i2, i3 = idx
        so0, so1, so2, so3, sod = so
        off = i0 * so0 + i1 * so1 + i2 * so2 + i3 * so3 + offs_d * sod
        return tl.load(out_ptr + off, mask=d_mask, other=0.0).to(tl.float32)


@triton.jit
def _reduce_wave_lse(
    lse_ptr,
    row,
    idx,
    sl,
    has_nan,
    has_any,
    max_l,
    is_contiguous: tl.constexpr,
    allow_pos_inf: tl.constexpr,
):
    lse = _load_wave_lse(lse_ptr, row, idx, sl, is_contiguous)
    pos_inf = lse == float("inf")
    is_nan = (lse != lse) | (tl.full((), False, dtype=tl.int1) if allow_pos_inf else pos_inf)
    valid = (lse != -float("inf")) & (~is_nan)
    has_nan = has_nan | is_nan
    has_any = has_any | valid
    max_l = tl.maximum(max_l, tl.where(valid & (~pos_inf), lse, -float("inf")))
    return lse, valid, pos_inf, has_nan, has_any, max_l


@triton.jit
def _store_wave_fill(
    out_dst_ptr,
    row,
    head_dim,
    val,
    block_d: tl.constexpr,
    exact_d: tl.constexpr,
):
    for d_start in range(0, 1 if exact_d else head_dim, block_d):
        offs_d = d_start + tl.arange(0, block_d)
        d_mask = True if exact_d else (offs_d < head_dim)
        vec = tl.full((block_d,), val, dtype=tl.float32)
        tl.store(
            out_dst_ptr + row * head_dim + offs_d,
            vec.to(out_dst_ptr.dtype.element_ty),
            mask=d_mask,
        )


@triton.jit
def _store_wave_copy(
    out_dst_ptr,
    out_src_ptr,
    row,
    head_dim,
    idx,
    so,
    block_d: tl.constexpr,
    exact_d: tl.constexpr,
    is_contiguous: tl.constexpr,
):
    for d_start in range(0, 1 if exact_d else head_dim, block_d):
        offs_d = d_start + tl.arange(0, block_d)
        d_mask = True if exact_d else (offs_d < head_dim)
        vec = _load_wave_out(
            out_src_ptr,
            row,
            offs_d,
            head_dim,
            idx,
            so,
            d_mask,
            is_contiguous,
        )
        tl.store(
            out_dst_ptr + row * head_dim + offs_d,
            vec.to(out_dst_ptr.dtype.element_ty),
            mask=d_mask,
        )


@triton.jit
def _accum_wave_out(
    acc,
    weight,
    valid,
    out_ptr,
    row,
    offs_d,
    head_dim,
    idx,
    so,
    d_mask,
    is_contiguous: tl.constexpr,
):
    if valid:
        acc += weight * _load_wave_out(
            out_ptr,
            row,
            offs_d,
            head_dim,
            idx,
            so,
            d_mask,
            is_contiguous,
        )
    return acc


@triton.jit(do_not_specialize=["dim1", "dim2", "dim3", "head_dim"])
def _merge_two_attention_waves_kernel(  # noqa: PLR0912, PLR0915
    out_dst_ptr,
    lse_dst_ptr,
    out_0_ptr,
    lse_0_ptr,
    out_1_ptr,
    lse_1_ptr,
    out_2_ptr,
    lse_2_ptr,
    out_3_ptr,
    lse_3_ptr,
    out_4_ptr,
    lse_4_ptr,
    out_5_ptr,
    lse_5_ptr,
    out_6_ptr,
    lse_6_ptr,
    out_7_ptr,
    lse_7_ptr,
    dim1,
    dim2,
    dim3,
    head_dim,
    so0_0,
    so1_0,
    so2_0,
    so3_0,
    sod_0,
    sl0_0,
    sl1_0,
    sl2_0,
    sl3_0,
    so0_1,
    so1_1,
    so2_1,
    so3_1,
    sod_1,
    sl0_1,
    sl1_1,
    sl2_1,
    sl3_1,
    so0_2,
    so1_2,
    so2_2,
    so3_2,
    sod_2,
    sl0_2,
    sl1_2,
    sl2_2,
    sl3_2,
    so0_3,
    so1_3,
    so2_3,
    so3_3,
    sod_3,
    sl0_3,
    sl1_3,
    sl2_3,
    sl3_3,
    so0_4,
    so1_4,
    so2_4,
    so3_4,
    sod_4,
    sl0_4,
    sl1_4,
    sl2_4,
    sl3_4,
    so0_5,
    so1_5,
    so2_5,
    so3_5,
    sod_5,
    sl0_5,
    sl1_5,
    sl2_5,
    sl3_5,
    so0_6,
    so1_6,
    so2_6,
    so3_6,
    sod_6,
    sl0_6,
    sl1_6,
    sl2_6,
    sl3_6,
    so0_7,
    so1_7,
    so2_7,
    so3_7,
    sod_7,
    sl0_7,
    sl1_7,
    sl2_7,
    sl3_7,
    block_d: tl.constexpr,
    exact_d: tl.constexpr,
    num_waves: tl.constexpr,
    is_contiguous: tl.constexpr,
    in_place_0: tl.constexpr,
    single_pos_inf_wins: tl.constexpr = False,
):
    """Merge 1..8 attention waves and natural-log LSE values into ``(out_dst, lse_dst)``."""
    row = tl.program_id(0).to(tl.int64)
    if not is_contiguous:
        rem = row
        i3 = rem % dim3
        rem = rem // dim3
        i2 = rem % dim2
        rem = rem // dim2
        i1 = rem % dim1
        i0 = rem // dim1
    else:
        zero64 = tl.zeros((), dtype=tl.int64)
        i0, i1, i2, i3 = zero64, zero64, zero64, zero64
    idx = (i0, i1, i2, i3)

    wave_outs = (
        out_0_ptr,
        out_1_ptr,
        out_2_ptr,
        out_3_ptr,
        out_4_ptr,
        out_5_ptr,
        out_6_ptr,
        out_7_ptr,
    )
    wave_lses = (
        lse_0_ptr,
        lse_1_ptr,
        lse_2_ptr,
        lse_3_ptr,
        lse_4_ptr,
        lse_5_ptr,
        lse_6_ptr,
        lse_7_ptr,
    )
    wave_so = (
        (so0_0, so1_0, so2_0, so3_0, sod_0),
        (so0_1, so1_1, so2_1, so3_1, sod_1),
        (so0_2, so1_2, so2_2, so3_2, sod_2),
        (so0_3, so1_3, so2_3, so3_3, sod_3),
        (so0_4, so1_4, so2_4, so3_4, sod_4),
        (so0_5, so1_5, so2_5, so3_5, sod_5),
        (so0_6, so1_6, so2_6, so3_6, sod_6),
        (so0_7, so1_7, so2_7, so3_7, sod_7),
    )
    wave_sl = (
        (sl0_0, sl1_0, sl2_0, sl3_0),
        (sl0_1, sl1_1, sl2_1, sl3_1),
        (sl0_2, sl1_2, sl2_2, sl3_2),
        (sl0_3, sl1_3, sl2_3, sl3_3),
        (sl0_4, sl1_4, sl2_4, sl3_4),
        (sl0_5, sl1_5, sl2_5, sl3_5),
        (sl0_6, sl1_6, sl2_6, sl3_6),
        (sl0_7, sl1_7, sl2_7, sl3_7),
    )

    has_nan = tl.full((), False, dtype=tl.int1)
    has_any = tl.full((), False, dtype=tl.int1)
    max_l = tl.full((), -float("inf"), dtype=tl.float32)

    l_vals = ()
    v_vals = ()
    pos_inf_vals = ()
    allow_pos_inf: tl.constexpr = single_pos_inf_wins and (num_waves == 2)
    for w in tl.static_range(num_waves):
        l_w, v_w, pos_inf_w, has_nan, has_any, max_l = _reduce_wave_lse(
            wave_lses[w],
            row,
            idx,
            wave_sl[w],
            has_nan,
            has_any,
            max_l,
            is_contiguous,
            allow_pos_inf,
        )
        l_vals += (l_w,)
        v_vals += (v_w,)
        pos_inf_vals += (pos_inf_w,)

    if allow_pos_inf:
        has_nan = has_nan | (pos_inf_vals[0] & pos_inf_vals[1])
        two_a_wins = (v_vals[0] & (~v_vals[1])) | (pos_inf_vals[0] & (~pos_inf_vals[1]))
        two_b_wins = ((~v_vals[0]) & v_vals[1]) | (pos_inf_vals[1] & (~pos_inf_vals[0]))
    elif num_waves == 2:
        two_a_wins = v_vals[0] & (~v_vals[1])
        two_b_wins = (~v_vals[0]) & v_vals[1]
    else:
        two_a_wins = tl.full((), False, dtype=tl.int1)
        two_b_wins = tl.full((), False, dtype=tl.int1)

    skip_inplace = (two_a_wins & (~has_nan)) if (in_place_0 and num_waves == 2) else False
    if not skip_inplace:
        if has_nan:
            tl.store(lse_dst_ptr + row, float("nan"))
            _store_wave_fill(out_dst_ptr, row, head_dim, float("nan"), block_d, exact_d)
        elif not has_any:
            tl.store(lse_dst_ptr + row, -float("inf"))
            _store_wave_fill(out_dst_ptr, row, head_dim, 0.0, block_d, exact_d)
        elif num_waves == 1 or ((num_waves == 2) and two_a_wins):
            tl.store(lse_dst_ptr + row, l_vals[0])
            _store_wave_copy(
                out_dst_ptr,
                wave_outs[0],
                row,
                head_dim,
                idx,
                wave_so[0],
                block_d,
                exact_d,
                is_contiguous,
            )
        elif (num_waves == 2) and two_b_wins:
            tl.store(lse_dst_ptr + row, l_vals[1])
            _store_wave_copy(
                out_dst_ptr,
                wave_outs[1],
                row,
                head_dim,
                idx,
                wave_so[1],
                block_d,
                exact_d,
                is_contiguous,
            )
        else:
            w_vals = ()
            denom = tl.zeros((), dtype=tl.float32)
            for w in tl.static_range(num_waves):
                w_w = tl.where(v_vals[w], tl.math.exp(l_vals[w] - max_l), 0.0)
                w_vals += (w_w,)
                denom = w_w if w == 0 else (denom + w_w)

            merged_lse = max_l + tl.math.log(denom)
            tl.store(lse_dst_ptr + row, merged_lse)

            nw_vals = ()
            for w in tl.static_range(num_waves):
                nw_vals += (w_vals[w] / denom,)

            for d_start in range(0, 1 if exact_d else head_dim, block_d):
                offs_d = d_start + tl.arange(0, block_d)
                d_mask = True if exact_d else (offs_d < head_dim)
                acc = tl.zeros((block_d,), dtype=tl.float32)
                for w in tl.static_range(num_waves):
                    acc = _accum_wave_out(
                        acc,
                        nw_vals[w],
                        v_vals[w],
                        wave_outs[w],
                        row,
                        offs_d,
                        head_dim,
                        idx,
                        wave_so[w],
                        d_mask,
                        is_contiguous,
                    )
                tl.store(
                    out_dst_ptr + row * head_dim + offs_d,
                    acc.to(out_dst_ptr.dtype.element_ty),
                    mask=d_mask,
                )


def _coalesce_4d_strides(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
    leading_dims: tuple[int, ...],
) -> tuple[tuple[int, int, int], list[tuple[int, ...]]] | None:
    """Coalesce leading dimensions to <= 4 axes across all strided waves, or return None."""
    dims = list(leading_dims)
    out_strides = [list(out.stride()[:-1]) for out in outputs]
    lse_strides = [list(lse.stride()) for lse in lses]
    idx = len(dims) - 2
    while len(dims) > 4 and idx >= 0:
        can_merge = all(
            dims[idx] == 1 or dims[idx + 1] == 1 or s[idx] == s[idx + 1] * dims[idx + 1]
            for s in (*out_strides, *lse_strides)
        )
        if can_merge:
            merged_dim = dims[idx] * dims[idx + 1]
            for s in (*out_strides, *lse_strides):
                kept = (
                    s[idx + 1] if dims[idx] == 1 else (s[idx] if dims[idx + 1] == 1 else s[idx + 1])
                )
                s[idx : idx + 2] = [kept]
            dims[idx : idx + 2] = [merged_dim]
            if idx >= len(dims) - 1:
                idx -= 1
        else:
            idx -= 1
    if len(dims) > 4:
        return None
    pad = 4 - len(dims)
    padded_dims = ([1] * pad) + dims
    per_wave_strides: list[tuple[int, ...]] = []
    for out, os, ls in zip(outputs, out_strides, lse_strides, strict=True):
        padded_os = ([0] * pad) + os
        padded_ls = ([0] * pad) + ls
        per_wave_strides.append((*padded_os, out.stride(-1), *padded_ls))
    return (padded_dims[1], padded_dims[2], padded_dims[3]), per_wave_strides


def can_launch_merge_attention_lse(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
) -> bool:
    """Return whether ``outputs`` and ``lses`` can run in the fused Triton merge kernel."""
    leading_dims = tuple(outputs[0].shape[:-1])
    if len(leading_dims) <= 4 or all(t.is_contiguous() for t in (*outputs, *lses)):
        return True
    return _coalesce_4d_strides(outputs, lses, leading_dims) is not None


def launch_merge_attention_lse_chunk(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
    out_dst: torch.Tensor,
    lse_dst: torch.Tensor,
    *,
    in_place_0: bool = False,
    single_pos_inf_wins: bool = False,
) -> None:
    """Merge 1..8 waves into contiguous ``(out_dst, lse_dst)`` in one Triton launch."""
    num_waves = len(outputs)
    assert 1 <= num_waves <= MAX_MERGE_WAVES
    head_dim = out_dst.shape[-1]
    leading_dims = tuple(out_dst.shape[:-1])
    total_rows = 1
    for dim in leading_dims:
        total_rows *= dim
    if total_rows == 0 or head_dim == 0:
        return

    block_d = min(256, max(16, triton.next_power_of_2(head_dim)))
    exact_d = head_dim == block_d
    is_contiguous = all(t.is_contiguous() for t in (*outputs, *lses))
    if is_contiguous:
        dim1 = 1
        dim2 = 1
        dim3 = 1
        stride_args: list[object] = [None] * (MAX_MERGE_WAVES * 9)
    else:
        coalesced = _coalesce_4d_strides(outputs, lses, leading_dims)
        assert coalesced is not None
        (dim1, dim2, dim3), per_wave = coalesced
        stride_args = []
        for idx in range(MAX_MERGE_WAVES):
            if idx < num_waves:
                stride_args.extend(per_wave[idx])
            else:
                stride_args.extend([None] * 9)

    ptr_args: list[object] = []
    for idx in range(MAX_MERGE_WAVES):
        if idx < num_waves:
            ptr_args.extend((outputs[idx], lses[idx]))
        else:
            ptr_args.extend((None, None))

    with device_context(out_dst.device):
        _merge_two_attention_waves_kernel[(total_rows,)](
            out_dst,
            lse_dst,
            *ptr_args,
            dim1,
            dim2,
            dim3,
            head_dim,
            *stride_args,
            block_d=block_d,
            exact_d=exact_d,
            num_waves=num_waves,
            is_contiguous=is_contiguous,
            in_place_0=in_place_0,
            single_pos_inf_wins=single_pos_inf_wins,
            num_warps=REDUCE_NUM_WARPS,
        )


def merge_two_attention_waves(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> None:
    """Merge ``(out_b, lse_b)`` in place into contiguous ``(out_a, lse_a)``."""
    launch_merge_attention_lse_chunk(
        (out_a, out_b),
        (lse_a, lse_b),
        out_a,
        lse_a,
        in_place_0=True,
        single_pos_inf_wins=True,
    )
