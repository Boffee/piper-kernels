"""FP16x2 persistent Viterbi search in Gluon: L=12 cost-only and L=16 walks.

One Gluon launch executes all ``T`` steps for a batch in shared memory, packing two
rows per 32-bit word (``half2``) in both registers and shared memory. Weights are
pre-normalized once by the caller as ``(w / p2) * FP16_SCALE``, so decoded INT8 code
values shift by an exact power of two (``0.03125``) and every step uses native
``half2`` arithmetic (``sub.f16x2``, ``fma.rn.f16x2``, ``min.f16x2``) without
float-mode duplication or host-managed step loops.

Rows are sorted by bitwise ``alpha`` and padded per bucket to ``rows_per_cta``, so
every CTA decodes each 16-state tile once for all of its rows. Backpointers are
stored coalesced as ``uint16`` pairs and traced back by ``_traceback_kernel``.
"""

# Kernel constexprs are short upper-case names (KE = k_t, KA = k_{t+1}). Gluon's
# frontend has no starred unpacking, so per-pair tuples grow by concatenation.
# ruff: noqa: N803, N806, RUF005, ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false

from __future__ import annotations

import functools
import re
from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from triton.compiler import CompiledKernel
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context

from .._layout import LCG_INCREMENT, LCG_MULTIPLIER

FP16_SCALE = 0.03125
"""Exact 2^-5 multiplier for normalized weights and clamped INT8 codes in FP16."""
_RESERVED_SHARED = 1024
SHARED_NEED = 96 * 1024 + 128
"""Opt-in shared memory per CTA of the largest kernel (96 KiB prefixes + 128 B reduction)."""

_ONE_MAD_A = gl.constexpr(LCG_MULTIPLIER)
_ONE_MAD_B = gl.constexpr(LCG_INCREMENT)
_IDENTITY = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))
_U32 = 0xFFFFFFFF

_MIN_TRITON = (3, 9)
_TRITON_VERSION = tuple(int(part) for part in re.findall(r"\d+", triton.__version__)[:2])
_BACKPOINTER_BYTES = 1 << 32
_TRACEBACK_SEQUENCES = 64


@gluon.constexpr_function
def _geometry(L: int, min_k: int) -> tuple[int, int, int]:
    """``(threads, loop_bits, pairs_per_cta)`` for ``(L, min_k)``.

    Each pair holds two rows packed in ``half2``, giving 12 rows/CTA except at
    ``L=16, min_k=2`` (64 KiB/pair fits 1 pair = 2 rows in 64 KiB) and
    ``L=16, min_k=3`` (32 KiB/pair fits 3 pairs = 6 rows in 96 KiB).
    """
    if L == 12:
        return (64, 2, 6)
    if min_k == 2:
        return (1024, 2, 1)
    if min_k == 3:
        return (512, 3, 3)
    return (256, 4, 6)


@gluon.constexpr_function
def _thread_layout(threads: int):
    return gl.BlockedLayout([1], [32], [threads // 32], [0])


@gluon.constexpr_function
def _low_pairs(pairs: int) -> int:
    return 1 << (pairs.bit_length() - 1)


@gluon.constexpr_function
def _pair_offset_bytes(p: int, pairs: int, cap: int) -> int:
    low = _low_pairs(pairs)
    return (p - low if p >= low else p) * cap * 4


@gluon.constexpr_function
def _phys_const(v: int) -> int:
    """1-way bank-conflict-free XOR swizzle across all L in {12, 16} and k in {2..6}."""
    return v ^ (((v >> 5) ^ (v >> 7)) & 31)


@gluon.constexpr_function
def _split_count(ka: int) -> int:
    return 1 << (4 - ka) if ka < 4 else 1


@gluon.constexpr_function
def _in_offsets(group: int, split: int) -> tuple[int, ...]:
    if split == 1:
        return tuple(_phys_const(i << group) for i in range(16))
    split_bits = split.bit_length() - 1
    mask = (16 // split) - 1
    return tuple(
        _phys_const(((i & mask) << group) | ((i >> (4 - split_bits)) << (group - split_bits)))
        for i in range(16)
    )


@gluon.constexpr_function
def _state_steps(out_bits: int, split: int) -> tuple[int, ...]:
    if split == 1:
        return tuple(_ONE_MAD_A.value * (i << out_bits) & _U32 for i in range(16))
    split_bits = split.bit_length() - 1
    mask = (16 // split) - 1
    return tuple(
        _ONE_MAD_A.value
        * (((i & mask) << out_bits) | ((i >> (4 - split_bits)) << (out_bits - split_bits)))
        & _U32
        for i in range(16)
    )


@gluon.constexpr_function
def _split_out_offsets(out_bits: int, split: int) -> tuple[int, ...]:
    if split == 1:
        return (0,)
    shift = out_bits - (split.bit_length() - 1)
    return tuple(4 * _phys_const(s << shift) for s in range(split))


@gluon.constexpr_function
def _split_bp_offsets(idx_bits: int, split: int) -> tuple[int, ...]:
    if split == 1:
        return (0,)
    shift = idx_bits - (split.bit_length() - 1)
    return tuple(s << shift for s in range(split))


@gluon.constexpr_function
def _st_u32_asm(offset: int, predicated: bool) -> str:
    store = f"st.shared.u32 [$0+{offset}], $1;"
    return "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p " + store + " }" if predicated else store


@gluon.constexpr_function
def _decode_h2_asm() -> str:
    """1MAD numerator -> clamped INT8 * 0.03125 packed identically in both FP16 halves."""
    return (
        "{ .reg .b32 a; .reg .f32 n, s; "
        "dp4a.u32.u32 a, $1, 16843009, 1262484994; "
        "add.rn.f32 n, a, 0fCB400000; "
        "mul.rn.f32 s, n, $2; "
        "cvt.rni.f32.f32 s, s; "
        "max.f32 s, s, 0fC2FE0000; min.f32 s, s, 0f42FE0000; "
        "mul.rn.f32 s, s, 0f3D000000; "
        "cvt.rn.f16x2.f32 $0, s, s; }"
    )


@gluon.constexpr_function
def _h2_min_only_first_asm() -> str:
    return (
        "{ .reg .b32 d0, d1, t0, t1; "
        "sub.f16x2 d0, $1, $2; sub.f16x2 d1, $1, $3; "
        "fma.rn.f16x2 t0, d0, d0, $4; fma.rn.f16x2 t1, d1, d1, $5; "
        "min.f16x2 $0, t0, t1; }"
    )


@gluon.constexpr_function
def _h2_min_only_step_asm() -> str:
    return (
        "{ .reg .b32 d0, d1, t0, t1, m; "
        "sub.f16x2 d0, $1, $2; sub.f16x2 d1, $1, $3; "
        "fma.rn.f16x2 t0, d0, d0, $4; fma.rn.f16x2 t1, d1, d1, $5; "
        "min.f16x2 m, t0, t1; min.f16x2 $0, $6, m; }"
    )


@gluon.constexpr_function
def _h2_argmin_first_asm() -> str:
    return (
        "{ .reg .b32 d0, d1, t0, t1, msk; "
        "sub.f16x2 d0, $2, $3; sub.f16x2 d1, $2, $4; "
        "fma.rn.f16x2 t0, d0, d0, $5; fma.rn.f16x2 t1, d1, d1, $6; "
        "set.lt.u32.f16x2 msk, t1, t0; "
        "min.f16x2 $0, t0, t1; "
        "and.b32 $1, msk, 0x00010001; }"
    )


@gluon.constexpr_function
def _h2_argmin_single_asm(c: int) -> str:
    cc = (c << 16) | c
    return (
        "{ .reg .b32 d, t, p; "
        "sub.f16x2 d, $2, $3; "
        "fma.rn.f16x2 t, d, d, $4; "
        "set.lt.u32.f16x2 p, t, $5; "
        "min.f16x2 $0, $5, t; "
        f"lop3.b32 $1, p, {cc}, $6, 0xCA; }}"
    )


@gluon.constexpr_function
def _merge_shfl_asm(mask_val: int) -> str:
    return (
        "{ .reg .b32 ob, oa, p; "
        f"shfl.sync.bfly.b32 ob, $2, {mask_val}, 31, 0xffffffff; "
        f"shfl.sync.bfly.b32 oa, $3, {mask_val}, 31, 0xffffffff; "
        "set.lt.u32.f16x2 p, ob, $2; "
        "min.f16x2 $0, $2, ob; "
        "lop3.b32 $1, p, oa, $3, 0xCA; }"
    )


@gluon.constexpr_function
def _merge_shfl_cost_asm(mask_val: int) -> str:
    return (
        "{ .reg .b32 ob; "
        f"shfl.sync.bfly.b32 ob, $1, {mask_val}, 31, 0xffffffff; "
        "min.f16x2 $0, $1, ob; }"
    )


@gluon.jit
def _phys(v):
    return v ^ (((v >> 5) ^ (v >> 7)) & 31)


@gluon.jit
def _thread_index(THREADS: gl.constexpr):
    """Read ``%tid.x`` inside each step so LLVM does not hoist per-kind address maps."""
    tid = gl.inline_asm(
        "mov.u32 $0, %tid.x;",
        ("=r",),
        [],
        gl.distributed_type(gl.int32, [THREADS], _thread_layout(THREADS)),
        is_pure=False,
    )
    return tid & (THREADS - 1)


@gluon.jit
def _shared_address(buffer):
    return gl.inline_asm("mov.u32 $0, $1;", ("=r", "r"), [buffer], gl.int32, is_pure=False)


@gluon.jit
def _st_shared_u32(addr, val, offset: gl.constexpr):
    gl.inline_asm(_st_u32_asm(offset, False), ("r", "r"), [addr, val], (), is_pure=False)


@gluon.jit
def _st_shared_u32_if(addr, val, keep, offset: gl.constexpr):
    gl.inline_asm(_st_u32_asm(offset, True), ("r", "r", "r"), [addr, val, keep], (), is_pure=False)


@gluon.jit
def _decode_h2(x, alpha):
    return gl.inline_asm_elementwise(
        _decode_h2_asm(), "=r,r,f", [x, alpha], dtype=gl.uint32, is_pure=True, pack=1
    )


@gluon.jit
def _argmin_h2(
    wp, q, ins, p: gl.constexpr, FIRST: gl.constexpr, COUNT: gl.constexpr, COST_ONLY: gl.constexpr
):
    if COST_ONLY:
        best = gl.inline_asm_elementwise(
            _h2_min_only_first_asm(),
            "=r,r,r,r,r,r",
            [wp, q[FIRST], q[FIRST + 1], ins[p * 16 + FIRST], ins[p * 16 + FIRST + 1]],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
        for i in gl.static_range(2, COUNT, 2):
            best = gl.inline_asm_elementwise(
                _h2_min_only_step_asm(),
                "=r,r,r,r,r,r,r",
                [
                    wp,
                    q[FIRST + i],
                    q[FIRST + i + 1],
                    ins[p * 16 + FIRST + i],
                    ins[p * 16 + FIRST + i + 1],
                    best,
                ],
                dtype=gl.uint32,
                is_pure=True,
                pack=1,
            )
        return best, best
    best, arg = gl.inline_asm_elementwise(
        _h2_argmin_first_asm(),
        "=r,=r,r,r,r,r,r",
        [wp, q[FIRST], q[FIRST + 1], ins[p * 16 + FIRST], ins[p * 16 + FIRST + 1]],
        dtype=(gl.uint32, gl.uint32),
        is_pure=True,
        pack=1,
    )
    for i in gl.static_range(2, COUNT):
        best, arg = gl.inline_asm_elementwise(
            _h2_argmin_single_asm(i),
            "=r,=r,r,r,r,r,r",
            [wp, q[FIRST + i], ins[p * 16 + FIRST + i], best, arg],
            dtype=(gl.uint32, gl.uint32),
            is_pure=True,
            pack=1,
        )
    return best, arg


@gluon.jit
def _reduce_parts(best, arg, part, PARTS: gl.constexpr, COST_ONLY: gl.constexpr):
    if COST_ONLY:
        if PARTS >= 2:
            best = gl.inline_asm_elementwise(
                _merge_shfl_cost_asm(1), "=r,r", [best], dtype=gl.uint32, is_pure=True, pack=1
            )
        if PARTS == 4:
            best = gl.inline_asm_elementwise(
                _merge_shfl_cost_asm(2), "=r,r", [best], dtype=gl.uint32, is_pure=True, pack=1
            )
        return best, arg
    add_a = ((16 * part) | ((16 * part) << 16)).to(gl.uint32)
    arg = arg + add_a
    if PARTS >= 2:
        best, arg = gl.inline_asm_elementwise(
            _merge_shfl_asm(1),
            "=r,=r,r,r",
            [best, arg],
            dtype=(gl.uint32, gl.uint32),
            is_pure=True,
            pack=1,
        )
    if PARTS == 4:
        best, arg = gl.inline_asm_elementwise(
            _merge_shfl_asm(2),
            "=r,=r,r,r",
            [best, arg],
            dtype=(gl.uint32, gl.uint32),
            is_pure=True,
            pack=1,
        )
    return best, arg


@gluon.jit
def _store_step_output(
    base_addr, out_off, bp_ptr, arg, best, part, is_owner, do_bp,
    poff: gl.constexpr, PARTS: gl.constexpr, COST_ONLY: gl.constexpr,
):  # fmt: skip
    """Store one pair's winning shared-memory cost and optional packed uint16 backpointer."""
    if PARTS == 1:
        _st_shared_u32(base_addr + out_off, best, poff)
        if not COST_ONLY and do_bp:
            gl.store(bp_ptr, ((arg & 63) | ((arg >> 8) & 0x3F00)).to(gl.uint16))
    else:
        best, arg = _reduce_parts(best, arg, part, PARTS, COST_ONLY)
        _st_shared_u32_if(base_addr + out_off, best, is_owner, poff)
        if not COST_ONLY and do_bp:
            gl.store(bp_ptr, ((arg & 63) | ((arg >> 8) & 0x3F00)).to(gl.uint16), mask=part == 0)


@gluon.jit
def _step_h2(
    t, low, high, low_addr, high_addr, pair_idx, alpha, w_pair_ptr, bp_pair_ptr,
    num_pairs, outputs, write_bp_from,
    L: gl.constexpr, CAP: gl.constexpr, PAIRS: gl.constexpr, MIN_K: gl.constexpr,
    KE: gl.constexpr, KA: gl.constexpr, COST_ONLY: gl.constexpr,
):  # fmt: skip
    THREADS: gl.constexpr = _geometry(L, MIN_K)[0]
    LB: gl.constexpr = _geometry(L, MIN_K)[1]
    SPLIT: gl.constexpr = _split_count(KA)
    COUNT: gl.constexpr = 16 // SPLIT
    PARTS: gl.constexpr = 1 << (KA - 4) if KA > 4 else 1
    PART_BITS: gl.constexpr = KA - 4 if KA > 4 else 0
    OUT_BITS: gl.constexpr = L - KA
    GROUP: gl.constexpr = L - KE - KA
    LOW_P: gl.constexpr = _low_pairs(PAIRS)
    IDX_BITS: gl.constexpr = OUT_BITS - LB
    OFFSETS: gl.constexpr = _in_offsets(GROUP, SPLIT)
    STEPS: gl.constexpr = _state_steps(OUT_BITS, SPLIT)
    SPLIT_OUT_OFFS: gl.constexpr = _split_out_offsets(OUT_BITS, SPLIT)
    SPLIT_BP_OFFS: gl.constexpr = _split_bp_offsets(IDX_BITS, SPLIT)

    tid = _thread_index(THREADS)
    part = tid & (PARTS - 1)
    idx = tid >> PART_BITS
    j0 = idx << LB

    wp = ()
    bp_base = ()
    do_bp = (not COST_ONLY) and (t >= write_bp_from)
    bp_t = t - write_bp_from
    for p in gl.static_range(PAIRS):
        wp = wp + (gl.load(w_pair_ptr + t * num_pairs + pair_idx[p]),)
        if not COST_ONLY:
            pair_off = (bp_t * num_pairs + pair_idx[p]).to(gl.int64) * CAP + idx
            bp_base = bp_base + (bp_pair_ptr + pair_off,)

    in_base = _phys(((part * 16) << GROUP) | (j0 >> KE))
    ins = ()
    for p in gl.static_range(PAIRS):
        buf = low if p < LOW_P else high
        first = (p if p < LOW_P else p - LOW_P) * CAP
        for i in gl.static_range(16):
            ins = ins + (buf.gather(first + (in_base ^ OFFSETS[i]), 0),)
    gl.barrier()

    out_base = _phys(j0)
    x_base = (j0 + ((16 * part) << OUT_BITS)).to(gl.uint32) * _ONE_MAD_A + _ONE_MAD_B
    is_owner = (part == 0).to(gl.int32)

    for e in range(outputs):
        xe = x_base + _ONE_MAD_A * e.to(gl.uint32)
        q = ()
        for i in gl.static_range(16):
            q = q + (_decode_h2(xe + STEPS[i], alpha),)
        out_off = 4 * (out_base ^ e)
        bp_off = e << IDX_BITS
        for p in gl.static_range(PAIRS):
            base_addr = low_addr if p < LOW_P else high_addr
            for s in gl.static_range(SPLIT):
                best, arg = _argmin_h2(wp[p], q, ins, p, s * COUNT, COUNT, COST_ONLY)
                _store_step_output(
                    base_addr,
                    out_off ^ SPLIT_OUT_OFFS[s],
                    bp_base[p] + bp_off + SPLIT_BP_OFFS[s] if not COST_ONLY else bp_pair_ptr,
                    arg, best, part, is_owner, do_bp, _pair_offset_bytes(p, PAIRS, CAP),
                    PARTS, COST_ONLY,
                )  # fmt: skip
    gl.barrier()


@gluon.jit
def _init_smem(
    low_addr, high_addr, pair_idx, start_pair_ptr, constrained,
    THREADS: gl.constexpr, CAP: gl.constexpr, PAIRS: gl.constexpr, WIDTH0: gl.constexpr,
):  # fmt: skip
    TL: gl.constexpr = _thread_layout(THREADS)
    LOW_P: gl.constexpr = _low_pairs(PAIRS)
    tid = gl.arange(0, THREADS, TL)
    zero_u32 = gl.full([THREADS], 0, gl.uint32, TL)
    h_inf_lo = gl.full([THREADS], 0x00007C00, gl.uint32, TL)
    h_inf_hi = gl.full([THREADS], 0x7C000000, gl.uint32, TL)
    for p in gl.static_range(PAIRS):
        sp = gl.load(start_pair_ptr + pair_idx[p], mask=constrained != 0, other=-1)
        s0 = sp & 0xFFFF
        s1 = (sp >> 16) & 0xFFFF
        base_addr = low_addr if p < LOW_P else high_addr
        for i in gl.static_range((1 << WIDTH0) // THREADS):
            v = i * THREADS + tid
            lo = gl.where((v == s0) | (constrained == 0), zero_u32, h_inf_lo)
            hi = gl.where((v == s1) | (constrained == 0), zero_u32, h_inf_hi)
            _st_shared_u32(base_addr + _phys(v) * 4, lo | hi, _pair_offset_bytes(p, PAIRS, CAP))


@gluon.jit
def _merge_h2_candidates(b0, b1, a0, a1, COST_ONLY: gl.constexpr):
    if COST_ONLY:
        best = gl.inline_asm_elementwise(
            "min.f16x2 $0, $1, $2;", "=r,r,r", [b0, b1], dtype=gl.uint32, is_pure=True, pack=1
        )
        return best, a0
    return gl.inline_asm_elementwise(
        "{ .reg .b32 m; set.lt.u32.f16x2 m, $3, $2; min.f16x2 $0, $2, $3; "
        "lop3.b32 $1, m, $5, $4, 0xCA; }",
        "=r,=r,r,r,r,r",
        [b0, b1, a0, a1],
        dtype=(gl.uint32, gl.uint32),
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _reduce_final_min(
    low, high, pair_idx, out_ptr,
    THREADS: gl.constexpr, CAP: gl.constexpr, PAIRS: gl.constexpr, WIDTH0: gl.constexpr,
    COST_ONLY: gl.constexpr,
):  # fmt: skip
    TL: gl.constexpr = _thread_layout(THREADS)
    LOW_P: gl.constexpr = _low_pairs(PAIRS)
    NUM_WARPS: gl.constexpr = THREADS // 32
    tid = _thread_index(THREADS)
    red_buf = gl.allocate_shared_memory(gl.uint32, [2 * NUM_WARPS], _IDENTITY)
    red_addr = _shared_address(red_buf)
    lane = tid & 31
    warp = tid >> 5
    is_lane0 = (lane == 0).to(gl.int32)
    z_i32 = gl.zeros_like(tid)

    for p in gl.static_range(PAIRS):
        buf = low if p < LOW_P else high
        first = (p if p < LOW_P else p - LOW_P) * CAP
        best = buf.gather(first + _phys(tid), 0)
        arg = (tid | (tid << 16)).to(gl.uint32)
        for i in gl.static_range(1, (1 << WIDTH0) // THREADS):
            v = i * THREADS + tid
            val = buf.gather(first + _phys(v), 0)
            best, arg = _merge_h2_candidates(
                best, val, arg, (v | (v << 16)).to(gl.uint32), COST_ONLY
            )
        for sh in gl.static_range(5):
            if COST_ONLY:
                best = gl.inline_asm_elementwise(
                    _merge_shfl_cost_asm(1 << sh),
                    "=r,r",
                    [best],
                    dtype=gl.uint32,
                    is_pure=True,
                    pack=1,
                )
            else:
                best, arg = gl.inline_asm_elementwise(
                    _merge_shfl_asm(1 << sh),
                    "=r,=r,r,r",
                    [best, arg],
                    dtype=(gl.uint32, gl.uint32),
                    is_pure=True,
                    pack=1,
                )
        _st_shared_u32_if(red_addr + warp * 4, best, is_lane0, 0)
        if not COST_ONLY:
            _st_shared_u32_if(red_addr + (NUM_WARPS + warp) * 4, arg, is_lane0, 0)
        gl.barrier()
        if NUM_WARPS > 1:
            b = red_buf.gather(z_i32, 0)
            a = (
                red_buf.gather(gl.full([THREADS], NUM_WARPS, gl.int32, TL), 0)
                if not COST_ONLY
                else b
            )
            for w_i in gl.static_range(1, NUM_WARPS):
                bw = red_buf.gather(gl.full([THREADS], w_i, gl.int32, TL), 0)
                aw = (
                    red_buf.gather(gl.full([THREADS], NUM_WARPS + w_i, gl.int32, TL), 0)
                    if not COST_ONLY
                    else a
                )
                b, a = _merge_h2_candidates(b, bw, a, aw, COST_ONLY)
            res = b if COST_ONLY else a
        else:
            res = best if COST_ONLY else arg
        gl.store(out_ptr + pair_idx[p] + z_i32, res, mask=tid == 0)
        gl.barrier()


@gluon.jit(
    do_not_specialize=["num_pairs", "length", "outputs", "k0", "constrained", "write_bp_from"]
)
def _forward_kernel(
    w_pair_ptr, alpha_cta_ptr, start_pair_ptr, kinds_ptr, out_ptr, bp_pair_ptr,
    num_pairs, length, outputs, k0, constrained, write_bp_from,
    L: gl.constexpr, MIN_K: gl.constexpr, KINDS: gl.constexpr, COST_ONLY: gl.constexpr,
):  # fmt: skip
    """All ``length`` steps for one CTA's ``PAIRS`` pairs sharing ``alpha_cta[cta]``."""
    THREADS: gl.constexpr = _geometry(L, MIN_K)[0]
    PAIRS: gl.constexpr = _geometry(L, MIN_K)[2]
    MIXED: gl.constexpr = KINDS != 1
    CAP: gl.constexpr = 1 << (L - MIN_K)
    LOW_P: gl.constexpr = _low_pairs(PAIRS)
    HIGH_P: gl.constexpr = PAIRS - LOW_P

    cta = gl.program_id(0)
    low = gl.allocate_shared_memory(gl.uint32, [LOW_P * CAP], _IDENTITY)
    high = gl.allocate_shared_memory(gl.uint32, [HIGH_P * CAP], _IDENTITY) if HIGH_P > 0 else low
    low_addr = _shared_address(low)
    high_addr = _shared_address(high)

    alpha = gl.load(alpha_cta_ptr + cta)
    pair_idx = ()
    for p in gl.static_range(PAIRS):
        pair_idx = pair_idx + (cta * PAIRS + p,)

    if not MIXED:  # noqa: SIM114 (compile-time vs run-time width selection)
        _init_smem(
            low_addr, high_addr, pair_idx, start_pair_ptr, constrained, THREADS, CAP, PAIRS,
            L - MIN_K,
        )  # fmt: skip
    elif k0 == MIN_K:
        _init_smem(
            low_addr, high_addr, pair_idx, start_pair_ptr, constrained, THREADS, CAP, PAIRS,
            L - MIN_K,
        )  # fmt: skip
    else:
        _init_smem(
            low_addr, high_addr, pair_idx, start_pair_ptr, constrained, THREADS, CAP, PAIRS,
            L - MIN_K - 1,
        )  # fmt: skip
    gl.barrier()

    for t in range(length):
        kind = gl.load(kinds_ptr + t)
        for k in gl.static_range(4):
            if (KINDS >> k) & 1:  # noqa: SIM102
                if kind == k:
                    _step_h2(
                        t, low, high, low_addr, high_addr, pair_idx, alpha, w_pair_ptr, bp_pair_ptr,
                        num_pairs, outputs, write_bp_from, L, CAP, PAIRS, MIN_K,
                        MIN_K + (k >> 1), MIN_K + (k & 1), COST_ONLY,
                    )  # fmt: skip

    if COST_ONLY or (constrained == 0):
        if not MIXED:  # noqa: SIM114
            _reduce_final_min(
                low, high, pair_idx, out_ptr, THREADS, CAP, PAIRS, L - MIN_K, COST_ONLY
            )
        elif k0 == MIN_K:
            _reduce_final_min(
                low, high, pair_idx, out_ptr, THREADS, CAP, PAIRS, L - MIN_K, COST_ONLY
            )
        else:
            _reduce_final_min(
                low, high, pair_idx, out_ptr, THREADS, CAP, PAIRS, L - MIN_K - 1, COST_ONLY
            )


@triton.jit
def _traceback_kernel(
    bp_pair_ptr, slot_ptr, j_ptr, widths_ptr, states_ptr, batch, num_pairs,
    length: tl.constexpr, trace_steps: tl.constexpr, state_bits: tl.constexpr,
    width: tl.constexpr, lb: tl.constexpr, block: tl.constexpr,
):  # fmt: skip
    """Trace each sequence backward over the last ``trace_steps`` steps from coalesced pairs."""
    seq = (tl.program_id(0) * block + tl.arange(0, block)).to(tl.int64)
    valid = seq < batch
    slot = tl.load(slot_ptr + seq, mask=valid, other=0).to(tl.int64)
    pair = slot >> 1
    shift_byte = ((slot & 1) << 3).to(tl.int32)
    j = tl.load(j_ptr + seq, mask=valid, other=0)
    last_bp = trace_steps - 1
    offset_t = length - trace_steps
    for i in range(trace_steps):
        bp_t = last_bp - i
        t = offset_t + bp_t
        bits = state_bits - tl.load(widths_ptr + t + 1)
        bp_pos = ((j & ((1 << lb) - 1)) << (bits - lb)) | (j >> lb)
        packed = tl.load(
            bp_pair_ptr + (bp_t * num_pairs + pair) * width + bp_pos, mask=valid, other=0
        ).to(tl.int32)
        a = ((packed >> shift_byte) & 0xFF).to(tl.int64)
        state = (a << bits) | j
        tl.store(states_ptr + bp_t * batch + seq, state, mask=valid)
        j = state >> tl.load(widths_ptr + t)


def _supports_properties(capability: tuple[int, int], shared_optin: int) -> bool:
    """Whether an NVIDIA GPU with these properties runs ``_forward_kernel``."""
    return capability >= (8, 0) and shared_optin >= SHARED_NEED


@functools.cache
def _supports_index(index: int) -> bool:
    props = torch.cuda.get_device_properties(index)
    return _supports_properties((props.major, props.minor), props.shared_memory_per_block_optin)


def _index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def supports_device(device: torch.device) -> bool:
    """Whether ``run_viterbi`` runs on ``device``: NVIDIA SM80+ with ``SHARED_NEED`` bytes
    of opt-in shared memory per CTA under Triton 3.9+."""
    if device.type != "cuda" or torch.version.hip is not None or _TRITON_VERSION < _MIN_TRITON:
        return False
    return _supports_index(_index(device))


def _kinds(step_widths: Sequence[int]) -> list[int]:
    n, k = len(step_widths), min(step_widths)
    return [2 * (step_widths[t] - k) + (step_widths[(t + 1) % n] - k) for t in range(n)]


def _kind_set(step_widths: Sequence[int]) -> int:
    return sum(1 << k for k in set(_kinds(step_widths)))


@functools.cache
def _kinds_tensor(step_widths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(_kinds(step_widths), dtype=torch.int32, device=device)


@functools.cache
def _widths_tensor(step_widths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    n = len(step_widths)
    extended = [step_widths[t % n] for t in range(n + 2)]
    return torch.tensor(extended, dtype=torch.int64, device=device)


type _Specialization = tuple[int, int, int, bool]


def _specialization(
    state_bits: int, step_widths: Sequence[int], cost_only: bool
) -> _Specialization:
    return state_bits, min(step_widths), _kind_set(step_widths), cost_only


def _compile_args(L: int, min_k: int, kinds: int, cost_only: bool) -> dict[str, int | bool]:
    threads = _geometry(L, min_k)[0]
    return {
        "L": L,
        "MIN_K": min_k,
        "KINDS": kinds,
        "COST_ONLY": cost_only,
        "num_warps": threads // 32,
    }


@functools.cache
def _compiled(specialization: _Specialization, index: int) -> CompiledKernel:
    L, min_k, kinds, cost_only = specialization
    u32, f32, i32, u16 = torch.uint32, torch.float32, torch.int32, torch.uint16
    outputs = 1 << _geometry(L, min_k)[1]
    with torch.cuda.device(index):
        kernel = _forward_kernel.warmup(
            u32, f32, i32, i32, u32, u16, 1, 1, outputs, min_k, 0, 0, grid=(1,),
            **_compile_args(L, min_k, kinds, cost_only),
        )  # fmt: skip
        # Private: n_regs is populated only once the CUDA module handle is initialized.
        kernel._init_handles()
    return kernel


def _resident_ctas(specialization: _Specialization, index: int) -> int:
    kernel = _compiled(specialization, index)
    props = torch.cuda.get_device_properties(index)
    threads = _geometry(specialization[0], specialization[1])[0]
    warp_registers = -(-kernel.n_regs * 32 // 256) * 256
    by_threads = props.max_threads_per_multi_processor // threads
    by_registers = props.regs_per_multiprocessor // (warp_registers * threads // 32)
    by_shared = props.shared_memory_per_multiprocessor // (
        kernel.metadata.shared + _RESERVED_SHARED
    )
    return min(by_threads, by_registers, by_shared)


def _wave_ctas(
    state_bits: int, step_widths: Sequence[int], cost_only: bool, device: torch.device
) -> int:
    index = _index(device)
    ctas = _resident_ctas(_specialization(state_bits, step_widths, cost_only), index)
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    return sms * ctas


def _bucket_pack_rows(
    alpha: torch.Tensor, rows_per_cta: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group rows by bitwise ``alpha``, padding each bucket to a multiple of ``rows_per_cta``."""
    n = alpha.shape[0]
    device = alpha.device
    sorted_key, order = torch.sort(alpha.contiguous().view(torch.int32), stable=True)
    uniq_keys, counts = torch.unique_consecutive(sorted_key, return_counts=True)
    ctas_per_bucket = (counts + rows_per_cta - 1) // rows_per_cta
    padded_sizes = ctas_per_bucket * rows_per_cta
    total_slots = int(padded_sizes.sum().item())

    bucket_starts_src = torch.cumsum(counts, 0) - counts
    bucket_starts_dst = torch.cumsum(padded_sizes, 0) - padded_sizes
    bucket_ids = torch.arange(uniq_keys.numel(), device=device)

    bucket_of_slot = torch.repeat_interleave(bucket_ids, padded_sizes)
    offset_in_bucket = torch.arange(total_slots, device=device) - bucket_starts_dst[bucket_of_slot]
    slot_to_row = order[
        bucket_starts_src[bucket_of_slot]
        + torch.minimum(offset_in_bucket, counts[bucket_of_slot] - 1)
    ]

    bucket_of_sorted = torch.repeat_interleave(bucket_ids, counts)
    sorted_to_slot = bucket_starts_dst[bucket_of_sorted] + (
        torch.arange(n, device=device) - bucket_starts_src[bucket_of_sorted]
    )
    row_to_slot = torch.empty(n, dtype=torch.int64, device=device)
    row_to_slot[order] = sorted_to_slot

    alpha_cta = torch.repeat_interleave(uniq_keys.view(torch.float32), ctas_per_bucket)
    return slot_to_row, row_to_slot, alpha_cta


def _pack_w_pairs(w_norm_h: torch.Tensor, length: int, num_pairs: int) -> torch.Tensor:
    return (
        w_norm_h.T.contiguous()
        .view(length, num_pairs, 2)
        .view(torch.uint32)
        .squeeze(-1)
        .contiguous()
    )


def _run_cost_only(
    w_norm_h: torch.Tensor,
    alpha: torch.Tensor,
    widths: tuple[int, ...],
    kinds_t: torch.Tensor,
    state_bits: int,
    min_k: int,
    lb: int,
    pairs_per_cta: int,
    kind_set: int,
    num_warps: int,
) -> torch.Tensor:
    device = w_norm_h.device
    length = w_norm_h.shape[1]
    rows_per_cta = 2 * pairs_per_cta
    with device_context(device):
        slot_to_row, row_to_slot, alpha_cta = _bucket_pack_rows(alpha, rows_per_cta)
        num_ctas = alpha_cta.shape[0]
        num_pairs = num_ctas * pairs_per_cta
        w_packed = _pack_w_pairs(w_norm_h[slot_to_row], length, num_pairs)
        start_pair = torch.empty(1, dtype=torch.int32, device=device)
        out_pair = torch.empty(num_pairs, dtype=torch.uint32, device=device)
        bp_pair = torch.empty(1, dtype=torch.uint16, device=device)
        _forward_kernel[(num_ctas,)](
            w_packed, alpha_cta, start_pair, kinds_t, out_pair, bp_pair,
            num_pairs, length, 1 << lb, widths[0], 0, 0,
            L=state_bits, MIN_K=min_k, KINDS=kind_set, COST_ONLY=True, num_warps=num_warps,
        )  # fmt: skip
        return out_pair.view(torch.float16).view(-1)[row_to_slot]


def _run_walk(
    w_norm_h: torch.Tensor,
    alpha: torch.Tensor,
    overlap: torch.Tensor | None,
    widths: tuple[int, ...],
    kinds_t: torch.Tensor,
    widths_t: torch.Tensor,
    state_bits: int,
    min_k: int,
    lb: int,
    pairs_per_cta: int,
    kind_set: int,
    num_warps: int,
    write_bp_from: int,
) -> torch.Tensor:
    device = w_norm_h.device
    length = w_norm_h.shape[1]
    trace_steps = length - write_bp_from
    rows_per_cta = 2 * pairs_per_cta
    cap = 1 << (state_bits - min_k)

    with device_context(device):
        slot_to_row, row_to_slot, alpha_cta = _bucket_pack_rows(alpha, rows_per_cta)
        total_ctas = alpha_cta.shape[0]
        total_slots = total_ctas * rows_per_cta

        wave_ctas = max(1, _wave_ctas(state_bits, widths, False, device))
        max_ctas = max(1, _BACKPOINTER_BYTES // (rows_per_cta * trace_steps * cap))
        waves = max(1, max_ctas // wave_ctas)
        batch_ctas = min(total_ctas, waves * wave_ctas)
        if batch_ctas < total_ctas and batch_ctas >= 4:
            batch_ctas = (batch_ctas // 4) * 4

        slot_states = torch.empty(total_slots, trace_steps, dtype=torch.int64, device=device)
        bp_storage = torch.empty(
            trace_steps * batch_ctas * pairs_per_cta * cap, dtype=torch.uint16, device=device
        )
        slot_ids: torch.Tensor | None = None

        for c0 in range(0, total_ctas, batch_ctas):
            c1 = min(total_ctas, c0 + batch_ctas)
            nb_ctas = c1 - c0
            nb_pairs = nb_ctas * pairs_per_cta
            nb_slots = nb_ctas * rows_per_cta
            s_slice = slot_to_row[c0 * rows_per_cta : c1 * rows_per_cta]
            w_packed = _pack_w_pairs(w_norm_h[s_slice], length, nb_pairs)
            if overlap is None:
                start_pair = torch.empty(1, dtype=torch.int32, device=device)
                constrained = 0
            else:
                ov_b = overlap[s_slice].to(torch.int32).view(nb_pairs, 2)
                start_pair = (ov_b[:, 0] | (ov_b[:, 1] << 16)).contiguous()
                constrained = 1

            out_pair = torch.empty(nb_pairs, dtype=torch.uint32, device=device)
            bp_pair = bp_storage[: trace_steps * nb_pairs * cap].view(trace_steps, nb_pairs, cap)
            _forward_kernel[(nb_ctas,)](
                w_packed, alpha_cta[c0:c1], start_pair, kinds_t, out_pair, bp_pair,
                nb_pairs, length, 1 << lb, widths[0], constrained, write_bp_from,
                L=state_bits, MIN_K=min_k, KINDS=kind_set, COST_ONLY=False, num_warps=num_warps,
            )  # fmt: skip
            j_end = (
                out_pair.view(torch.uint16).view(-1).to(torch.int64).contiguous()
                if overlap is None
                else overlap[s_slice].contiguous()
            )
            if slot_ids is None or slot_ids.numel() != nb_slots:
                slot_ids = torch.arange(nb_slots, dtype=torch.int64, device=device)
            states_t = torch.empty(trace_steps, nb_slots, dtype=torch.int64, device=device)
            _traceback_kernel[(triton.cdiv(nb_slots, _TRACEBACK_SEQUENCES),)](
                bp_pair, slot_ids, j_end, widths_t, states_t, nb_slots, nb_pairs,
                length=length, trace_steps=trace_steps, state_bits=state_bits,
                width=cap, lb=lb, block=_TRACEBACK_SEQUENCES, num_warps=2,
            )  # fmt: skip
            slot_states[c0 * rows_per_cta : c1 * rows_per_cta] = states_t.T
        return slot_states[row_to_slot]


def run_viterbi(
    state_bits: int,
    step_widths: Sequence[int],
    w_norm_h: torch.Tensor,
    alpha: torch.Tensor,
    overlap: torch.Tensor | None = None,
    *,
    cost_only: bool = False,
    write_bp_from: int = 0,
) -> torch.Tensor:
    """Run FP16x2 Viterbi over ``w_norm_h [N, T]`` (float16) and folded ``alpha [N]`` (float32).

    With ``cost_only=True``, returns minimum final costs ``[N]`` (float16) without writing
    backpointers. Otherwise traces back steps ``write_bp_from..T-1`` and returns states
    ``[N, T - write_bp_from]`` (int64).
    """
    if cost_only and overlap is not None:
        raise ValueError("overlap is not supported with cost_only=True")
    n, length = w_norm_h.shape
    device = w_norm_h.device
    trace_steps = length - write_bp_from
    if n == 0:
        if cost_only:
            return torch.empty(0, dtype=torch.float16, device=device)
        return torch.empty(0, trace_steps, dtype=torch.int64, device=device)

    widths = tuple(step_widths)
    min_k = min(widths)
    threads, lb, pairs_per_cta = _geometry(state_bits, min_k)
    spec = _specialization(state_bits, widths, cost_only)
    num_warps = threads // 32
    kinds_t = _kinds_tensor(widths, device)
    if cost_only:
        return _run_cost_only(
            w_norm_h, alpha, widths, kinds_t, state_bits, min_k, lb, pairs_per_cta,
            spec[2], num_warps,
        )  # fmt: skip
    return _run_walk(
        w_norm_h, alpha, overlap, widths, kinds_t, _widths_tensor(widths, device),
        state_bits, min_k, lb, pairs_per_cta, spec[2], num_warps, write_bp_from,
    )  # fmt: skip


__all__ = ["FP16_SCALE", "SHARED_NEED", "run_viterbi", "supports_device"]
