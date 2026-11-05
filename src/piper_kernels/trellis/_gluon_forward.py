"""Persistent Viterbi forward: every step of a search batch in one Gluon launch.

The step loop (``_viterbi._step_kernel``) runs each search as one Triton launch per
position, so every step reads and writes all rows' ``2^(L-k)`` prefix costs through
global memory. ``run_forward`` runs all ``T`` steps of a batch in one launch: a CTA
keeps its rows' prefix costs in shared memory, computes the 1MAD hash in registers
and decodes each state once for all of its rows when they share ``p1``. Final costs
and backpointers are identical to the step loop's and in the layouts the loop and
the traceback use. The kernel is written in Gluon, Triton's explicit-layout dialect,
with inline PTX. It runs on NVIDIA GPUs from sm_80 with Triton 3.9 and the shared
memory of its largest kernel (``supports_device``); other devices take the step loop.

Scope: the 1MAD code; L = 12 or 16 with any schedule of one step width of 3 to 6
bits or of two adjacent ones, which covers every layout of the format, its L = 12
search and the windows of either; float and INT8 decodes; free and overlap starts;
one ``p1`` per row for all ``T`` steps. The backpointers take
``T · rows · 2^(L - min k)`` bytes; a launch takes about
``ceil(rows / compute_wave_rows(...))`` waves, so the caller sizes batches in whole
waves.

Design: one CTA runs ``compute_rows_per_cta`` rows (6, or 3 at L = 16 with 3-bit steps, whose
32 KiB of prefix costs per row allow no more) through all ``T`` steps with 256
threads at L = 16 and 64 at L = 12 (``_GEOMETRY``; with 3-bit steps 512 at L = 16,
``_geometry``). Step ``t`` maps the prefix ``M[v]``, ``v`` the low ``L - k_t`` bits of
the previous state, to ``M'[j']``, ``j'`` the low ``L - k_{t+1}`` bits of
``s_t = (a, j')``: the minimum over the ``k_{t+1}`` dropped bits ``a`` of
``M[s_t >> k_t] + (w_t - value(s_t))²``. With the group ``g = j' >> k_t``,
``M[s_t >> k_t] = M[(a, g)]``: the outputs of one group share the same
``2^k_{t+1}`` inputs.

- Tensors are ``[threads]`` with one element per thread (``_thread_layout``), so a
  Gluon tensor is a per-thread scalar: per step, thread ``(part, q, mid)`` loads its
  group's 16 inputs ``(a, g)`` per row into registers, then loops over its
  ``2^loop_bits`` outputs ``j'`` of that group (16 at L = 16, each output's numerators
  computed during the previous one; 4 at L = 12 and 8 with 3-bit steps, without that
  prefetch). The ``2^(k_{t+1} - 4)`` lanes of ``part`` split an output's values of
  ``a``, 16 each; at ``k_{t+1} = 3`` an iteration takes the 8 of each of two outputs,
  ``j'`` and ``j' + 2^LOW`` in two groups (``PAIR``: adjacent stored positions, so one
  16-bit backpointer store), so every thread does the same 16 states per iteration at
  every step width. A thread's outputs share their group, so with 3-bit steps the
  loop bits drop to 3 and the threads double. Consecutive lanes take consecutive top
  ``k_{t+2}`` bits of ``j'`` (the stored position's lowest bits), so a warp's
  backpointer stores are contiguous runs. Per-row and per-``a`` arrays are tuples of
  tensors, unrolled at trace time. Each step kind ``(k_t, k_{t+1}, k_{t+2})`` of the
  schedule (relative to its smallest step, ``MIN_K``) is a constexpr specialization of
  ``_step``, chosen per step at run time.
- Decode, per CTA (``_order`` sorts rows by bitwise ``p1``; a CTA tests whether its
  rows share one value and runs one of two specializations of its loop over the
  steps, ``_run``): each state's 1MAD numerator comes from a ``dp4a`` of its hash;
  with one ``p1`` the CTA decodes it once for all rows, else each row decodes its
  own. Float mode computes the code value ``n / D`` by a 3-op Newton form exact for
  all 1021 numerators, then multiplies by ``p1``. The arithmetic is pinned (launched
  with ``enable_fp_fusion=False``): ``v = fl(c · p1)`` (INT8: then rint and clamp),
  ``d = fma(v, -p2, w)``, ``cost = fma(d, d, prefix)``, the step loop's arithmetic.
- Costs are non-negative and never NaN for finite inputs: prefixes start at +0 or
  +inf, and ``fma(d, d, prefix)`` adds the exact, non-negative ``d · d`` to a
  non-negative prefix (so never -0). The minimum's bits are therefore those of an
  ascending scan however the comparisons are grouped, and costs order like their
  bits as unsigned integers (``_argmin``, ``_owner``).
- Shared memory: the rows' prefix costs (``2^(L - min k)`` each) in an identity
  layout indexed by the swizzled position ``phys`` (``_IMAGES``), computed in
  registers; loads are native ``gather``s.
- Outputs: backpointers ``bp[t, row, pos]`` at
  ``pos = _store_index(j', L - k_{t+1}, k_{t+2})`` and the final costs at
  ``_store_index(v, L - k_0, k_1)``.

Gluon 3.9 workarounds, each measured necessary: prefix stores are ``st.shared``
inline PTX (Gluon's barrier analysis puts a ``bar.sync`` between any two native
stores to one allocation it cannot prove disjoint, here between every output's
stores; the two ``gl.barrier`` calls per step separate a step's loads from its
stores and its stores from the next step's loads); compares and selects are inline
PTX (``_argmin``: LLVM's SLP vectorizer packs them into bit vectors); a per-step
``%tid.x`` read keeps LICM from hoisting every step kind's thread mapping
(``_thread_index``); trip counts and ``k_0``, ``k_1`` and the start mode are run-time
arguments (a constant 16-trip output loop unrolls into about 130 KB of code; the rest
would multiply the specializations); shared allocations are powers of two (a CTA's
prefixes take two buffers: 4 rows and 2, or 2 and 1).

Kernels compile per (L, mode, smallest step, set of step kinds) on first use
(``compute_wave_rows`` compiles the one it measures) and are cached by Triton.
"""

# Kernel constexprs are short upper-case names (KE = k_t, KA = k_{t+1}, ...). Gluon's
# frontend has no starred unpacking, so per-row tuples grow by concatenation. Gluon
# device types and layouts are not Python runtime values.
# ruff: noqa: N803, N806, RUF005, ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
# Static ranges unroll at trace time, so their first iteration binds what later ones read.
# pyright: reportPossiblyUnboundVariable=false

from __future__ import annotations

import functools
import re
from collections.abc import Sequence

import torch
import triton
from triton.compiler import CompiledKernel
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._layout import LCG_INCREMENT, LCG_MULTIPLIER

STATE_BITS = (12, 16)
STEP_BITS = (3, 4, 5, 6)
"""Step widths ``k_t``; a schedule takes one of them or two adjacent ones (``supports_trellis``)."""
MAX_LENGTH = 256
"""Longest sequence: the format's largest block."""
ROWS = 6
"""Rows per CTA where their prefix costs fit (``compute_rows_per_cta``): at L = 16 with 4-bit
steps they take 6 · 16 KiB of the 99 KiB a CTA may use on sm_86 to sm_120, so one CTA runs per
SM."""
SHARED_LIMIT = 99 * 1024
"""Opt-in shared memory per CTA on sm_86 to sm_120, which a CTA's prefix costs must fit."""
_RESERVED_SHARED = 1024
"""Shared memory the driver reserves per CTA from sm_80 on (``_resident_ctas``)."""

_GEOMETRY = {16: (256, 4), 12: (64, 2)}
"""Threads per CTA and loop bits per state width for steps of 4 bits and more (``_geometry``): a
thread's ``2^loop_bits`` outputs per step times 16 states each times the threads cover the
``2^L`` states."""


def _rows(L: int, min_k: int) -> int:
    """Rows per CTA (``compute_rows_per_cta``)."""
    return ROWS if 4 * ROWS << (L - min_k) <= SHARED_LIMIT else ROWS // 2


SHARED_NEED = max(4 * _rows(L, k) << (L - k) for L in STATE_BITS for k in STEP_BITS)
"""Shared memory per CTA of the largest kernel (L = 16: 96 KiB of prefix costs at 4 bits, and at 3
bits with 3 rows): a GPU that cannot opt in to it takes the step loop (``supports_device``)."""


@gluon.constexpr_function
def _geometry(L: int, min_k: int) -> tuple[int, int]:
    """Threads per CTA and loop bits of a kernel whose smallest step is ``min_k``: ``_GEOMETRY``'s,
    except that a thread's outputs share their group ``g = j' >> k_t`` (``_step``), so with 3-bit
    steps the loop bits drop to 3 and the threads double (L = 16). Keeping the threads and giving
    each two sets of outputs instead (3 rows: the 4-bit kernels' 96 inputs per thread) ran 1.4-10%
    slower: 16 warps per SM hide more latency than twice the work per thread and step."""
    threads, bits = _GEOMETRY[L]
    return (threads << (bits - min_k), min_k) if bits > min_k else (threads, bits)


# A Gluon kernel reads only constexpr globals.
_ONE_MAD_A = gl.constexpr(LCG_MULTIPLIER)
_ONE_MAD_B = gl.constexpr(LCG_INCREMENT)
_ROUND_MAGIC = gl.constexpr(12582912.0)  # 1.5 * 2^23
_IDENTITY = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))
_U32 = 0xFFFFFFFF
_IMAGES = {
    4: {
        6: (1,),
        7: (1, 4),
        8: (1, 2, 12),
        10: (3, 1, 4, 8, 19),
        11: (1, 2, 16, 4, 8, 19),
        12: (2, 4, 9, 1, 2, 4, 25),
    },
    3: {
        8: (1, 2, 24),
        9: (20, 16, 1, 2),
        12: (6, 12, 20, 17, 1, 4, 2),
        13: (3, 8, 4, 24, 18, 3, 12, 6),
    },
}
"""Swizzle images of a ``W``-bit prefix index's bits 5 and up (``_phys``), per swizzle family
``min(MIN_K, 4)``. ``phys(v)`` is ``v`` XOR the images of its set bits 5 and up: linear over GF(2)
and the identity on bits 5 and up, so a bijection. Every load and store instruction of a step is
conflict free when its distinct lane addresses span bits whose bank images (the low 5 bits) are
linearly independent; each family's thread mapping (``_step``, ``_geometry``) needs its own
images for that, found by search over every step kind of the family at L = 12 and 16 and checked
by the forward tests. Family 4 serves the 4/5 and 5/6 kernels, family 3 the 3/4 kernels."""


@gluon.constexpr_function
def _phys(v: int, width: int, family: int) -> int:
    """``v`` XOR the images of its set bits 5 and up (``_IMAGES``)."""
    low = 0
    for bit in range(5, width):
        if v >> bit & 1:
            low ^= _IMAGES[family][width][bit - 5]
    return v ^ low


_IN_OFFSETS = {
    (family, width, group, select): tuple(
        _phys((i & 7) << group | (i >> 3) << select if select >= 0 else i << group, width, family)
        for i in range(16)
    )
    for family, images in _IMAGES.items()
    for width in images
    for group in range(width - 3 + 1)
    for select in range(-1, group)
    if select >= 0 or group <= width - 4
}


@gluon.constexpr_function
def _family(min_k):
    """The swizzle family of a kernel whose smallest step is ``min_k`` (``_IMAGES``)."""
    return min(min_k, 4)


@gluon.constexpr_function
def _images(width, family):
    return _IMAGES[family][width]


@gluon.constexpr_function
def _in_offsets(width, group, select, family):
    """Swizzled positions of a thread's 16 inputs of a ``width``-bit prefix with ``group`` bits of
    ``g``, relative to its first group ``g0`` (``phys`` is linear: input ``(a, g)`` is
    ``phys(g0) ^`` this): ``(a = i, g0)``, or with two groups (``select``: the bit of ``g`` they
    differ in, else -1) ``(a = i mod 8, g0 + (i // 8 << select))``."""
    return _IN_OFFSETS[family, width, group, select]


@gluon.constexpr_function
def _threads(L, min_k):
    return _geometry(L, min_k)[0]


@gluon.constexpr_function
def _loop_bits(L, min_k):
    return _geometry(L, min_k)[1]


@gluon.constexpr_function
def _parts(ka):
    """Lanes per output at a ``ka``-bit step ``k_{t+1}``: 16 candidates each, or one of 8 at 3."""
    return 1 << max(ka - 4, 0)


@gluon.constexpr_function
def _low_rows(rows):
    """The rows in the first prefix buffer (``_row_buffer``): the largest power of two of them."""
    return 1 << (rows.bit_length() - 1)


@gluon.constexpr_function
def _row_offset(r, rows, cap):
    """The byte offset of row ``r``'s prefix costs in its buffer (``_row_buffer``)."""
    low = _low_rows(rows)
    return (r - low if r >= low else r) * cap * 4


@gluon.constexpr_function
def _thread_layout(threads):
    """One element per thread: a ``[threads]`` tensor is a per-thread scalar."""
    return gl.BlockedLayout([1], [32], [threads // 32], [0])


@gluon.constexpr_function
def _warp_lane_layout(threads):
    """``_thread_layout`` viewed as ``[warp, lane]``: a gather along the lane axis is a shuffle."""
    return gl.BlockedLayout([1, 1], [1, 32], [threads // 32, 1], [1, 0])


@gluon.constexpr_function
def _state_steps(out_bits, pair_bit):
    """1MAD LCG increments of a thread's 16 states over ``(0, j')``, ``out_bits = L - k_{t+1}``:
    ``(a = i, j')``, ``A · (i << out_bits)`` mod ``2^32``, or with two outputs per iteration
    (``pair_bit``: the bit of ``j'`` they differ in, else -1) ``(a = i mod 8, j' + (i // 8 <<
    pair_bit))``."""
    return tuple(
        _ONE_MAD_A.value
        * ((i & 7) << out_bits | (i >> 3) << pair_bit if pair_bit >= 0 else i << out_bits)
        & _U32
        for i in range(16)
    )


@gluon.constexpr_function
def _store_asm(offset, predicated):
    store = f"st.shared.f32 [$0+{offset}], $1;"
    return "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p " + store + " }" if predicated else store


@gluon.constexpr_function
def _scan_asm(i):
    """PTX of one pair-then-scan step at pair ``(i, i + 1)`` (the candidates' indices): ``$0, $1``
    the new minimum and argument from ``$2, $3`` (the running ones) and the pair's costs ``$4,
    $5``."""
    return (
        "{ .reg .pred p, q; .reg .f32 m; .reg .b32 a; "
        "setp.lt.f32 p, $5, $4; selp.f32 m, $5, $4, p; "
        f"selp.b32 a, {i + 1}, {i}, p; "
        "setp.lt.f32 q, m, $2; selp.f32 $0, m, $2, q; selp.b32 $1, a, $3, q; }"
    )


@gluon.jit
def _phys_of(v, width: gl.constexpr, MIN_K: gl.constexpr):
    """``_phys`` of a tensor of ``width``-bit prefix indices in a kernel whose smallest step is
    ``MIN_K``."""
    images: gl.constexpr = _images(width, _family(MIN_K))
    low = gl.zeros_like(v)
    for bit in gl.static_range(5, width):
        low ^= ((v >> bit) & 1) * images[bit - 5]
    return v ^ low


@gluon.jit
def _thread_index(THREADS: gl.constexpr):
    """``%tid.x`` per element of a ``_thread_layout`` tensor (``gl.arange(0, THREADS)``), read so
    that LLVM cannot hoist what depends on it out of ``_run``'s loop over the steps: hoisted,
    every step kind's thread mapping stays live across the whole loop (VR: 255 registers; all 8
    kinds: 72 spills, 87% of the one-kind speed). The mask restores the range ``arange`` implies,
    which lets LLVM drop the gathers' bounds masks and fold each row's offset into the load."""
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
    """The shared-memory byte address of ``buffer``'s first element (a uniform scalar)."""
    return gl.inline_asm("mov.u32 $0, $1;", ("=r", "r"), [buffer], gl.int32, is_pure=False)


@gluon.jit
def _store_shared(address, value, offset: gl.constexpr):
    """``st.shared.f32 [address + offset], value`` per thread, outside the barrier analysis."""
    gl.inline_asm(_store_asm(offset, False), ("r", "f"), [address, value], (), is_pure=False)


@gluon.jit
def _store_shared_if(address, value, keep, offset: gl.constexpr):
    """``_store_shared`` where ``keep`` (int32) is nonzero."""
    gl.inline_asm(
        _store_asm(offset, True), ("r", "f", "r"), [address, value, keep], (), is_pure=False
    )


@gluon.jit
def _numerator(x):
    """A state's 1MAD numerator ``n`` (byte sum of ``x`` - 510) as FP32: ``dp4a`` puts the byte
    sum in the mantissa of 1.5 · 2^23, and subtracting it is exact."""
    return gl.inline_asm_elementwise(
        "{ .reg .b32 a; dp4a.u32.u32 a, $1, 16843009, 1262484994; add.rn.f32 $0, a, 0fCB400000; }",
        "=f,r",
        [x],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _numerators(x_e, out_bits: gl.constexpr, PAIR_BIT: gl.constexpr):
    """``_numerator`` of a thread's 16 states (``_state_steps``) from ``x_e``, the 1MAD input of
    its first."""
    steps: gl.constexpr = _state_steps(out_bits, PAIR_BIT)
    out = ()
    for i in gl.static_range(16):
        out = out + (_numerator(x_e + steps[i]),)
    return out


@gluon.jit
def _code_value(n):
    """``fp32(n / 147.800537109375)``, correctly rounded for every numerator: the Newton correction
    ``fma(fma(-q, D, n), R, q)`` of ``q = fl(n · R)``, ``R = fp32(1 / D)`` (``q · (-D)`` is
    ``-q · D`` exactly)."""
    return gl.inline_asm_elementwise(
        "{ .reg .f32 q, r; mul.rn.f32 q, $1, 0f3BDDB447; fma.rn.f32 r, q, 0fC313CCF0, $1; "
        "fma.rn.f32 $0, r, 0f3BDDB447, q; }",
        "=f,f",
        [n],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _decode_shared(alpha, n):
    """``clamp(rint(alpha · n), ±127)`` (the format's INT8 decode) for a value every row shares:
    ``rint`` issues on its own pipe."""
    rounded = gl.inline_asm_elementwise(
        "cvt.rni.f32.f32 $0, $1;", "=f,f", [alpha * n], dtype=gl.float32, is_pure=True, pack=1
    )
    return gl.minimum(gl.maximum(rounded, -127.0), 127.0)


@gluon.jit
def _decode_row(alpha, n):
    """The same per row: ``rint(clamp(alpha · n))`` by the FP32 add and subtract of 1.5 · 2^23,
    at full rate on the FMA pipe; it differs only in the sign of zero, which ``d · d`` discards."""
    clamped = gl.minimum(gl.maximum(alpha * n, -127.0), 127.0)
    return (clamped + _ROUND_MAGIC) - _ROUND_MAGIC


@gluon.jit
def _shared_values(n, p1, INT8: gl.constexpr, SHARED: gl.constexpr):
    """What the CTA's rows share of the 16 states' decodes, from their numerators ``n``: their
    values at the one ``p1`` (``SHARED``), else the numerators (INT8) or code values (float) that
    each row decodes (``_row_values``)."""
    c = n
    if not INT8:
        c = ()
        for i in gl.static_range(16):
            c = c + (_code_value(n[i]),)
    if SHARED:
        out = ()
        for i in gl.static_range(16):
            out = out + (_decode_shared(p1, c[i]) if INT8 else p1 * c[i],)
    else:
        out = c
    return out


@gluon.jit
def _row_values(shared, p1, INT8: gl.constexpr, SHARED: gl.constexpr):
    """One row's decoded values of the 16 states from ``_shared_values`` (``p1``: the row's)."""
    if SHARED:
        out = shared
    else:
        out = ()
        for i in gl.static_range(16):
            out = out + (_decode_row(p1, shared[i]) if INT8 else p1 * shared[i],)
    return out


@gluon.jit
def _argmin(value, ins, r: gl.constexpr, minus_p2, w, FIRST: gl.constexpr, COUNT: gl.constexpr):
    """Row ``r``'s minimum over the ``COUNT`` candidates ``c`` of ``ins[r, i] + (w - value[i] ·
    p2)²``, ``i = FIRST + c``, and its lowest minimizing ``c``: pairs ``(c, c + 1)``, then a strict
    less-than scan over the pair minima in ascending ``c``, which finds the same minimum and lowest
    minimizer as one ascending scan (the pair holding the lowest minimizer has it as its minimum,
    and every earlier pair is larger). The compares and selects are PTX: written as ``gl.where``,
    LLVM's SLP vectorizer (on for every target but sm_80, with no per-kernel switch) packs the six
    rows' compares into ``<32 x i1>`` vectors, which NVPTX keeps as bits of a register (693 instead
    of 521 instructions per output, 74% of the speed)."""
    for c in gl.static_range(0, COUNT, 2):
        d0 = gl.fma(value[FIRST + c], minus_p2, w)
        d1 = gl.fma(value[FIRST + c + 1], minus_p2, w)
        t0 = gl.fma(d0, d0, ins[r * 16 + FIRST + c])
        t1 = gl.fma(d1, d1, ins[r * 16 + FIRST + c + 1])
        if c == 0:
            best, arg = gl.inline_asm_elementwise(
                "{ .reg .pred p; setp.lt.f32 p, $3, $2; selp.f32 $0, $3, $2, p; "
                "selp.b32 $1, 1, 0, p; }",
                "=f,=r,f,f",
                [t0, t1],
                dtype=(gl.float32, gl.int32),
                is_pure=True,
                pack=1,
            )
        else:
            best, arg = gl.inline_asm_elementwise(
                _scan_asm(c),
                "=f,=r,f,r,f,f",
                [best, arg, t0, t1],
                dtype=(gl.float32, gl.int32),
                is_pure=True,
                pack=1,
            )
    return best, arg


@gluon.jit
def _partner(x, lane_partner, THREADS: gl.constexpr):
    """``x`` of lane ``lane_partner`` (``[warps, 32]``) of each lane's warp (one warp shuffle)."""
    pairs = gl.reshape(x, [THREADS // 32, 32])
    pairs = gl.convert_layout(pairs, _warp_lane_layout(THREADS), assert_trivial=True)
    swapped = gl.reshape(gl.gather(pairs, lane_partner, 1), [THREADS])
    return gl.convert_layout(swapped, _thread_layout(THREADS), assert_trivial=True)


@gluon.jit
def _owner(best, part, partners, THREADS: gl.constexpr, PARTS: gl.constexpr):
    """Whether this lane stores its lane group's minimum (int32): the ``PARTS`` lanes of an output
    hold ascending sixteens of ``a``, and the lane holding the minimum with the lowest ``a`` on ties
    stores it. Costs order like their bits, so per butterfly level ``bits(other) >= bits(mine) +
    upper`` (``upper``: this lane is the level's upper half) says "this lane holds its half's
    minimum" in both lanes of the level's pair, exactly one lane of the group passes every level,
    and only costs are exchanged: one shuffle (``partners[0]``, lane ^ 1) for 2 parts, two (then
    ``partners[1]``, lane ^ 2, on the pair minima) for 4."""
    bits = best.to(gl.uint32, bitcast=True)
    other = _partner(bits, partners[0], THREADS)
    owner = other >= bits + (part % 2).to(gl.uint32)
    if PARTS == 4:
        pair = gl.minimum(bits, other)
        other_pair = _partner(pair, partners[1], THREADS)
        owner = owner & (other_pair >= pair + (part // 2).to(gl.uint32))
    return owner.to(gl.int32)


@gluon.jit
def _row_buffer(low, high, r: gl.constexpr, CAP: gl.constexpr, ROWS: gl.constexpr):
    """Row ``r``'s prefix buffer and its first element's index there: the first ``_low_rows``
    rows share ``low``, the rest ``high`` (Gluon allocations are powers of two: 6 rows are 4 and
    2, 3 rows 2 and 1)."""
    LOW_ROWS: gl.constexpr = _low_rows(ROWS)
    if r < LOW_ROWS:
        return low, r * CAP
    else:
        return high, (r - LOW_ROWS) * CAP


@gluon.jit
def _output(stored, offsets, n, ins, w, bp, low_address, high_address, part, partners, p1,
            minus_p2, THREADS: gl.constexpr, CAP: gl.constexpr, ROWS: gl.constexpr,
            PARTS: gl.constexpr, PAIR: gl.constexpr, INT8: gl.constexpr,
            SHARED: gl.constexpr):  # fmt: skip
    """One iteration's outputs of every row from its 16 states' numerators ``n`` (one output, or
    a ``PAIR`` of 8 states each at adjacent stored positions): the minimum over ``a`` and its
    backpointer, stored. ``offsets``: per output, its byte offset in a row's prefix buffer;
    ``stored``: the (first) output's stored position past the thread's first. A pair's
    backpointers are one 16-bit store (the first output's position is even)."""
    shared = _shared_values(n, p1[0], INT8, SHARED)
    LOW_ROWS: gl.constexpr = _low_rows(ROWS)
    for r in gl.static_range(ROWS):
        value = _row_values(shared, p1[r], INT8, SHARED)
        base = low_address if r < LOW_ROWS else high_address
        if PAIR:
            best0, arg0 = _argmin(value, ins, r, minus_p2[r], w[r], 0, 8)
            best1, arg1 = _argmin(value, ins, r, minus_p2[r], w[r], 8, 8)
            _store_shared(base + offsets[0], best0, _row_offset(r, ROWS, CAP))
            _store_shared(base + offsets[1], best1, _row_offset(r, ROWS, CAP))
            pair = bp[r] + stored
            pair = pair.to(gl.pointer_type(gl.uint16), bitcast=True)
            gl.store(pair, (arg0 | (arg1 << 8)).to(gl.uint16))
        else:
            best, arg = _argmin(value, ins, r, minus_p2[r], w[r], 0, 16)
            if PARTS == 1:
                _store_shared(base + offsets[0], best, _row_offset(r, ROWS, CAP))
                gl.store(bp[r] + stored, arg.to(gl.uint8))
            else:
                owner = _owner(best, part, partners, THREADS, PARTS)
                _store_shared_if(base + offsets[0], best, owner, _row_offset(r, ROWS, CAP))
                arg = arg + 16 * part
                gl.store(bp[r] + stored, arg.to(gl.uint8), mask=owner != 0)


@gluon.jit
def _stored(e, LOW: gl.constexpr, LB: gl.constexpr, KAP: gl.constexpr):
    """The stored position of a thread's output ``j0 + e`` past that of ``j0``
    (``_store_index``): ``e << KAP`` when ``e`` lies in the ``LOW`` bits below ``j'``'s top
    ``KAP``, else its bits from ``LOW`` up are the low bits of the top ``KAP`` (``_step``)."""
    return e << KAP if LOW >= LB else ((e & ((1 << LOW) - 1)) << KAP) | (e >> LOW)


@gluon.jit
def _step(t, low, high, low_address, high_address, row, p1, minus_p2, w_ptr, bp_ptr, rows, outputs,
          L: gl.constexpr, CAP: gl.constexpr, ROWS: gl.constexpr, INT8: gl.constexpr,
          SHARED: gl.constexpr, MIN_K: gl.constexpr, KE: gl.constexpr, KA: gl.constexpr,
          KAP: gl.constexpr):  # fmt: skip
    """Step ``t`` for the CTA's rows, ``KE = k_t``, ``KA = k_{t+1}``, ``KAP = k_{t+2}``. Thread
    mapping, fastest first: ``part`` (which sixteen of the ``2^KA`` values of ``a``: ``PARTS``
    lanes share an output), ``q`` (the bits of ``j'`` from ``QB`` up), ``mid`` (the bits of ``j'``
    between ``QS`` and the thread's loop bits ``e``, its low ``LB``). ``QS`` is ``LOW``, so that
    ``q`` is the top ``KAP`` bits of ``j'`` (the stored position's lowest bits: a warp's
    backpointer stores are contiguous runs), unless ``LB`` is larger (L = 12 with 6-bit steps):
    then ``e`` takes the low bits of the top ``KAP`` and ``q`` the rest of them. ``QB`` is ``QS``,
    except with a ``PAIR`` of outputs per iteration (``KA = 3``), whose second output takes ``j'``
    bit ``LOW = QS``, a bit of the group above ``KE`` and bit 0 of the stored position."""
    THREADS: gl.constexpr = _threads(L, MIN_K)
    LB: gl.constexpr = _loop_bits(L, MIN_K)
    WARP_LANE: gl.constexpr = _warp_lane_layout(THREADS)
    GROUP: gl.constexpr = L - KE - KA  # bits of g = j' >> KE
    LOW: gl.constexpr = L - KA - KAP  # bits of j' below its top KAP
    QS: gl.constexpr = LOW if LOW >= LB else LB
    PAIR: gl.constexpr = KA == 3  # two outputs per iteration, j0 + e and j0 + e + 2^LOW
    PAIR_BIT: gl.constexpr = LOW if PAIR else -1
    QB: gl.constexpr = QS + 1 if PAIR else QS  # q's lowest j' bit
    PARTS: gl.constexpr = _parts(KA)
    IN_BITS: gl.constexpr = L - KE
    OUT_BITS: gl.constexpr = L - KA
    offsets: gl.constexpr = _in_offsets(IN_BITS, GROUP, PAIR_BIT - KE if PAIR else -1,
                                        _family(MIN_K))  # fmt: skip
    tid = _thread_index(THREADS)
    part = tid % PARTS
    q = (tid // PARTS) & ((1 << (OUT_BITS - QB)) - 1)
    mid = (tid // PARTS) >> (OUT_BITS - QB)
    j0 = (q << QB) | (mid << LB)  # the thread's outputs: j0 + e, e < 2^LB
    pos = ((j0 & ((1 << LOW) - 1)) << KAP) | (q << (QB - LOW))  # stored position of j0

    w = ()
    bp = ()
    for r in gl.static_range(ROWS):
        w = w + (gl.load(w_ptr + t * rows + row[r]),)
        bp = bp + (bp_ptr + (t * rows + row[r]).to(gl.int64) * CAP + pos,)
    in_base = _phys_of((part << (GROUP + 4)) | (j0 >> KE), IN_BITS, MIN_K)  # (16 part + i, g)
    ins = ()
    for r in gl.static_range(ROWS):
        buffer, first = _row_buffer(low, high, r, CAP, ROWS)
        for i in gl.static_range(16):
            ins = ins + (buffer.gather(first + (in_base ^ offsets[i]), 0),)
    gl.barrier()

    out_base = _phys_of(j0, OUT_BITS, MIN_K)  # j0 + e is out_base ^ e: e lies in the identity bits
    x_base = (j0 + ((16 * part) << OUT_BITS)).to(gl.uint32) * _ONE_MAD_A + _ONE_MAD_B
    lanes = gl.arange(0, 32, gl.SliceLayout(0, WARP_LANE))[None, :]
    lanes = lanes + gl.zeros([THREADS // 32, 1], gl.int32, WARP_LANE)
    partners = (lanes ^ 1, lanes ^ 2)
    # A pair's second output j0 + e + 2^LOW: j' bit LOW, clear in j0 + e, is a bit of the group
    # (LOW >= KE) and bit 0 of the stored position.
    second_offset: gl.constexpr = 4 * _phys(1 << LOW, OUT_BITS, _family(MIN_K))
    if LB >= 4:
        # Each output's numerators are computed during the previous output (one extra set per
        # step); with 4 or 8 outputs per step (L = 12; 3-bit steps) the extra set costs more than
        # the latency it hides.
        nxt = _numerators(x_base, OUT_BITS, PAIR_BIT)
        for e in range(outputs):
            n = nxt
            x_next = x_base + _ONE_MAD_A * (e + 1).to(gl.uint32)
            nxt = _numerators(x_next, OUT_BITS, PAIR_BIT)
            offset = 4 * (out_base ^ e)
            _output(
                _stored(e, LOW, LB, KAP), (offset, offset ^ second_offset), n, ins, w, bp,
                low_address, high_address, part, partners, p1, minus_p2, THREADS, CAP, ROWS,
                PARTS, PAIR, INT8, SHARED,
            )  # fmt: skip
    else:
        for e in range(outputs):
            n = _numerators(x_base + _ONE_MAD_A * e.to(gl.uint32), OUT_BITS, PAIR_BIT)
            offset = 4 * (out_base ^ e)
            _output(
                _stored(e, LOW, LB, KAP), (offset, offset ^ second_offset), n, ins, w, bp,
                low_address, high_address, part, partners, p1, minus_p2, THREADS, CAP, ROWS,
                PARTS, PAIR, INT8, SHARED,
            )  # fmt: skip
    gl.barrier()


@gluon.jit
def _run(low, high, low_address, high_address, row, p1, minus_p2, w_ptr, bp_ptr, kinds_ptr, rows,
         length, outputs, L: gl.constexpr, CAP: gl.constexpr, ROWS: gl.constexpr,
         INT8: gl.constexpr, MIN_K: gl.constexpr, KINDS: gl.constexpr,
         SHARED: gl.constexpr):  # fmt: skip
    """Every step for a CTA whose rows share one ``p1`` (``SHARED``) or not: step ``t``'s kind
    ``kinds[t]`` selects its specialization."""
    for t in range(length):
        kind = gl.load(kinds_ptr + t)
        for k in gl.static_range(8):
            if (KINDS >> k) & 1:  # noqa: SIM102 (compile time: only the schedule's kinds)
                if kind == k:
                    _step(
                        t, low, high, low_address, high_address, row, p1, minus_p2, w_ptr, bp_ptr,
                        rows, outputs, L, CAP, ROWS, INT8, SHARED, MIN_K, MIN_K + (k >> 2),
                        MIN_K + ((k >> 1) & 1), MIN_K + (k & 1),
                    )  # fmt: skip


@gluon.jit
def _initial(low_address, high_address, row, start_ptr, mod, constrained, THREADS: gl.constexpr,
             CAP: gl.constexpr, ROWS: gl.constexpr, MIN_K: gl.constexpr,
             WIDTH0: gl.constexpr):  # fmt: skip
    """Initial prefixes over ``v``, the low ``WIDTH0 = L - k_0`` bits of the last state: free (all
    zero) or constrained to ``v = start`` (an out-of-range start leaves every cost infinite)."""
    tid = gl.arange(0, THREADS, _thread_layout(THREADS))
    LOW_ROWS: gl.constexpr = _low_rows(ROWS)
    for r in gl.static_range(ROWS):
        start = gl.load(start_ptr + row[r] % mod, mask=constrained != 0, other=-1)
        for i in gl.static_range((1 << WIDTH0) // THREADS):
            v = i * THREADS + tid
            cost = gl.where((v == start) | (constrained == 0), 0.0, float("inf"))
            index = _phys_of(v, WIDTH0, MIN_K)
            address = (low_address if r < LOW_ROWS else high_address) + 4 * index
            _store_shared(address, cost, _row_offset(r, ROWS, CAP))


@gluon.jit
def _final(low, high, row, m_out_ptr, k1, THREADS: gl.constexpr, CAP: gl.constexpr,
           ROWS: gl.constexpr, MIN_K: gl.constexpr, WIDTH0: gl.constexpr):  # fmt: skip
    """Final costs at their stored positions ``pos = (low L - k_0 - k_1 bits of v, top k_1
    bits)``."""
    tid = gl.arange(0, THREADS, _thread_layout(THREADS))
    for r in gl.static_range(ROWS):
        buffer, first = _row_buffer(low, high, r, CAP, ROWS)
        for i in gl.static_range((1 << WIDTH0) // THREADS):
            stored = i * THREADS + tid
            v = ((stored & ((1 << k1) - 1)) << (WIDTH0 - k1)) | (stored >> k1)
            cost = buffer.gather(first + _phys_of(v, WIDTH0, MIN_K), 0)
            gl.store(m_out_ptr + row[r].to(gl.int64) * (1 << WIDTH0) + stored, cost)


@gluon.jit(do_not_specialize=["rows", "mod", "length", "outputs", "k0", "k1", "constrained"])
def _forward_kernel(
    w_ptr, p1_ptr, p2_ptr, start_ptr, order_ptr, kinds_ptr, m_out_ptr, bp_ptr, rows, mod, length,
    outputs, k0, k1, constrained, L: gl.constexpr, ROWS: gl.constexpr, INT8: gl.constexpr,
    MIN_K: gl.constexpr, KINDS: gl.constexpr,
):  # fmt: skip
    """All ``length`` steps for one CTA's ``ROWS`` row slots (``order``, ``compute_rows_per_cta``).
    ``MIN_K``: the schedule's smallest step ``k``; ``kinds[t]`` is step ``t``'s kind ``4 (k_t - k)
    + 2 (k_{t+1} - k) + k_{t+2} - k`` and ``KINDS`` the set of kinds as bits; ``k0``, ``k1``:
    ``k_0``, ``k_1``. ``outputs`` is ``2^loop_bits``, passed at run time so that the output loop
    stays a loop (LLVM unrolls a constant 16-trip loop into about 130 KB of code)."""
    THREADS: gl.constexpr = _threads(L, MIN_K)
    MIXED: gl.constexpr = KINDS != 1  # both step widths: k_0 known at run time
    CAP: gl.constexpr = 1 << (L - MIN_K)
    LOW_ROWS: gl.constexpr = _low_rows(ROWS)
    cta = gl.program_id(0)
    low = gl.allocate_shared_memory(gl.float32, [LOW_ROWS * CAP], _IDENTITY)
    high = gl.allocate_shared_memory(gl.float32, [(ROWS - LOW_ROWS) * CAP], _IDENTITY)
    low_address = _shared_address(low)
    high_address = _shared_address(high)

    row = ()
    p1 = ()
    minus_p2 = ()
    for r in gl.static_range(ROWS):
        slot_row = gl.load(order_ptr + cta * ROWS + r)
        row = row + (slot_row,)
        p1 = p1 + (gl.load(p1_ptr + slot_row),)
        minus_p2 = minus_p2 + (-gl.load(p2_ptr + slot_row),)
    first_bits = p1[0].to(gl.int32, bitcast=True)
    one = p1[1].to(gl.int32, bitcast=True) == first_bits  # the slots share one bitwise p1
    for r in gl.static_range(2, ROWS):
        one = one & (p1[r].to(gl.int32, bitcast=True) == first_bits)

    if not MIXED:  # noqa: SIM114 (a compile-time branch, then a run-time one)
        _initial(
            low_address, high_address, row, start_ptr, mod, constrained, THREADS, CAP, ROWS,
            MIN_K, L - MIN_K,
        )  # fmt: skip
    elif k0 == MIN_K:
        _initial(
            low_address, high_address, row, start_ptr, mod, constrained, THREADS, CAP, ROWS,
            MIN_K, L - MIN_K,
        )  # fmt: skip
    else:
        _initial(
            low_address, high_address, row, start_ptr, mod, constrained, THREADS, CAP, ROWS,
            MIN_K, L - MIN_K - 1,
        )  # fmt: skip
    gl.barrier()

    if one:
        _run(
            low, high, low_address, high_address, row, p1, minus_p2, w_ptr, bp_ptr, kinds_ptr,
            rows, length, outputs, L, CAP, ROWS, INT8, MIN_K, KINDS, True,
        )  # fmt: skip
    else:
        _run(
            low, high, low_address, high_address, row, p1, minus_p2, w_ptr, bp_ptr, kinds_ptr,
            rows, length, outputs, L, CAP, ROWS, INT8, MIN_K, KINDS, False,
        )  # fmt: skip

    if not MIXED:  # noqa: SIM114 (a compile-time branch, then a run-time one)
        _final(low, high, row, m_out_ptr, k1, THREADS, CAP, ROWS, MIN_K, L - MIN_K)
    elif k0 == MIN_K:
        _final(low, high, row, m_out_ptr, k1, THREADS, CAP, ROWS, MIN_K, L - MIN_K)
    else:
        _final(low, high, row, m_out_ptr, k1, THREADS, CAP, ROWS, MIN_K, L - MIN_K - 1)


_MIN_TRITON = (3, 9)
"""The Gluon release the kernel and its workarounds were written and measured for."""
_TRITON_VERSION = tuple(int(part) for part in re.findall(r"\d+", triton.__version__)[:2])


def _supports_properties(capability: tuple[int, int], shared_optin: int) -> bool:
    """Whether an NVIDIA GPU with these properties runs ``run_forward``."""
    return capability >= (8, 0) and shared_optin >= SHARED_NEED


@functools.cache
def _supports_index(index: int) -> bool:
    props = torch.cuda.get_device_properties(index)
    capability = (props.major, props.minor)
    return _supports_properties(capability, props.shared_memory_per_block_optin)


def supports_device(device: torch.device) -> bool:
    """Whether ``run_forward`` runs on ``device``: an NVIDIA GPU from sm_80 with
    ``SHARED_NEED`` bytes of opt-in shared memory per CTA, under Triton 3.9 or later.
    Checked once per device."""
    if device.type != "cuda" or torch.version.hip is not None or _TRITON_VERSION < _MIN_TRITON:
        return False
    return _supports_index(_index(device))


def supports_trellis(state_bits: int, step_widths: Sequence[int]) -> bool:
    """Whether ``run_forward`` runs this trellis: ``state_bits`` in ``STATE_BITS`` and
    ``1 <= T <= MAX_LENGTH`` steps in any order of one or two adjacent widths of
    ``STEP_BITS`` (3/4, 4/5 or 5/6 bits)."""
    widths = set(step_widths)
    return (
        state_bits in STATE_BITS
        and 0 < len(step_widths) <= MAX_LENGTH
        and widths <= set(STEP_BITS)
        and max(widths) - min(widths) <= 1
    )


def compute_rows_per_cta(state_bits: int, step_widths: Sequence[int]) -> int:
    """Rows per CTA of a trellis's kernel (``_order`` groups rows into CTAs of this
    many, and a wave holds a multiple): ``ROWS``, or half of them where ``ROWS`` rows'
    prefix costs do not fit in ``SHARED_LIMIT`` (L = 16 with 3-bit steps)."""
    return _rows(state_bits, min(step_widths))


def _kinds(step_widths: Sequence[int]) -> list[int]:
    """Each step's kind ``4 (k_t - k) + 2 (k_{t+1} - k) + k_{t+2} - k`` with ``k`` the
    smallest step, cyclic."""
    n, k = len(step_widths), min(step_widths)
    return [
        4 * (step_widths[t] - k) + 2 * (step_widths[(t + 1) % n] - k) + step_widths[(t + 2) % n] - k
        for t in range(n)
    ]


def _kind_set(step_widths: Sequence[int]) -> int:
    return sum(1 << k for k in set(_kinds(step_widths)))


@functools.cache
def _kinds_tensor(step_widths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """``_kinds`` on ``device``, made once per schedule."""
    return torch.tensor(_kinds(step_widths), dtype=torch.int32, device=device)


type _Specialization = tuple[int, bool, int, int]


def _specialization(state_bits: int, step_widths: Sequence[int], int8: bool) -> _Specialization:
    """The compiled kernel a trellis runs: its constexprs ``(L, INT8, MIN_K, KINDS)``."""
    return state_bits, int8, min(step_widths), _kind_set(step_widths)


def _compile_args(L: int, int8: bool, min_k: int, kinds: int) -> dict[str, int | bool]:
    """A specialization's constexprs and launch options."""
    options = {"num_warps": _geometry(L, min_k)[0] // 32, "enable_fp_fusion": False}
    constexprs = {"L": L, "ROWS": _rows(L, min_k), "INT8": int8, "MIN_K": min_k, "KINDS": kinds}
    return constexprs | options


def _index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


@functools.cache
def _compiled(specialization: _Specialization, index: int) -> CompiledKernel:
    """A specialization compiled for device ``index`` and loaded, with its register
    count."""
    f32, i32, i64, u8 = torch.float32, torch.int32, torch.int64, torch.uint8
    with torch.cuda.device(index):
        kernel = _forward_kernel.warmup(
            f32, f32, f32, i64, i32, i32, f32, u8, 0, 1, 1, 1, 4, 4, 0, grid=(1,),
            **_compile_args(*specialization),
        )  # fmt: skip
        # Private: a compiled kernel's register count (n_regs) is known only once its
        # module is loaded, which Triton does in this method on the first launch;
        # compute_wave_rows needs it before.
        kernel._init_handles()
    return kernel


def _resident_ctas(specialization: _Specialization, index: int) -> int:
    """Resident CTAs per SM of a specialization: the fewest that its threads,
    registers (per warp in units of 256) and shared memory (with
    ``_RESERVED_SHARED``) leave room for."""
    kernel = _compiled(specialization, index)
    props = torch.cuda.get_device_properties(index)
    threads = _geometry(specialization[0], specialization[2])[0]
    warp_registers = -(-kernel.n_regs * 32 // 256) * 256
    by_threads = props.max_threads_per_multi_processor // threads
    by_registers = props.regs_per_multiprocessor // (warp_registers * threads // 32)
    shared = kernel.metadata.shared + _RESERVED_SHARED
    by_shared = props.shared_memory_per_multiprocessor // shared
    return min(by_threads, by_registers, by_shared)


def compute_wave_rows(
    state_bits: int, step_widths: Sequence[int], int8: bool, device: torch.device
) -> int:
    """Rows that occupy every SM of ``device`` exactly once.

    A launch takes about ``ceil(rows / wave_rows)`` times as long as one wave, so
    batches should be whole multiples (RTX 5090, L = 16: 170 SMs x 1 CTA x 6 rows =
    1020; with 3-bit steps x 3 rows = 510). Compiles the kernel on first use.
    """
    index = _index(device)
    ctas = _resident_ctas(_specialization(state_bits, step_widths, int8), index)
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    return sms * ctas * compute_rows_per_cta(state_bits, step_widths)


def _order(p1: torch.Tensor, rows_per_cta: int) -> torch.Tensor:
    """The row of each CTA slot, int32 ``[ctas · rows_per_cta]``.

    Rows are sorted by bitwise ``p1`` so a CTA's rows share one value where they can,
    the last CTA is padded with the last row (a repeated slot computes and stores the
    same bytes), and the CTAs that hold more than one value come first. Those decode
    per row, which takes about 1.7x the time of a shared decode in INT8; starting them
    first keeps them out of the last wave, where one would extend the launch. Device
    ops only (no host synchronization).
    """
    rows = p1.shape[0]
    ctas = -(-rows // rows_per_cta)
    key = p1.view(torch.int32)
    order = torch.sort(key, stable=True).indices
    order = torch.cat([order, order[-1:].expand(ctas * rows_per_cta - rows)])
    slots = key[order].view(ctas, rows_per_cta)
    mixed = (slots[:, 1:] != slots[:, :-1]).any(1).to(torch.int8)
    slowest_first = torch.sort(mixed, descending=True, stable=True).indices
    return order.view(ctas, rows_per_cta)[slowest_first].reshape(-1).to(torch.int32)


def run_forward(
    state_bits: int,
    step_widths: Sequence[int],
    *,
    int8: bool,
    w: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    overlap: torch.Tensor | None,
    bp: torch.Tensor,
) -> torch.Tensor:
    """All ``T`` steps of the Viterbi forward for ``rows`` independent rows.

    Returns the final prefix costs and fills ``bp``, identical to the step loop from
    the same initial prefix. Inputs are contiguous and on one CUDA device that
    ``supports_device``, where the kernel launches:

    - ``state_bits``, ``step_widths``: the trellis, with ``supports_trellis``;
    - ``int8``: the format's INT8 decode (``p1`` is the folded ``alpha``) instead of the
      float value ``(c(s)·p1)·p2``;
    - ``w``: FP32 ``[T, rows]``, the weight of each row at each position. The kernel
      indexes it by int32 ``t · rows + row``, which the caller's batches keep below
      ``2^31``;
    - ``p1``, ``p2``: FP32 ``[rows]``, the row's scale for every position and its
      multiplier. ``w``, ``p1`` and ``p2`` must be finite: the kernel groups its
      comparisons on the premise that costs are never NaN, so a row with a non-finite
      input gets unspecified results;
    - ``overlap``: int64 ``[mod]`` with ``rows % mod == 0``, or ``None``. ``None``:
      free start, every initial prefix cost 0. Otherwise row ``r`` starts from
      ``overlap[r % mod]``, below ``2^(L - k_0)``: cost 0 there and infinity elsewhere;
    - ``bp``: uint8 ``[T, rows, 2^(L - min k)]`` to receive the backpointers:
      ``bp[t, r, pos]`` is the minimizing dropped bits ``a`` at the stored position
      ``pos`` of each prefix after step ``t`` (the first ``2^(L - k_(t+1))``
      positions; the rest are not written).

    Returns FP32 ``[rows, 2^(L - k_0)]``: the prefix costs after the last step in the
    stored layout (``_store_index(j, L - k_0, k_1)``), as the loop's. Rows are grouped
    by bitwise ``p1`` internally; outputs are in the caller's row order and each row's
    result depends only on its own inputs.
    """
    length, rows = w.shape
    out = torch.empty(rows, 1 << (state_bits - step_widths[0]), device=w.device)
    if rows == 0:
        return out
    per_cta = compute_rows_per_cta(state_bits, step_widths)
    with torch.cuda.device(w.device):
        order = _order(p1, per_cta)
        kinds = _kinds_tensor(tuple(step_widths), w.device)
        free = overlap is None
        outputs = 1 << _geometry(state_bits, min(step_widths))[1]
        _forward_kernel[(order.shape[0] // per_cta,)](
            w, p1, p2, torch.empty(1, dtype=torch.int64, device=w.device) if free else overlap,
            order, kinds, out, bp, rows, 1 if free else overlap.shape[0], length, outputs,
            step_widths[0], step_widths[1 % length], int(not free),
            **_compile_args(*_specialization(state_bits, step_widths, int8)),
        )  # fmt: skip
    return out


__all__ = [
    "compute_rows_per_cta",
    "compute_wave_rows",
    "run_forward",
    "supports_device",
    "supports_trellis",
]
