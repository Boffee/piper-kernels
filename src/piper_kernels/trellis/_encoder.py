"""Coarse-to-fine trellis encoder: rotated FP32 rows to trellis-coded blocks.

The encoder minimizes each block's plain squared weight error (data-free, no
Hessian) against the format's exact INT8 decode. Each step below runs a Viterbi
search (``_viterbi.compute_walks``) or a refit of the block scale for a fixed walk,
and keeps a block's new walk or scale only where it lowers its FP64 error:

1. **Coarse float stage**, on the layout's step widths with 12-bit states: QTIP
   Algorithm 4's overlap at the scale ``rms(block) / 0.909``, one Viterbi at that
   scale, then the best point of a 97-point log grid around it for the walk. The
   row scale starts at ``s_c0 = max |ŵ| / 127`` of that reconstruction.
2. **Row scale:** ``m · s_c0`` for the ``m`` among 29 multipliers from 0.84 to 2.83
   at which the coarse walks decode to INT8 with the least error (no Viterbi).
3. **Coarse INT8 stage** at that row scale: an FP8 refit of the block scale, then
   one Viterbi over its five nearest E4M3 values and another refit.
4. **Final pass** on the stored 16-bit trellis at those scales: the overlap again,
   from the 128 positions around the wrap, one constrained Viterbi at the block
   scale and an FP8 refit. Its walk replaces the coarse one.

A 12-bit pass costs about 0.06 of a 16-bit one on the persistent forward, so the
final pass dominates. On ConvRot-rotated, Gaussian-like rows this ends 1.3-2.0%
above the MSE of the research study's full multi-round encoder. Rows with heavy
outliers encode much worse, because the one-candidate float stage starts at
``rms / 0.909``. The constants are the study's (``piper-research``,
``projects/e8-w4a8-vq``), where this encoder is ``trellis_vr_fast``; tests check
that both produce the same bytes.
"""

# Triton's JIT launcher and constexpr arguments are not fully represented by its
# Python typing surface.
# pyright: reportCallIssue=false, reportIndexIssue=false, reportArgumentType=false
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.runtime import device_context

from ._layout import STATE_BITS, TrellisLayout
from ._viterbi import (
    SearchTrellis,
    build_numerator_table,
    build_value_table,
    compute_overlap,
    compute_walks,
    fold_block_scales,
)

_CENTER_RMS = 0.909
"""The float stage's initial block scale is ``rms(block) / 0.909``, calibrated on
Gaussian rows (about 1.1 · rms, as for one global scale)."""

_COARSE_STATE_BITS = 12
"""State bits of the coarse search. 14 gained 0.010-0.012 dB over 12 for 1.2-1.4x
the time."""

_OVERLAP_WINDOW = 64
"""The final pass's overlap comes from the ``2 · 64`` positions around the wrap
point: 0.015 dB below the whole block for 0.82x the time. Shorter blocks use the
whole block."""

_FLOAT_SCALE_FACTORS = tuple(2.0 ** (k / 48) for k in range(-48, 49))
"""The float stage's grid: 97 factors, 1.45% apart, one octave either side of the
initial scale."""

_FP8_SEARCH_RADIUS = 2
"""The coarse INT8 Viterbi's candidates: E4M3 values within 2 grid steps."""

_FP8_REFIT_RADIUS = 8
"""An FP8 refit's candidates: E4M3 values within 8 grid steps of the least-squares
scale, about one octave either side."""

_ROW_MULTIPLIERS = tuple(2.0 ** (i / 16) for i in range(-4, 25))
"""Row-scale candidates ``m · s_c0``: 29 values in 4.4% steps from 0.84 to 2.83."""

_INT8_MAX = 127

_MAX_CHUNK_WEIGHTS = 1 << 24
"""Rows are encoded in chunks of at most this many weights, the research study's
chunk, to bound the per-block working tensors. Rows and blocks encode
independently, so chunking does not change the result."""


@functools.cache
def _build_fp8_values(device: torch.device) -> torch.Tensor:
    """All 126 finite positive E4M3 values, ascending (FP32)."""
    values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    values = values[(values > 0) & torch.isfinite(values)].unique().sort().values
    return values.to(device)


def _round_fp8(x: torch.Tensor) -> torch.Tensor:
    """The nearest E4M3 value (half-even, saturating at 448), as FP32."""
    return x.float().to(torch.float8_e4m3fn).float()


def _compute_fp8_window(center: torch.Tensor, radius: int) -> torch.Tensor:
    """The positive E4M3 values within ``radius`` grid steps of ``_round_fp8(center)``,
    ascending, ``[*center.shape, 2 · radius + 1]`` (clamped, so they may repeat at
    the grid ends)."""
    grid = _build_fp8_values(center.device)
    nearest = _round_fp8(center).clamp(min=float(grid[0]))
    index = torch.searchsorted(grid, nearest.contiguous())
    offsets = torch.arange(-radius, radius + 1, device=center.device)
    positions = (index.unsqueeze(-1) + offsets).clamp(0, grid.numel() - 1)
    return grid[positions]


@triton.jit
def _squared_error_kernel(
    w_ptr, entries_ptr, factor_ptr, p2_ptr, out_ptr, total, length,
    int8: tl.constexpr, double: tl.constexpr, block: tl.constexpr,
):  # fmt: skip
    """``(w - v · p2)²`` per position of contiguous ``[N, T]`` inputs, in FP64 when
    ``double``. ``v`` is ``clamp(rint(n · alpha), ±127)`` (INT8, ``factor`` holding
    ``alpha``) or ``c · scale``, each product rounded to FP32 (launched with
    ``enable_fp_fusion=False``); ``factor`` and ``p2`` are per block."""
    i = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = i < total
    w = tl.load(w_ptr + i, mask=valid, other=0.0)
    entry = tl.load(entries_ptr + i, mask=valid, other=0.0)
    factor = tl.load(factor_ptr + i // length, mask=valid, other=0.0)
    p2 = tl.load(p2_ptr + i // length, mask=valid, other=0.0)
    value = entry * factor
    if int8:
        value = tl.minimum(tl.maximum(libdevice.nearbyint(value), -127.0), 127.0)
    recon = value * p2
    d = w.to(tl.float64) - recon.to(tl.float64) if double else w - recon
    tl.store(out_ptr + i, d * d, mask=valid)


_ERROR_BLOCK = 1024


@dataclass
class _Walks:
    """A solution per block: walks, block scales (FP32 or FP8-valued) and FP64
    errors."""

    states: torch.Tensor  # [N, T] int64
    scale: torch.Tensor  # [N] FP32
    err: torch.Tensor  # [N] FP64

    def take(
        self, mask: torch.Tensor, states: torch.Tensor, scale: torch.Tensor, err: torch.Tensor
    ) -> None:
        """Replace the blocks where ``mask`` is set."""
        self.states = torch.where(mask[:, None], states, self.states)
        self.scale = torch.where(mask, scale, self.scale)
        self.err = torch.where(mask, err, self.err)


@dataclass(frozen=True)
class _Mode:
    """A reconstruction rule over per-position table entries: float
    ``(c · scale) · p2``, or the format's INT8 decode ``q(n, scale) · p2``, where
    ``scale`` is the block scale."""

    trellis: SearchTrellis
    int8: bool
    p2: torch.Tensor  # [N]: 1 for the float stage, else the row scale

    def compute_entries(self, states: torch.Tensor) -> torch.Tensor:
        """The table entries of ``states``: ``c(s)``, or ``n(s)`` in INT8 mode."""
        table = build_numerator_table if self.int8 else build_value_table
        return table(self.trellis.state_bits, states.device)[states]

    def compute_error(
        self,
        w: torch.Tensor,
        entries: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Per-block squared error ``[N]`` of the reconstruction from ``entries``
        against contiguous ``w [N, T]``, summed in ``dtype``."""
        factor = (fold_block_scales(scale) if self.int8 else scale).contiguous()
        out = torch.empty(w.shape, dtype=dtype, device=w.device)
        total = w.numel()
        with device_context(w.device):
            _squared_error_kernel[(triton.cdiv(total, _ERROR_BLOCK),)](
                w, entries, factor, self.p2, out, total, w.shape[1],
                int8=self.int8, double=dtype == torch.float64, block=_ERROR_BLOCK,
                enable_fp_fusion=False,
            )  # fmt: skip
        return out.sum(-1)


@dataclass
class _Search:
    """The encoder's state for one trellis: the blocks ``w [N, T]``, the float
    stage's grid center ``[N]`` and the tail-biting overlap of its searches."""

    trellis: SearchTrellis
    w: torch.Tensor
    center: torch.Tensor
    overlap: torch.Tensor | None = None

    def compute_values(self, states: torch.Tensor) -> torch.Tensor:
        """``c(s)`` per position ``[N, T]``."""
        return build_value_table(self.trellis.state_bits, states.device)[states]

    def evaluate(self, states: torch.Tensor, scale: torch.Tensor, mode: _Mode) -> _Walks:
        error = mode.compute_error(self.w, mode.compute_entries(states), scale)
        return _Walks(states, scale, error)

    def compute_grid_position(self, scale: torch.Tensor) -> torch.Tensor:
        """Real-valued index of ``scale`` on the float grid (0 = ``center``)."""
        half = (len(_FLOAT_SCALE_FACTORS) - 1) // 2
        octaves = torch.log2(scale.double() / self.center.double())
        return octaves * (half / math.log2(_FLOAT_SCALE_FACTORS[-1]))

    def compute_grid_scale(self, index: torch.Tensor) -> torch.Tensor:
        """``center · factor`` at signed grid ``index`` ``[N]`` or ``[N, S]``, clamped to
        the grid."""
        factors = torch.tensor(_FLOAT_SCALE_FACTORS, device=index.device)
        half = (len(_FLOAT_SCALE_FACTORS) - 1) // 2
        center = self.center if index.dim() == 1 else self.center[:, None]
        return center * factors[index.clamp(-half, half) + half]

    def search(self, walks: _Walks, mode: _Mode, candidates: torch.Tensor) -> None:
        """Optimal walk and candidate scale per block ``[N, S]``, kept where better."""
        states, choice = compute_walks(
            self.trellis, self.w, candidates, mode.p2, mode.int8, self.overlap
        )
        scale = candidates.gather(1, choice[:, None]).squeeze(1)
        found = self.evaluate(states, scale, mode)
        walks.take(found.err < walks.err, found.states, found.scale, found.err)

    def refit_float(self, walks: _Walks, mode: _Mode) -> None:
        """Best float-grid scale for the fixed walk, kept where better. The error is a
        convex quadratic in the scale, so the best grid point is one of the two that
        bracket the least-squares scale; both are scored exactly."""
        values = self.compute_values(walks.states)
        v64 = values.double()
        ls = (self.w.double() * v64).sum(-1) / v64.square().sum(-1).clamp(min=1e-300)
        # Beyond the grid (or ls <= 0) both bracket points clamp to its end.
        bounded = ls.clamp(min=self.center.double() / 4, max=self.center.double() * 4)
        below = self.compute_grid_position(bounded).floor().long()
        for index in (below, below + 1):
            scale = self.compute_grid_scale(index)
            err = mode.compute_error(self.w, values, scale)
            better = err < walks.err
            walks.scale = torch.where(better, scale, walks.scale)
            walks.err = torch.where(better, err, walks.err)

    def refit_fp8(self, walks: _Walks, mode: _Mode) -> None:
        """Best FP8 block scale for the fixed walk among the incumbent and the E4M3
        values within ``_FP8_REFIT_RADIUS`` of the least-squares scale, scored
        exactly in ``mode``."""
        v64 = self.compute_values(walks.states).double()
        numerator = (self.w.double() * v64).sum(-1)
        denominator = v64.square().sum(-1) * mode.p2.double()
        ls = numerator / denominator.clamp(min=1e-300)
        center = torch.where((denominator > 0) & (ls > 0), ls.float(), walks.scale)
        smallest = float(_build_fp8_values(center.device)[0])
        window = _compute_fp8_window(center.clamp(min=smallest), _FP8_REFIT_RADIUS)
        entries = mode.compute_entries(walks.states)
        for candidate_index in range(window.shape[-1]):
            scale = window[..., candidate_index]
            err = mode.compute_error(self.w, entries, scale)
            better = err < walks.err
            walks.scale = torch.where(better, scale, walks.scale)
            walks.err = torch.where(better, err, walks.err)


def _sum_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-row FP64 sum ``[R]`` of a per-block quantity ``[N]``."""
    return x.double().reshape(rows, -1).sum(-1)


def _compute_row_multiplier_indices(
    search: _Search, walks: _Walks, s_c0: torch.Tensor, rows: int
) -> torch.Tensor:
    """Per row, the index in ``_ROW_MULTIPLIERS`` of the row scale ``m · s_c0`` at
    which the coarse float walks decode to INT8 with the least error (FP32 sums).

    Each block takes the best of ``_round_fp8(scale / s_c)`` and its two E4M3
    neighbors. No Viterbi: it ranks the multipliers by the walk the INT8 search
    starts from. Ties take the smaller ``m``.
    """
    device = search.w.device
    per_row = search.w.shape[0] // rows
    entries = build_numerator_table(search.trellis.state_bits, device)[walks.states]
    multipliers = torch.tensor(_ROW_MULTIPLIERS, device=device)
    p2 = (s_c0 * multipliers[:, None]).repeat_interleave(per_row, dim=1)  # [M, N]
    window = _compute_fp8_window(walks.scale / p2, 1)  # [M, N, 3]
    errors = []
    for multiplier_index in range(len(_ROW_MULTIPLIERS)):
        mode = _Mode(search.trellis, True, p2[multiplier_index])
        err = [
            mode.compute_error(search.w, entries, window[multiplier_index, :, j], torch.float32)
            for j in range(3)
        ]
        errors.append(_sum_rows(torch.stack(err).amin(0), rows))
    return torch.stack(errors).argmin(0)


def _encode_rows(
    w_rows: torch.Tensor, layout: TrellisLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode finite rotated FP32 rows ``[R, K]`` on a CUDA device.

    Returns ``(states [R · K/T, T] int64, block_scales [R · K/T] FP8-valued FP32,
    scale [R] FP32)``: each block's 16-bit walk, its scale and the row scale.
    """
    rows, cols = w_rows.shape
    block_size = layout.block_size
    w = w_rows.reshape(-1, block_size)
    num_blocks = w.shape[0]
    stored = SearchTrellis(STATE_BITS, layout.compute_step_widths())
    coarse = stored.with_state_bits(_COARSE_STATE_BITS)
    center = (w.square().mean(-1).sqrt() / _CENTER_RMS).clamp(min=1e-8)
    search = _Search(coarse, w, center)

    # 1. Coarse float stage at the grid center.
    float_mode = _Mode(coarse, False, torch.ones(num_blocks, device=w.device))
    search.overlap = compute_overlap(coarse, w, center, float_mode.p2, int8=False)
    unset = torch.full_like(center, math.inf, dtype=torch.float64)
    walks = _Walks(torch.zeros_like(w, dtype=torch.int64), center, unset)
    index = search.compute_grid_position(walks.scale).round().long()
    search.search(walks, float_mode, search.compute_grid_scale(index[:, None]))
    search.refit_float(walks, float_mode)
    recon = search.compute_values(walks.states) * walks.scale[:, None] * float_mode.p2[:, None]
    s_c0 = (recon.reshape(rows, cols).abs().amax(-1) / _INT8_MAX).clamp(min=1e-8)

    # 2. Row scale.
    multiplier_index = _compute_row_multiplier_indices(search, walks, s_c0, rows)
    scale = s_c0 * torch.tensor(_ROW_MULTIPLIERS, device=w.device)[multiplier_index]

    # 3. Coarse INT8 stage.
    p2 = scale.repeat_interleave(cols // block_size)
    int8_mode = _Mode(coarse, True, p2)
    walks = search.evaluate(walks.states, _round_fp8(walks.scale / p2), int8_mode)
    search.refit_fp8(walks, int8_mode)
    search.search(walks, int8_mode, _compute_fp8_window(walks.scale, _FP8_SEARCH_RADIUS))
    search.refit_fp8(walks, int8_mode)

    # 4. Final pass on the stored trellis.
    final = _Search(stored, w, center)
    final_mode = _Mode(stored, True, p2)
    window = min(_OVERLAP_WINDOW, block_size // 2)
    final.overlap = compute_overlap(stored, w, walks.scale, p2, int8=True, window=window)
    handed = _Walks(walks.states, walks.scale, torch.full_like(walks.err, math.inf))
    final.search(handed, final_mode, _compute_fp8_window(handed.scale, 0))
    final.refit_fp8(handed, final_mode)
    return handed.states, handed.scale, scale


def pack_states(states: torch.Tensor, layout: TrellisLayout) -> torch.Tensor:
    """Tail-biting bitstreams ``[N, bytes]`` uint8 of walks ``[N, T]``.

    Step ``t`` stores the ``k_t`` bits it shifts in, the low bits of ``s_t``, at
    stream bits ``o_t + 16 - k_t`` to ``o_t + 15`` (mod ``B``), most significant
    first, so state ``t`` is the cyclic window at ``o_t`` the decoder reads. Those
    ranges tile the stream, so every bit is written once.
    """
    total = layout.block_bits
    sources = [0] * total
    shifts = [0] * total
    widths = layout.compute_step_widths()
    for t, (offset, width) in enumerate(zip(layout.compute_state_offsets(), widths, strict=True)):
        for i in range(width):
            position = (offset + STATE_BITS - width + i) % total
            sources[position] = t
            shifts[position] = width - 1 - i
    device = states.device
    source_index = torch.tensor(sources, device=device).view(-1, 8)
    shift_amount = torch.tensor(shifts, dtype=torch.int32, device=device).view(-1, 8)
    narrow = states.to(torch.int32)
    packed = torch.zeros(states.shape[0], layout.bytes_per_block, dtype=torch.int32, device=device)
    for bit in range(8):
        value = (narrow[:, source_index[:, bit]] >> shift_amount[:, bit]) & 1
        packed |= value << (7 - bit)
    return packed.to(torch.uint8)


def encode_matrix(
    rotated: torch.Tensor, layout: TrellisLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trellis-coded storage of a rotated FP32 matrix ``[R, K]`` on a CUDA device.

    Returns ``(bitstreams [R, K/T, bytes] uint8, block_scales [R, K/T] float8_e4m3fn,
    scale [R, 1] float32)``. The values must be finite.
    """
    rows, cols = rotated.shape
    blocks_per_row = cols // layout.block_size
    device = rotated.device
    bitstreams = torch.empty(
        rows, blocks_per_row, layout.bytes_per_block, dtype=torch.uint8, device=device
    )
    block_scales = torch.empty(rows, blocks_per_row, dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty(rows, 1, dtype=torch.float32, device=device)
    chunk_rows = max(1, _MAX_CHUNK_WEIGHTS // cols)
    for start_row in range(0, rows, chunk_rows):
        end_row = min(rows, start_row + chunk_rows)
        states, chunk_block_scales, chunk_scale = _encode_rows(
            rotated[start_row:end_row].contiguous(), layout
        )
        packed = pack_states(states, layout)
        bitstreams[start_row:end_row] = packed.view(end_row - start_row, blocks_per_row, -1)
        block_scales[start_row:end_row] = chunk_block_scales.view(-1, blocks_per_row).to(
            torch.float8_e4m3fn
        )
        scale[start_row:end_row] = chunk_scale[:, None]
    return bitstreams, block_scales, scale


__all__ = ["encode_matrix", "pack_states"]
