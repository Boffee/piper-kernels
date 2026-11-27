# Trellis-coded weights

Trellis coding stores a ConvRot INT8 weight in 2 to 6 bits per weight. It can be
executed directly in memory via `ConvRotTrellisTensor` (`F.linear`) and
`piper_kernels.moe` (`"convrot_trellis"` across uniform 4-D, 2-D variable-row, and
1-D variable-row + `slot_offsets` expert slabs) or decoded at load time via
`ConvRotInt8Tensor.from_trellis` and `copy_from_trellis_` into ordinary ConvRot
INT8 storage. This README owns the format and its encoder; the
[weights guide](../../../docs/weights.md#encode-execute-and-decode-trellis-coded-weights)
shows the API.

The code is QTIP's bitshift trellis with its 1MAD code ([Tseng et al., 2024,
arXiv:2406.11235](https://arxiv.org/abs/2406.11235)). It is tail-biting, has one
FP8 scale and two wrap guard bytes per block, and supports variable rates.

## Stored tensors

A logical `[out, in]` weight, already rotated for its ConvRot group size, is cut
into blocks of `T = block_size` consecutive weights of a row. Each block is one
tail-biting trellis sequence (QTIP's `T`) with its own scale and two trailing
wrap guard bytes (copies of bytes `0` and `1`), the same block unit GGUF and
NVFP4 use. `TrellisStorage` holds the tensors and the block size:

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `bitstreams` | `[out, in / T, bytes]` | uint8 | Little-endian block bitstream (`code_bytes + 2` guard bytes) |
| `block_scales` | `[out, in / T]` | float8_e4m3fn | One scale per block |
| `scale` | `[out]` or `[out, 1]` | float32 | Row scale, the ConvRot INT8 `scale` unchanged |

The stored bytes per block set the rate: each block stores
`code_bytes = bytes - 2` payload code bytes (`B = 8 · (bytes - 2)` code bits)
plus 2 trailing guard bytes. Besides the tensors, a checkpoint needs only the
block size `T`, the ConvRot group size, and the logical dtype; the last two are
stored for ConvRot INT8 already. Because each tensor's rate comes from its own
shape, rates can differ between tensors.

`ConvRotTrellisTensor.from_quantized`, `ConvRotInt8Tensor.from_trellis`, and
`copy_from_trellis_` also accept a variable-row layout via `row_widths`
(`uint8 [out]`): `bitstreams` becomes 1-D `uint8 [total_bytes]`, storing each row
`r`'s `in / T` blocks of `row_widths[r]` stored bytes back-to-back with row byte
offsets `(in / T) · prefix_sum(row_widths)`. Each row width must lie in
`[⌈2·T / 8⌉ + 2, ⌊6·T / 8⌋ + 2]` and `(in / T) · sum(row_widths)` must fit within
`bitstreams.numel()`; both are caller preconditions (no host readback or
synchronization at decode time). `"convrot_trellis"` MoE 2-D and 1-D
variable-row slabs follow the same per-slot `row_widths` bounds and require
non-negative `slot_offsets` whose per-slot ranges fit within
`bitstreams.numel()`.

`piper_kernels.trellis` exports `TrellisHessian`, `TrellisLayout`, `resolve_block_size`,
and `resolve_layout` alongside `TrellisStorage`.

## Decoding a block

A block's `B` code bits are stored little-endian (`bit 0` is the LSB of byte `0`),
and bytes `[code_bytes, code_bytes + 1]` duplicate bytes `[0, 1]` so every
16-bit window reads linearly without modulo wrap-around logic.

1. **Step widths.** Weight `t` (from 0) shifts in `k_t` bits: `step_bits`, plus one
   at `w = B - step_bits·T` evenly spaced steps, where `step_bits = ⌊B/T⌋` and
   `k_t = step_bits + ⌊(t+1)·w/T⌋ - ⌊t·w/T⌋`. For example, 141-code-byte
   (143-stored-byte) blocks of 256 weights (`B = 1128`) have 104 5-bit steps among
   152 4-bit ones.
2. **States.** Weight `t`'s 16-bit state `s_t` is the linear little-endian window
   of bits `o_t, o_t + 1, …, o_t + 15` (with `bits 0..15` mapped from LSB to MSB)
   at `o_t = k_1 + … + k_t = step_bits·t + ⌊(t+1)·w/T⌋`. Consecutive states share
   `16 - k_t` bits, which is what makes this a trellis; trailing guard bytes
   handle tail-biting boundary windows seamlessly.
3. **Code.** `x = (34038481·s_t + 76625530) mod 2^32`, and the numerator `n_t` is the
   sum of `x`'s four bytes minus 510, an integer in `[-510, 510]`. The code value
   `n_t / 147.8` is approximately standard normal.
4. **Dequantization & INT8 decoding.** With `R` the FP32 value `0x1.bb688ep-8`
   nearest `1 / 147.800537109375`, execution uses one of two exact paths:

   - **Continuous block-scaled FP32** (small-batch linear A16 with `M <= 4`, or
     `M <= 8` on AMD RDNA4, all batches on targets without an optimized ConvRot
     INT8 W8A8 backend such as CPU or SM70, MoE A16, and
     `ConvRotTrellisTensor.dequantize()`): skips per-weight INT8 rounding and
     accumulates `x_t · n_t` per block before multiplying once by `fp32(block_scale · R)`
     and applying `scale`.
   - **Prescaled rounded INT8** (large-batch W8A8 linear with an INT8 backend at
     `M >= 5`, or `M >= 9` on AMD RDNA4, MoE prefill, and
     `ConvRotInt8Tensor.from_trellis` / `copy_from_trellis_`):
     `q_t = clamp(round_half_even(fp32(n_t) · fp32(block_scale · R)), -127, 127)` in
     the rotated basis, as for any ConvRot INT8 weight.

Products are single IEEE FP32 multiplies of exactly representable values, so CUDA,
HIP, and the PyTorch reference match by construction. Tests verify little-endian
bitstreams, decoding, `ConvRotTrellisTensor` linear execution, and
`piper_kernels.moe` `"convrot_trellis"` execution live on AMD Radeon RX 9070 XT
(`gfx1201`) and CPU for the RDNA4 optimizations (with prior live verification on
NVIDIA RTX 5090 `sm120` on the shared CUDA schedule) and offline zero-spill compile +
PTX/AMDGCN inspection across NVIDIA `sm70`, `sm75`, `sm80`, `sm86`, `sm89`, and `sm120`
and AMD `gfx942`, `gfx1100`, `gfx1151`, `gfx1200`, and `gfx1201`. Live NVIDIA (after
the RDNA4 refactor), non-RDNA4 AMD GPUs, multi-GPU, and Windows execution remain
unexercised locally on this branch.
Scales must not be NaN encodings (`0x7F`, `0xFF`); this is a caller precondition, not
a runtime check.

## Layout limits

| Value | Supported | Why |
|---|---|---|
| Block size `T` | 16 to 256 | A 16-bit window never crosses more than 2 guard bytes; one decode tile covers a block |
| Code bits per block `B` | Whole bytes from `2·T` to `6·T` bits (`bytes = B/8 + 2`) | 2- to 6-bit steps plus 2 trailing wrap guard bytes |

Whole bytes cost at most 1/32 bit per weight of rate granularity at `T = 256`, and
let the rate come from the tensor shape instead of more metadata.

The block size must divide the weight's input width. The encoder chooses the
largest multiple of 16 up to 256 that divides it: 256 for most widths, 224 for 2688.

## Encoding

`TrellisStorage.from_hp(weight, group_size=..., bits_per_weight=...)` (or an exact
whole-byte stored `bytes_per_block=...` in place of `bits_per_weight`) rotates the
weight in FP32 and encodes each block. When using `bits_per_weight`, the rate
counts each block's 2 trailing wrap guard bytes and 1 FP8 scale (24 bits overhead):
a block of `T` weights allocates `⌊bits_per_weight · T⌋ - 24` bits, rounded down to
whole code bytes, plus 2 guard bytes. On `T = 256`, valid `bits_per_weight` spans
`2.09375` to `6.09375`: at 4.5 bits per weight a block stores 141 code bytes + 2
guard bytes = 143 stored bytes (`bytes_per_block=143`), 4- and 5-bit steps; at 3.5
it stores 109 code bytes + 2 guard bytes = 111 stored bytes, 3- and 4-bit steps; at
2.5 it stores 77 code bytes + 2 guard bytes = 79 stored bytes, 2- and 3-bit steps.

The [encoder](_encoder.py) reverses step order into the Gluon Viterbi solver so
its native left-shift transitions produce exact little-endian walks, then packs
little-endian bitstreams with 2 trailing wrap guard bytes and minimizes squared error
in the rotated basis. Without calibration data, Stage 1 analytically estimates
per-block scales from a Bennett 1MAD + tail-clipping proxy and selects each
row's scale across 29 multipliers. Stage 2 runs a cost-only 12-bit Viterbi check over
the 3 nearest E4M3 block scales. Stage 3 then runs one tail-biting 16-bit Viterbi pass
and refits each block's E4M3 scale in FP32. The module docstring describes each stage.

When `hessian` is passed (raw unrotated `H [in, in]` or a pre-factored
`TrellisHessian`), `TrellisHessian.from_hessian` symmetrizes `H`, rotates it in FP64
into the ConvRot basis (`M^T H M`), adds `damping * mean(diag_pos) * I` (retried up to
`25x` if ill-conditioned) for Cholesky factor `L`, and forms the block unit-lower
triangular FP32 feedback matrix (`feedback[e:, s:e] = L[e:, s:e] @ L[s:e, s:e]^{-1}`).
The encoder fixes `scale` once across each rotated row and sweeps blocks from right
to left, propagating `(W_orig - W_hat_b) @ feedback[s:e, :s]` into earlier blocks
before selecting each block's FP8 scale and Viterbi walk.

On seeded BF16 Gaussian rows (`48 × 1024`, `group_size=256`), which ConvRot-rotated
weights resemble, the weight SQNR is:

| Bits per weight (total) | 2.09 | 2.5 | 3.09 | 3.5 | 4.09 | 4.5 | 5.5 | 6.09 |
|---|---|---|---|---|---|---|---|---|
| Prescaled INT8 SQNR (dB) | 11.5 | 13.9 | 17.5 | 19.9 | 23.4 | 25.8 | 31.8 | 35.3 |
| Unrounded A16 SQNR (dB) | 11.5 | 13.9 | 17.5 | 19.9 | 23.4 | 25.7 | 31.4 | 34.4 |

Peak-aware scale selection prevents outlier blocks from clipping, though extreme
post-rotation outliers remain bounded by 1MAD's ~3.2σ tail range.

## Decode implementation

[`triton.py`](triton.py) decodes a tile of 4 blocks per program (2 warps, or 1 warp
on AMD RDNA4) via one unified little-endian kernel (`decode_int8_kernel` /
`decode_slots_int8_kernel`). Each 4-step group aligns to 32-bit little-endian words
(`w0`, `w1`, and `w2` only when `code_bytes > 5 · T / 8` or `variable_rows=True`),
extracts 16-bit windows by pure shift-and-mask (2-word `uint64` shifts up to 5.0
code bpw, 3-word above), computes 1MAD byte sums (`dp4a.s32.u32` via
[`_nvidia/isa.py`](_nvidia/isa.py) on CUDA and `v_sad_u8` via [`_amd/isa.py`](_amd/isa.py)
on AMD RDNA4, with pure-Triton byte masking on other targets), and either rounds to
packed 4x INT8 or writes unrounded block-scaled FP32. Power-of-2 block sizes specialize away
runtime integer division, and one compiled kernel serves uniform rates (and one serves
variable rows).

On an RTX 5090 (`sm120`, cold-L2), it decodes at **925–956 G weights/s** on uniform
`14336×4096` 256-weight blocks and **538–819 G weights/s** on variable-row streams
(`4096×2688` at 538 Gw/s eager/graph; `4096×4096` at 746 Gw/s eager / 819 Gw/s CUDA Graph),
consuming packed bytes well above pinned PCIe copy bandwidth. Small-batch linear A16
execution fuses the same 4-step little-endian decode into registers with continuous
block-scaled accumulation, reaching **896 GB/s** (`0.0366 ms`) cold-L2 on
`14336×4096` `M = 1` at 4.5 bpw. Wave64 targets such as `gfx942` have compile coverage only.
[`_reference.py`](_reference.py) is the PyTorch oracle.

## Encode implementation

The [persistent Gluon forward](_gluon_forward.py) runs all steps of a batch in one
launch, packing two rows per 32-bit word (`FP16x2`) in registers and shared memory
with normalized weights `(w / row_scale) · 0.03125`. Rows are sorted and bucket-padded
by bitwise E4M3 block scale so every CTA decodes states once for all its rows and
writes coalesced `uint16` backpointer pairs; a Triton traceback kernel reconstructs
walks. It requires an NVIDIA SM80+ GPU with 99 KiB of opt-in shared memory per CTA and
Triton 3.9+, and its inline PTX is NVIDIA-only.

Encoding runs offline on NVIDIA SM80+ GPUs. On an RTX 5090 with PyTorch 2.15 and Triton
3.9, `TrellisStorage.from_hp` encoded a 4096×5120 Gaussian BF16 weight with group size 256
at 89.8 to 113.8 M weights/s across 3.09 to 6.09 bits per weight (108.6 M weights/s at
4.5 bits per weight; 113.8 M weights/s at 4.09 bits per weight) and 56.0 to 56.5 M
weights/s at 2.09 to 2.5 bits per weight (1 pair = 2 rows/CTA in shared memory). Search
batches target 4 GiB of backpointers; peak allocation delta on the Gaussian matrix was
4.4 GiB beyond the input weight. Batches and row chunks give identical results.
