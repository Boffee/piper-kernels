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
FP8 scale per block and a variable rate. Decoding is folded into the INT8
values the GEMM consumes.

## Stored tensors

A logical `[out, in]` weight, already rotated for its ConvRot group size, is cut
into blocks of `T = block_size` consecutive weights of a row. Each block is one
tail-biting trellis sequence (QTIP's `T`) with its own scale, the same unit GGUF and
NVFP4 call a block. `TrellisStorage` holds the tensors and the block size:

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `bitstreams` | `[out, in / T, bytes]` | uint8 | One tail-biting bitstream per block |
| `block_scales` | `[out, in / T]` | float8_e4m3fn | One scale per block |
| `scale` | `[out]` or `[out, 1]` | float32 | Row scale, the ConvRot INT8 `scale` unchanged |

The bytes per block set the rate: a block stores `B = 8 · bytes` bits. Besides
the tensors, a checkpoint needs only the block size `T`, the ConvRot group size
and the logical dtype; the last two are stored for ConvRot INT8 already. Because
each tensor's rate comes from its own shape, rates can differ between tensors.

`ConvRotTrellisTensor.from_quantized`, `ConvRotInt8Tensor.from_trellis`, and
`copy_from_trellis_` also accept a variable-row layout via `row_widths`
(`uint8 [out]`): `bitstreams` becomes 1-D `uint8 [total_bytes]`, storing each row
`r`'s `in / T` blocks of `row_widths[r]` bytes back-to-back with row byte offsets
`(in / T) · prefix_sum(row_widths)`. Each row width must lie in `[⌈2·T / 8⌉,
⌊6·T / 8⌋]` and `(in / T) · sum(row_widths)` must fit within `bitstreams.numel()`;
both are caller preconditions (no host readback or synchronization at decode time).
`"convrot_trellis"` MoE 2-D and 1-D variable-row slabs follow the same per-slot
`row_widths` bounds and require non-negative `slot_offsets` whose per-slot ranges fit
within `bitstreams.numel()`.

`piper_kernels.trellis` exports `TrellisLayout`, `resolve_block_size`, and
`resolve_layout` alongside `TrellisStorage`.

## Decoding a block

A block's `B` bits are stored most significant bit first.

1. **Step widths.** Weight `t` (from 0) shifts in `k_t` bits: `step_bits`, plus one
   at `w = B - step_bits·T` evenly spaced steps, where `step_bits = ⌊B/T⌋` and
   `k_t = step_bits + ⌊(t+1)·w/T⌋ - ⌊t·w/T⌋`. For example, 143-byte blocks of 256
   weights (`B = 1144`) have 120 5-bit steps among 136 4-bit ones.
2. **States.** Weight `t`'s 16-bit state `s_t` is the cyclic window of stream bits
   `o_t, o_t + 1, …, o_t + 15 (mod B)`, most significant first, at
   `o_t = k_1 + … + k_t = step_bits·t + ⌊(t+1)·w/T⌋`. Consecutive states share
   `16 - k_t` bits, which is what makes this a trellis. Because the stream is cyclic
   (tail-biting), it stores no initial state.
3. **Code.** `x = (34038481·s_t + 76625530) mod 2^32`, and the numerator `n_t` is the
   sum of `x`'s four bytes minus 510, an integer in `[-510, 510]`. The code value
   `n_t / 147.8` is approximately standard normal.
4. **INT8.** With `R` the FP32 value `0x1.bb688ep-8` nearest `1 / 147.800537109375`:

   ```text
   alpha = fp32(block_scale · R)
   q_t   = clamp(round_half_even(fp32(n_t) · alpha), -127, 127)
   ```

   The decoded weight is `q_t · scale` in the rotated basis, as for any ConvRot INT8
   weight.

Both products are single IEEE FP32 multiplies of exactly representable values, so
CUDA, HIP and the PyTorch reference decode identical INT8 values by construction.
Tests confirm decoding on NVIDIA hardware and decoding, direct
`ConvRotTrellisTensor` linear execution, and `piper_kernels.moe` `"convrot_trellis"`
execution on AMD RX 9070 XT (`gfx1201`); NVIDIA direct linear and MoE execution
(`sm75`, `sm80`, `sm89`, `sm120`) and AMD `gfx942`, `gfx1100`, and `gfx1151` have
offline compile coverage only. Scales must not be NaN encodings (`0x7F`, `0xFF`);
this is a caller precondition, not a runtime check.

## Layout limits

| Value | Supported | Why |
|---|---|---|
| Block size `T` | 16 to 256 | A 16-bit window never wraps twice; one decode tile covers a block |
| Bits per block `B` | Whole bytes from `2·T` to `6·T` bits | 2- to 6-bit steps, the rates the encoder supports |

Whole bytes cost at most 1/32 bit per weight of rate granularity at `T = 256`, and
let the rate come from the tensor shape instead of more metadata.

The block size must divide the weight's input width. The encoder chooses the
largest multiple of 16 up to 256 that divides it: 256 for most widths, 224 for 2688.

## Encoding

`TrellisStorage.from_hp(weight, group_size=..., bits_per_weight=...)` (or an exact
whole-byte `bytes_per_block=...` in place of `bits_per_weight`) rotates the
weight in FP32 and encodes each block. When using `bits_per_weight`, the rate
counts the block's FP8 scale: a block of `T` weights stores
`⌊bits_per_weight · T⌋ - 8` bits, rounded down to whole bytes. At 4.5 bits per
weight a 256-weight block takes 143 bytes, 4- and 5-bit steps; at 3.5 it takes
111 bytes, 3- and 4-bit steps; at 2.5 it takes 79 bytes, 2- and 3-bit steps.

The [encoder](_encoder.py) minimizes each block's squared error in the rotated
basis against the exact INT8 decode, without calibration data. Stage 1 analytically
estimates per-block scales from a Bennett 1MAD + tail-clipping proxy and selects each
row's scale across 29 multipliers. Stage 2 runs a cost-only 12-bit Viterbi check over
the 3 nearest E4M3 block scales. Stage 3 then runs one tail-biting 16-bit Viterbi pass
and refits each block's E4M3 scale in FP32. The module docstring describes each stage.

On Gaussian rows, which ConvRot-rotated weights resemble, the weight SQNR is:

| Bits per weight | 2.03 | 2.5 | 3.03 | 3.5 | 4.03 | 4.5 | 5.5 | 6.03 |
|---|---|---|---|---|---|---|---|---|
| SQNR (dB) | 11.5 | 14.3 | 17.5 | 20.3 | 23.4 | 26.2 | 32.1 | 35.2 |

Peak-aware scale selection prevents outlier blocks from clipping, though extreme
post-rotation outliers remain bounded by 1MAD's ~3.2σ tail range.

## Decode implementation

[`triton.py`](triton.py) decodes a tile of 4 blocks per program (2 warps). Each block
loads its first two wrap bytes once, and each weight computes its offset in closed form
and gathers the at most three contiguous bytes holding its window before merging any
wrap bits from those cached head bytes. Power-of-2 block sizes specialize away runtime
integer division. The kernel takes the rate as runtime arguments (and computes row
offsets from a prefix sum of `row_widths` when `variable_rows=True`), so one compiled
kernel serves every uniform rate and one serves every variable-row layout.

On an RTX 5090 it decodes about 680–730 G weights/s for uniform 256-weight blocks
(up from ~465–520 G weights/s) and about 490–590 G weights/s for variable-row streams
(up from ~240–265 G weights/s), consuming packed bytes well above pinned PCIe copy
bandwidth (33 GB/s measured), so blocks are stored packed rather than padded to words.
Wave64 targets such as `gfx942` have compile coverage only.
[`_reference.py`](_reference.py) is the PyTorch oracle; tests check both against
blocks encoded by the research encoder.

## Encode implementation

The [persistent Gluon forward](_gluon_forward.py) runs all steps of a batch in one
launch, packing two rows per 32-bit word (`FP16x2`) in registers and shared memory
with normalized weights `(w / row_scale) · 0.03125`. Rows are sorted and bucket-padded
by bitwise E4M3 block scale so every CTA decodes states once for all its rows and
writes coalesced `uint16` backpointer pairs; a Triton traceback kernel reconstructs
walks. It requires an NVIDIA SM80+ GPU with 99 KiB of opt-in shared memory per CTA and
Triton 3.9+, and its inline PTX is NVIDIA-only.

Encoding runs offline on NVIDIA SM80+ GPUs. On an RTX 5090 with PyTorch 2.15 and Triton
3.9, `TrellisStorage.from_hp` encoded a 4096×5120 Gaussian BF16 weight with group
size 256 at 91 to 103 M weights/s across 3 to 6 bits per weight (102.9 M weights/s at
4.5 bits per weight; 103.5 M weights/s at 4.03 bits per weight) and 52 to 58 M
weights/s at 2.03 to 2.5 bits per weight (1 pair = 2 rows/CTA in shared memory). Search batches target 4
GiB of backpointers; peak allocations on the Gaussian matrix were 4.4 GiB beyond the
input weight. Batches and row chunks give identical results.
