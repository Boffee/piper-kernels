# Trellis-coded weights

Trellis coding stores a ConvRot INT8 weight in 3 to 6 bits per weight. It is a
storage and transfer format: `TrellisStorage.from_hp` encodes a weight once, and
`ConvRotInt8Tensor.from_trellis` and `copy_from_trellis_` decode it into ordinary
ConvRot INT8 storage, so execution uses the INT8 kernels unchanged. This README
owns the format and its encoder; the
[weights guide](../../../docs/weights.md#encode-and-decode-trellis-coded-storage)
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
Tests confirm this on NVIDIA hardware; on AMD the kernel has so far only been
compiled (gfx942, gfx1201), not run. Scales must not
be NaN encodings (`0x7F`, `0xFF`); this is a caller precondition, not a runtime
check.

## Layout limits

| Value | Supported | Why |
|---|---|---|
| Block size `T` | 16 to 256 | A 16-bit window never wraps twice; one decode tile covers a block |
| Bits per block `B` | Whole bytes from `3·T` to `6·T` bits | 3- to 6-bit steps, the rates the encoder supports |

Whole bytes cost at most 1/32 bit per weight of rate granularity at `T = 256`, and
let the rate come from the tensor shape instead of more metadata.

The block size must divide the weight's input width. The encoder chooses the
largest multiple of 16 up to 256 that divides it: 256 for most widths, 224 for 2688.

## Encoding

`TrellisStorage.from_hp(weight, group_size=..., bits_per_weight=...)` rotates the
weight in FP32 and encodes each block. The rate counts the block's FP8 scale: a
block of `T` weights stores `⌊bits_per_weight · T⌋ - 8` bits, rounded down to whole
bytes. At 4.5 bits per weight a 256-weight block takes 143 bytes, 4- and 5-bit
steps; at 3.5 it takes 111 bytes, 3- and 4-bit steps.

The [encoder](_encoder.py) minimizes each block's squared error in the rotated
basis against the exact INT8 decode, without calibration data. A coarse search
with 12-bit states picks each block's scale and each row's scale; one tail-biting
Viterbi pass on the stored 16-bit trellis then sets the walk. The module docstring
describes the stages and where they come from.

On Gaussian rows, which ConvRot-rotated weights resemble, the weight SQNR is:

| Bits per weight | 3.03 | 3.5 | 4.03 | 4.5 | 5.5 | 6.03 |
|---|---|---|---|---|---|---|
| SQNR (dB) | 17.5 | 20.3 | 23.5 | 26.3 | 32.2 | 35.3 |

Rows with heavy outliers encode worse: the search starts each block at a scale
from its RMS.

## Decode implementation

[`triton.py`](triton.py) decodes a tile of 8 blocks per program. Each weight
computes its offset in closed form and gathers the at most three bytes holding its
window. The few windows within 16 bits of the stream's end join its last bits to
its first. The kernel takes the rate as runtime arguments, so one compiled kernel
serves every rate.

On an RTX 5090 it decodes about 600 G weights/s at every rate. That consumes
packed bytes about ten times faster than a pinned PCIe copy delivers them (33 GB/s
measured), so blocks are stored packed rather than padded to words. For the same
reason the launch shape (8 blocks per program, 4 warps) was not tuned: decode is
not on the transfer's critical path. It has not been measured on AMD, including
wave64 targets such as gfx942.
[`_reference.py`](_reference.py) is the PyTorch oracle; tests check both against
blocks encoded by the research encoder.

## Encode implementation

The [Viterbi search](_viterbi.py) finds the exact minimum over every walk of a
block and a few candidate scales, with pinned FP32 arithmetic. Its forward runs
in one of two ways, with identical results:

- **Persistent forward** ([`_gluon_forward.py`](_gluon_forward.py)): one Gluon
  launch runs every step of a batch, keeping each row's prefix costs in shared
  memory. It needs an NVIDIA GPU from sm_80 with 99 KiB of opt-in shared memory per
  CTA, and Triton 3.9, whose Gluon it was written and measured for. Its inline PTX
  is NVIDIA-only.
- **Step loop:** one Triton launch per step. It runs on other NVIDIA GPUs and is
  the persistent forward's bitwise reference.

Encoding requires an NVIDIA GPU: there is no ROCm encoder, since checkpoints are
encoded once, offline. On an RTX 5090 with Triton 3.9, a 4096×5120 weight encodes
at 37 to 46 M weights/s from 3 to 6 bits per weight with the persistent forward,
about 13 minutes for 34 G weights. The step loop runs at 6.6 M weights/s.
Search batches hold up to 4 GiB of backpointers, for a peak of about 4.6 GiB
beyond the weight. Batches and row chunks do not change the result.
