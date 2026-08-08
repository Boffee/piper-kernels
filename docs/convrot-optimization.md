# ConvRot core optimization findings

This document records the ConvRot investigation on the `convrot-optimization`
branch. It is an experimental reference, not a request to merge the branch
mechanically. The intended next step is to reimplement the selected design from
the main branch with a smaller, review-oriented patch.

Start with [the incremental adoption roadmap](convrot-adoption.md). It separates
the branch-implemented ConvRot operator from the later benchmark-only
[follow-on experiments](convrot-follow-on-experiments.md).

```text
Base commit:       acdd9a6
Checkpoint date:   2026-08-07
Python:            3.14.6
Implementation:    pure Triton/Python; Comfy Kitchen is benchmark-only
```

## Scope and notation

For a linear projection:

```text
activation A:  [M, K]
weight W:      [N, K]
output Y:      [M, N]

M = flattened batch and sequence rows
K = input feature dimension
N = output feature dimension
```

For video DiTs, `M` contains the spatial tokens across frames. A 128K sequence
therefore means `M = 131072`; `N` remains the output width of the projection.

The investigated MiniMax H3 shapes are:

| Projection | M (5-second preset) | N | K | Input activation |
|:---|---:|---:|---:|:---|
| QKV | 37,710 | 21,504 | 5,376 | none |
| Attention output | 37,710 | 5,376 | 7,168 | none |
| MLP FC1 | 37,710 | 28,672 | 5,376 | none |
| MLP FC2 | 37,710 | 5,376 | 14,336 | SwiGLU over raw `[gate | up]` |

All four model projections are bias-free. The exact row count can vary slightly
with model inputs because text and reference-conditioning tokens are added to the
visual-token rows.

## End-to-end math

Let `R` be a block-diagonal normalized Hadamard rotation with one `H256 / 16`
block per 256 input features. The packed weight already stores its rotated INT8
representation and one FP32 scale per output channel.

For each activation row:

```text
rotated[m, :] = A[m, :] R
act_scale[m]  = max(max(abs(rotated[m, :])) / 127, 1e-30)
act_q[m, :]   = clamp(round(rotated[m, :] / act_scale[m]), -128, 127)

acc[m, n] = sum_k int32(act_q[m, k]) * int32(weight_q[n, k])
Y[m, n]   = cast(acc[m, n] * act_scale[m] * weight_scale[n] + bias[n])
```

The optimized path retains one scale for the complete row. It does not change to
per-group or per-chunk activation scaling.

For FC2, the fused input is `[gate | up]`. Piper deliberately preserves the
logical eager boundaries:

```text
gate_act = cast_input_dtype(silu(float32(gate)))
A        = cast_input_dtype(float32(gate_act) * float32(up))
```

The result then enters the same rotation and rowwise quantization. Comfy
Kitchen's fused CUDA kernel instead keeps SwiGLU and rotation in FP32 and uses
CUDA fast math, so the two fused implementations are close but not expected to
be byte-identical.

## Branch-implemented core design

### 1. Factorize every H4 quartet

The important mathematical change was to compute all four outputs from a
quartet together:

```text
p = a + b        q = a - b
r = c + d        s = c - d

y0 = p + s       y1 = p - s
y2 = q + r       y3 = r - q
```

This is eight add/subtract operations per quartet. The original flat expression
computed output lanes independently and repeatedly gathered the same four
inputs. Expressing the tensor as `(outer, stride, 4)` quartets lets Triton see
operand reuse and keeps the compiler dependency graph tractable.

This layout change was the decisive unlock. The first flat one-pass prototype
compiled into pathological state and could take roughly 2--8 ms at the H3
widths. With the factorized representation, the same exact one-pass formulation
takes about 0.4--1.1 ms at 37,710 rows.

### 2. Fuse rotation, row maximum, and INT8 emission

The split preparation path moves at least seven bytes per logical value:

```text
rotation:      BF16/FP16 read (2) + rotated write (2)
quantization:  rotated read (2) + INT8 write (1)
total:         7 bytes/value, excluding the small scale output
```

The selected fused path moves three:

```text
BF16/FP16 input read (2) + INT8 output write (1)
```

One Triton program owns a complete logical row, applies only the four H4 stages
inside each H256 group, performs the row-wide maximum, and emits INT8. K values
are padded to the next power of two for the program shape, but the maximum
butterfly stride is 64, so padded groups cannot mix with valid groups.

The row is rounded back to the logical input dtype before the maximum and
normalization. This preserves the established split-path quantization semantics.

### 3. Retile and reorder the INT8 GEMM

Large-M H3 projections use a `128 x 256 x 128` output/K tile, eight warps, and
three stages. Exact tile multiples bypass masks. The flattened launch grid groups
M tiles so adjacent programs reuse activation data across N:

```text
narrow N:  strict M-major scheduling
wide N:    a band of M tiles for activation and weight reuse
```

The wide QKV and FC1 projections gain most of their absolute end-to-end time
from this GEMM work. For the narrower attention-output and FC2 projections,
activation preparation supplies the win and offsets a small GEMM disadvantage
relative to Comfy Kitchen.

### 4. Fuse raw FC2 input-SwiGLU only at an explicit API boundary

`linear_input_act(input, weight, "swiglu", bias)` accepts a raw `[gate | up]`
tensor whose final dimension is `2K`. It avoids implicit graph matching and
leaves ordinary `torch.nn.functional.linear` behavior unchanged.

The best possible separate materialization moves at least nine bytes per logical
value: two BF16 inputs, one BF16 activated write, that BF16 read, and one INT8
write. The fused path moves five bytes: two BF16 inputs and one INT8 output.

For `K = 14336` SwiGLU on Blackwell, 16 warps remove the spills caused by the
extra exponential and multiply. The wider launch is guarded to large row counts
and compute capability 12 or newer. Other cases use eight or four warps.

### 5. Use 64-bit address products for large sequences

At `M = 131072`, two H3 tensors cross signed-32-bit element indexing:

```text
QKV output:           M * N = 2,818,572,288 elements
FC1 output:           M * N = 3,758,096,384 elements
FC2 raw SwiGLU input: M * 2K = 3,758,096,384 elements
```

Row indices are cast to 64 bits before multiplying by row strides in fused
preparation and GEMM pointer expressions. This is a correctness requirement, not
a speed optimization.

## Branch runtime dispatch and fallbacks

The exact one-pass path is intentionally conservative:

| Condition | Selected behavior |
|:---|:---|
| CUDA compute capability 12 or newer | eligible for measured SM120 schedules |
| group size 256 | eligible for one-pass fusion |
| FP16 or BF16 activation | eligible for one-pass fusion |
| at least 512 flattened rows | eligible for one-pass fusion |
| next power of two of K at most 16,384 | eligible for one-pass fusion |
| older CUDA, other groups/dtypes, small M, or larger K | conservative split fallback |
| explicit SwiGLU with the same H256 limits | fused SwiGLU preparation |
| unsupported device or shape | portable PyTorch materialization and reference path |

The split fallback remains useful for correctness, small shapes, non-H256 groups,
and compiler/resource limits. On SM120, its large H256 rotation uses a small
dense tensor-core matmul; smaller and older-device cases use the factorized
butterfly kernels.

## Benchmark environment and protocol

Unless noted otherwise:

```text
GPU:              NVIDIA GeForce RTX 5090 (SM120)
Torch:            2.12.1+cu130
Triton:           3.7.1
Comfy Kitchen:    0.2.28
activation dtype: BF16
group size:       256
timing:           warmed CUDA events, p50 [p20, p80]
```

In `benchmark_convrot.py`, “prepared execution” means the complete warmed public
operator on already-created inputs: output/scratch allocation, activation
preparation, INT8 GEMM, and scaled epilogue. It does not mean GEMM-only timing.
Inputs, packed weights, and scales are seeded synthetic tensors rather than
checkpoint distributions.

Piper is asserted against the portable reference with default dtype tolerances,
except the explicitly fused SwiGLU output, which must remain within 1% relative
L2 error and whose preparation codes and scales have stricter direct tests.
Comfy Kitchen is rejected if the sampled output has non-finite mismatches or
exceeds 2% relative L2 error.

Comfy Kitchen is an optional benchmark dependency. On NVIDIA it compares against
its native CUDA ConvRot preparation and CUTLASS INT8 GEMM, not its ordinary
Triton fallback. Its relevant fused kernel is in
[`int8_linear.cu`](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.28/comfy_kitchen/backends/cuda/ops/int8_linear.cu).

### MiniMax H3 5-second projection results

Command:

```shell
uv run --with comfy-kitchen==0.2.28 \
  python benchmarks/benchmark_convrot.py \
  --preset minimax-h3-5s \
  --compare-comfy-kitchen \
  --warmup-ms 200 \
  --measurement-time-ms 1000
```

| Projection | Piper | Comfy Kitchen | CK / Piper |
|:---|---:|---:|---:|
| QKV | 12.944 ms | 15.438 ms | 1.19x |
| Attention output | 4.678 ms | 5.045 ms | 1.08x |
| MLP FC1 | 17.348 ms | 20.646 ms | 1.19x |
| MLP FC2 including SwiGLU | 9.989 ms | 10.495 ms | 1.05x |

### Preparation phase results at 37,710 rows

Command:

```shell
uv run --with comfy-kitchen==0.2.28 \
  python benchmarks/benchmark_convrot_preparation.py \
  --rows 37710 \
  --in-features 5376 7168 14336 \
  --compare-comfy-kitchen \
  --warmup-ms 100 \
  --measurement-time-ms 300
```

| K | Split Piper | Fused Piper | Comfy Kitchen | CK / fused Piper |
|---:|---:|---:|---:|---:|
| 5,376 | 1.163 ms | 0.399 ms | 0.832 ms | 2.09x |
| 7,168 | 1.549 ms | 0.529 ms | 0.909 ms | 1.72x |
| 14,336 | 3.098 ms | 1.094 ms | 1.682 ms | 1.54x |

The fused kernel sustains approximately 1.48--1.54 TB/s of algorithmic minimum
traffic. This is a normalization of the minimum bytes above, not a measured DRAM
counter.

Fused SwiGLU preparation is reproducible with:

```shell
uv run --with comfy-kitchen==0.2.28 \
  python benchmarks/benchmark_convrot_preparation.py \
  --rows 37710 --in-features 14336 --input-act swiglu \
  --compare-comfy-kitchen --warmup-ms 100 --measurement-time-ms 300
```

It measured 1.705 ms for Piper and 2.029 ms for Comfy Kitchen, a 1.19x ratio.

### 128K-sequence results

`128K` means `M = 131072`. Each shape was run in a fresh process, with both
providers initialized and warmed independently on the same seeded tensors. The full
portable reference was not timed because its FP32 QKV epilogue alone requests a
10.5 GiB temporary. Correctness was checked separately on sampled rows, including
rows immediately around the signed-32-bit address boundary and the final row.

Use `--skip-reference-timing` for each custom shape. For example:

```shell
uv run --with comfy-kitchen==0.2.28 \
  python benchmarks/benchmark_convrot.py \
  --rows 131072 --out-features 21504 --in-features 5376 \
  --no-bias --compare-comfy-kitchen --skip-reference-timing \
  --warmup-ms 100 --measurement-time-ms 300
```

Repeat in separate processes for `(N, K)` of `(5376, 7168)` and
`(28672, 5376)`. FC2 uses `(5376, 14336)` plus `--input-act swiglu`. The sampled
reference mode records rows around both `2^31` boundaries and the final row.

| Projection | Piper | Comfy Kitchen | CK / Piper | Piper live tensors |
|:---|---:|---:|---:|---:|
| QKV | 45.432 ms | 53.467 ms | 1.18x | 7.33 GiB |
| Attention output | 16.257 ms | 17.287 ms | 1.06x | 3.97 GiB |
| MLP FC1 | 60.126 ms | 70.324 ms | 1.17x | 9.11 GiB |
| MLP FC2 including SwiGLU | 34.319 ms | 35.454 ms | 1.03x | 10.14 GiB |

Preparation at the same row count:

| Preparation | Piper | Comfy Kitchen | CK / Piper |
|:---|---:|---:|---:|
| K=5,376 | 1.373 ms | 2.871 ms | 2.09x |
| K=7,168 | 1.822 ms | 3.131 ms | 1.72x |
| K=14,336 | 3.735 ms | 5.811 ms | 1.56x |
| SwiGLU + K=14,336 | 5.864 ms | 7.026 ms | 1.20x |

The PyTorch-visible operation allocation is identical for Piper and Comfy
Kitchen. Comfy Kitchen additionally uses about 21 MiB of raw-CUDA Stream-K
workspace on the tested 170-SM GPU for QKV, attention output, and FC2; that
workspace is not visible to `torch.cuda.max_memory_allocated`.

## Resource observations

For ordinary one-pass BF16 preparation on SM120:

| K | Warps | Registers/thread | Compiler spills/thread | Shared/CTA |
|---:|---:|---:|---:|---:|
| 5,376 / 7,168 | 4 | 166 | 0 | 32 KiB |
| 14,336 | 4 | 255 | 24 | 32 KiB |

The 14,336-wide kernel remains faster with four warps despite the spills. Capping
registers or increasing ordinary-preparation warps regressed latency. SwiGLU at
the same K changes the result: 16 warps use about 90 registers/thread without
spills and are approximately 2--3% faster than four or eight warps at large M.

Comfy Kitchen uses a 1,024-thread CTA and an FP32 shared row buffer. Its dynamic
shared allocation is `(K + 8192) * 4` bytes: 54,272, 61,440, and 90,112 bytes for
the three H3 widths. That normally limits it to one CTA per SM. The exact SM120
binary inspected during this study reported 29 registers/thread without an input
activation and 37 with SwiGLU, with no local stack spill.

## Core formulation decisions

| Experiment | Finding | Decision |
|:---|:---|:---|
| Flat gather-based fused row | Huge expression graph and pathological spills/state | Replace with explicit quartet factorization |
| Split dense H256 rotation | Tensor cores improved the fallback; a 256-wide output tile avoided duplicate input loads | Retain for split fallback, not the main H3 path |
| Two-pass recompute | Avoids the BF16 intermediate but performs the rotation twice; minimum traffic is 5 bytes/value | Not needed after exact one-pass became fast |
| Partial-max hierarchy | Useful diagnostic, but standalone quantization was already near memory bandwidth | Keep benchmark insight; do not add another buffer/pass |
| Per-group or 1,792-feature scales | Very fast and likely more accurate, but changes GEMM accumulation and quantization semantics | Reject for the exact-rowwise design |
| Persistent multi-CTA row coordination | Can avoid K-wide CTA state, but needs atomics, a global completion protocol, and scheduler assumptions | Defer |
| Triton experimental Gluon explicit shared memory | Could reproduce Comfy Kitchen's layout exactly, but adds an experimental dependency and was unnecessary | Defer |
| Register caps at K=14,336 | Every tested cap below the compiler default slowed the kernel | Leave uncapped |
| 32-warps ordinary preparation | More occupancy did not overcome synchronization/layout overhead | Keep four warps |
| 16-warps large SwiGLU | Removed spills and improved large-M Blackwell latency | Retain behind an architecture/size guard |
| Fusing eager SwiGLU | Removes the dominant remaining real-FC2 materialization cost | Retain explicit public API |

The useful idea borrowed from attention optimization was hierarchical thinking:
separate global-memory traffic, on-chip state, reduction dependencies, compiler
resource usage, and downstream GEMM cost. Attention-specific persistent barriers
and partial statistics were not directly required once the H4 algebra exposed
the correct local reuse.

## Correctness and semantic checks

The branch checks:

- fused rotation/quantization against the split path for FP16 and BF16 at K of
  512, 5,376, 7,168, and 14,336;
- fused SwiGLU preparation against materialized eager SwiGLU;
- ordinary and fused-SwiGLU public linears against the portable reference;
- grouped GEMM tile ordering and exact-dimension mask specialization;
- `torch.compile(..., fullgraph=True)` for the explicit SwiGLU API;
- public package exports and benchmark shape/traffic calculations.

At 128K, sampled QKV rows 99,864--99,866 and 131,071 straddle the first
overflowing output row and matched the reference exactly. Sampled SwiGLU rows
74,898--74,900 and 131,071 straddle the first overflowing raw-input row and also
matched exactly.

FP16 SwiGLU can differ from materialized PyTorch by one INT8 code in rare values
because `tl.exp` and PyTorch's SiLU implementation are not guaranteed to use the
same approximation. Tests therefore constrain preparation codes/scales directly
and use an output tolerance for the fused public operation.

## Core adoption ladder

For the integration branch, reimplement the design in this order instead of
copying the experimental diff wholesale:

1. Add the benchmark presets and preparation-phase tool, then capture the main
   branch baseline.
2. Add the factorized H4 helper and verify it against `rotate_groups` for every
   supported group and logical dtype.
3. Add the exact one-row fused preparation kernel behind the conservative H256
   dispatch guard. Keep the existing split path unchanged as fallback.
4. Benchmark and land GEMM tiling/grid-order changes independently from
   preparation, so their wins and portability are reviewable.
5. Add `linear_input_act(..., "swiglu")` as a separate explicit API change, with
   portable fallback and `torch.compile` coverage.
6. Use 64-bit row-base products from the first implementation, including output
   pointers and raw `2K` SwiGLU inputs.
7. Rerun correctness and performance on at least one pre-Blackwell NVIDIA GPU
   before generalizing the SM120 tuning rules.

Suggested integration boundaries are therefore: benchmark infrastructure,
factorized/fused preparation, GEMM scheduling, then SwiGLU API/fusion.

## Known limitations and follow-ups

- Performance tuning was performed on one RTX 5090. Correctness is portable,
  but the launch thresholds and wide-SwiGLU warp choice need measurements on
  Ada, Ampere, Hopper, and AMD before being treated as general defaults.
- The ordinary benchmark times the complete portable reference by default. At
  128K, use one shape per process with `--skip-reference-timing`; otherwise
  reference intermediates and retained provider outputs can exhaust 32 GiB.
- Compiler resource counts are Triton-version-specific and should be rechecked
  when upgrading from Triton 3.7.1.
- Comfy Kitchen's fast-math FP32 SwiGLU semantics differ from Piper's
  eager-compatible logical dtype boundaries. Quality comparisons must state
  which semantic target is used.
- The fused path intentionally stops at padded K of 16,384. Larger K produces
  severe compiler spill pressure and must retain the split fallback unless a
  different formulation is introduced.
