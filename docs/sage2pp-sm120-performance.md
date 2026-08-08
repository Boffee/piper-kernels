# SageAttention2++ SM120 performance findings

This document records the pure-Triton SageAttention2++ optimization investigation on
consumer Blackwell. It explains the selected implementation, the experiments that were
rejected, and the evidence behind the remaining gap to the canonical CUDA kernel. The
results are a regression checkpoint for this hardware and software stack, not a portable
performance guarantee.

```text
Base commit:          9736eca
Checkpoint date:      2026-08-08
GPU:                  NVIDIA GeForce RTX 5090 (SM120)
Driver:               595.71.05
Python:               3.14.6
Torch:                2.12.1+cu130
Triton:               3.7.1
Canonical Sage:       2.2.0 at d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
Primary long shape:   BF16 B1/H8/D128 self-attention
```

## Benchmark protocol

The comparison uses `benchmark_attention.py` and its `prepared_execution` phase. For
both Sage providers, preparation is an identity callable and `run()` includes statistics,
Q/K/V quantization, attention, and the output epilogue. The CUDA-event p50 is therefore a
fair comparison of complete device work. `operator_end_to_end` additionally includes host
dispatch and synchronized wall overhead.

Long-sequence results use one second of warmup and two seconds of measurement per provider.
The provider order was reversed in a second process. The ranges below are the two resulting
p50 values, not p20/p80 confidence intervals. The GPU was checked for unrelated compute
activity before every retained run.

Effective operations are reported as:

```text
noncausal = 4 * B * H * Nq * Nk * D
causal    = 4 * B * H * D * N * (N + 1) / 2
```

This counts the INT8 QK and FP8 PV multiply-adds. It is an algorithmic normalization, not
a claim that every auxiliary scalar instruction is a floating-point operation.

## Selected result

### Long noncausal attention

| Sequence | Triton p50 range | Canonical p50 range | Triton effective throughput | Canonical effective throughput | Throughput parity |
|---:|---:|---:|---:|---:|---:|
| 32K | 8.198–8.266 ms | 7.856–7.888 ms | 532–536 TFLOP/s | 558–560 TFLOP/s | 95.4–95.8% |
| 64K | 31.854–32.060 ms | 29.660–29.802 ms | 549–552 TFLOP/s | 590–593 TFLOP/s | 93.0–93.1% |
| 128K | 124.835–125.809 ms | 115.109–115.834 ms | 559–564 TFLOP/s | 608–611 TFLOP/s | 92.1–92.2% |

At 128K, the selected raw-score recurrence was 1.57–1.92% faster than the otherwise
identical materialized-score recurrence in alternating paired trials.

### Long causal attention

| Sequence | Triton p50 range | Canonical p50 range | Triton effective throughput | Canonical effective throughput | Throughput parity |
|---:|---:|---:|---:|---:|---:|
| 32K | 4.563–4.589 ms | 4.219–4.229 ms | 479–482 TFLOP/s | 520–521 TFLOP/s | 91.9–92.7% |
| 64K | 16.926–16.974 ms | 15.557–15.578 ms | 518–520 TFLOP/s | 565 TFLOP/s | 91.8–91.9% |
| 128K | 64.890–65.023 ms | 59.041–59.061 ms | 541–542 TFLOP/s | 596 TFLOP/s | 90.8–91.0% |

The causal prefix split plus raw-score recurrence improved the previous Triton causal path
by 11.2–11.9% in clean paired trials from 32K through 128K.

### 8K checkpoint

FP16 B1/H8/N8192/D128 noncausal attention remains close to canonical while retaining the
long-sequence specializations:

| Provider | Device p50 [p20, p80] | Complete wall p50 [p20, p80] | Effective throughput | MAE vs SDPA |
|:---|---:|---:|---:|---:|
| Pure Triton SageAttention2++ | 0.614 [0.613, 0.616] ms | 0.639 [0.637, 0.641] ms | 447.37 TFLOP/s | 0.000563 |
| Canonical CUDA SageAttention2++ | 0.607 [0.605, 0.608] ms | 0.602 [0.601, 0.603] ms | 452.91 TFLOP/s | 0.000563 |
| PyTorch SDPA | 1.687 [1.684, 1.703] ms | 1.709 [1.707, 1.714] ms | 162.98 TFLOP/s | 0 |

The device-throughput parity at this checkpoint is 98.9%.

### Numerical quality

The selected changes preserve canonical SageAttention2++ quality. Representative BF16
128K measurements were:

| Mode | Triton MAE / SQNR | Canonical MAE / SQNR |
|:---|---:|---:|
| Noncausal | 0.0001433 / 28.08 dB | 0.0001434 / 28.08 dB |
| Causal | 0.0002805 / 28.71 dB | 0.0002805 / 28.70 dB |

Ragged descriptor storage, strided inputs, and the aligned causal boundary were validated
separately. The raw-score recurrence differs from the materialized form only at FP32 rounding
scale: a direct 4K comparison measured mean error `5.29e-8` and a maximum difference of one
BF16 ULP.

## Accepted implementation changes

### 1. Shift the probability range into the softmax frame

Canonical SageAttention2++ subtracts `log2(448)` from each online-softmax block maximum.
The largest probability is then approximately 448 before FP8 conversion:

```text
block_max = max(scores) - log2(448)
next_max = max(running_max, block_max)
probability = exp2(scores - next_max)
probability_fp8 = cast_fp8(probability)
```

The denominator accumulates the same shifted probabilities, so the constant cancels in the
normalized output. V uses its raw per-channel scale. This replaces the earlier full-tile
`probability * 448` and compensating `value_scale / 448` formulation. It is algebraically
generic across lengths, causality, and GPU architectures.

On the audited SM120 loop, the change removed 64 static FP32 multiply instructions from
the compiled loop body without adding spills.

### 2. Quantize the uniquely owned Q tile in the attention prologue

Each attention CTA uniquely owns its Q rows. On SM120, it now loads the original FP16/BF16
tile, computes one INT8 scale per 32 rows, and retains the quantized tile for the QK loop.
This removes a global INT8 Q round trip and one preprocessing launch.

Restoring external per-warp Q quantization was 0.67% slower at 32K in the broad sweep.
Tighter alternating runs measured it 0.85% slower at 64K and 1.25% slower at 128K. A
simpler row-scale layout also lost despite producing shorter static code. The selected
implementation remains SM120-specific because other architectures use a different Q/K
scale layout.

### 3. Dispatch K and V quantization roles from one grid

On SM12x, K and V still use independent CTAs and their established quantization coordinates.
A single uniform grid assigns each program either a K or V role, removing one host launch
without forcing the two tensors into the same CTA. At 8K this saved about 2 microseconds,
or 0.33%. Other architectures retain their independent per-thread K and V launches.

Directly fusing K and V into the same CTA was slower because its combined working set and
synchronization outweighed the launch saving.

### 4. Reduce raw scores before applying a positive grouped scale

For SM120 per-warp Q/K quantization, the dequantization scale is positive and constant for
each 32-row Q group across one 64-key tile. Therefore:

```text
max(raw_scores * scale) = max(raw_scores) * scale
```

The selected recurrence reduces raw FP32 scores first and forms exponent arguments with:

```text
fma(raw_scores, score_scale, -next_max)
```

This avoids a long-lived materialized scaled-score tensor. At 128K noncausal it reduced the
attention specialization from 22 to 12 compiler-reported spill slots and produced the
1.57–1.92% paired gain. It was 0.23–0.37% slower at a 64K key length, so D128
noncausal dispatch enables it at a 128K key length.

The transformation is generic only when the scale is positive and constant across the
reduced columns. It is invalid for the per-thread K scale layout where scale varies by column.

### 5. Separate the causal prefix from its boundary

For a causal query block, every complete key tile strictly before the block is known valid.
The selected kernel uses a mask-free prefix loop, followed by a masked diagonal or ragged-tail
loop. Only one or two boundary tiles pay for causal comparisons and selects.

The prefix endpoint is rounded down to a 64-key boundary before omitting the mask. This is a
correctness requirement when the query block has 32 rows: starting a K tile at row 32 would
overlap keys and use the wrong 64-key scale group.

The split alone improved the 8K causal kernel by about 6%. Enabling the raw-score recurrence
only in the mask-free prefix added roughly another 6% at long lengths. D128 causal dispatch
selects it from a 32K key length onward.

### 6. Retain the measured descriptor schedule

The selected D128 SM120 attention tile remains:

```text
BLOCK_M = 128
BLOCK_N = 64
warps   = 4
stages  = 3
K/V loads = tensor descriptors
```

For the measured B1/H8 schedule, causal calls retain at most 64 query rows through 4K,
then use 128 rows where K/V reuse outweighs the larger accumulator footprint. Other B/H
shapes may retain 64 or 32 rows when `select_query_block()` needs more grid parallelism.

## Resource and code-generation observations

| Specialization | Registers/thread | Spill slots | Shared/CTA | Resource ceiling |
|:---|---:|---:|---:|---:|
| 8K noncausal | 255 | 22 | 49,304 B | 2 CTAs/SM, 8 warps |
| 128K noncausal raw-score | 255 | 12 | 49,304 B | 2 CTAs/SM, 8 warps |
| 128K causal split/raw-prefix | 255 | 18 | 49,304 B | 2 CTAs/SM, 8 warps |

The 8K global spill count is higher than an earlier external-Q kernel because query
quantization now lives in the prologue. Its steady attention loop has no local stores, which
is more important than the whole-program spill count for long sequences.

Each key tile contains the expected 64 signed-INT8 QK MMA instructions and 64 E4M3 PV MMA
instructions in both Triton and canonical CUDA. Occupancy is not the main difference: both
are register-limited to two CTAs per SM. The remaining hot-loop gap is ordinary instruction
and layout work around those tensor operations.

The largest identified difference is FP8 probability-fragment packing. Triton emits roughly
32 shuffles, 32 permutes, and 18 selects per tile, versus about 8 shuffles and one permute in
canonical CUDA. Triton also reloads Q MMA fragments from shared memory inside the K/V loop,
where canonical retains its Q fragments across the loop.

## Roofline interpretation

The RTX 5090 architecture specification reports 838 dense FP8 Tensor TFLOP/s with FP16
accumulation, 838 dense INT8 Tensor TOPS, 1,792 GB/s of memory bandwidth, and 96 MiB of L2.
The paired INT8 QK and FP8 PV work therefore has a dense mixed-tensor ceiling of 838 effective
TFLOP/s. See the
[NVIDIA RTX Blackwell architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

At 128K:

| Mode | Triton / dense peak | Canonical / dense peak | Dense compute floor |
|:---|---:|---:|---:|
| Noncausal | 67.3% | 72.9% | 83.97 ms |
| Causal | 64.7% | 71.1% | approximately 41.99 ms |

A naive DRAM model is misleading. With `BLOCK_M=128`, logical K/V traffic gives 256
effective operations per byte and a 459 TFLOP/s DRAM-only roof. Both implementations exceed
that value. One head's quantized K and V tensors occupy about 32 MiB at 128K, so scheduler-local
reuse through the 96 MiB L2 materially reduces physical DRAM traffic.

Nsight Compute counters were unavailable because the machine denied performance-counter
access. These conclusions therefore use published hardware ceilings, timings, compiler
resources, SASS, and Nsight Systems kernel durations rather than measured DRAM/L2 counters.

## Rejected experiments

The table records negative results so future work does not repeat source shapes that were
already measured. A lower static instruction count was never accepted without a runtime win.

| Experiment | Result and reason for rejection |
|:---|:---|
| Descriptor stages 1/2/4 | At 64K: +13.5% / +2.6% / +27.8%. At 128K: +11.7% / +1.8% / +26.5%. Stage 4 also reduced residency to one CTA/SM. |
| Pointer K/V loads | Best pointer form was +9.4% at 64K and +5.8% at 128K. Descriptor loads remain selected. |
| `BLOCK_M=256`, eight warps | +23.3% at 64K and +11.5% at 128K despite halving logical K/V traffic. It lost the second resident CTA. |
| `BLOCK_M=256`, four warps | Compiled with 324 spills and only four resident warps; timing was skipped. |
| External or row-shaped Q scales | External Q was 0.85% slower at 64K and 1.25% slower at 128K in tighter alternating runs. The row-shaped inline form also lost. |
| Same-CTA K/V fusion | Quantization took 36.9 microseconds versus 32.5 microseconds separately and made the full operator about 1.3 microseconds slower. |
| One reusable host workspace | Device time was neutral and synchronized wall time regressed by 3.3 microseconds. |
| Broad aligned-mask removal | Removed 247 static instructions but added four spill slots and two hot-loop local stores; runtime regressed about 10 microseconds at 8K. |
| Narrow aligned-mask removal | Retained the stack size but still introduced a hot local store/load dependency. |
| Raw-FP32 recurrence below its gate | 0.23–0.37% slower at 64K noncausal, despite winning at 128K. |
| Raw-integer maximum | Produced pathological compiler lowering: about 492 ms at 64K and 1,919 ms at 128K. |
| Load V before FP8 probability conversion | Reduced static probability permutes but regressed the integrated inline-Q kernel by about 17 microseconds at 8K. |
| Probability/PV layout reshapes | Transpose, split, and row/K permutations added shared traffic, permutes, or loop-local spills. No candidate passed the resource filter. |
| Eight-warps, `BLOCK_M=64`, or alternate causal stages | None beat the selected two-CTA/four-warp descriptor schedule. |
| Loop unrolling or broad LICM | Increased spills or provided no runtime benefit. |
| Cluster launch or warp specialization | The current Triton compiler rejected these forms during lowering. |

## Methodological lessons

- Compare complete device work. A faster attention kernel can be offset by slower quantization,
  and a launch-saving prototype can lose inside its merged kernel.
- Alternate provider order and bracket candidates with controls. GPU temperature and unrelated
  processes were large enough to contaminate microsecond-scale conclusions.
- Inspect where spills occur. Prologue spills are amortized; local stores inside every 64-key
  iteration are often fatal at 128K.
- Static SASS counts are diagnostic, not an objective. Several shorter kernels ran slower because
  register allocation or instruction scheduling changed.
- Preserve algebraic preconditions explicitly. Reducing before scaling requires a positive,
  row-constant scale; causal mask removal requires complete aligned key tiles.
- Validate ragged lengths, strided inputs, both head dimensions and dtypes, and causal query-block
  sizes. The `BLOCK_M=32` regression was invisible at the long `BLOCK_M=128` benchmark shape.
- Keep causal and noncausal source bodies separate when codegen differs materially. A DRY refactor
  can perturb Triton register allocation even when the generated mathematics is identical.

## Reproduction

Install the revision-pinned canonical benchmark dependency:

```shell
TORCH_CUDA_ARCH_LIST=12.0 uv sync --group benchmark
```

Run long noncausal and causal comparisons:

```shell
uv run python benchmarks/benchmark_attention.py \
  --providers pure-triton-sage2pp canonical-cuda-sage2pp \
  --sequence 32768 65536 131072 \
  --batch-size 1 --heads 8 --head-dim 128 --dtype bfloat16 \
  --warmup-ms 1000 --measurement-time-ms 2000 --seed 0 \
  --json artifacts/sage2pp-sm120-long-noncausal.json

uv run python benchmarks/benchmark_attention.py \
  --providers pure-triton-sage2pp canonical-cuda-sage2pp \
  --sequence 32768 65536 131072 \
  --batch-size 1 --heads 8 --head-dim 128 --dtype bfloat16 --causal \
  --warmup-ms 1000 --measurement-time-ms 2000 --seed 0 \
  --json artifacts/sage2pp-sm120-long-causal.json
```

Repeat each command with the provider order reversed. Run one provider and shape per process for
compiler inspection:

```shell
uv run python benchmarks/benchmark_attention.py \
  --providers pure-triton-sage2pp --sequence 131072 \
  --batch-size 1 --heads 8 --head-dim 128 --dtype bfloat16 \
  --compiler-report --compiler-json artifacts/sage2pp-sm120-128k-compiler.json
```

## Validation and remaining work

The selected implementation passed the complete portable and GPU suites, Ruff, Pyright,
`git diff --check`, and package builds. GPU coverage explicitly forces the normally long-only
raw-score recurrence at a small shape so both causal and noncausal variants remain practical to
test in CI.

The next credible improvements are below the ordinary launch-policy layer:

1. Teach Triton to pack the E4M3 probability fragment directly into the PV dot operand layout.
2. Retain Q MMA fragments across the K/V loop without creating additional accumulator spills.
3. Keep scalar K scales in registers rather than software-pipelining them through shared memory.
4. Explore a canonical-like single-buffer asynchronous K/V pipeline in the compiler/backend.
5. Re-run the matrix on SM89, other head counts, and D64 before expanding the measured D128
   dispatch policy.

Until one of those changes is available, the selected descriptor/four-warp/three-stage schedule
is the measured local optimum. Further launch-grid tuning is unlikely to close the remaining
8–10% long-sequence gap.
