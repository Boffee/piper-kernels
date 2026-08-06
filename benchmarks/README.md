# Benchmarks

Operator benchmarks live here rather than in the correctness test suite. Each benchmark
should report hardware, dtype, shapes, cold compilation time, and warmed execution time.

Run the ConvRot provider comparison with:

```shell
uv run python benchmarks/benchmark_convrot.py
```

Use `--help` to select activation rows, weight dimensions, group size, dtype, and timing
windows. The script verifies exact agreement before reporting Triton and reference timings.

Run the end-to-end SageAttention2++ comparison with:

```shell
uv run python benchmarks/benchmark_sage_attention.py
```

The Sage timing includes K smoothing and Q/K/V quantization. Use `--help` to select query
sequence lengths, a fixed `--kv-sequence` for cross-attention, dtype, head dimension, and
causal mode. The comparison reports warmed PyTorch SDPA
latency and error relative to SDPA as well as cold Triton compilation time. Warmed latency is
reported as the median with p20/p80 bounds; increase `--repeat-ms` for a larger timing sample.

The official SageAttention2++ CUDA implementation is an opt-in benchmark-only dependency. It
is pinned to an exact upstream revision, and its build needs NVCC plus an explicit target when
the desired GPU is not visible. Install it and include it in the comparison with:

```shell
# RTX 4090 (SM89)
TORCH_CUDA_ARCH_LIST=8.9 uv sync --group benchmark

# RTX 5090 (SM120)
TORCH_CUDA_ARCH_LIST=12.0 uv sync --group benchmark

uv run --group benchmark python benchmarks/benchmark_sage_attention.py --canonical
```

The canonical comparison runs both official INT8 QK + FP8 PV accumulator strategies:
SageAttention2++'s `fp32+fp16` path and SageAttention2's `fp32+fp32` path. It uses the official
production Q/K granularity for each target: per-thread on SM89 and per-warp on SM12x. Building
for SM89 requires CUDA 12.4 or newer; SM120 requires CUDA 12.8 or newer. The dependency stays
out of the default development group so CPU-only development and CI do not need a CUDA
compiler.

On RTX 5090 with FP16 B1/H8/D128 noncausal self-attention at N=8192, a one-second warmed sample
measured canonical Sage2++ at 0.606 ms and Piper's pure-Triton Sage2++ at 0.637 ms, leaving Triton
about 5.1% slower. Their mean absolute error was identical at 0.000563; maximum error was
0.004890 canonical and 0.004883 Piper. Canonical Sage2 measured 0.702 ms and PyTorch SDPA
1.687 ms. The matched native-UINT8 scale-forward experiment measured 0.7657 ms hot and
0.8119 ms including preparation, with 36.23 dB SQNR, so it remains slower than both
production FP8 Sage2++ implementations at this length.

The production Triton Sage2++ path now preserves tensor descriptors for ragged SM120 lengths by
rounding only its quantized INT8 K and FP8 V storage up to K64. The attention kernel still receives
the original semantic K length, so its score mask is unchanged and the zero-filled storage tail
never enters softmax. On BF16 B1/H24/D128, reversed-order E2E comparisons against the previous
masked pointer fallback improved N=8191 from 4.40-4.42 to 1.924-1.925 ms and N=8193 from
4.41-4.42 to 1.950-1.952 ms. At N=32769 the same comparison improved 62.09-62.37 to
25.999-26.041 ms. Outputs were bitwise identical. Aligned shapes keep their existing allocation
and descriptor path. Predicate-free and split-query variants were not selected for FP8 Sage2++:
they regressed 1.6-2.5% around 8K and 0.7-2.5% around 32K despite helping the more complicated
per-key INT8 recurrence.

Run the experimental Sage2++ INT4-range ConvRot QK comparison with:

```shell
uv run python benchmarks/benchmark_sage_int4_convrot.py
```

This keeps Sage2++'s FP32 softmax and FP8 PV path, but rotates Q/K and limits their integer
codes to the symmetric INT4 range. Triton currently stores those codes as INT8 and emits an
INT8 `tl.dot`, so the reported latency measures quantization quality and ConvRot overhead—not
native packed-INT4 MMA performance.

Run the corresponding 8+4 PV experiment with:

```shell
uv run python benchmarks/benchmark_sage_uint4_pv_convrot.py
```

This comparison keeps INT8 QK and evaluates per-row UINT4 P with block-scaled,
INT4-range V. It includes no rotation, feature-axis V rotation with an inverse
output rotation, and paired block-local P/V rotation with an affine UINT4
zero-point correction. All four-bit codes remain unpacked in INT8 storage.

Isolate the PV datatype and accumulator cost with the direct signed-INT8 baseline:

```shell
uv run python benchmarks/benchmark_sage_int8_pv.py
```

This keeps Sage2++'s INT8 QK and FP32 online softmax. It compares FP8 PV with FP16 and FP32
partial accumulators against a fastest-case fixed-scale INT8 control and a quality-oriented INT8
variant using block-local P normalization and 64-token V scales. For D128 it also tunes an exact
two-D64-output-slice fixed-INT8 schedule. The script reports both prequantized-kernel and full
Q/K/V preprocessing latency, plus SQNR.

On RTX 5090, BF16 B1/H24/D128 noncausal self-attention, the short-context split schedule uses
M64/N128/S2 at N=512 and M64/N64/S3 through N=4096. When the self-attention grid consists entirely
of full tiles, removing query/key boundary predicates lets Triton fuse the FP32 numerator rescale
and add into FFMA. This lowered the 4K split kernel from 0.461 to 0.429 ms and the 512 kernel from
0.0188 to 0.0178 ms. From N=8192, M128/N64/S3 with two D64 PV output slices is fastest. It spills
28 slots, but the friendlier accumulator layout outweighs that traffic and is faster than the
full-D128 layout. The public experimental baseline selects this predicate-free path only for full-
tile SM120 noncausal self-attention; RTX 40-series, causal, boundary, and non-D128 shapes retain the
safe reference schedule until separately tuned.

A same-process comparison against canonical SageAttention2 (FP8 PV with FP32 accumulation) gives
the remaining long-context gap directly:

| N | canonical SA2 hot ms | optimized fixed INT8 hot ms | hot gap | canonical SA2 E2E ms | fixed INT8 E2E ms | E2E gap |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1.6038 | 1.6278 | +1.50% | 1.8129 | 1.8004 | -0.69% |
| 32768 | 24.2118 | 24.7916 | +2.39% | 25.0973 | 25.4297 | +1.32% |
| 131072 | 385.8113 | 390.8620 | +1.31% | 386.5820 | 391.1973 | +1.19% |

Each entry is the mean of two long same-process measurements in opposite provider order. The hot
comparison uses prequantized inputs; E2E includes smoothing and Q/K/V quantization. The optimized
pure-Triton fixed-INT8 path is therefore within 2.4% of canonical SA2 hot and within 1.4% E2E on
all three long contexts tested.

The fixed-scale kernel also has an opt-in native-UINT8 probability specialization for the pinned
mixed-sign Triton compiler patch described below. It maps nonnegative P to all 256 UINT8 codes and
uses native U8-by-S8 PV MMA, while QK remains signed S8-by-S8. Compared with signed P, generated
SM120 SASS drops 64 static `LOP3` instructions (118 to 54) without changing the 255-register,
28-spill D64 schedule. A random BF16 B1/H24/N4096/D128 comparison against exact SDPA improved
SQNR from 29.39 dB to 33.33 dB because P gains one bit of resolution.

Same-process reversed-order measurements against Triton Sage2++ show a long-context crossover:

| N | Triton SA2++ hot ms | native-UINT8 fixed hot ms | hot difference | Triton SA2++ E2E ms | native-UINT8 fixed E2E ms | E2E difference |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 24.0471 | 24.2590 | +0.88% | 24.6594 | 25.0817 | +1.71% |
| 65536 | - | - | - | 98.2024 | 97.6354 | -0.58% |
| 131072 | 386.7416 | 384.0799 | -0.69% | 390.1706 | 386.0661 | -1.05% |

Thus fixed integer PV can slightly beat pure-Triton Sage2++ in the 64K-128K video-generation
regime, but not at 32K or below on this RTX 5090. This path remains experimental because released
Triton 3.7.1 does not expose mixed-sign integer dot; the compiler patch is pure Triton/compiler
work and adds no CUDA kernel or inline PTX. Reproduce the complete comparison after installing the
patch with:

```shell
PYTHONPATH=/tmp/piper-triton-mixed uv run python \
  benchmarks/benchmark_sage_int8_pv.py \
  --sequence 32768 65536 131072 --native-uint8-mma \
  --warmup-ms 300 --repeat-ms 1500
```

Long-context recurrence and layout controls explain why the remaining conversion cost cannot be
removed cheaply in ordinary Triton. Accumulating same-coordinate PV tiles in a persistent INT32
fragment until the online maximum advances required simultaneous FP32 and INT32 D128 numerators;
M128 reached 280 spill slots and 9.27 ms at N=8192, while the best M64 form was still slower than
the ordinary recurrence. Splitting D128 into four D32 slices reduced spills from 28 to 20 but
doubled descriptor barriers and regressed N=8192 to 1.71 ms. Loading D128 V once before two D64
dots did not reduce generated loads and also regressed. The selected two-D64 K64 recurrence is
therefore still the best measured schedule.

The raw INT32 control still establishes the arithmetic upper bound: feeding every PV MMA into one
persistent INT32 accumulator and converting only in the epilogue previously came within 1.8-2.5%
of canonical SA2, but it is not numerically usable because it does not rescale old contributions
when the online-softmax maximum advances. The exact paired-K64 experiment halves PV conversions,
but retaining the first FP16 P tile blocks the efficient M128 schedule; its best 8K result was
2.09 ms. Retaining a prequantized INT8 first P tile instead did not reduce resources and measured
34.26 ms at N=32768, so that control is also rejected. Direct M64/K128 reached 1.76 ms, while
M128/K128 spilled more than 100 slots. Replacing
I2FP with an exact IEEE-754 magic-bias conversion merely trades those instructions for FP32 adds,
and narrowed FP16/BF16 conversions add shuffle work; neither improves latency. The remaining hot
1-2% is consequently still the INT32-to-FP32 boundary plus the final CUDA-vs-Triton schedule gap,
not INT8 tensor-core throughput.

Inspect one fixed schedule's compiler resources and SASS instruction mix with:

```shell
uv run python benchmarks/profile_sage_pv_variant.py \
  fp8-fp16-transposed --sequence 4096 --block-m 128
uv run python benchmarks/profile_sage_pv_variant.py \
  fp8-fp16-descriptor --sequence 4096 --block-m 128
uv run python benchmarks/profile_sage_pv_variant.py \
  int8-block-transposed --sequence 4096 --block-m 128
uv run python benchmarks/profile_sage_pv_variant.py \
  int8-log-signed-descriptor --sequence 4096 --block-m 128 --num-stages 2
uv run python benchmarks/profile_sage_pv_variant.py \
  int8-fixed-int32-raw-descriptor --sequence 8192 --block-m 128 --num-stages 2
```

The JSON report includes latency, registers, spills, shared memory, a resource-based residency
ceiling, MMA opcodes, and selected static SASS instruction families. Fixed schedules make
datatype and recurrence comparisons meaningful; use the autotuning benchmarks above for the
best latency of each variant. Row-major controls isolate the cost of Triton's RHS layout
conversion, `*-transposed` variants consume feature-major V with ordinary pointer loads, and
`*-descriptor` variants use Triton tensor descriptors. Descriptor schedules are currently an
SM120 experiment; production enables them only for measured D128/M128 shapes.

Capture launch-by-launch end-to-end GPU time for Piper or canonical Sage with Nsight Systems:

```shell
nsys profile \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --output=/tmp/sage-piper \
  uv run --group benchmark python benchmarks/profile_sage_e2e_variant.py \
    piper --sequence 4096

nsys stats --report cuda_gpu_kern_sum /tmp/sage-piper.nsys-rep
```

Select `canonical2pp` or `canonical2` in place of `piper` for the official kernels. The capture
excludes warmup and compilation. Nsight Compute can add hardware utilization and stall reasons
when the host permits access to NVIDIA performance counters.

Measure the same variants on real post-normalization, post-RoPE attention activations from
FLUX.2 Klein 4B with:

```shell
uv run \
  --with 'diffusers>=0.39' \
  --with transformers \
  --with accelerate \
  --with safetensors \
  --with sentencepiece \
  python benchmarks/benchmark_sage_int4_convrot_model.py
```

The real-model benchmark leaves the model's exact attention output in the inference path and
shadow-evaluates Sage variants at selected layers and denoising steps. Use `--variants` to select
the QK and PV experiments, including `int8_pv_fixed`, `int8_pv_block`,
`int8_pv_key_log_signed`, and `uint8_pv_key_log`; or use `--pv-diagnostics` to isolate UINT4-P
error from INT4-V error. It reports score and attention-output SQNR without allowing local error
to alter subsequent layers. The default checkpoint download requires roughly 15 GB of local
storage; pass `--local-files-only` after it has been cached.

Analyze paired-Hadamard INT8 PV scale distributions and fixed-scale alternatives on the same
real-model activations with:

```shell
uv run \
  --with 'diffusers>=0.39' \
  --with transformers \
  --with accelerate \
  --with safetensors \
  --with sentencepiece \
  python benchmarks/analyze_sage_pv_convrot_scales.py \
  --local-probability
```

The first prompt calibrates absolute and RMS-relative scales; a different prompt evaluates
global, per-layer, per-head/channel, and RMS-factor INT8 quantizers. `--local-probability` uses
the equivalent tile-local softmax normalization that removes the running maximum from the
probability quantizer's absolute scale.

Benchmark the resulting paired-Hadamard signed-INT8 prototype with one V RMS scale per one or
two K=64 tiles with:

```shell
uv run python benchmarks/benchmark_sage_int8_pv_convrot_rms.py
```

Both variants retain K=64 MMA and rotation tiles. The two-tile variant reduces V-scale metadata
by half; the benchmark reports the hot attention loop and full Q/K/V preprocessing separately so
metadata reduction is not mistaken for an MMA speedup.

Avoid the quadratic P rotation by rotating only V's feature dimension and using an affine
UINT8-equivalent probability encoding with:

```shell
uv run python benchmarks/benchmark_sage_uint8_pv_feature_convrot.py
```

Run the exact per-key recurrence with nonnegative signed-INT8 probabilities instead of affine
UINT8 with:

```shell
uv run python benchmarks/benchmark_sage_uint8_pv_feature_convrot.py \
  --scale-axes key --rotations 0 --probability-scale-modes log \
  --no-affine-probability
```

The primary variant rotates V over features, selects one V scale per key row, and folds those
K-dependent scales into P before quantization. The signed-INT8 MMA then evaluates
``(P_uint8 - 128) @ V_int8`` with a precomputed ``128 * sum(V_int8)`` correction as its
accumulator. The benchmark retains the original per-feature scales as a control and compares no
rotation, H16, and H64 against FP8 PV and direct block-scaled signed-INT8 PV. Inverse feature
rotations are fused into the attention epilogue; end-to-end timings also include Q/K/V
preprocessing. `--probability-scale-modes tile` tests a cheaper K-tile-wide P scale, and
`--value-scale-floors` tests whether deliberately coarsening small-norm V rows can recover its
probability resolution. The per-key default uses an exact log-domain recurrence: it shifts QK
scores by `log2(scale_v)` and accumulates the denominator with `1 / scale_v`, producing the same
UINT8 codes and quality as dynamic `P * scale_v` normalization without its second per-query
reduction. Dynamic and tile modes remain reproducible controls.
The signed-probability option uses codes `[0, 127]`, removes the affine correction and its
metadata, and keeps the same per-key log recurrence. On SM120 D128 it enables the lower-spill
M128 tensor-descriptor schedule; the tradeoff is one fewer bit of probability precision.

Test native mixed-sign integer MMA with the pinned Triton 3.7.1 compiler patch:

```shell
git clone --depth 1 --branch v3.7.1 \
  https://github.com/triton-lang/triton.git /tmp/piper-triton-src
git -C /tmp/piper-triton-src apply \
  "$PWD/benchmarks/patches/triton-3.7.1-mixed-int8-dot.patch"

TRITON_HOME=/tmp/piper-triton-build-cache MAX_JOBS=24 \
  uv build --python "$PWD/.venv/bin/python" --wheel \
  --out-dir /tmp/piper-triton-dist /tmp/piper-triton-src
uv pip install --python "$PWD/.venv/bin/python" --no-deps \
  --target /tmp/piper-triton-mixed /tmp/piper-triton-dist/*.whl

PYTHONPATH=/tmp/piper-triton-mixed uv run python \
  benchmarks/benchmark_mixed_int8_dot.py native
PYTHONPATH=/tmp/piper-triton-mixed uv run python \
  benchmarks/benchmark_sage_uint8_pv_feature_convrot.py \
  --sequence 4096 --scale-axes key --rotations 0 \
  --probability-scale-modes log --value-scale-floors 0 \
  --native-uint8-mma
```

The patch carries each integer dot operand's signedness through Triton IR and the
`AccelerateMatmul` rewrite, then selects the corresponding MMAv2 PTX opcode. It is intentionally
limited to the MMA path used by consumer Ada and Blackwell; it does not add mixed-sign WGMMA for
Hopper. The direct benchmark checks exact output and reports the emitted SASS. The attention
benchmark keeps signed INT8 QK while using native UINT8-by-INT8 PV, so a correct SM120 build
reports both `IMMA.16832.S8.S8.SAT` and `IMMA.16832.U8.S8.SAT` in the fixed-schedule profiler.

Evaluate the exact D128 PV computation as two sequential D64 output-feature slices with:

```shell
PYTHONPATH=/tmp/piper-triton-mixed uv run python \
  benchmarks/benchmark_sage_uint8_pv_feature_convrot.py \
  --sequence 512 1024 2048 4096 8192 --scale-axes key --rotations 0 \
  --probability-scale-modes log --value-scale-floors 0 \
  --native-uint8-mma --split-pv-head-dim
```

This does not split K, change the per-key V scale, or introduce another quantization. It reuses
the same probability codes for both feature slices and converts each completed D64 INT32 dot to
its corresponding FP32 accumulator immediately. On SM120, the split selects an M64/S3 tensor-
descriptor schedule that avoids the register spills seen in the D128 partial. It is an exact-
quality scheduling experiment; the ordinary affine UINT8 emulation can use it too, although its
separate zero-point correction remains.

With the D64 schedule fixed at M64/N64/S3, native mixed-sign MMA isolates the remaining
key-scaling cost. At N=4096 the fixed INT8 control measured 0.460 ms, local-P UINT8 with neither
the log score shift nor weighted denominator measured 0.476 ms, adding the log shift reached
0.514 ms, and the exact weighted denominator reached 0.558 ms. Thus native UINT8 removes the
affine correction, but the two log-domain softmax terms account for most of the remaining gap;
changing the PV datatype or MMA schedule alone cannot remove them.

`--scale-forward-log-recurrence` rewrites the exact weighted denominator without approximating
it. If `s[k]` is V's per-key quantization scale and
`m = max(score[k] + log2(s[k]))`, the original form uses
`r[k] = exp2(score[k] + log2(s[k]) - m)` for P and sums `r[k] / s[k]` for the
denominator. The scale-forward form instead computes `b[k] = exp2(score[k] - m)`, uses
`b[k] * s[k]` for P, and sums plain `b[k]`. The represented numerator and denominator are the
same; reassociation can move a small number of UINT8 rounding decisions by one code.

This materially improves Triton's register layout. In the native split-D64 profile it reduced
registers from 226 to 180 and static global loads from 14 to 6. On SM120, limiting the S2 kernel
to 168 registers crosses the three-resident-CTA boundary (180 registers permits only two CTAs).
The cap introduces 12 spill stores but still improved stable one-second profiles by about 22% at
N=1152, 11% at N=2048, 7% at N=3072, 2% at N=4096, and 4% at N=8192 relative to the uncapped S2
kernel. It is counterproductive at N=512, so the selected schedule uses M64/S2/R168 only from
N=1024 and retains M64/S3 without a cap below that threshold.

The captured FLUX.2 sample retained 41.68 dB mean output SQNR; worst output SQNR changed from
38.02 to 38.00 dB and mean relative L1 remained 0.0071. The scale-forward rewrite is exact up to
UINT8 rounding and the register cap does not change its arithmetic.

The PV-side scale-forward implementation now precomputes the final `255 * s_v[k]` multiplier
during V quantization. The attention loop can therefore encode P with one
`P * multiplier + 0.5` operation instead of forming `(P * s_v) * 255`. For the then-selected
M64/S2/R168 schedule, FP32 metadata is counterintuitively faster than FP16 on SM120: it avoids
Triton's packed-half broadcast layout, reduces compiler spill slots from 12 to 8, and removes the
hot-loop scale multiply. With H24, D128, and prequantized inputs:

| N | prior scale-forward ms | optimized PV scale ms | omit-scale ceiling ms | speedup |
|---:|---:|---:|---:|---:|
| 1152 | 0.05047 | 0.04885 | 0.04788 | 3.2% |
| 4224 | 0.56709 | 0.54439 | 0.53246 | 4.0% |
| 8192 | 2.09618 | 2.01573 | 1.98742 | 3.8% |
| 32768 | 32.78539 | 31.92899 | 31.35981 | 2.6% |

The omit-scale result is an intentionally incorrect profiler control that bounds all recoverable
PV-scaling overhead. The optimized path recovers most of that bound while preserving the exact
recurrence. End-to-end preparation plus attention improved from 0.64372 to 0.62170 ms at N=4224
and from 2.27507 to 2.21003 ms at N=8192. Ten captured FLUX.2 calls measured the same 41.68 dB
mean output SQNR and 0.0071 relative L1; worst output SQNR was 38.02 dB versus 38.00 dB before the
rewrite. `optimize_pv_scaling` selects this path automatically for noncausal SM120 native-UINT8
split-D128 scale-forward attention and accepts an explicit boolean override.

The residual 1.5-2% to the invalid no-scale ceiling is the K64 multiplier broadcast itself. An
explicit tensor descriptor added barriers and did not improve latency. Expanding the multiplier
four times to match warp-row groups caused Triton to materialize an M64xK64 tile in shared memory:
shared use rose from about 25 KiB to 41 KiB, CTA residency fell from three to two, and N=4224
slowed to 1.25 ms. FP16-only PV scaling changed the instruction mix but not registers, spills, or
latency. Those controls are not selected by the production policy.

SM120 noncausal, rotation-free, affine/native UINT8 D128 exact-log shapes select scale-forward
automatically from N=1024; other architectures and shapes remain explicit experiments. Pass an
explicit boolean to override selection. RTX 4090/SM89 requires separate tuning before promotion.

Complete noncausal self-attention tiles also select a predicate-free version of that schedule.
The specialization removes query/key boundary tests from the score, log-scale, probability, and
PV-multiplier paths without changing the recurrence. Reversed-order, same-process A/B runs found
3.2-3.8% lower hot latency at N=8192, 3.1-4.1% at N=32768, and 3.5-5.2% at N=131072. The N=8192
kernel retained its 168-register limit while spill slots fell from eight to six; the generated
kernel also removed 31 IMADs and 33 FP32 multiplies. BF16 outputs differed only at rounding scale
(at most 4.88e-4 in these runs). The public path enables it only for exact full-tile SM120
self-attention, leaving causal, cross-attention, and boundary tiles on the masked kernel.

Ragged SM120 lengths previously fell off a much larger cliff because the selected split-D128
path disabled tensor descriptors whenever K was not 16-aligned. The preparation path now rounds
only its quantized K/V storage up to K64, zero-fills at most 63 INT8 rows, and describes that padded
storage while continuing to pass the original semantic length to every score and metadata mask.
This is exact padding of internal storage, not attention padding: invalid keys never enter the
softmax. At N=8191/8193, descriptor-backed hot latency fell from 7.04/7.20 ms to 2.23/2.27 ms;
end-to-end latency fell from 7.17/7.31 ms to 2.48/2.52 ms. Direct output comparisons were bitwise
identical to the original masked pointer path.

For ragged contexts from N=16384, complete query blocks are launched through a query-predicate-
free specialization and only the final partial query block uses the general kernel. This second
step was neutral or shape-sensitive near 8K, but reduced hot latency by 3.2-4.0% around 16K,
3.5-4.2% around 32K, and 4.3-5.3% around 131K. Splitting the K loop itself into a predicate-free
bulk plus an inlined masked tail was rejected: it was 2.4-6.5% slower around 8K and 3.7-4.1%
slower around 32K because the altered Triton pipeline outweighed its simpler boundary logic.

The profiler accepts `--maxnreg` so the occupancy boundary can be reproduced without exposing
the cap as a public attention option. An attempted paired-K64 scale-forward recurrence is kept as
the `int8-log-split-scale-forward-paired-native-descriptor` profiler control. At N=4096 it used
255 registers, spilled, and took 0.868 ms versus about 0.54 ms for the selected recurrence. The
extra pending D128 partial defeats the register reduction, so it is not a production path.

`int8-log-split-scale-forward-pair-p-native-descriptor` moves the pair alignment before PV using
`a * (P @ V) = (a * P) @ V`. It retains the first K64 probability tile in FP16, finds one maximum
for the K128 pair, quantizes both probability tiles in that coordinate system, and feeds both
MMAs into one INT32 accumulator. This eliminates the D128-wide Q10 rescale and reduced the paired
kernel from 255 registers with spills to 225 registers without spills. The larger two-tile body
uses 41,588 bytes of shared memory and therefore still permits only two resident CTAs on SM120.

Three alternating 500-ms profiles measured 0.5145 ms versus 0.5338 ms for the selected unpaired
kernel at N=4096, and 2.0248 versus 2.0523 ms at N=8192. At the FLUX-relevant N=1152 it instead
took 0.0566 versus 0.0503 ms. Pair-wide probability normalization also trades away local UINT8
range: on ten captured FLUX.2 calls, the corresponding K128 reference measured 40.94 dB mean and
37.60 dB worst output SQNR with 0.0082 relative L1, versus 41.68 dB, 38.00 dB, and 0.0071 for the
selected scale-forward kernel. It remains a reproducible long-sequence research control rather
than a diffusion schedule.

For video-scale contexts, the pair was revisited after the K64 PV-scaling optimization. The
`int8-log-split-scale-forward-pair-p-precomputed-pv-scale-native-descriptor` variant prepares
`255*s_v[k]` once in FP16 instead of forming it in every MxK probability tile. FP16 is preferable
here: it reduces the pair from 227 to 225 registers, whereas FP32 metadata raises it to 246. The
best paired schedule remains M64/S2/W4; M32 is 48% slower at N=32768, M128/W8 is 5% slower, S1
loses load overlap, and S3 drops to one resident CTA.

Long-context H24/D128 profiles show that the optimized pair reaches parity rather than a clear
asymptotic win:

| N | selected K64 ms | optimized K128 pair ms | pair delta |
|---:|---:|---:|---:|
| 8192 | 2.0036 | 2.0328 | +1.46% |
| 32768 | 31.5950 | 31.8719 | +0.88% |
| 65536 | 126.7678 | 127.0346 | +0.21% |
| 131072 | 509.32 | 509.09 | -0.05% |

The 131K row is the mean of three multi-second samples and is well inside run-to-run variance.
Pairing halves probability-coordinate/INT32-conversion work, but retaining the first K64 P tile
expands shared memory from about 25 KiB to 41.6 KiB and reduces residency from three CTAs to two.
A worthwhile next iteration must therefore pipeline loads at K64 granularity while sharing one
K128 maximum; further algebraic changes alone are unlikely to move the long-context result.

`int8-log-split-scale-forward-sampled-pair-p-native-descriptor` tests estimating that K128
coordinate from only one physically interleaved K64 sample. Preparation packs the even keys of
each original K128 group before its odd keys so both hot-kernel loads stay contiguous. The first
half is quantized immediately and retained as UINT8 codes; the held-out half uses the sampled
maximum plus `--sampled-headroom-log2`, clipping codes above 255. This removes Q10 and avoids
retaining first-half FP16 probabilities, but Triton does not keep the long-lived UINT8 tile packed:
the kernel used 232 registers, zero spills, 41,588 bytes of shared memory, and two resident CTAs,
versus 225 registers for exact K128 normalization.

With three log2 units of headroom, hot H24/D128 timings were 0.0564, 2.0049, and 31.2700 ms at
N=1152, 8192, and 32768, versus 0.0566, 2.0465, and 31.8612 ms for exact K128. The small speedup is
not quality-viable. On ten captured FLUX.2 calls, zero headroom clipped 51.01% of query/pair rows
and measured 11.71 dB mean output SQNR; three-bit headroom still clipped 3.67% and measured only
18.33 dB, versus 40.94 dB for exact K128. The worst zero-headroom held-out maximum gap was 18.36
log2 units: 16.98 came from the score maximum alone and 3.69 from V-scale maxima. Interleaving is
representative of averages, not query-dependent extremes, so physical key reordering cannot make
one half's maximum a reliable normalization for the other. Packing is intentionally excluded from
the reported hot timings and was not promoted into production preprocessing.

The `int8-log-split-int32-lazy-h{0,1,2}-native-descriptor` controls test a persistent
block-floating INT32 numerator. Each row keeps an integer base-2 exponent, shifts the D128
accumulator only when that exponent advances, and passes the shifted accumulator directly into
the next PV MMA. H0 compiled to 179 registers without spills; R168 enabled three resident CTAs
with 10 spills. It measured 0.0520, 0.5254, and 1.9582 ms at N=1152, 4096, and 8192, versus
0.0504, 0.5341, and 2.0712 ms for the selected scale-forward recurrence.

The speedup is not quality-viable. Quantizing every later tile in the global exponent's coordinate
system discards small probabilities before their K64 contributions can sum in INT32. On ten
captured FLUX.2 calls, H0 measured 29.48 dB mean and 22.47 dB worst output SQNR with 0.0251
relative L1, versus 41.68 dB, 38.00 dB, and 0.0071 for scale-forward. Reserving one exponent bit
of UINT8 headroom worsened those results to 25.28 dB, 17.46 dB, and 0.0412. These remain
profiler-only negative controls; tile-local probability normalization is required for diffusion
quality.

`--tile-common-log-denominator` tests replacing the exact per-key reciprocal denominator with
one geometric-mean inverse scale per K64 tile. On RTX 5090 at N=4096 this reduced the split-D64
affine hot kernel from 0.657 to 0.602 ms, registers from 228 to 214, and static global-load
instructions from 30 to 23. It is not quality-viable: on ten captured FLUX.2 Klein attention
calls, mean output SQNR fell from 41.68 dB to 10.93 dB and the worst call reached -7.51 dB.
The tile scalar cannot represent the query-dependent, probability-weighted inverse scale when
real attention concentrates on exceptional keys.

`--narrow-int8-log-denominator` retains that query dependence by quantizing `1 / scale_v[k]`
to signed INT8 and evaluating the denominator with an additional Mx64 by 64x16 MMA using the
same UINT8 P codes as PV. Its quality is much stronger: the same FLUX.2 sample measured 40.08 dB
mean and 32.60 dB worst output SQNR, versus 41.68 and 38.02 dB for the exact denominator. It is
also not a performance win in ordinary Triton on SM120. At N=4096 the fixed M64/S3 profile rose
from 0.657 to 0.718 ms, registers from 228 to 238, barriers from 14 to 16, and shared-memory loads
from one to three. Triton materializes layout traffic around the minimum-width N16 denominator
MMA, making it slower than the FP32 multiply-and-reduce it replaces.

The real-model benchmark also contains math-first K-tile and recurrence controls:

```shell
uv run \
  --with 'diffusers>=0.39' \
  --with transformers \
  --with accelerate \
  --with safetensors \
  --with sentencepiece \
  python benchmarks/benchmark_sage_int4_convrot_model.py \
  --local-files-only \
  --variants int8_pv_key_log_ref64 int8_pv_key_log_ref128 \
    int8_pv_key_log_pair_q10 int8_pv_key_log_ref64_fp16_p \
    int8_pv_key_log_ref64_fp16_acc int8_pv_key_log_ref64_fp16_norm_acc
```

Sharing one probability normalization across K128 saves a prospective conversion but loses
about 0.9 dB mean output SQNR on the sampled FLUX.2 activations. Keeping independent K64 scales
and merging two INT32 partials with a Q10 row multiplier is quality-neutral, but its Triton
prototype is slower because the loop-carried partial adds integer shifts and register spills.
Raw FP16 numerator recurrence can overflow on real layers even when random inputs look safe.
Storing FP16 probabilities or a bounded, normalized FP16 running output avoids that failure, but
the generated SM120 kernels are no faster: conversions offset the smaller representation and do
not improve CTA residency. These remain research ablations rather than production schedules.

Keep the PV numerator in INT32 across the online-softmax loop with:

```shell
PYTHONPATH=/tmp/piper-triton-mixed uv run python \
  benchmarks/benchmark_sage_uint8_pv_feature_convrot.py \
  --sequence 4096 --scale-axes key --rotations 0 \
  --probability-scale-modes log --value-scale-floors 0 \
  --native-uint8-mma --integer-output-recurrence
```

This common-scale recurrence lets each PV MMA consume the previous INT32 numerator directly and
converts it only in the epilogue. It is a speed/quality ablation: probabilities below the global
UINT8 quantization step disappear, which is substantially worse on real FLUX.2 activations than
random-input SQNR suggests. `--integer-tile-exponent-recurrence` preserves tile-local probability
resolution and real-model quality by aligning completed INT32 partials with power-of-two shifts,
but the separate partial increases register pressure and is currently slower than FP32 recurrence.

The raw conversion overhead can be isolated without changing operands or MMA count using:

```shell
PYTHONPATH=/tmp/piper-triton-mixed uv run python \
  benchmarks/profile_int8_pv_conversion.py --sequence 4096 8192
```

This bounded PV-only control compares converting every K64 INT32 partial into an FP32 accumulator
against feeding every MMA into one persistent INT32 accumulator and converting only in the
epilogue. The outputs are bit-exact because the control has no online-softmax rescaling and avoids
overflow. At M64, persistence improved N=4096/8192 by 6.2%/6.7%, reduced registers from 166 to
125, and raised residency from three to four CTAs. At M128, where both variants remained at two
CTAs, it improved performance by 24.2%/23.6%, reduced registers from the 255-register ceiling to
204, and removed eight spill slots. This is an upper bound, not a usable attention recurrence:
the matched full-attention Q8 recurrence needed per-tile integer multiply/shift alignment and
measured 2.595 ms versus 2.538 ms for FP32 recurrence at N=8192, 2.3% slower.

Power-of-two tile coordinates were also tested as a cheaper alternative to FP32 recurrence. The
exact version normalizes each K64 probability tile to an integer base-2 exponent, completes PV in
INT32, and merges the result with one rounded shift of whichever D128 accumulator has the smaller
exponent. Selecting the dominant operand first halves the static shift instructions from 156 to
80, but it does not remove the simultaneous persistent accumulator and completed partial. The
best split-D64 S2 kernel therefore still uses 215 registers and only two CTAs per SM.

Moving the alignment before PV replaces the shifted MxD partial with a shifted MxK UINT8 operand
and lets MMA accumulate directly into the persistent INT32 numerator. Real FLUX.2 exponent
statistics explain the quality tradeoff: only 4.35% of row/tile merges have equal exponents,
while 66.25% differ by at least five bits. The running exponent advances on only 4.97% of merges,
so pre-alignment is cheap structurally, but it removes most locally resolved nonzero probability
codes. Nearest rounding retained 42.11% of nonzero codes and measured only 30.02 dB mean / 22.78
dB worst output SQNR. Ten-bit stratified dither retained 44.10% and improved to 38.89 / 34.87 dB;
one threshold per key matched per-(query,key) dithering, avoiding an unnecessary MxK random phase.
The actual Triton kernel measured 38.80 / 34.84 dB, closely matching the reference.

On RTX 5090, H24/D128, the final same-run hot profiles were:

| N | selected FP32 recurrence ms | exact post-dot INT32 ms | dithered pre-dot INT32 ms |
|---:|---:|---:|---:|
| 1152 | 0.04864 | 0.06820 | 0.05352 |
| 4096 | 0.51727 | 0.57991 | 0.55815 |
| 8192 | 2.00359 | 2.25120 | 2.13357 |

The pre-dot path reduces the natural live set from 215 to 188 registers; an R168 cap reaches three
CTAs with 14 spill slots. It nevertheless remains slower than FP32 recurrence and loses about
2 dB mean SQNR relative to exact post-dot alignment (and more relative to the selected 41.68 dB
scale-forward path). Coarser one-, two-, and four-bit dithers reached only 33.45, 35.69, and 37.26
dB mean SQNR. M32 raised natural occupancy to three CTAs but took 3.10 ms instead of 2.25 ms at
N=8192; S1 lost load overlap, and S3 did not raise occupancy. These formulations remain explicit
research controls and are not selected by the production policy.

Converting each K32 PV MMA result immediately was tested as a way to shorten the lifetime of the
K64 INT32 partial. The control keeps the same K64 softmax normalization and the same 64 total
integer MMA instructions, but issues two independent K32 PV dots per D64 half and converts/adds
each result to the persistent FP32 numerator immediately. It is available only through the
`k32-immediate` profiler variant. The result is mathematically equivalent to the selected K64
path: on a random N=192 case it differed by 8.7e-9 mean absolute error and 2.44e-4 maximum error.

This does not remove the hardware-required INT32 MMA fragment and is slower on RTX 5090. With
unconstrained registers, the K32 control increased registers from 181 to 238, barriers from 11
to 15, INT32-to-FP32 instructions from 96 to 160, and FFMA instructions from 98 to 162. Direct
K32 descriptor loads were used, so this is not an artifact of loading and retaining a full K64 V
tile. Under the production R168 cap, spills rose from 8 to 44 slots:

| N | selected K64 ms | immediate K32 ms | K32 penalty |
|---:|---:|---:|---:|
| 1152 | 0.04854 | 0.07443 | 53.4% |
| 8192 | 1.98359 | 3.02605 | 52.6% |
| 32768 | 31.65452 | 47.65067 | 50.5% |
| 131072 | 506.28711 | 760.94769 | 50.3% |

Without the register cap, eliminating spill amplification still left the K32 control 10.1%,
10.2%, and 10.6% slower at N=8192, 32768, and 131072. The selected formulation is therefore the
right granularity: let both hardware K32 operations accumulate into one K64 INT32 fragment, then
pay for one conversion and one scaled FP32 numerator update. An N=8192 schedule sweep did not
find a compensating tile choice: M64/S2 remained best; M32/S2 was about 36% slower and M128/S2
with eight warps was about 12% slower than the uncapped M64/S2 K32 control.

An output-side scaling formulation was then evaluated against a revised quality target: matching
canonical SageAttention2++ is sufficient, rather than preserving the per-key path's extra SQNR.
For noncausal attention, K and V can be jointly permuted without changing exact attention. Stable
one-bit-per-octave buckets of each key row's V range make adjacent K tiles more homogeneous. A
fixed H64 feature basis reduces coordinate outliers and can be folded exactly into the V and output
projection weights, so it need not execute in the attention hot loop. After this preparation, V is
quantized with one INT8 scale per K tile and feature group; the scale is applied to the completed
INT32 PV output rather than to P.

On twenty captured FLUX.2 Klein calls, canonical SageAttention2++ measured 36.49 dB mean / 32.64
dB worst output SQNR. The useful quality boundary was:

| P/V K tile | V feature group | mean output SQNR | worst output SQNR |
|---:|---:|---:|---:|
| 64 | 16 | 39.62 dB | 33.57 dB |
| 64 | 32 | 39.27 dB | 32.97 dB |
| 64 | 64 | 38.91 dB | 32.37 dB |
| 64 | 128 (scalar) | 38.65 dB | 32.02 dB |
| 128 | 4 | 38.88 dB | 33.60 dB |
| 128 | 8 | 38.43 dB | 32.80 dB |
| 128 | 16 | 37.99 dB | 32.05 dB |

K64/group32 and K128/group8 therefore clear the initial sampled canonical mean and worst-case
thresholds. A larger 60-call sample exposed a lower tail for group8 (37.59 dB mean / 31.16 dB
worst), so the packed validation path conservatively defaults to K128/group4.
Ignoring even one V outlier per K64 scale group was not a viable substitute: mean SQNR collapsed
from 41.26 to 26.42 dB and the worst call fell from 36.37 to 16.50 dB. Attention sometimes assigns
substantial weight to those values, so robust-percentile clipping is explicitly rejected.

The native-UINT8 grouped-output kernel supports both K64 and K128. K128 is expressed
as one Mx128 QK/PV tile, so four K32 PV MMAs accumulate in one INT32 fragment before one FP32
conversion and scaled recurrence update. This avoids the retained first-half probability tile that
made the earlier hand-paired K64 formulation register-heavy. The selected K128/group4 M64/S2
kernel uses 248 registers, no spills, 41,512 bytes of shared memory, and two CTAs per SM. The
K128/group8 control uses 230 registers. The K64/group32
M64/S2/R168 kernel uses three CTAs per SM with ten spill slots and remains preferable for short
sequences. Direct reconstruction tests measured 55.6 dB agreement with both kernels.

The accompanying pure-Triton packer now rotates V by H64, histograms log2 row ranges, computes
bucket prefixes, and atomically scatters paired K/V rows. On the expanded 60-call FLUX.2 sample,
the actual K128/group4 atomic path measured 38.01 dB mean output SQNR and 0.0108 mean relative L1,
compared with canonical SageAttention2++ at 36.54 dB and 0.0132. Its single worst call was 32.20
dB versus canonical's 32.64 dB; half-octave buckets raised it to 32.37 dB without changing the
mean and also reduced atomic contention, so half-octave bucketing is the default. Group4 is
therefore the current quality/performance point, while group8 is retained only as
an ablation. H64 should be folded into the V and output projection weights in model integration;
the validation wrapper explicitly inverse-rotates the output.

Same-run RTX 5090 end-to-end timings, including packing and quantization but assuming the H64
weight fold, were:

| N | packed K128/group4 ms | selected per-key ms | fixed INT8 ms | pack only ms |
|---:|---:|---:|---:|---:|
| 8192 | 2.12620 | 2.15094 | 1.99444 | 0.17637 |
| 32768 | 29.77856 | 32.00790 | 29.01510 | 0.70378 |
| 131072 | 469.54495 | 510.40973 | 454.32422 | 2.98132 |

At 131K this is 8.0% faster than the selected per-key formulation and 3.4% slower than fixed INT8;
the packer itself is only 0.6% of total latency. Group8 can narrow the hot-kernel gap further, but
its expanded-sample tail rejects it as the default. The grouped-output path
remains experimental and the existing per-key production policy is unchanged.

Sorted runs also permit amortizing the output-scale coordinate. The K512 control uses one V scale
for four consecutive K128 tiles, keeps the FP32 numerator in that scale coordinate, and multiplies
it by the precomputed adjacent-scale ratio only at run boundaries. Flattening this formulation
back into one pipelined K loop reduced the attention kernel from 248 to 236 registers with no
spills. A split quantizer first reduces K128 tiles into K512 FP32 maxima and then quantizes through
K128 write tiles; this reduced K512 preparation from 12.91 to 6.25 ms at N=131072.

| N | K128 hot ms | K512-run hot ms | K512-run preparation ms | K512-run E2E ms |
|---:|---:|---:|---:|---:|
| 8192 | 1.82503 | 1.79060 | 0.32777 | 2.21021 |
| 32768 | 28.88132 | 28.24182 | 1.50156 | 29.72408 |
| 131072 | 463.96861 | 451.94833 | 6.24979 | 458.44379 |

The K512 run is about 0.9% slower end-to-end than the 454.32 ms fixed-INT8 reference at 131K,
while substantially narrowing the original grouped-output gap. K1024 did not improve long-context
latency. On the actual 20-call FLUX.2 atomic-packer sample, K512 measured 37.92 dB mean / 32.93 dB
worst output SQNR and 0.0110 relative L1, versus 38.05/33.06 dB and 0.0108 for K128. The earlier
canonical SageAttention2++ sample measured 36.49/32.64 dB, so K512 remains within the chosen
quality target. It is exposed as the long-context `scale_run_n=512` experiment rather than the
short-sequence default.

The long-context run was subsequently retuned against the predicate-free M128 fixed-UINT8
baseline. The useful recurrence stores the unnormalized numerator in FP16 after multiplying each
completed INT32 PV partial by `2^-16`. For `K <= 131072`, the conservative bound
`K * 255 * 127 * 2^-16 < 65504` prevents FP16 overflow. The conversion is performed through FP32,
so this is not an integer right shift: locally resolved small partials retain their fractional
value before the FP16 recurrence rounding. The common factor is restored once in the epilogue.

This smaller persistent state makes M128/K64/S2 viable with 254 registers and 16 spill slots. In
three independent 700-ms profiles, the medians were:

| N | fixed UINT8 hot ms | K512 scaled-FP16 hot ms | remaining gap |
|---:|---:|---:|---:|
| 32768 | 24.22599 | 25.76030 | 6.33% |
| 131072 | 383.10809 | 409.98605 | 7.02% |

At 131K, packing, quantization, and attention measured 415.52 ms versus 384.23 ms for fixed-UINT8
end-to-end. Thus the new recurrence cuts the prior hot gap from roughly 17% to 7%, but does not
reach parity. The residual is the tile-local probability-coordinate merge: fixed UINT8 encodes P
in the running-max coordinate and adds each INT32 partial directly, while the quality-preserving
block path must apply `current_weight` across the MxD partial.

On the two-prompt reference sample, the scaled recurrence measured 39.49 dB mean / 33.43 dB worst
output SQNR, versus canonical SageAttention2++ at 36.49/32.64 dB. The actual packed Triton kernel
measured 38.72/33.24 dB on the one-prompt check, versus 36.58/33.03 dB for canonical. Contiguous
per-feature K512 scales were rejected despite similar average quality because one real layer fell
to 22.25 dB; range sorting remains necessary for the tail.

The following alternatives did not improve the selected schedule: FP32 scale metadata, K128 and
K256 local-P tiles, normalized FP16 state, unnormalized BF16 state, dominant-operand selects,
conditional maximum-advance branches, wider K1024/K2048 scale runs, loop unrolling, LICM, and
whole-loop software pipelining. Triton's automatic warp-specialization pass currently crashes on
this mixed QK/softmax/PV loop on SM120 and is limited to simple matmul loops in the installed
compiler. The selected experiment is exposed as `scaled_fp16_numerator=True` with K64/K512 and an
explicit 131072-key safety guard.

The same fixed-coordinate recurrence also applies to the exact per-key scale-forward path. In
that formulation each `s_v[k]` is already folded into the UINT8 P operand before MMA, so the
persistent numerator never changes V-scale coordinates. Keeping the completed INT32 partial in a
`2^-16` FP16 numerator and restoring `2^16 / 255` in the epilogue is therefore simpler than the
sorted-run case: only the ordinary online-softmax weights touch the persistent state. This is
distinct from the older normalized-FP16 ablation, which computed a new denominator reciprocal
and normalized the output after every K64 tile.

For the selected key-scaled M64/S2/R168 kernel, the fixed-coordinate recurrence reduced resources
from 168 registers and six spill slots to 144 registers with no spills. More importantly, it made
M128/S2 usable: that schedule uses 254 registers and ten spill slots, but processes twice as many
query rows per CTA. Three independent 1.2-second fixed-schedule profiles measured:

| N | FP32 M64/S2 hot ms | scaled-FP16 M128/S2 hot ms | latency reduction |
|---:|---:|---:|---:|
| 32768 | 31.11688 | 26.58483 | 14.56% |
| 131072 | 501.41389 | 421.49173 | 15.94% |

The production-style autotuning benchmark selected M128/S2 from N=8192. Its matched end-to-end
measurements, including Q/K/V preparation, were 1.96534, 28.11904, and 436.29366 ms at N=8192,
32768, and 131072, versus 2.10137, 31.84694, and 509.24953 ms for the FP32 key-scaled path. The
new recurrence therefore improves end-to-end latency by 6.5%, 11.7%, and 14.3% respectively.

Quality remains comfortably above the canonical target. Across twenty captured FLUX.2 Klein
calls, the FP32 key-scaled path measured 41.61 dB mean / 37.60 dB worst output SQNR and 0.0073
relative L1; the scaled-FP16 numerator measured 41.55/37.58 dB and 0.0074. Canonical
SageAttention2++ measured 36.49/32.64 dB and 0.0133 on the same calls. The public experiment is
exposed as `scaled_fp16_numerator=True`, selects M128/S2 for noncausal SM120 D128 sequences from
8192 tokens, and retains the explicit 131072-key bound.

An FP16 denominator was also tested and rejected. A shared `2^-16` coordinate first exposed
underflow, while a wider `2^-4` coordinate bounded the worst-case 131K denominator by 8192 and
retained much smaller contributions. Both nevertheless reached only 24.38 dB SQNR at N=131072,
versus 35.38 dB with the FP32 denominator. The failure is relative precision, not range: once the
persistent denominator is large, a K64 contribution can be smaller than its FP16 ULP regardless
of power-of-two scaling. The FP16 state reduced registers by only one, increased spills from ten
to fourteen, and improved fixed-schedule latency by just 0.5% at 32K and 0.9% at 131K. It remains
an explicit `scaled_fp16_denominator=True` negative control rather than a selected policy.

The useful denominator-side change is algebraic and keeps the recurrence FP32. Applying the
`255 / 65536` conversion to the M-row denominator in the epilogue, rather than multiplying the
M-by-D numerator by `65536 / 255`, reduced static FP32 multiplies from 653 to 529 and total static
SASS instructions from 2752 to 2608 without changing registers or spills. Repeated profiles
improved another 0.5-0.6%; the final timings above include this rewrite.

The register headroom was then tested against wider state and schedules. Keeping the scaled-FP16
numerator without split-D64 made M64 fit in 167 registers with no spills, but generated 28
barriers and substantially more shared-memory traffic; it was 5.7% and 5.3% slower than split
M128 at N=32768 and 131072. M256/S3/W8 also compiled without catastrophic spilling, but measured
28.61 and 447.02 ms versus 26.41 and 417.61 ms for M128/S2/W4. Register caps from R208 through
R248 did not change the M128 residency ceiling and were neutral or slower.

Applying the same recurrence to the exact K128 probability-pair control reduced it from 225 to
180 registers without spills. Shared memory remained 41,588 bytes, however, so residency stayed
at two CTAs. It was essentially neutral within the pair itself: 31.64 versus 31.70 ms at N=32768
and 505.06 versus 507.80 ms at N=131072. That is still far behind the selected independent-K64
path. The result confirms that the freed registers can hold wider state, but K128 pairing is
limited by its retained first probability tile and shared-memory schedule rather than its
persistent numerator.

A larger public-path issue was found during this sweep. The scaled recurrence selected M128, but
the split-D64 tensor-descriptor policy still admitted only the older M64 schedule. Consequently,
the profiler used descriptors while the end-to-end launcher silently used pointer loads. The
SM120 scaled-FP16 M128 path now enables its measured D64 descriptors as well. This is deliberately
scoped to that schedule; the portable M64 policy is unchanged.

With descriptors, FP16 storage for the precomputed `255*s_v[k]` multiplier becomes faster than
FP32, reversing the earlier M64 result. The M128 kernels use 253 registers and 14 spill slots for
FP16 metadata versus 254 registers and 10 slots for FP32, but the FP16 descriptor/broadcast path
wins consistently. Alternating same-process end-to-end measurements were:

| N | FP32 multiplier ms | FP16 multiplier ms | reduction |
|---:|---:|---:|---:|
| 8192 | 1.910 | 1.864 | 2.4% |
| 32768 | 27.050 | 26.404 | 2.4% |
| 131072 | 422.098 | 410.664 | 2.7% |

The preparation kernels alone were identical at 0.708 ms for N=32768; the gain is in attention,
not cheaper quantization. On ten captured FLUX.2 Klein calls, FP16 and FP32 multiplier metadata
both measured 41.62 dB mean / 37.99 dB worst output SQNR and 0.0072 relative L1. The scaled-FP16
numerator therefore selects FP16 multiplier metadata by default, while
`fp32_pv_scale_metadata=True` retains the old control.

The final M128 schedule was then profiled specifically to separate key-scale arithmetic from
the recurrence and compiler schedule. Removing the `log2(s_v[k])` score shift while retaining
the per-key `255*s_v[k]` PV multiplier is an intentionally low-quality ceiling. Three repeated
profiles measured:

| N | exact key-scaled ms | no-log-shift ceiling ms | log-shift cost |
|---:|---:|---:|---:|
| 8192 | 1.6745 | 1.6517 | 1.38% |
| 32768 | 25.6858 | 25.2731 | 1.63% |
| 131072 | 405.9269 | 400.5775 | 1.34% |

The PV multiplier is not the bottleneck in this schedule. Forming `255*s_v[k]` inside the loop
was tied with loading the prepared multiplier at 32K, while omitting the multiplication or using
a separate scale descriptor was slower. The original weighted-denominator recurrence and an
INT8-denominator control were also 1.4% and 2.8% slower at 8K. Therefore the remaining useful
key-specific ceiling is only the roughly 1.5% log-coordinate shift; the earlier M64 result that
attributed another 1-2% to the PV multiplier does not transfer to M128.

That ceiling is not quality-viable even under the relaxed canonical-quality target. On twenty
captured FLUX.2 Klein calls, omitting only the log shift reduced mean/worst output SQNR to
20.38/3.84 dB with 0.1672 relative L1, versus 36.49/32.64 dB and 0.0133 for canonical
SageAttention2++. The shift is therefore structurally required; the key-scaled path's otherwise
large quality margin cannot absorb normalizing P in the wrong V-scale coordinate.

Several ways of approximating or rescheduling the log maximum were rejected. A separable
`max(score)+max(log_scale)` bound introduced a reduction barrier; preparing its K64 scalar and
issuing the load before QK still measured 1.696 ms. Computing the maximum in FP16 measured
1.688 ms, deriving the log from `255*s_v` measured 1.855 ms, and prefetching the exact log vector
before QK measured 1.736 ms. These either add conversion/reduction work or lengthen metadata
live ranges, so none was retained in the attention kernel.

The recurrence boundary can be isolated from input preparation and unrelated attention policy
with:

```shell
PYTHONPATH=/tmp/piper-triton-mixed:$PYTHONPATH uv run python \
  benchmarks/profile_int8_pv_recurrence.py attention-local \
  --sequence 32768 --probability-dtype int8
```

This M128/K64/D128 control runs the real INT8 QK reduction, forms quantized P, completes two D64
INT8 PV dots, and varies only the persistent numerator merge. Stable RTX 5090 measurements were:

| N | running-coordinate P ms | local-coordinate P ms | exact local penalty |
|---:|---:|---:|---:|
| 8192 | 2.4608 | 2.5595 | 4.01% |
| 32768 | 37.2316 | 38.8336 | 4.30% |

Both variants issue the same three MMA operations per K64 tile and have the same two-CTA
residency ceiling. Local normalization adds 64 FP16 multiply instructions and four row-level
exponent/conversion operations to merge each completed PV partial into the running softmax
coordinate. The full production-shaped M128/S2 kernel showed the same boundary at N=32768:
25.8688 ms for exact key scaling versus 24.8953 ms for fixed UINT8, a 3.91% gap.

Because `next_max = max(running_max, block_max)`, one recurrence weight is exactly one. An exact
control selected the weighted operand per row and used one shared weight plus one FMA. It was
slower: 2.7440 ms at N=8192 and 41.6767 ms at N=32768. Triton materialized the row-wise choice
across both D64 accumulator fragments, increasing static permutation instructions from 448 to
740 and static SASS from 3426 to 3903. The ordinary two-weight expression is therefore the
better lowering despite doing one apparently redundant multiply.

A narrower shared-weight control retained the ordinary MxD multiply/FMA merge and changed only
the two row-vector exponentials into `exp2(-abs(block_max-running_max))` plus row-vector selects.
It successfully reduced static `MUFU` instructions from 76 to 72 with unchanged registers,
spills, MMAs, and layout conversions. The absolute-value/compare/select sequence added twelve
other instructions, however. Alternating profiles measured 2.5715/2.5830 ms for the ordinary
form versus 2.5816/2.5870 ms for the shared form at N=8192, and 38.8016/38.8267 versus
38.8789/38.9002 ms at N=32768. The shared exponential is therefore a small regression even
without the expensive accumulator-fragment selection and remains a profiler-only negative
control.

The compiler boundary is not specific to the pinned Triton 3.7.1 build. Triton main at commit
`707fc2ca` (reported as 3.8.0) measured 37.3755/38.6573 ms for running/local coordinates at 32K,
still a 3.43% penalty. It changed register allocation slightly but did not introduce an
output-scaled INT8 MMA or a cheaper row-wise merge.

Quantizing P directly in the running-max coordinate removes the local merge but is not a usable
quality trade. On the twenty-call FLUX.2 Klein sample it measured 31.73 dB mean / 24.90 dB worst
output SQNR and 0.0197 relative L1, versus 41.55/37.58 dB and 0.0074 for the selected local
scale-forward recurrence. The canonical SageAttention2++ target on the same sample was
36.49/32.64 dB and 0.0133. Revisiting the quality-viable dithered pre-dot INT32 alignment under
M128 was also negative: its 74 spill slots and extra conversion/shift chain measured 40.2750 and
634.5155 ms at N=32768/131072, versus 25.8181 and 405.5987 ms for the selected exact path.

Together these controls localize the remaining key-scaled cost to the required per-query
coordinate merge. It is no longer primarily a metadata-load, V-scale multiplication, occupancy,
or MMA-selection problem. Closing it without losing canonical quality would require either a
different P representation that preserves local UINT8 resolution or hardware/compiler support
for scaling an INT32 MMA result as it enters the persistent accumulator.

The explicit UINT8 probability clamp is not another hidden hot-loop cost. A guarded control
derived the FP16 `255*s_v` multiplier from the same rounded FP16 `log2(s_v)`, biased the
multiplier downward by `2^-10`, compensated that common factor in the epilogue, and removed
`min(255, code + 0.5)`. The metadata bound and output agreement passed, but the clamped and
clamp-free specializations both compiled to 253 registers, 14 spills, 2612 static instructions,
and 140 `FMNMX` instructions. Alternating measurements were tied at roughly 1.70 ms for N=8192
and 25.90 ms for N=32768. Triton already folds the clamp into the UINT8 conversion/lowering; the
remaining `FMNMX` instructions implement required score and running-maximum reductions. The
guarded metadata path was therefore reverted rather than retaining numerical complexity without
a generated-code change.

A matched 32K generated-code comparison measured 24.710 ms for fixed UINT8, 25.277 ms for the
no-log ceiling, and 25.753 ms for exact key scaling. The residual after removing the log shift
comes primarily from maintaining the rescalable online numerator, not from loading key-scale
metadata. FP32 M128 recurrence spilled 54 slots and took 34.67 ms; an unscaled BF16 prototype
spilled 66 slots and took 37.18 ms. Register caps from R248 through R176 retained two-CTA
residency and slowed the kernel, while R168 reached three CTAs only by spilling 54 slots and took
2.06 ms at 8K. The selected uncapped scaled-FP16 numerator is consequently the best measured
quality-preserving key-scaled formulation on SM120.

The K512 run kernel likewise benefits from specializing complete noncausal self-attention tiles:
same-process A/B measurements improved hot latency by 0.4-1.8% at N=8192, 2.2% at N=32768, and
1.8-2.6% at N=131072, with only BF16-rounding-level output differences. This is selected for the
K512 experiment. Applying the same source-level rewrite to the ordinary K128 grouped-output
kernel was 3.5-4.3% slower at N=8192 and 8.7-8.9% slower at N=32768, despite simpler generated
code. That kernel therefore retains its boundary predicates: this optimization is compiler-
schedule-sensitive, not a blanket property of every INT8 recurrence.

Quarter-octave K512 bucketing did not tighten the effective group scales enough to help. On the
same 20-call FLUX.2 sample it measured 37.91/32.92 dB versus 37.94/32.96 dB for half-octave K512,
with identical 0.0110 relative L1. It was also 0.8%, 0.4%, and 0.2% slower end-to-end at N=8192,
32768, and 131072. The bucket key is a row-wide maximum, whereas each V scale covers one four-
feature group; finer scalar buckets therefore do not necessarily make all 32 group maxima more
homogeneous. Half-octave remains selected.

Sorting also makes a block-floating INT32 recurrence numerically plausible. A power-of-two-only
reference measured 36.73/31.05 dB, and its FP32 and INT32 alignment results were identical: the
rounded post-dot shifts added no measurable underflow error. Keeping full-range local P and exact
V quantization, then representing only the completed dot's coefficient as a Q8 mantissa plus
exponent, recovered 38.82/33.58 dB. It is not a speed optimization on SM120. The direct grouped
Triton recurrence used 255 registers, spilled 28 slots, and took 2.59 ms at N=8192. A shared
per-query exponent retained 38.74/33.54 dB, removed spills, and used 234 registers, but still took
2.23 ms versus 1.80 ms for the simpler FP32-rescaled kernel. The integer recurrence is therefore
kept as a numerical ablation rather than selected.

Inspect whether feature Hadamard rotations improve the lower tail of tile-normalized per-key V
scales on real FLUX activations with:

```shell
uv run \
  --with 'diffusers>=0.39' \
  --with transformers \
  --with accelerate \
  --with safetensors \
  --with sentencepiece \
  python benchmarks/analyze_sage_feature_convrot_scales.py \
  --local-files-only
```

The report separates coordinate outliers, which regular or randomized signed Hadamard rotations
can reduce, from irreducible per-token RMS variation. It also reports the query-conditioned
`max(P * r)` distribution by layer, where `r` is a key's V scale divided by its K-tile maximum.
That statistic is the actual fraction of the UINT8 range used by tile-wide P scaling.

Evaluate calibrated feature bases for fixed-scale UINT8 P and per-feature INT8 V with:

```shell
uv run \
  --with 'diffusers>=0.39' \
  --with transformers \
  --with accelerate \
  --with safetensors \
  --with sentencepiece \
  python benchmarks/analyze_sage_per_feature_transforms.py \
  --local-files-only
```

The first prompt calibrates per-layer/per-head transforms; all remaining prompts are held out.
The analyzer compares identity, structured and random orthogonal rotations, PCA, diagonal
feature balancing, bounded-condition ZCA/Cholesky whitening, and the per-key dynamic Triton
control. It simulates the full INT8-QK, tile-local UINT8-P, per-feature-INT8-V recurrence rather
than reporting V reconstruction alone. Dense calibrated transforms are intended to be folded
into the V and output projections; applying them as standalone runtime kernels would invalidate
the attention timing comparison.

The calibrated basis was also evaluated specifically for sorted K512 scale runs, K128 UINT8-P
tiles, and grouped INT8 V. Calibration uses ten calls from prompt 0; the reported results use 30
held-out calls from three other prompts. Plain H64 remains effectively neutral, but bounded ZCA
equalizes the group coordinates enough to support much wider groups:

| basis | group4 mean/worst | group16 mean/worst | group32 mean/worst | group64 mean/worst |
|:---|---:|---:|---:|---:|
| identity | 38.75/33.16 | 37.88/31.48 | 37.49/30.78 | 37.13/30.12 |
| H64 | 38.77/33.10 | 37.90/31.46 | 37.52/30.80 | 37.16/30.04 |
| ZCA cond4 | 39.98/35.77 | 39.64/35.30 | 39.47/35.07 | 39.28/34.56 |
| ZCA cond8 | 39.97/35.92 | 39.77/35.62 | 39.64/35.33 | 39.52/35.13 |

The production-shaped atomic packer and Triton kernel, with transformed V rounded to BF16 and
the inverse basis applied to the output, measured 38.87/35.29 dB for ZCA-cond8/group16. The
same held-out canonical FP8 SageAttention2++ control measured 36.57/32.64 dB. Group32 and group64
also passed at 38.95/35.27 and 38.87/34.87 dB, but did not improve kernel speed over group16.

Requesting R224 gives group16 a better compiler schedule: 224 registers and six spill slots,
versus the poor uncapped 212-register schedule. Same-input RTX 5090 end-to-end measurements were:

| N | K512 group4 ms | ZCA-ready group16/R224 ms |
|---:|---:|---:|
| 8192 | 2.24030 | 2.20406 |
| 32768 | 30.12639 | 29.72963 |
| 131072 | 463.52385 | 459.27319 |

The roughly 1% gain is small; at 131K the isolated attention kernels are effectively tied. The
benefit does not justify a runtime dense transform. For model integration, fold a calibrated
per-layer/per-head `Z` into `W_V` as `W_V @ Z` and fold `Z^-1` into the output projection as
`Z^-1 @ W_O`. Group16/R224 is the selected calibrated experiment; group4 remains the portable
uncalibrated path.

Measure hardware-granular PV skipping on the same real FLUX.2 Klein activations with:

```shell
uv run \
  --with 'diffusers>=0.39' \
  --with transformers \
  --with accelerate \
  --with safetensors \
  --with sentencepiece \
  python benchmarks/analyze_sage_pv_skipping.py \
  --local-files-only
```

The first prompt calibrates per-layer/per-head skip thresholds and three held-out prompts measure
the resulting skip rate and combined quantization-plus-sparsity error. The analyzer compares the
original key order, range sorting within local K512 regions, and global range sorting. Its score
gap is the inexpensive SpargeAttention-style online gate; the value gate additionally incorporates
the transformed per-key V range. The online-mass gate uses the tile row sum and running denominator
that online softmax already computes. Final softmax mass and the actual tile-output norm are
included as offline oracles rather than proposed hot-loop implementations. Decisions are
aggregated over hardware-shaped query groups and K128 PV tiles, and the softmax denominator always
includes every key even when a numerator contribution is omitted.

On FLUX.2 Klein at N=1152, thresholds were calibrated from ten prompt-0 captures and evaluated on
30 captures from three held-out prompts. Global range sorting was necessary for the ZCA-cond8,
group16 dense control: it measured 39.77/35.62 dB mean/worst SQNR, versus 36.57/32.64 dB for
canonical SageAttention2++. The finer K64/Q16 analysis found only a very small safe PV-skip region:

| gate | calibration target | held-out skip | mean/worst SQNR | relative L1 |
|:---|---:|---:|---:|---:|
| online mass | 1% | 1.5% | 39.43/32.80 dB | 0.0093 |
| online mass | 2% | 2.6% | 38.10/30.03 dB | 0.0101 |
| final-mass oracle | 1% | 1.3% | 39.72/34.18 dB | 0.0089 |
| final-mass oracle | 2% | 2.3% | 39.07/31.93 dB | 0.0092 |
| contribution oracle | 5% | 5.0% | 39.06/27.69 dB | 0.0103 |

The online-mass gate is therefore mathematically useful but not a worthwhile kernel optimization
for this model and sequence length: the roughly 1.5% safe PV sparsity cannot repay Q16 scheduling,
runtime control flow, or additional state. The analyzer remains useful for long-video captures,
where sparsity may change materially with sequence length and spatial-temporal structure.

Pure V-magnitude pruning was also tested after global sorting. For every K tile, `v_only` takes
the maximum transformed row range in that tile and removes the lowest calibrated tiles for every
query, without consulting QK scores or probabilities. It failed even at very low aggregate skip
rates:

| resolution / N | K tile | calibration target | held-out skip | mean/worst SQNR |
|:---|---:|---:|---:|---:|
| 512px / 1152 | 64 | 5% | 1.8% | 31.78/10.62 dB |
| 512px / 1152 | 128 | 5% | 1.5% | 31.74/10.83 dB |
| 768px / 2432 | 64 | 1% | 0.4% | 36.10/16.86 dB |
| 768px / 2432 | 64 | 5% | 1.9% | 29.95/11.27 dB |

Resolution does change block-selection granularity: one K64 block is 5.6% of an N=1152 head and
2.6% of an N=2432 head. The higher-resolution run shows that finer granularity does not repair the
underlying selector. Some low-range V blocks receive decisive attention, so V magnitude alone is
not a safe proxy for PV contribution. Global sorting remains useful for grouped quantization, but
it does not justify unconditional removal of the low end of the sorted sequence.

### Stock-Triton affine UINT8 execution

The signed-MMA affine proxy now stores each exact K64 `sum(Vq)` correction as INT16 rather than
`128*sum(Vq)` as INT32. The sum is bounded by 8128, so reconstructing the INT32 MMA accumulator
with a seven-bit shift is exact while halving metadata storage. For the selected scaled-FP16
numerator, `scaled_fp16_correction=True` goes further: preparation stores
`sum(Vq)/512` directly in the numerator's 2^-16 FP16 coordinate, and the attention kernel adds it
after converting the signed MMA partial. This changes only FP16 reassociation. A direct test
matches the exact INT32-correction output within 0.002 absolute/0.004 relative tolerance, and the
N=2048 synthetic attention benchmark retained 36.00 dB SQNR versus 28.27 dB for its FP8 control.

Profile the selected stock-Triton path with:

```shell
uv run python benchmarks/profile_sage_pv_variant.py \
  int8-log-split-scale-forward-precomputed-pv-scale-scaled-fp16-numerator-fp16-correction-unmasked-descriptor \
  --sequence 32768 --block-m 128 --num-stages 2 --maxnreg 240
```

Matched RTX 5090 hot results were:

| N | original INT32 metadata ms | FP16-coordinate correction ms | native U8xS8 ms |
|---:|---:|---:|---:|
| 4096 | 0.57191 | 0.50590 | 0.46115 |
| 8192 | - | 1.87228 | 1.70435 |
| 32768 | - | 28.43666 | 25.92803 |
| 131072 | - | 448.99 | 408.45 |

The remaining stock-Triton penalty is about 10%. Generated code attributes it to the affine
correction path: 16 additional global-load instructions and 64 packed FP16 additions per loop
body. It is not an INT8 tensor-core throughput difference. Three exact alternatives were negative
controls: reconstructing `sum(Vq)` from the loaded V tile took 0.625 ms at N=4096, moving the
correction through a separate tensor descriptor took 0.595 ms, and spelling the correction as a
post-dot add generated the same SASS as supplying it as MMA C. Loading the correction earlier also
caused spills. Stock Triton 3.7.1 and current upstream both reject U8xU8 `tl.dot`, so moving the
zero point to V and using a row-wise `sum(P)` correction is not currently expressible without a
compiler change.

The matched N=8192 end-to-end benchmark, including Q/K/V preparation, measured 2.04952 ms for
stock Triton versus 1.85847 ms for patched native U8xS8. Hot attention was 1.89832 versus
1.71106 ms, and both paths measured 36.06-36.07 dB SQNR.

The exact affine correction can also be delayed across eight K64 tiles with
`delayed_fp16_correction_group=8` (benchmark CLI:
`--delayed-fp16-correction-group 8`). The kernel retains each tile's block maximum, reconstructs
the eight query-dependent recurrence weights at the group boundary, and contracts them with the
eight precomputed correction vectors. This does not approximate the attention math; only the
placement and FP16 association of the correction changes. A direct test covers both G8 and G16,
and the N=2048 synthetic benchmark measured 35.98 dB for G8 versus 36.00 dB for per-tile
correction.

On RTX 5090, G8 removes most per-tile correction loads and packed additions while adding one
group-boundary barrier and eight static `exp2` instructions. Registers/spills remain 254/10.
Matched hot profiles were:

| N | per-tile correction ms | delayed G8 ms | patched native U8xS8 ms |
|---:|---:|---:|---:|
| 4096 | 0.50564 | 0.46913 | 0.46115 |
| 8192 | 1.87228 | 1.77645 | 1.70435 |
| 32768 | 28.40416 | 26.95103 | 25.92803 |
| 131072 | 448.99 | 425.82 | 408.45 |

Thus delayed correction recovers over half of the stock-Triton affine penalty and leaves roughly
a 4% long-context gap to native mixed-sign MMA. In the public N=8192 H24 benchmark, hot/E2E
latency improved from 1.90408/2.04939 ms to 1.80354/1.96090 ms. G16 was slower at 0.48362 ms
for N=4096 because it reached 255 registers and 16 spills. Two exact log-metadata controls were
also negative: storing block maxima in FP16 took 0.47936 ms, and incrementally reusing the online
softmax weights took 0.48196 ms because its packed multiplies execute on every tile. The selected
G8 path therefore retains FP32 block maxima and reconstructs weights once per group.

A hierarchical G16 control treated correction alignment as a second mini-recurrence: it retained
the first G8's FP16 weights and maximum while collecting the second G8's maxima, then rescaled the
first weights for one final K16 contraction. The extra Mx8 state remained live beside both D64
attention accumulators, producing 255 registers, 16 spills, and 0.49368 ms at N=4096. M64 avoided
spills but its best 0.51650 ms remained slower than M64 G8 at 0.50769 ms and much slower than the
M128 G8 schedule. Combining the two D64 correction contractions into one interleaved D128 dot was
also counterproductive: join/split layout conversions increased static permutations to 387,
barriers to 49, spills to 22, and latency to 0.56355 ms. Both controls were removed after
measurement. Factoring the selected G8 correction into one helper and removing its redundant
maxima reset reduced static SASS by eight instructions without changing its measured latency.

### Optimized nonnegative signed-INT8 probability path

When one probability bit can be traded for speed, `affine_probability=False` encodes P directly
as signed INT8 codes in `[0, 127]` and removes affine correction entirely. The scale-forward
preparation now precomputes `127 * s_v[k]` just as the UINT8 path precomputes `255 * s_v[k]`, and
the fixed 2^-16 FP16 numerator epilogue uses the selected probability range rather than assuming
255. On SM120, `triton_sage_attention_int8_pv_per_key_log` selects this optimized split-D64 path
for noncausal D128 attention through N=131072.

Profile it with:

```shell
uv run python benchmarks/profile_sage_pv_variant.py \
  int8-log-signed-split-scale-forward-precomputed-pv-scale-scaled-fp16-numerator-unmasked-descriptor \
  --sequence 32768 --block-m 128 --num-stages 2
```

Matched prequantized RTX 5090 profiles show that removing correction brings stock signed INT8 to
the same performance class as the compiler-patched native U8xS8 path:

| N | signed INT8 ms | delayed affine G8 ms | patched native U8xS8 ms |
|---:|---:|---:|---:|
| 4096 | 0.46063 | 0.47047 | 0.46115 |
| 8192 | 1.68076 | 1.77645 | 1.70435 |
| 32768 | 25.71550 | 26.95103 | 25.92803 |
| 131072 | 404.89 | 425.82 | 408.45 |

The generated N=4096 kernel uses 253 registers, 14 spills, 128 signed IMMA instructions, and no
correction HMMA. At the FLUX.2 Klein model's N=1152 shape, signed INT8 measured 0.04852 ms hot and
0.07252 ms E2E, versus 0.05180/0.07580 ms for per-tile affine UINT8 and 0.088 ms E2E for canonical
SageAttention2++. At longer context it retains the exact key-scale recurrence cost: N=8192 E2E
was 1.86577 ms versus 1.752 ms for pure-Triton SageAttention2++ and 1.647 ms for canonical CUDA;
N=32768 was 26.50122 versus 24.629 and 22.904 ms, respectively.

Quality remains above the chosen canonical target. On the established two-prompt, twenty-call
FLUX.2 Klein sample, signed key-scaled INT8 measured 39.81 dB mean / 36.62 dB worst output SQNR
and 0.0089 relative L1. Canonical SageAttention2++ measured 36.49/32.64 dB and 0.0133. Thus the
seventh probability bit costs roughly 1.5-2 dB versus affine UINT8 but retains a 3-4 dB margin
over canonical SageAttention2++ on this sample.
