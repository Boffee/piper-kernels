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
variant using block-local P normalization and 64-token V scales. The script tunes its listed
launch configurations per shape and reports prequantized-kernel latency and SQNR.

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
