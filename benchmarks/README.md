# Benchmarks

Operator benchmarks live here rather than in the correctness test suite. Each benchmark
reports hardware, software, Git state, shapes, kernel configuration, numerical quality where
applicable, and consistently named timing phases. The support code in `lib/` is development-only;
it is not part of the installed `piper_kernels` API.

## Common provider and timing model

A provider has two explicit callables:

- `prepare()` performs per-invocation preprocessing such as quantization, packing, or
  scale construction and returns the prepared inputs.
- `run(prepared)` executes the operator using already-prepared inputs. It may launch
  one or more kernels.

“Prepared” describes the benchmark boundary, not an operator's internals. Work performed
inside a public operator—including ConvRot activation rotation and quantization—remains part
of `run(prepared)`. A provider may use a no-op `prepare()` when it repeatedly invokes the
complete operator on fixed source tensors.

The common runner reports these phases:

- `first_call_ms`: synchronized wall time for the first operator invocation, including
  any lazy compilation. It is not compiler CPU time in isolation and does not claim
  that compiler caches were initially empty.
- `preparation`: warmed, synchronized wall latency of preparation-only work.
- `prepared_execution`: warmed device-event latency of `run(prepared)` on fixed source
  objects, including any preparation internal to the timed operator.
- `operator_end_to_end`: warmed, synchronized wall latency of `run(prepare())`.

Synchronized wall timing captures host dispatch, allocation, packing, and device work.
Device-event timing isolates elapsed work on the accelerator stream. Every latency
distribution serializes its `clock`, and `first_call_clock` describes the scalar first call.

Warmed latencies are displayed as `p50 [p20, p80]`. A phase is `null` in machine output
when it does not apply to a provider. The configured warmup and measurement-time windows
are stored alongside every phase result. Benchmark code can use the shared model directly:

```python
provider = BenchmarkProvider(
    name="my-kernel",
    prepare=prepare_inputs,
    run=launch_kernel,
    synchronize=torch.cuda.synchronize,
    configuration={"block_m": 64, "num_warps": 4},
)
measurement = measure_provider(provider, warmup_ms=100, measurement_time_ms=500)
```

`AttentionShape` records batch size, Q/KV head counts, Q/KV sequence lengths, and head
dimension without assuming self-attention or MHA. `AttentionConfig` records dtype,
causality, scale, and an explicit QKV layout such as `BHSD`.

## Offline configuration tuning

`tune_candidates()` provides a small offline search loop for development. Kernel-specific
adapters define named configurations and construct `BenchmarkProvider` instances; the shared
runner compiles each candidate, applies an optional quality gate, measures either prepared
execution or the complete operator, and selects the fastest passing candidate. Unsupported and
out-of-resource candidates are recorded rather than aborting the search. Unexpected failures
still propagate so implementation and compiler bugs remain visible.

This tooling never changes production dispatch or autotunes in a user's hot path. Every candidate
is available as a versioned record accepted by the common JSON/JSONL writer, and the winner is
marked with `selected: true`.

The executable Piper Attention tuner consumes the same immutable execution plan as production.
Omitted axes retain their production values, so the default invocation measures exactly the
production plan:

```shell
uv run python benchmarks/tune_piper_attention.py \
  --sequence 8192 \
  --json artifacts/piper_attention_tuning.json
```

Use `--phase operator_end_to_end` to include preprocessing in the ranking. The default
`prepared_execution` phase compares only the prepared fused recurrence. On targets where a
candidate is unsupported, it remains in the report with `status: skipped`. Candidates with
non-finite output mismatches or less than 20 dB SQNR are rejected by default; use
`--minimum-sqnr-db` to change the finite threshold.

Explicit Piper axes form a deduplicated Cartesian search, capped at 256 candidates. This makes
Triton's native loop-pipeline and loop-invariant-code-motion controls measurable without
silently applying a SageAttention2++ schedule to Piper Attention:

```shell
uv run python benchmarks/tune_piper_attention.py \
  --sequence 8192 --head-dim 128 --causal \
  --load-path pointer tensor-descriptor \
  --block-m 64 128 \
  --num-stages 2 3 \
  --causal-block-order forward reverse \
  --loop-num-stages none 3 \
  --loop-licm disabled enabled \
  --json artifacts/piper_attention_execution_plan.json
```

The SageAttention2++ adapter searches the same immutable execution-plan fields used by
production dispatch. Omitted axes retain the production value; values supplied for multiple
axes form a Cartesian search, capped at 256 candidates by default:

```shell
uv run python benchmarks/tune_sage_attention_2pp.py \
  --sequence 8192 --head-dim 128 \
  --block-m 64 128 \
  --num-stages 2 3 \
  --load-path pointer tensor-descriptor \
  --json artifacts/sage_attention_2pp_sm120_tuning.json
```

As in `benchmark_attention.py`, SageAttention2++'s `prepared_execution` phase is the
complete public operator—including statistics and Q/K/V quantization—not only the final
recurrence kernel.
Use `operator_end_to_end` when synchronized host and allocation overhead should participate in
the ranking. Every candidate must also clear the configurable SQNR and non-finite quality gate.

The SageAttention2++ tuner currently runs on the optimized NVIDIA SM89+ backend, including
SM120. It can search a future CUDA target once that target is supported by the kernel, but it
does not enable a new backend. In particular, AMD `gfx1200`/`gfx1201` remain unsupported until
the inline PTX FP8 conversion and NVIDIA FP8-MMA path have HIP equivalents.

Selected records are evidence, not runtime policy: review the result across representative
shapes and repeated processes, then deliberately freeze an accepted winner in the production
execution-plan selector. The project does not use `triton.autotune` in the public operator path;
doing so would add first-use compilation/search latency, multiply cache entries across dynamic
shapes, and cannot apply the tuner's full-operator quality gate. Triton autotuning remains useful
for narrow, opt-in single-kernel experiments, but offline search is the production-policy tool.

## Quality and reproducibility

`measure_quality()` centralizes mean/max absolute error, relative L1 and L2 error,
SQNR, cosine similarity, and actual/reference non-finite counts. Providers can attach
endpoint saturation counts for quantized tensors with `measure_saturation()`.
Integer quality inputs through 32 bits are promoted to FP64. Full-width INT64 and UINT64
inputs are rejected because no floating comparison dtype preserves every possible value;
providers needing them should use a domain-specific exact comparison.

Every `BenchmarkRecord` includes:

- GPU name, accelerator backend, and architecture;
- Python, Torch, Triton, CUDA or ROCm runtime, and available driver versions;
- Git revision and dirty-worktree state;
- logical shape, provider configuration, phase timings, quality, and optional extras.

All benchmark CLIs retain their Markdown or terminal summaries. Add `--json PATH` to
write a versioned JSON array or `--jsonl PATH` to write one compact record per line.
Serialization is strict JSON; non-finite floating-point metrics such as infinite SQNR
for an exact result are represented as `null`.

The schema starts at version 1. Consumers should check `schema_version` before relying
on field names. A shortened record looks like:

```json
{
  "schema_version": 1,
  "benchmark": "integer-pv-dot",
  "provider": "triton-native",
  "shape": {"tiles": 2048, "key_tile": 64},
  "configuration": {
    "lhs_dtype": "int8",
    "rhs_dtype": "int8",
    "accumulator_dtype": "int32",
    "implementation": "native",
    "block_m": 64,
    "block_n": 128,
    "num_warps": 4,
    "seed": 0
  },
  "timings": {
    "warmup_ms": 500,
    "measurement_time_ms": 2000,
    "first_call_ms": 310.2,
    "first_call_clock": "synchronized_wall",
    "preparation": {
      "median_ms": 0.004,
      "p20_ms": 0.004,
      "p80_ms": 0.005,
      "clock": "synchronized_wall"
    },
    "prepared_execution": {
      "median_ms": 0.031,
      "p20_ms": 0.030,
      "p80_ms": 0.032,
      "clock": "device_event"
    },
    "operator_end_to_end": {
      "median_ms": 0.036,
      "p20_ms": 0.035,
      "p80_ms": 0.037,
      "clock": "synchronized_wall"
    }
  }
}
```

## Included benchmarks

Run the ConvRot provider comparison with:

```shell
uv run python benchmarks/benchmark_convrot.py
```

Use `--help` to select activation rows, weight dimensions, group size, dtype,
deterministic input seed, and timing windows. The existing custom-shape interface remains
the default. Custom-shape options such as `--rows` and `--input-activation` cannot be mixed
with a named preset, because the preset supplies those values. `--in-features` is linear and
weight width `K`; a raw `[up | gate]` SwiGLU input has `2K` features. The script validates
ordinary linears with the output dtype's standard tolerance; the SwiGLU path uses a declared
relative-L2 bound. Quality is computed over at most 256 rows stratified across `M`, including
the final row and rows around every first signed-32-bit input or output element-offset crossing.
Machine output records the exact sampled indices.

The Piper provider times the complete public entrypoint on fixed source tensors, so its
`prepared_execution` includes ConvRot's internal activation preparation and GEMM. Provider
configuration records the public entrypoint and whether SwiGLU dispatch selected fused or
materialized input preparation. Record shapes contain only the case name and logical dimensions;
provider configuration distinguishes the logical input layout from the layout passed to that
provider. The optional provider is recorded as provider-managed when its internal choice is not
observable.

Exercise the four principal bias-free MiniMax H3 transformer projections at the measured
5-second row count or at 128K rows with:

```shell
uv run python benchmarks/benchmark_convrot.py --preset minimax-h3-5s
uv run python benchmarks/benchmark_convrot.py --preset minimax-h3-128k
```

Both presets cover QKV `(N, K) = (21504, 5376)`, attention output `(5376, 7168)`,
MLP FC1 `(28672, 5376)`, and MLP FC2 `(5376, 14336)`. FC2 consumes the explicit
raw `[up | gate]` input contract with 28672 features, applies SwiGLU, and supplies linear
`K = 14336` to ConvRot. The 5-second preset uses `M = 37710`; the 128K preset uses
`M = 131072`. Row count remains a reproducible workload choice rather than a universal
model constant.

The 128K preset automatically skips full portable-reference timing because its FP32 output
temporary can exceed 10 GiB. It still runs each complete optimized tensor and validates the
sampled boundary rows. Custom memory-intensive shapes can select the same behavior with
`--skip-reference-timing`.

Compare against the optional Comfy Kitchen CUDA provider with:

```shell
uv run --with comfy-kitchen==0.2.28 \
  python benchmarks/benchmark_convrot.py \
  --preset minimax-h3-5s --compare-comfy-kitchen
```

Comfy Kitchen is a benchmark-only dependency and is loaded only when requested. Its 0.2.x
SwiGLU API consumes `[gate | up]`, so the benchmark prepares that provider's reordered input
once outside the timed operator while keeping Piper's public `[up | gate]` contract. Provider
metadata records the adapter and the installed package version under `installed_version`.

Diagnose activation preparation independently, using preallocated outputs, with:

```shell
uv run python benchmarks/benchmark_convrot_preparation.py

uv run python benchmarks/benchmark_convrot_preparation.py \
  --rows 131072 --in-features 14336 --input-activation swiglu

uv run --with comfy-kitchen==0.2.28 \
  python benchmarks/benchmark_convrot_preparation.py \
  --rows 37710 --in-features 5376 --compare-comfy-kitchen
```

The preparation benchmark reports rotation, rowwise quantization, their two-launch split,
and the one-pass fused candidate for the H3 widths. The traffic column is an algorithmic
minimum, not a measured DRAM-transaction count. Unlike the permissive public comparison above,
the preparation adapter calls a private native entrypoint and accepts exactly
`comfy-kitchen==0.2.28`. Its records include both the installed package version and the private
adapter-contract version. The final column names its Piper baseline explicitly: split
preparation without an input activation and fused preparation for SwiGLU.

Add `--json PATH` or `--jsonl PATH` to serialize one common `BenchmarkRecord` per width and
phase. Each record distinguishes linear `K` from raw input width and includes the phase,
operation provenance, baseline, device timing, minimum traffic, and effective bandwidth. Piper
records additionally include the selected fused block size, warp count, and production-policy
eligibility. Piper timing and compiler records use the same `piper-triton` provider identifier
and plan configuration.
Compiler output remains independently selectable with `--compiler-json` or
`--compiler-jsonl`, so both record types can be written by one invocation. Benchmark and
compiler records must use different output paths.

The common compiler-report adapter can inspect one width per fresh process:

```shell
uv run python benchmarks/benchmark_convrot_preparation.py \
  --rows 37710 --in-features 5376 \
  --compiler-report --no-sass \
  --compiler-json artifacts/convrot-preparation-5376.json
```

Compiler records include specialization fingerprints, registers, spills, shared memory,
warps, stages, and resource-based residency ceilings. Repeat in separate processes for each
width so process-wide Triton specialization caches remain unambiguous.

Run the stock-Triton integer P x V microbenchmark with:

```shell
uv run python benchmarks/benchmark_integer_pv_dot.py s8-s8
uv run python benchmarks/benchmark_integer_pv_dot.py u8-s8-native
uv run python benchmarks/benchmark_integer_pv_dot.py u8-s8-affine-proxy
```

The `u8-s8-native` variant uses Piper Attention's stock-Triton compiler extension to emit native
`UINT8 x INT8 -> INT32` MMAv2. The extension is packaged in the normal Python wheel and
requires no patched Triton, CUDA extension, native build, or executable inline PTX. It is tested
with Triton 3.7.1 and validates its compiler hook and generated MMA fail-closed, allowing newer
Triton versions only while the same lowering remains compatible.

Native mixed-sign lowering currently requires NVIDIA SM8x or consumer Blackwell SM12x and the
`m16n8k32` MMAv2 path. Turing, Hopper WGMMA, datacenter Blackwell, and ROCm mixed-sign lowering
are not supported by this extension. The native benchmark installs the hook automatically before
JIT compilation; production native-UINT8 launchers use the same selection-time installation.
Unsupported targets should select the exact affine signed-INT8 proxy instead. The benchmark
records the LHS, RHS, and accumulator dtypes explicitly, checks exact INT32 output including
UINT8 values above 127, and records operand saturation.

Inspect the generated mixed-sign MMA while verifying exact output with:

```shell
uv run python benchmarks/benchmark_integer_pv_dot.py u8-s8-native \
  --compiler-report --no-sass
```

The PTX report contains `mma.sync.aligned.m16n8k32...s32.u8.s8.s32`. Add SASS inspection when
`nvdisasm` is available to verify the corresponding native `U8.S8` machine instruction.
Backend-specific PTX, SASS, and AMDGCN inspection belongs to compiler/profiling tooling
rather than this portable benchmark runner.

Run full-attention comparisons with:

```shell
uv run python benchmarks/benchmark_attention.py
```

The hardware-aware default always includes PyTorch SDPA, adds Piper Attention where its
mixed-sign MMA is supported, and adds pure-Triton SageAttention2++ where FP8 tensor cores
are supported. Choose any subset with `--providers`; `--help` lists the stable provider
names. For example:

```shell
uv run python benchmarks/benchmark_attention.py \
  --sequence 8192 \
  --providers piper_attention piper_attention_affine \
              sage_attention_2pp pytorch-sdpa
```

Add the revision-pinned official CUDA SageAttention2++ and SageAttention2 providers
with:

```shell
TORCH_CUDA_ARCH_LIST=12.0 uv sync --group benchmark
uv run python benchmarks/benchmark_attention.py --canonical
```

Replace `12.0` with `8.9` on RTX 40-series GPUs. The benchmark dependency is
SageAttention 2.2.0 at commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` and is never
imported by package production code. SM89 comparisons use canonical per-thread Q/K
quantization; SM12x comparisons use canonical per-warp Q/K quantization. Both canonical
providers enable K smoothing and differ only in their P x V accumulator strategy.

Each row uses the common provider lifecycle and records first-call synchronized wall time,
preparation, warmed device-event execution, complete operator latency, quality against SDPA,
and effective TFLOP/s. Use `--sequence`, `--kv-sequence`, `--head-dim`, `--dtype`, and
`--causal` to build a shape matrix. JSON and JSONL output use the shared versioned benchmark
schema and identify the algorithm and implementation in each provider's configuration.
Pure-Triton SageAttention2++ records also serialize their selected launch, fusion, loop, and
packed-probability-conversion choices.

Piper Attention exposes a more granular lifecycle than SageAttention2++ and SDPA operators:

- `preparation` includes compact K/V mean reduction, Q/K/V quantization, scale metadata,
  and affine correction metadata when requested;
- `prepared_execution` is the hot fused QK, FP32 online-softmax, integer PV recurrence,
  and centered-mean epilogue;
- `operator_end_to_end` runs preparation and the fused kernel as one complete call.

Machine records also identify Q/K granularity and native versus affine mixed-sign execution.
Historical fixed-INT8, block-INT8, sorted-group, and key-scaled research controls remain
reproducible from the `wip/sage-integer-attention` checkpoint at `b75f3ee`; they are not copied
into the installed package.

Compiler inspection and external profiling are available for one shape at a time:

```shell
uv run python benchmarks/benchmark_attention.py \
  --sequence 8192 --providers sage_attention_2pp \
  --compiler-report --compiler-json artifacts/sage_attention_2pp_compiler.json

uv run python benchmarks/benchmark_attention.py \
  --sequence 8192 --providers piper_attention --compiler-report --no-sass

nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  uv run python benchmarks/benchmark_attention.py \
  --sequence 8192 --profile --profile-provider sage_attention_2pp
```

When more than one Triton provider is selected, use `--compiler-provider` to choose which
one to inspect. Combined profiling and compiler inspection must target the same provider.

### SageAttention2++ regression baseline

The issue #8 productionization was validated on an RTX 5090 (SM120) with Torch
2.12.1+cu130 and Triton 3.7.1. For FP16 B1/H8/D128 non-causal self-attention at
N=8192, a one-second warmed sample measured:

| provider | device p50 [p20, p80] (ms) | synchronized wall p50 [p20, p80] (ms) | mean absolute error vs SDPA |
|:---|---:|---:|---:|
| pure Triton SageAttention2++ | 0.637 [0.635, 0.641] | 0.666 [0.665, 0.668] | 0.000563 |
| canonical CUDA SageAttention2++ | 0.610 [0.608, 0.611] | 0.618 [0.612, 0.620] | 0.000563 |
| canonical CUDA SageAttention2 | 0.707 [0.705, 0.708] | 0.707 [0.706, 0.709] | 0.000561 |
| PyTorch SDPA | 1.692 [1.689, 1.695] | 1.702 [1.699, 1.717] | 0 |

The pure-Triton attention specialization used 255 registers per thread, 8 compiler-reported
spills, 49,704 bytes of shared memory per workgroup, and four warps. Its SASS contained the
expected 64 signed INT8 QK MMA instructions and 64 E4M3 x E4M3 to FP16 PV MMA instructions.
The complete GPU suite passed 155 tests. These measurements are a regression reference for
this hardware/software stack, not a portable performance guarantee.

### PyTorch 2.13 migration checkpoint

The minimum-version upgrade was benchmarked on an RTX 5090 (SM120), driver 595.71.05,
Python 3.14.6, CUDA 13.0, and Triton 3.7.1. Each result is the median of three process-level
medians using BF16 B1/H8/D128 non-causal self-attention, a 300 ms warmup window, and a
1.5 second measurement window. Positive deltas mean Torch 2.13 was slower.

| sequence | Torch 2.12.1 hot (ms) | Torch 2.13.0 hot (ms) | hot delta | Torch 2.12.1 complete (ms) | Torch 2.13.0 complete (ms) | complete delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.0366 | 0.0372 | +1.57% | 0.0931 | 0.0927 | -0.46% |
| 2,048 | 0.0822 | 0.0825 | +0.31% | 0.1352 | 0.1349 | -0.24% |
| 4,096 | 0.1741 | 0.1741 | +0.00% | 0.2238 | 0.2231 | -0.30% |
| 8,192 | 0.6124 | 0.6124 | +0.00% | 0.6479 | 0.6471 | -0.13% |
| 16,384 | 2.2415 | 2.2395 | -0.09% | 2.2600 | 2.2558 | -0.18% |

The 1K hot result differs by one device-timer quantum (about 0.0006 ms). At larger shapes,
Torch 2.13 changed hot latency by at most 0.31% and slightly improved every complete-operator
median. All reported quality metrics were identical between versions.

### SageAttention2++ SM89 tuning checkpoint

The pure-Triton SageAttention2++ path was tuned on an RTX 4070 Ti SUPER (SM89)
under Windows 11, driver 596.49, Python 3.14.7, Torch 2.12.1+cu130, CUDA 13.0,
and Triton 3.7.1.post27. BF16 B1/H8/D128 warmed device-event medians measured:

| sequence | execution | pure Triton (ms) | canonical CUDA (ms) | gap |
|---:|:---|---:|---:|---:|
| 8,192 | non-causal | 1.320 | 1.326 | -0.5% |
| 8,192 | causal | 0.914 | 0.924 | -1.1% |
| 32,768 | non-causal | 20.241 | 19.987 | +1.3% |
| 32,768 | causal | 11.060 | 11.130 | -0.6% |
| 131,072 | non-causal | 310.977 | 302.304 | +2.9% |
| 131,072 | causal | 164.773 | 157.340 | +4.7% |

Negative gaps mean Triton was faster. The retained D128 causal schedule uses 128
query rows, four warps, two launch stages, and reverse CTA ordering from 8K onward.
The long non-causal path uses 128 query rows, four warps, 64-key tiles,
loop-invariant-code motion, and a three-stage loop pipeline. Packed native
`cvt.rn.satfinite.e4m3x2.f32` replaces stock Triton's software E4M3 conversion
on the SM89 probability and V paths. These imported measurements establish a 5%
non-inferiority checkpoint; they were not reproduced locally without SM89 hardware.

### Packed E4M3 conversion SM120 portability check

The packed conversion was separately A/B tested on an RTX 5090 (SM120), driver
595.71.05, Python 3.14.6, Torch 2.12.1+cu130, and Triton 3.7.1. Each result below is
the median process result from three rounds with BF16 B1/H8 inputs, 300 ms warmup
windows, and 1.5 second measurement windows. The 8K/32K runs alternated baseline and
packed worktrees; the 128K runs rotated clean, ungated, and selective worktrees:

| execution | head dim | sequence | ungated packed hot-latency change |
|:---|---:|---:|---:|
| non-causal | 64 | 8,192 / 32,768 | +0.27% / -0.12% |
| causal | 64 | 8,192 / 32,768 | -0.88% / -0.40% |
| non-causal | 128 | 8,192 / 32,768 | -0.51% / -0.38% |
| causal | 128 | 8,192 / 32,768 | +1.50% / +1.61% |
| non-causal | 128 | 131,072 | -0.94% |
| causal | 128 | 131,072 | +1.40% |

Compiler inspection showed the packed D128 causal attention kernel increasing from
22 to 24 spills, while the packed fused-V quantizer added eight SASS instructions.
Production therefore keeps stock conversion for SM120 fused V and D128 causal
probabilities, while retaining packed probabilities on the beneficial paths. A final
three-round comparison of this selective policy measured +0.02% / -0.05% hot deltas
at causal D128 8K / 32K. At 128K, the selective policy retained a -1.19% non-causal
gain and measured -0.07% causal versus clean; the corresponding ungated attention
kernel raised causal spills from 18 to 20. Quality metrics were unchanged.

### Piper Attention regression baseline

Issue #6 was validated on an RTX 5090 (SM120) with Torch 2.12.1+cu130 and
Triton 3.7.1. BF16 non-causal self-attention measured the following warmed
latencies; Piper Attention's hot column is its prepared fused recurrence, while the complete
column includes all preprocessing. The uncentered rows are historical development controls;
the production operator now always centers V.

| shape | provider | hot device p50 [p20, p80] (ms) | complete wall p50 [p20, p80] (ms) | SQNR vs SDPA (dB) |
|:---|:---|---:|---:|---:|
| B1/H8/N8192/D128 | Piper Attention centered | 0.674 [0.672, 0.676] | 0.775 [0.772, 0.779] | 36.08 |
| B1/H8/N8192/D128 | Piper Attention uncentered | 0.675 [0.674, 0.677] | 0.776 [0.774, 0.778] | 36.05 |
| B1/H8/N8192/D128 | Piper Attention affine fallback | 0.706 [0.703, 0.710] | 0.785 [0.784, 0.787] | 36.08 |
| B1/H8/N8192/D128 | pure Triton SageAttention2++ | 0.637 [0.636, 0.639] | 0.669 [0.668, 0.671] | 28.12 |
| B1/H8/N8192/D128 | canonical CUDA SageAttention2++ | 0.609 [0.607, 0.610] | 0.614 [0.607, 0.617] | 28.13 |
| B1/H1/N131072/D128 | Piper Attention uncentered | 19.579 [19.371, 19.600] | 19.824 [19.806, 19.845] | 35.48 |
| B1/H1/N131072/D128 | pure Triton SageAttention2++ | 17.647 [17.611, 17.708] | 17.823 [17.777, 17.926] | 28.33 |
| B1/H1/N131072/D128 | canonical CUDA SageAttention2++ | 17.155 [16.992, 17.171] | 17.177 [17.023, 17.195] | 28.33 |

At N=8192 the fused Piper Attention specialization used 254 registers per thread, 12
compiler-reported spills, 33,588 bytes of shared memory, and four warps. Its PTX
contained 64 signed INT8 QK MMA instructions and 64 native `U8.S8` PV MMA
instructions. Preprocessing kernels reported no spills. These measurements are a
regression checkpoint, not a cross-device performance guarantee.

The production implementation was also replayed on the cached Diffusers BF16
LTX-2.3 attention call used during development (`B1/H32/N6144/D128`). The table
reports global quality and the lowest per-head SQNR; the ignored local capture is
not a repository fixture because the versioned capture/replay format belongs to
issue #11.

| provider | global SQNR (dB) | relative L1 | mean absolute error | max absolute error | worst-head SQNR (dB) |
|:---|---:|---:|---:|---:|---:|
| Piper Attention centered | 38.96 | 0.960% | 0.000907 | 0.0703 | 33.72 |
| Piper Attention uncentered | 38.96 | 0.962% | 0.000909 | 0.0781 | 33.70 |
| pure Triton SageAttention2++ | 32.43 | 2.292% | 0.002166 | 0.1250 | 28.18 |

This ordinary call has little V bias, so centering is nearly neutral. The committed
adversarial biased-V regression covers the centered path, and the constant-V regression
requires exact restoration.

## Triton compiler inspection

Providers register the Triton JIT functions they launch through
`triton_jit_functions`. After the provider has run at least once, the shared inspector
discovers its compiled specialization and reports:

- registers per thread, compiler-reported spills, shared memory and warps per workgroup,
  stages, and CUDA CTAs per cluster;
- a resource-only workgroup and warp residency ceiling per compute unit from the device
  limits exposed by PyTorch, including the limiting resource;
- static PTX instruction-family and MMA-opcode counts when PTX is available;
- static SASS instruction-family and MMA-opcode counts for NVIDIA CUDA kernels.

The residency value is a ceiling, not achieved occupancy. It does not model every
architecture's allocation granularity or replace hardware profiling. For CUDA
specializations with more than one CTA per cluster, resources and residency remain
workgroup/CTA-level values; they do not claim to predict active cluster residency.
Static instruction counts describe one compiled program, not dynamic execution counts.

The integer P x V benchmark is the executable reference integration:

```shell
uv run python benchmarks/benchmark_integer_pv_dot.py s8-s8 \
  --compiler-report \
  --compiler-json artifacts/s8-s8-compiler.json
```

SASS inspection invokes `nvdisasm` from the NVIDIA CUDA Toolkit. It is enabled
automatically for CUDA compiler reports and disabled for other Triton backends. If the
tool is absent, the inspector gives an actionable error; use `--no-sass` when only the
portable resource report and available compiler IR are needed, or use
`--nvdisasm /path/to/nvdisasm` when the toolkit binary is not on `PATH`. ROCm resource
reporting uses the same provider and specialization model, while AMDGCN disassembly
remains a separate future backend adapter.

All Triton-cache and compiled-metadata access lives in `lib/triton_inspection.py`.
Specialized diagnostics can read an artifact without depending on Triton internals:

```python
from lib.triton_inspection import compiled_artifact

ttgir = compiled_artifact(jit_kernel, "ttgir")
```

Compiler JSON has its own versioned `triton_compiler` record type and includes provider
configuration, environment and Git metadata, specialization fingerprints, resources,
and instruction summaries for comparison across commits. `--compiler-json` writes an
array and `--compiler-jsonl` writes one compiler record per line through the same output
machinery as benchmark records.

Compiler reporting requires each registered JIT function to have one specialization in
the current process by default. This prevents one provider from silently claiming
specializations compiled earlier by another provider. Run compiler comparisons as one
provider/configuration per process; advanced diagnostics that intentionally inspect an
entire process-wide cache must opt out explicitly.

## External profiler captures

`profile_provider()` launches either `prepared_execution` or `operator_end_to_end` for
any `BenchmarkProvider`. By default, its initial compilation call and warmup iterations
finish and synchronize before the CUDA profiler starts. Passing
`--profile-include-setup` explicitly includes them in a separate `profile/setup` NVTX
range.

For example, capture the integer P x V provider with Nsight Systems:

```shell
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  uv run python benchmarks/benchmark_integer_pv_dot.py s8-s8 \
  --profile --profile-phase prepared_execution
```

The launch loop accepts an injected capture controller so a future ROCTracer/ROCTx
adapter can reuse its provider-phase and setup-exclusion behavior. The built-in
controller intentionally reports a clear unsupported-backend error on ROCm rather than
presenting CUDA profiler APIs as portable.
