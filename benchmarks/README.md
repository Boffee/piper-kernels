# Benchmarks

Use these runners to compare operators, investigate regressions, or select an
execution plan. They record shapes, configuration, quality where applicable,
timings, and hardware/software metadata. Synthetic operator results do not
establish full-model speed or checkpoint quality.

Run from the repository root with the Python for the accelerator being tested.
See [development setup](../docs/development.md#accelerator-environments) for CUDA
and ROCm environments. With an external environment, replace `uv run python` below
with `PYTHONPATH=src /path/to/env/bin/python`. Production runners are shared across
CUDA and ROCm, subject to each operator's backend support.

## Choose a runner

Start with an explicit shape and save the result:

```shell
uv run python benchmarks/benchmark_attention.py \
  --sequence 8192 --heads 16 --head-dim 128 --dtype bfloat16 \
  --json artifacts/attention.json
```

| Workload | Runner | Useful controls |
|---|---|---|
| Dense attention | [benchmark_attention.py](benchmark_attention.py) | `--providers`, `--sequence`, `--kv-sequence`, `--causal` |
| ConvRot INT8 linear | [benchmark_convrot_int8.py](benchmark_convrot_int8.py) | `--rows`, `--in-features`, `--out-features`, `--phases` |
| ConvRot INT8 Conv3D | [benchmark_convrot_int8_conv3d.py](benchmark_convrot_int8_conv3d.py) | Repeated `--shape N,C,T,H,W,O`, `--tune`, `--vendor-benchmark` |
| NVFP4 / ConvRot NVFP4 FFN | [benchmark_nvfp4_ffn.py](benchmark_nvfp4_ffn.py) | `--format`, repeated `--shape M K N`, `--activation`, `--scaling` |
| Dense projection/attention/output fusion | [benchmark_piper_fusion.py](benchmark_piper_fusion.py) | `--heads`, `--kv-heads`, `--width`, `--chunk-rows` |
| Sparse attention | [benchmark_sparse_piper.py](benchmark_sparse_piper.py) | `--ratios`, `--routing`, `--sequence` |
| Sparse routing and projections | [scores](benchmark_sparse_piper_scores.py), [projections](benchmark_sparse_piper_projection.py) | Query-block windows, routing mode, projection dimensions |
| Complete sparse fusion | [benchmark_sparse_piper_fusion.py](benchmark_sparse_piper_fusion.py) | `--sequence`, `--query-chunk-rows` |
| Integer probability/value dot | [benchmark_integer_pv_dot.py](benchmark_integer_pv_dot.py) | Arithmetic variants, compiler inspection, profiling |

Use `--help` for the full argument list. The ConvRot `small_m`, `tail`,
`realistic`, and `preparation` scripts are specialized NVIDIA ablations and
compiler diagnostics. Use the ordinary linear runner for production comparisons
on either accelerator. Add workloads to these runners and reuse the
development-only support in [`lib`](lib).

### ConvRot INT8

Include activation rotation, quantization, GEMM, and the scale epilogue when
comparing the public linear operator. `--phases` adds separate preparation,
prepared GEMM, and complete-call device measurements:

```shell
uv run python benchmarks/benchmark_convrot_int8.py \
  --rows 8192 --in-features 6144 --out-features 4096 --no-bias --phases \
  --json artifacts/convrot-int8.json
```

The Conv3D runner compares native and portable quantized operations, plus vendor
FP16 convolution in contiguous and channels-last layouts. FP16 is a performance
baseline; the quantized reference is the correctness oracle. Both include
matching padding and, for fused cases, GroupNorm/SiLU. Weight reconstruction and
initial layout conversion are outside timing. `--skip-reference-timing` retains
correctness checks. On ROCm, record `PYTORCH_MIOPEN_SUGGEST_NHWC` and
`--vendor-benchmark`, which affect the baseline.

### Attention and projection fusion

By default, dense attention chooses supported providers for the device and
includes PyTorch SDPA. Piper's prepared phase times the recurrence after Q/K/V preparation;
SageAttention2++ and SDPA time their complete operator in that phase. Compare
complete operator timings when judging integration cost. The ordinary dense
runner uses equal Q/KV head counts; GQA needs separate coverage, such as the
projection-fusion benchmark.

The optional canonical SageAttention providers require the pinned benchmark
dependency and an NVIDIA build for the target architecture:

```shell
TORCH_CUDA_ARCH_LIST=12.0 uv sync --group benchmark
uv run python benchmarks/benchmark_attention.py --canonical
```

Use `8.9` for SM89. The pinned revision is in [pyproject.toml](../pyproject.toml);
production code does not import this benchmark dependency.

Fusion runners compare complete projection/attention pipelines, verify the
advertised graph rewrites, and exclude compilation and reference checks from
timing. Dense fusion also reports CUDA-graph replay separately. Peak extra
allocation includes outputs and workspace, excluding resident inputs/weights;
dense graph pools are excluded too.

```shell
uv run python benchmarks/benchmark_piper_fusion.py \
  --sequence 8192 8193 --heads 4 --kv-heads 2 --head-dim 64 --width 1024 \
  --chunk-rows 4096 --json artifacts/dense-fusion.json
```

Keep model-shaped results explicit about what is represented. H3 fusion defaults
use B1/H56/D128 and hidden width 5376, not captured checkpoint inputs. Krea2-shaped
dense runs need Hq48/Hkv12/D128, width 6144, and `--rotary-dim 128`; they omit the
model's text padding mask and sigmoid output gate. Sparse `--ratios 1.0` retains
sparse quantization and is not identical to dense Piper.

### NVFP4 FFN

The FFN runner measures the complete fused FFN on SM120. Shapes are input rows,
input/output width, and intermediate width:

```shell
uv run python benchmarks/benchmark_nvfp4_ffn.py --format convrot-nvfp4 \
  --shape 1024 2048 8192 --shape 1797 2048 8192 \
  --dtype bfloat16 --scaling dynamic --json artifacts/nvfp4-ffn.json
```

Its `prepared_execution` is CUDA-graph device timing of the complete FFN.
Compilation, capture, Python dispatch, and allocation are excluded. Shape,
dtype, and finiteness checks run before timing; numerical accuracy belongs to
the FFN correctness tests. The runner stops if another compute process occupies
the GPU. Match activation, scales, seed, chunking, and sample settings across
revisions. Recorded synthetic static down scales are not checkpoint calibration
evidence.

## Interpret and reproduce results

The common provider model has `prepare()` and `run(prepared)` callables. A
provider may prepare nothing and invoke the complete operator in `run`.
Consequently, phase names alone do not establish equivalent work:

| Phase | Measurement boundary |
|---|---|
| `first_call_ms` | Synchronized wall time of the first invocation, including lazy compilation; compiler caches may already be warm. |
| `preparation` | Warmed synchronized wall time of preparation alone. |
| `prepared_execution` | Warmed device-event time of `run(prepared)`, including preparation inside that callable. |
| `operator_end_to_end` | Warmed synchronized wall time of `run(prepare())`. |

Synchronized wall time includes host dispatch, allocation, and device work.
Device events measure elapsed stream work. Graph replay removes per-call host
work; compare it only with matching graph measurements. Records identify the
`clock`, timing windows, and sample counts. Common summaries are `p50 [p20, p80]`;
unsupported phases are `null`. Device-phase runners separately record
cache-flushed `device_event` and `graph_device_event` samples. Fusion runners
can use shuffled fixed-count wall samples instead of timed windows.

Compare the same shapes, configuration, seed, reference, timing scope, and
environment on baseline and candidate. Confirm a promising result with repeated
fresh-process measurements, alternating their order on an otherwise idle accelerator.
Use complete operator timing for production decisions, and inspect memory as
well as latency. Validate architectures independently; a schedule measured on
one GPU is evidence for that device, not its whole architecture family.

Quality records include absolute/relative error, SQNR, cosine similarity, and
actual/reference non-finite counts where supported. Check the reference and
whether comparisons cover all elements or sampled queries. Synthetic inputs
cannot replace representative model activations. Integer quality comparisons
support values through 32 bits using FP64; full-width INT64/UINT64 require an
exact domain-specific comparison.

Save reports with `--json PATH` or `--jsonl PATH`. Common `BenchmarkRecord`
metadata includes GPU/backend/architecture, Python/Torch/Triton/runtime and
available driver versions, Git revision and dirty state, logical shape, provider
configuration, timings, and quality. Preserve the command, model/shape source,
and clock or power settings when sharing results. Non-finite metrics such as
infinite SQNR serialize as `null`; inspect non-finite counts separately.
Check `schema_version` before consuming artifacts.

Nested configurations use `execution_plan` for complete invocation choices and
`schedule` for an independently measured stage.

## Offline tuning

The [Piper Attention](tune_piper_attention.py),
[SageAttention2++](tune_sage_attention_2pp.py), and
[ConvRot linear](tune_convrot_int8_linear.py) tuners measure the immutable plans
used by production. Omitted axes keep production values; explicit axes form a
deduplicated, capped Cartesian search. Unsupported and out-of-resource
candidates are recorded; unexpected failures propagate. Winners must pass
quality checks. Nothing changes production policy or starts runtime autotuning.
The attention tuners exercise NVIDIA implementations; they do not enable new
backends.

```shell
uv run python benchmarks/tune_piper_attention.py \
  --sequence 8192 --head-dim 128 --block-m 64 128 --num-stages 2 3 \
  --phase operator_end_to_end --json artifacts/attention-tuning.json
```

Use `--help` for target-specific axes and quality thresholds. Attention tuners
default to a 20 dB SQNR gate plus non-finite checks. Piper's prepared phase is its
recurrence; ConvRot and SageAttention2++ include public-operator preprocessing.
State the phase when reporting a selected candidate.

### Attention tuning workload anchors

Use B1/BF16 with H16/H48 and D64/D128 as representative regimes. Measure square
Q=KV at 8192, 32768, and 131072, causal and non-causal. Include non-causal
rectangles 8192×32768 and their reverse. These are equal-head benchmarks;
GQA/MQA need separate coverage. FP16 provides secondary quality/code-generation
coverage where supported.

Use 2048 as a short-context guard, small boundaries such as 63/64/65 and
127/128/129 for masking, and 8193 plus near-square rectangles for realistic tails.
Confirm long behavior at 131072 and 131073 with H16/D64 and D128. Probe immediately
around any proposed applicability boundary. Anchors sample a continuous workload;
they are not model identities or dispatch keys.

### Large-M dense forward-linear tuning workload anchors

For `[M,K] × [N,K]`, use the eight combinations of M=8192/32768,
N=4096/16384, and K=6144/14336, with BF16 and no bias. ConvRot uses group 256
and includes rotation/quantization in the timed operator. Other formats retain
their preparation contract. Confirm the winner at M=131073 for
(N,K)=(16384,6144) and (4096,14336), when memory permits; the expansion also
tests output indexing beyond `2^31` elements.

If integration changes graph boundaries, also check SwiGLU, tanh-GELU, and
shared-input projections at a representative large shape. Small-M decode,
sparse/expert routing, backward, and fused pipelines need their own coverage.
Follow the [development guidance](../docs/development.md) when selecting a
general production policy from these measurements.

## Compiler inspection and profiling

Triton providers can report registers, spills, shared memory, warps, resource
residency ceilings, and available PTX/SASS instruction counts:

```shell
uv run python benchmarks/benchmark_attention.py \
  --sequence 8192 --providers piper_attention --compiler-report --no-sass
```

Residency is a resource ceiling, not achieved occupancy; static instruction
counts are not dynamic execution counts. Compiler comparisons use one
provider/configuration per process so cached specializations cannot be
misattributed. `--compiler-json` and `--compiler-jsonl` save versioned
`triton_compiler` records with environment, configuration, and specialization
fingerprints. NVIDIA SASS inspection requires `nvdisasm`; use `--nvdisasm`
to locate it or `--no-sass` for portable metadata/available IR. AMDGCN
disassembly is not integrated.

```shell
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  uv run python benchmarks/benchmark_integer_pv_dot.py s8-s8 \
  --profile --profile-phase prepared_execution
```

The default profile excludes compilation and warmup. `--profile-include-setup`
adds them in a separate NVTX range. This capture controller is CUDA-only.
When a runner has multiple providers, select the compiler/profile provider
explicitly and keep both selections aligned. Shared implementations live in
[`lib/timing.py`](lib/timing.py), [`lib/reporting.py`](lib/reporting.py),
[`lib/triton_inspection.py`](lib/triton_inspection.py), and
[`lib/profiling.py`](lib/profiling.py).
