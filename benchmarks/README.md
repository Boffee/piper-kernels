# Benchmarks

Use the [shared suite](benchmark.py) to compare operators across accelerators or
revisions. Cases have fixed shapes, input recipes, and mathematical options.
Hardware chooses the implementation, never a smaller workload. Synthetic
operator measurements do not establish full-model speed or checkpoint quality.

Run from the repository root using the environment for the accelerator under
test; see [CUDA and ROCm setup](../docs/development.md#accelerator-environments).
For an external environment, replace `uv run python` with
`PYTHONPATH=src /path/to/env/bin/python`.

## Run the suite

```shell
# Inspect cases without initializing an accelerator.
uv run python benchmarks/benchmark.py --list

# Exercise every family with small diagnostic workloads.
uv run python benchmarks/benchmark.py --case '*small*' --json artifacts/small.json

# Compare fixed production-shaped attention workloads.
uv run python benchmarks/benchmark.py --family attention --json artifacts/attention.json

# Run the entire standard suite.
uv run python benchmarks/benchmark.py --jsonl artifacts/operators.jsonl
```

`--case` accepts an identity or a quoted shell-style pattern. Repeat `--case`,
`--family`, or `--provider` to select a subset. The default suite includes both
endpoints of each production workload. Small and ragged diagnostics require
explicit selection; they are not substitutes for production measurements.
Use `--help` for controls. CUDA and ROCm both use PyTorch's `cuda` device name;
`--device cpu` is available for explicit reference checks.

## Workloads

The [typed catalog](lib/cases.py) is the authoritative case definition. Shapes
are rounded latent-space anchors informed by video, image, and decoder models:

| Workload | Approximate output range | Transformer tokens, low → high |
|---|---|---:|
| Video | 0.5 MP / 5 seconds → 1 MP / 15 seconds, at 24 FPS | 20,480 → 110,592 |
| Image | 768×768 → 2048×2048 | 2,816 → 16,896 |
| Decoder chunk | Chunk-sized work at the video spatial endpoints | 10,240 → 20,480 |

Video anchors use H3-like latent grids (32 output pixels per transformer token
axis and chunked temporal mapping), rounded with conditioning overhead. Image
anchors use a 16-pixel latent grid plus 512 conditioning tokens, informed by
Krea2/Qwen Image-like workloads. These are approximate workloads, not model
executions or exact prompt lengths. Longer videos increase decoder chunk count,
not the size of every decoder invocation. No model checkout is required to run.

Dense and sparse attention, linear, FFN, Conv3D, and projection/attention
pipelines use the matching catalog dimensions. Sparse keep ratios, GQA head
counts, GELU versus SwiGLU, and convolution normalization are explicit cases.
An optimization must execute the same operation as its comparison baseline.

Input tensors are generated on the CPU with a fixed recipe and transferred to
the selected device, so a seed does not depend on vendor RNG behavior. Weight
formats start from the same dense weights. Quantization formats are explicit
provider variants, accompanied by quality measurements.

Changing a case's meaning requires a catalog revision. Custom shape experiments
belong in diagnostics or tuning; do not change a standard case for one device.

## Measurements and outcomes

The primary measurement is the **complete warmed operator or pipeline** using
synchronized wall time. It includes host dispatch, output allocation, and all
required per-call activation preparation. Input creation, weight packing,
compilation, and correctness checks are outside timing. The common protocol
uses warm caches, a 100 ms warmup and a 500 ms measurement window; overrides are
recorded. Reports contain the median, p20, p80, and sample count.

Each requested implementation produces an outcome:

- `ok`: full-output finiteness and numerical checks passed; latency is reported.
- `unsupported`: the implementation cannot execute this case on this device.
- `oom`: setup, validation, or measurement exceeded available memory; the stage
  is recorded and the workload is unchanged.
- `failed`: an unexpected error or failed correctness check. The command exits
  unsuccessfully; this is not a skipped measurement.

Native support follows the library. For example, NVFP4 requires NVIDIA SM120,
and SageAttention2++ has no native AMD implementation. Unsupported providers
remain visible instead of being replaced by portable execution under their name.

Quality checks cover complete outputs for nonfinite values and deterministic
samples for numerical error. Attention samples retain the full key/value
context. Records identify the reference, coverage, tolerance, and any additional
comparison. Error against original floating weights includes quantization loss;
agreement with a matching quantized reference measures implementation correctness.
Pipeline comparisons verify the advertised compiler rewrites and compare the
fused operation with the same materialized operation.

Peak extra device allocation includes outputs and workspace, excluding resident
inputs and weights; compilation and validation allocations are excluded. It is
PyTorch allocator memory, not total process or device memory.

Results use the `operator_suite` record type and carry schema/catalog versions,
case identity, full workload, provider configuration, measurement protocol,
quality, outcome, and hardware/software/Git metadata. `--json` and `--jsonl`
save progress after every implementation. Nonfinite scalar metrics such as
infinite SQNR serialize as `null`; inspect the associated error/count fields.

Compare matching case revisions, providers, scopes, and protocols. Confirm a
promising result with repeated fresh-process runs on an otherwise idle device.
Record clock/power settings and validate each architecture independently.

## Diagnostics and offline tuning

These tools answer narrower questions and retain their labeled phase/clock
records. Use complete-call suite results for integration comparisons.

| Question | Tool |
|---|---|
| Attention preparation, compiler inspection, optional canonical Sage | [Dense attention diagnostic](benchmark_attention.py) |
| Sparse stages and routing scores | [Sparse attention](benchmark_sparse_piper.py), [scores](benchmark_sparse_piper_scores.py) |
| Q/K/V projection phases | [Sparse projections](benchmark_sparse_piper_projection.py) |
| Linear preparation, compiler inspection, optional external comparison | [Linear](benchmark_convrot_int8.py), [preparation](benchmark_convrot_int8_preparation.py) |
| Convolution layout and schedule investigation | [Conv3D](benchmark_convrot_int8_conv3d.py) |
| Integer arithmetic/compiler behavior | [Integer PV dot](benchmark_integer_pv_dot.py) |

Operation diagnostics accept `--case` to use a parent workload from the shared
catalog. Use full option names; `--case` rejects workload overrides. Saved records
store `case_id` and `catalog_version` at the top level, with workload dimensions
in `shape`. Custom experiments have null catalog metadata.
Device-event timing measures stream work; graph replay removes per-call host
work. Neither should be compared directly with the suite's synchronized wall
time. Compiler resource counts describe limits, not achieved occupancy.

The [Piper Attention](tune_piper_attention.py),
[SageAttention2++](tune_sage_attention_2pp.py), and
[INT8 linear](tune_convrot_int8_linear.py) tuners accept the same cases and search
implementation-specific schedules offline:

```shell
uv run python benchmarks/tune_piper_attention.py --case attention-video-low \
  --block-m 64 128 --num-stages 2 3 --phase operator_end_to_end \
  --json artifacts/attention-tuning.json
```

Candidate limits and quality gates keep searches explicit; unexpected failures
propagate. Nothing modifies production policy or enables runtime autotuning.
Attention schedule searches currently target NVIDIA. Use nearby shapes when
judging a policy change; see [performance tuning](../docs/development.md#performance-tuning).

The [ROCm workflow](../.github/workflows/rocm.yml) runs the same small cases and
saves their records on a provisioned RDNA4 runner. It requires the environment
described in the development guide; the workflow alone is not hardware evidence.

Add a workload to the catalog and its operation adapter, reusing the common
input, provider, quality, and reporting utilities in `lib/`. Suite operations
measure complete calls; diagnostic providers expose preparation and execution
phases. Both use the same deferred implementation factories and result metadata.
Avoid another standalone performance runner for a model or accelerator.
