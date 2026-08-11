# Changelog

All notable changes to Piper Kernels are documented here. Versions follow the policy in
[VERSIONING.md](VERSIONING.md).

## [Unreleased]

### Added

- Quality-gated offline ConvRot INT8 forward-linear execution-plan tuning across dynamic
  preparation and GEMM schedules, with stratified large-shape output validation.

### Changed

- Centralized ConvRot INT8 preparation and GEMM launch policy in one flat immutable execution
  plan shared by production, benchmark metadata, and offline tuning, with injectable preparation
  and launch boundaries for development measurements. Supported CUDA SwiGLU dispatch now leaves
  fused-versus-split preparation exclusively to that plan. ConvRot benchmarking and tuning now
  share one reproducible workload, sampled-reference, metadata, and provider-adapter harness;
  benchmark timing, offline tuning, and profiling also share one provider-phase launch contract.
  Benchmark support consumers now import helpers from their owning `lib.*` modules instead of a
  flattened package export surface. Offline tuners also share their common CLI controls, search
  axes, quality gate, result rendering, and record construction, while ConvRot tuning consumes
  legal schedule values directly from production policy. Attention benchmarks and tuners now
  share workload dtype and seed metadata, the SDPA quality reference, legal execution-plan values,
  candidate-budget validation, independently reproducible inputs, fixed-layout metadata, and
  smaller provider-construction orchestration boundaries. Piper Attention and SageAttention2++
  also share their input-validation contract and keep backend-independent execution policy outside
  their Triton launch modules. ConvRot CLIs now represent an absent input activation by omitting
  `--input-activation` rather than accepting a string sentinel.

## [0.2.1] - 2026-08-10

### Added

- Reproducible stochastic INT8 terminal-code selection for ConvRot `addmm_` LoRA merges,
  using deterministic row scales and the same unbiased probability formulation as Piper
  Offload.

## [0.2.0] - 2026-08-10

### Added

- Native Windows Triton setup through the platform-selected `triton-windows` package,
  including dependency/import CI and correct Windows benchmark version reporting.
- Piper Attention forward inference for NVIDIA SM8x and consumer Blackwell SM12x with
  Sage-style INT8 QK, per-key signed-INT8 V scales, FP32 probability multipliers,
  native UINT8 probability MMA, exact affine fallback, non-causal sequence-centered V,
  portable reference, and `torch.compile` support.
- Pure-Triton canonical SageAttention2++ 8+8 forward inference on NVIDIA GPUs with FP8
  tensor cores and FP16 accumulation, including a portable quantized reference,
  explicit `sage_attention_2pp` API, `torch.compile` support, and revision-pinned
  canonical CUDA benchmarks.
- Reusable Triton specialization, resource, PTX/SASS, and profiler-capture tooling for
  development benchmarks, with versioned compiler-report JSON and JSONL output.
- Stock-Triton native `UINT8 x INT8 -> INT32` support on NVIDIA SM8x and consumer
  Blackwell SM12x through a packaged, fail-closed `m16n8k32` MMAv2 compiler extension
  with exactness and generated-code validation.
- Reusable offline kernel-configuration tuning with quality gates, recorded candidate
  failures, deterministic winner selection, and executable Piper Attention and
  SageAttention2++ adapters.
- Explicit opt-in `convrot_linear(..., input_activation="swiglu")` with a portable
  `[up | gate]` reference, exact-SM120 fused preparation, and fake/meta plus
  `torch.compile` support.
- Canonical `ConvRotInt8Tensor.from_quantized(..., logical_dtype=...)` construction while
  retaining the `from_packed(..., dtype=...)` compatibility API.

### Changed

- Lowered the minimum supported Python version from 3.14 to 3.13.
- Centralized Piper Attention specialization and launch policy in an immutable execution plan
  shared by production, benchmark metadata, and quality-gated offline tuning. Triton loop
  pipelining, loop invariant code motion, and causal query-block ordering are explicit tuning
  dimensions while production schedules remain unchanged; in particular, SM89 does not inherit
  SageAttention2++ tuning without Piper-specific measurements.
- Selected packed four-code UINT8 probability conversion for measured SM120 Piper D64
  and non-causal D128 paths, while retaining stock conversion for causal D128.
- Raised the minimum supported PyTorch version from 2.12 to 2.13.
- Tuned the pure-Triton SageAttention2++ path on SM89 with native packed
  FP32-to-E4M3 conversion, 128-row two-stage reverse-order long-causal launches,
  and loop-invariant hoisting with a three-stage loop pipeline for long
  non-causal D128. SM120 keeps its stock fused-V and D128-causal probability
  conversion where local paired benchmarks found packed conversion neutral or slower.
- Consolidated the Piper Attention, SageAttention2++, canonical CUDA, and SDPA
  comparisons into the hardware-aware `benchmarks/benchmark_attention.py` development CLI.
- Standardized implementation packages, benchmark provider IDs, and tuner identifiers on
  `piper_attention` and `sage_attention_2pp`, with both public operators exported directly
  from `piper_kernels`.
- Optimized SageAttention2++ recurrence and causal scheduling across supported GPUs,
  with measured SM120 specializations for fused K/V quantization, fused query
  quantization, and long-sequence unscaled-score recurrence.
- Reduced ConvRot H4 rotation work through factorized quartets, added 64-bit-safe tensor
  addressing, and fused H256 rotation with rowwise INT8 quantization on measured SM120
  shapes while retaining the split preparation fallback everywhere else.
- Moved the format-neutral ConvRot functional API out of the INT8 storage module and
  separated semantic custom operators from their optional Triton implementations.

### Fixed

- Kept causal Piper Attention V uncentered so sequence-wide mean statistics and per-row
  INT8 rounding cannot make earlier outputs depend on future V rows.
- Honored positional, keyword, and mixed `torch.nn.functional.linear` calls for ConvRot
  weights, including keyword bias instead of silently dropping it.
- Enforced consistent ConvRot linear shape, device, logical-dtype, storage-layout, and
  inference-only contracts across reference, meta, and optimized execution.
- Made synthetic fake-CUDA dispatch independent of physical device availability and
  revalidated canonical storage after tensor-attribute replacement.
- Made zero-width ConvRot weights consistent across quantization, dequantization, ordinary
  and SwiGLU linear calls, in-place updates, CUDA execution, and `torch.compile`.

## [0.1.0] - 2026-08-03

### Added

- Initial ConvRot INT8 tensor, reference implementation, Triton backend, and in-place
  low-rank update support.

[Unreleased]: https://github.com/Boffee/piper-kernels/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Boffee/piper-kernels/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Boffee/piper-kernels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Boffee/piper-kernels/releases/tag/v0.1.0
