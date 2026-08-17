# Changelog

All notable changes to Piper Kernels are documented here. Versions follow the policy in
[VERSIONING.md](VERSIONING.md).

## [Unreleased]

### Added

- Quality-gated offline ConvRot INT8 forward-linear execution-plan tuning across dynamic
  preparation and GEMM schedules, with stratified large-shape output validation.
- `convrot_compile_options` for automatically sharing input preparation and folding packed
  `[up | gate]` SwiGLU into activated input preparation plus a prepared ConvRot linear in compiled
  inference graphs. This lets the packed input die before output allocation. The inference-only
  post-AOT rewrites preserve operation order and use explicit graph values and operators instead
  of an identity-based runtime cache.

### Changed

- The explicit ConvRot linear now defaults to an ordinary linear and accepts
  `activation_fn="swiglu"` for packed SwiGLU input preparation.
- ConvRot tensors and linear operators now live under `piper_kernels.linear.convrot`, mirroring
  the `piper_kernels.attention` package hierarchy. The former `piper_kernels.convrot` import path
  and `ConvRotInt8Tensor.from_packed` compatibility factory have been removed.
- Large-M dense forward-linear tuning guidance now standardizes a BF16, bias-free Cartesian
  matrix over `M=8K/32K`, `N=4K/16K`, and non-power-of-two `K=6144/14336` as model-neutral
  measurement anchors rather than dispatch keys. ConvRot INT8 applies group size 256 to this matrix.
  Its benchmark accepts Cartesian M/N/K axes, and its benchmark and tuner defaults now start from
  the lower-width anchor instead of power-of-two toy dimensions. `M=131073` is reserved for final
  long, ragged, and 64-bit-indexing validation.
- Exact SM120 ConvRot INT8 linears use one broadly selected large-M GEMM schedule with
  `128x256x128` tiles, eight warps, fixed `GROUP_M=16` launch ordering, and separate full-M and
  ragged-M-tail launches. Across the full BF16 `M=8K/32K`, `N=4K/16K`, `K=6144/14336` matrix,
  this delivered a 2.20-5.56x complete-operator speedup over the former generic schedule without
  a quality change; both `M=131073` expansion and contraction guards pass the same quality checks.
- SM120 Piper causal attention now traverses its mask-free prefix separately from the masked
  diagonal boundary and launches query blocks in reverse order. D128 uniformly uses split PV,
  with two FP32 accumulators for causal and non-causal attention at every sequence length. The
  scaled-FP16 outer recurrence and its 128K precision boundary have been removed after real
  MiniMax-H3 activations showed strongly layer-dependent quality loss. Non-causal D128 keeps
  uniform M128 tensor-descriptor tiling. All non-causal SM120 shapes derive the V-log bound, so the
  policy has no short-context or square/rectangular metadata boundary. The shared FP32 numerator
  recurrence keeps numerators in UINT8 probability-code units for D64 and D128 in both causal and
  non-causal attention,
  deferring the common normalization to the output epilogue. Aligned rectangular key tiles also
  use mask-free traversal without requiring square attention.
  Production attention plans now start from 128-row query tiles and use explicit target policy for
  the few smaller-tile paths instead of a runtime CTA-count heuristic. Attention tuning guidance
  standardizes H16/H48 and 8K/32K/128K as measurement anchors rather than exact dispatch shapes.
  Piper's never-selected fused Q/K/V preparation candidate has been removed, and its coupled causal
  prefix partition and reverse launch order are now represented by one traversal choice. Causal and
  non-causal key tiles now share one online-softmax recurrence implementation without changing
  generated kernel resources or instructions. Exact SM89 and SM120 execution plans are now
  constructed directly instead of first building and then mostly overwriting a generic plan. The
  measured SM89 non-causal D128 schedule is applied uniformly instead of retaining an 8K policy
  threshold. The ragged-tail launch also derives its block offset at runtime rather than compiling
  a distinct specialization for every query length. The unreachable unsplit-FP16 recurrence,
  scaled-FP16 recurrence, and benchmark-only affine Triton
  attention mode have been removed; unsupported accelerators continue to use the portable
  reference implementation.
- SM120 SageAttention2++ now uses uniform 128-row query tiles, matching the pinned canonical CUDA
  implementation at every sequence length. Grouped-scale tiles now make raw-score reduction
  intrinsic across D64 and D128, eliminating causal and non-causal sequence-length crossovers.
  D64 uses separate query quantization while D128 retains fused Q; K and V use one uniform pair of
  standalone quantizers instead of a neutral role-dispatch specialization. Stock probability
  conversion is uniform across SM120 shapes, and the unused M32 execution-plan specialization has
  been removed; attention tuning uses 2K only as a short-context sanity guard below the
  8K/32K/128K performance anchors. The measured SM89 D128 schedules are likewise applied uniformly
  rather than selected by an 8K query-length threshold.
- Shared Sage-style Q/K reference and Triton preparation now have one implementation across
  Piper Attention and SageAttention2++. Benchmark tooling now verifies the pinned canonical
  SageAttention installation before recording provenance, participates in the standard type and
  formatting checks, and shares common attention-tuner CLI and result-reporting paths.
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
  quantization, and long-sequence raw-score reduction before scaling.
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
