# Changelog

All notable changes to Piper Kernels are documented here. Versions follow the policy in
[VERSIONING.md](VERSIONING.md).

## [Unreleased]

### Added

- Piper Attention forward inference for NVIDIA SM8x and consumer Blackwell SM12x with
  Sage-style INT8 QK, per-key signed-INT8 V scales, FP32 probability multipliers,
  native UINT8 probability MMA, exact affine fallback, optional sequence-centered V,
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

[Unreleased]: https://github.com/Boffee/piper-kernels/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Boffee/piper-kernels/releases/tag/v0.1.0
