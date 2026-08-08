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
  failures, deterministic winner selection, and an executable Piper Attention example.

### Changed

- Consolidated the Piper, SageAttention2++, canonical CUDA, and SDPA comparisons into
  the hardware-aware `benchmarks/benchmark_attention.py` development CLI.
- Optimized SageAttention2++ recurrence and causal scheduling across supported GPUs,
  with measured SM12x specializations for fused K/V dispatch, inline query
  quantization, and long-sequence raw-score recurrence.

## [0.1.0] - 2026-08-03

### Added

- Initial ConvRot INT8 tensor, reference implementation, Triton backend, and in-place
  low-rank update support.

[Unreleased]: https://github.com/Boffee/piper-kernels/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Boffee/piper-kernels/releases/tag/v0.1.0
