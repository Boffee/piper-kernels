# Changelog

All notable changes to Piper Kernels are documented here. Versions follow the policy in
[VERSIONING.md](VERSIONING.md).

## [Unreleased]

### Added

- Experimental SM120 ConvRot H256 optimization checkpoint with factorized one-pass
  activation rotation/quantization, explicit fused-SwiGLU linear API, large-sequence-safe
  addressing, H3 benchmark presets, phase diagnostics, and a documented clean-room
  integration plan.
- Pure-Triton canonical SageAttention2++ 8+8 forward inference for SM89 and SM12x,
  including a portable quantized reference, explicit `sage_attention_2pp` API,
  `torch.compile` support, and revision-pinned canonical CUDA benchmarks.
- Reusable Triton specialization, resource, PTX/SASS, and profiler-capture tooling for
  development benchmarks, with versioned compiler-report JSON and JSONL output.
- Stock-Triton native `UINT8 x INT8 -> INT32` support on NVIDIA compute capability 8.0 and
  newer when Triton selects the supported `m16n8k32` MMAv2 lowering, through a packaged,
  fail-closed compiler extension with exactness and generated-code validation.

## [0.1.0] - 2026-08-03

### Added

- Initial ConvRot INT8 tensor, reference implementation, Triton backend, and in-place
  low-rank update support.

[Unreleased]: https://github.com/Boffee/piper-kernels/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Boffee/piper-kernels/releases/tag/v0.1.0
