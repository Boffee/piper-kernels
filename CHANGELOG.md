# Changelog

All notable changes to Piper Kernels are documented here. Versions follow the policy in
[VERSIONING.md](VERSIONING.md).

## [Unreleased]

### Added

- Pure-Triton SageAttention2++ 8+8 forward inference for SM89 and SM12x GPUs, including
  canonical quantization, delayed FP32 buffering, a portable reference, correctness tests,
  `torch.compile` support, and an end-to-end benchmark with an opt-in, revision-pinned
  canonical SageAttention CUDA baseline.

## [0.1.0] - 2026-08-03

### Added

- Initial ConvRot INT8 tensor, reference implementation, Triton backend, and in-place
  low-rank update support.

[Unreleased]: https://github.com/Boffee/piper-kernels/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Boffee/piper-kernels/releases/tag/v0.1.0
