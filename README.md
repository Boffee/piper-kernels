# Piper Kernels

Quantized attention, linear, and convolution operators for PyTorch inference.
Piper Kernels combines compact weight formats and GPU kernels with optional
compiler fusions that avoid intermediate tensors and repeated preparation.
Portable PyTorch references define the corresponding numerical operations.

Use the operators directly or install quantized weights in PyTorch modules.
Applications own model loading, checkpoint naming, and device placement;
[Piper Offload](https://github.com/Boffee/piper-offload) uses these weight formats
for its model runtime.

## Installation

Requires Python 3.13 or newer and PyTorch 2.14 or newer.

```bash
pip install piper-kernels
```

The base package requires only PyTorch. Optional extras provide weight-format
and accelerator dependencies:

| Extra | Use |
|---|---|
| `convrot` | ConvRot INT8 weights |
| `nvfp4` | Ordinary and ConvRot NVFP4 weights |
| `triton` | Optimized GPU backends |

For example, install `"piper-kernels[convrot,triton]"` for ConvRot INT8 GPU
execution. On Linux, install the CUDA or ROCm PyTorch build and its matching
Triton first; the extra does not install Linux Triton.

On 64-bit Windows, the `triton` extra selects Triton 3.8 through
[`triton-windows`](https://github.com/triton-lang/triton-windows). Optimized
execution needs Windows 10 or 11, a supported GPU with a current driver, and
the Visual C++ Redistributable for Visual Studio 2015–2022. For NVIDIA,
`triton-windows` bundles the CUDA toolchain and TinyCC; no separate CUDA toolkit
or Visual Studio installation is needed. For AMD, use
[TheRock's ROCm PyTorch build](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
and its matching GPU device packages; see
[AMD validation coverage](docs/development.md#accelerator-environments).

## Quick start

This runs dense Piper Attention on CUDA when available and on CPU otherwise.
Inputs use `[batch, heads, sequence, head_dim]` layout:

```python
import torch
from piper_kernels import piper_attention

device = "cuda" if torch.cuda.is_available() else "cpu"
query, key, value = (
    torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16) for _ in range(3)
)

with torch.inference_mode():
    output = piper_attention(query, key, value, is_causal=False)

print(output.shape)  # torch.Size([1, 2, 128, 64])
```

Supported GPUs with the required backend installed use an optimized path;
other configurations use the slower reference for the same quantized algorithm.
Both approximate floating-point attention, and compiler fusions can change
rounding. Measure numerical quality and full-operator latency on your workload;
see [attention](docs/attention.md) for input and backend support.

## Guides

| Task | Read |
|---|---|
| Choose and use dense, sparse, or SageAttention2++ attention | [Attention](docs/attention.md) |
| Construct, load, run, update, or shard quantized weights; enable compiler fusions | [Weights and quantized operators](docs/weights.md) |
| Understand implementation boundaries, extend kernels, or run checks | [Development](docs/development.md) |
| Measure performance and quality, tune, or inspect generated kernels | [Benchmarks](benchmarks/README.md) |

These docs describe the checked-out revision. Use the matching Git tag for an
installed release. [AGENTS.md](AGENTS.md) gives repository instructions for coding agents;
[VERSIONING.md](VERSIONING.md) covers compatibility and releases, and
[CHANGELOG.md](CHANGELOG.md) records changes between releases.

Licensed under the [Apache License, Version 2.0](LICENSE).
