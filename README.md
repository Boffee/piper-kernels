# Piper Kernels

Reusable PyTorch inference operators and optimized kernels for the Piper ecosystem and
other consumers.

Piper Kernels requires Python 3.14 or newer.

The package owns operator semantics, portable PyTorch references, tensor subclasses,
and optimized backends. It deliberately does not know about model repositories,
checkpoint metadata, pipeline frameworks, or device-offloading policy.

## Planned operators

| Package | Role |
|---|---|
| `piper_kernels.convrot` | ConvRot quantized tensors and linear operators; INT8 today, INT4 planned |
| `piper_kernels.attention` | SageAttention2++ 8+8 inference for consumer Ada and Blackwell GPUs |

## ConvRot INT8

Quantize a dense weight, or wrap existing checkpoint storage without dequantizing it,
then use the resulting tensor as a normal linear weight:

```python
import torch

from piper_kernels.convrot import ConvRotInt8Tensor

weight = ConvRotInt8Tensor.from_hp(dense_weight, group_size=64)
checkpoint_weight = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=64)
output = torch.nn.functional.linear(activation, weight, bias)

# In-place low-rank update with the standard Tensor.addmm_ contract.
weight.addmm_(lora_b, lora_a, alpha=lora_strength)
```

`addmm_` computes `weight = beta * weight + alpha * (mat1 @ mat2)` and requantizes
the result. It preserves the ConvRot tensor and packed storage identities, allowing
offload integrations to keep their existing buffers. Repeated updates are lossy, so
reload a pristine base weight before changing or removing a previously merged adapter.
This is an inference operation and does not support autograd.

The operator selects its Triton implementation on supported CUDA devices and otherwise
uses the portable PyTorch reference. Install the tensor format and optimized backend with
`piper-kernels[convrot,triton]`. The base package does not require TorchAO or Triton, so
future attention-only consumers do not inherit quantization-specific dependencies.

## SageAttention2++

The attention package provides a pure-Triton implementation of the canonical 8+8
SageAttention2++ forward path:

```python
from piper_kernels.attention import sage_attention

output = sage_attention(query, key, value, is_causal=False)
```

Inputs use `[batch, heads, sequence, head_dim]` layout and may be FP16 or BF16. The initial
optimized backend targets RTX 40-series SM89 and RTX 50-series SM12x GPUs, supports head
dimensions 64 and 128, equal query/KV head counts, causal or non-causal attention, and does
not support autograd.

The implementation follows SageAttention2++ rather than merely storing FP8 tensors: it
smooths K, applies architecture-tuned INT8 Q/K quantization, quantizes V per channel to E4M3,
quantizes online-softmax probabilities to E4M3, accumulates each 64-key PV tile in FP16, and
buffers the tile result in FP32. Quantized V is stored feature-major to avoid an expensive
RHS layout conversion; the SM120 D128 non-causal schedule additionally uses Triton tensor
descriptors. All device kernels are written in Triton; no CUDA extension or inline PTX is used.
Unsupported devices use a slow portable quantized reference.

Install the optimized backend with `piper-kernels[triton]`.
See [benchmarks/README.md](benchmarks/README.md) for the revision-pinned official CUDA
baseline and reproducible three-way comparison against Piper and PyTorch SDPA.

## Dependency direction

Applications such as Piper consume this package. Integrations such as torch-offload may
optionally recognize its tensor types, but `piper-kernels` does not depend on either
project.

## Development

```shell
uv sync --dev
uv run pytest
uv run ruff check .
uv run pyright
uv build
```

GPU tests use the `gpu` pytest marker. The pre-commit test hook hides CUDA so commits run
the portable suite; run `uv run pytest` directly to exercise installed GPU backends.

## Releases

Releases follow the compatibility and release policy in [VERSIONING.md](VERSIONING.md).
Distribution artifacts are built from version tags and published to PyPI by GitHub Actions
using Trusted Publishing; maintainers do not upload releases from local environments.
