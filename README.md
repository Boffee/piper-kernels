# Piper Kernels

Reusable PyTorch inference operators and optimized kernels for the Piper ecosystem and
other consumers.

Piper Kernels requires Python 3.14 or newer.

The package owns operator semantics, portable PyTorch references, tensor subclasses,
and optimized backends. It deliberately does not know about model repositories,
checkpoint metadata, pipeline frameworks, or device-offloading policy.

## Operators

| Package | Role |
|---|---|
| `piper_kernels` | Public Piper Attention and SageAttention2++ forward operators |
| `piper_kernels.convrot` | ConvRot quantized tensors and linear operators; INT8 today, INT4 planned |
| `piper_kernels.attention` | Attention dispatch, portable references, and optimized backends |

## ConvRot INT8

Quantize a dense weight, or wrap existing checkpoint storage without dequantizing it,
then use the resulting tensor as a normal linear weight:

```python
import torch

from piper_kernels.convrot import ConvRotInt8Tensor, convrot_linear

weight = ConvRotInt8Tensor.from_hp(dense_weight, group_size=256)
checkpoint_weight = ConvRotInt8Tensor.from_quantized(
    qdata,
    scale,
    group_size=256,
    logical_dtype=torch.bfloat16,
)
output = torch.nn.functional.linear(activation, weight, bias)

# Optionally fuse a raw [up | gate] SwiGLU input with ConvRot preparation.
mlp_output = convrot_linear(up_gate, weight, bias, input_activation="swiglu")

# In-place low-rank update with the standard Tensor.addmm_ contract.
weight.addmm_(lora_b, lora_a, alpha=lora_strength)
```

`from_quantized(..., logical_dtype=...)` is the preferred checkpoint-storage factory.
`from_packed(..., dtype=...)` remains available for compatibility with the 0.1 API.

For a weight with shape `[out_features, in_features]`, the SwiGLU input has shape
`[..., 2 * in_features]` and the output has shape `[..., out_features]`.
`convrot_linear(..., input_activation="swiglu")` computes `up * silu(gate)` before the
linear. Its optimized preparation fusion is selected only for measured SM120
configurations with group size 256. Other supported configurations materialize SwiGLU,
then dispatch through the ordinary ConvRot linear path, which may still use an optimized
backend. Ordinary `torch.nn.functional.linear` calls remain unchanged and do not apply an
activation. Both linear entry points are inference-only and reject autograd inputs.

`addmm_` computes `weight = beta * weight + alpha * (mat1 @ mat2)` and requantizes
the result. It preserves the ConvRot tensor and quantized storage identities, allowing
offload integrations to keep their existing buffers. Repeated updates are lossy, so
reload a pristine base weight before changing or removing a previously merged adapter.
This is an inference operation and does not support autograd.

The operator selects its Triton implementation on supported CUDA devices and otherwise
uses the portable PyTorch reference. Install the tensor format and optimized backend with
`piper-kernels[convrot,triton]`. The base package does not require TorchAO or Triton, and
attention-only consumers do not inherit the TorchAO dependency.

## Piper Attention

Piper Attention is the package's key-scaled integer-PV attention algorithm:

```python
from piper_kernels import piper_attention

output = piper_attention(query, key, value, is_causal=False)
```

It follows FlashAttention's fused online-softmax structure and SageAttention's K
smoothing plus INT8 QK quantization. Its distinct PV path quantizes each V key row
with one signed-INT8 scale, folds those scales into nonnegative probabilities, and
uses `UINT8 x INT8 -> INT32` tensor-core products. The probability multiplier remains
FP32 so every finite FP16 input scale is representable without a conversion in the hot
loop. FP32 also remains the softmax and denominator coordinate; the selected long SM12x
D128 schedule buffers a bounded PV numerator in FP16.

For centered V, Piper Attention uses the exact identity

```text
softmax(QK) @ V = softmax(QK) @ (V - mean_sequence(V)) + mean_sequence(V)
```

It stores only the compact FP32 `[batch, head, feature]` mean, subtracts it while
quantizing V, and restores it in the attention epilogue. This improves signed-INT8
precision when V has a large feature bias and preserves constant V exactly. The
default policy enables centering only for non-causal SM12x D128 calls with both
sequence lengths at least 1024; pass `center_value=True` or `False` to override it.
Centered non-causal SM12x D128 calls with at least 16384 keys additionally use the
selected stable low-to-high centered-row-range order. That permutation is exact
before quantization and reduces scale-coordinate variation in very long attention.

Native mixed-sign MMA is selected on the supported NVIDIA backend through the packaged
stock-Triton extension. The backend retains the exact affine identity
`u @ v = (u - 128) @ v + 128 * sum(v)` as its signed-INT8 correctness and portability
control. The public optimized dispatch supports NVIDIA SM8x and consumer Blackwell
SM12x, whose Triton lowering uses the MMAv2 instruction rewritten by the packaged
extension. SM89 and SM12x have measured schedules; Ampere currently uses the generic
schedule. Hopper lowers the operation through unsupported WGMMA and therefore uses the
slow portable quantized reference. Native ROCm mixed-sign lowering remains future work.

Piper Attention is an independently developed Sage-derived design. The per-key
quantizer, centering identity, and online-softmax lineage are not claimed as novel in
isolation; the name identifies this package's selected combination and fused recurrence.

## SageAttention2++

The package provides an independently written, pure-Triton backend for the
canonical [SageAttention2++](https://github.com/thu-ml/SageAttention) 8+8 algorithm:

```python
from piper_kernels import sage_attention_2pp

output = sage_attention_2pp(query, key, value, is_causal=False)
```

Inputs use `[batch, heads, sequence, head_dim]` layout and may be FP16 or BF16. The
optimized backend requires NVIDIA FP8 tensor cores with FP16 accumulation (SM89 or
newer); measured schedules currently cover consumer SM89 and SM120 GPUs, while other
SM12x targets retain grouped Q/K quantization with generic scheduling. It supports head
dimensions 64 and 128, equal query/KV head counts, arbitrary positive sequence lengths,
rectangular non-causal attention, strided sequence dimensions, and `torch.compile`. It
is inference-only and does not support autograd.

This is SageAttention2++, not a Piper Attention-specific algorithm: K is smoothed, Q/K
are quantized to INT8 with the canonical architecture-specific granularity, V and the
online-softmax probabilities are quantized to E4M3, each 64-key P x V tile accumulates
in FP16, and tile results are buffered in FP32. All optimized device code is Triton;
the package contains no CUDA extension. Unsupported devices use the slow
portable quantized reference.

Install either optimized attention backend with `piper-kernels[triton]`. The official CUDA
SageAttention package is a revision-pinned, optional benchmark dependency only; it is
not imported by production code. See [benchmarks/README.md](benchmarks/README.md) for
the reproducible provider comparison.

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
