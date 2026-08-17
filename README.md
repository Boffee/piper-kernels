# Piper Kernels

Reusable PyTorch inference operators and optimized kernels for the Piper ecosystem and
other consumers.

Piper Kernels requires Python 3.13 or newer.

The package owns operator semantics, portable PyTorch references, tensor subclasses,
and optimized backends. It deliberately does not know about model repositories,
checkpoint metadata, pipeline frameworks, or device-offloading policy.

## Operators

| Package | Role |
|---|---|
| `piper_kernels` | Public Piper Attention and SageAttention2++ forward operators |
| `piper_kernels.attention` | Attention dispatch, portable references, and optimized backends |
| `piper_kernels.linear` | Linear operators, tensor formats, and optimized backends |
| `piper_kernels.linear.convrot` | ConvRot quantized tensors and linear operators; INT8 today, INT4 planned |

## Triton setup

Install the optimized backends with `piper-kernels[triton]`, or include ConvRot's tensor
format with `piper-kernels[convrot,triton]`. The extra selects Triton 3.7 on each supported
platform: upstream `triton` on Linux and
[`triton-windows`](https://github.com/triton-lang/triton-windows) on 64-bit Windows.

Optimized Windows execution requires Windows 10 or 11, a supported NVIDIA GPU with a
current driver, and the Visual C++ Redistributable for Visual Studio 2015-2022. The
Windows wheel bundles its CUDA toolchain and TinyCC, so a separate CUDA toolkit or Visual
Studio install is not required for Piper's Triton kernels. The base package remains
portable and does not require either Triton distribution.

## ConvRot INT8

Quantize a dense weight, or wrap existing checkpoint storage without dequantizing it,
then use the resulting tensor as a normal linear weight:

```python
import torch

from piper_kernels.linear.convrot import (
    ConvRotInt8Tensor,
    convrot_int8_compile_options,
    convrot_int8_linear,
)

weight = ConvRotInt8Tensor.from_hp(dense_weight, group_size=256)
checkpoint_weight = ConvRotInt8Tensor.from_quantized(
    qdata,
    scale,
    group_size=256,
    logical_dtype=torch.bfloat16,
)
output = torch.nn.functional.linear(activation, weight, bias)

# Let Inductor optimize repeated inputs and absorb supported input activations.
compiled_block = torch.compile(block, options=convrot_int8_compile_options())

# Optionally fuse a raw [up | gate] SwiGLU input with ConvRot preparation.
mlp_output = convrot_int8_linear(up_gate, weight, bias, activation_fn="swiglu")

# GELU with tanh approximation uses the same activation/preparation boundary.
mlp_output = convrot_int8_linear(activation, weight, bias, activation_fn="gelu_tanh")

# The explicit API also supports an ordinary linear.
output = convrot_int8_linear(activation, weight, bias)

# In-place low-rank update with the standard Tensor.addmm_ contract.
weight.addmm_(lora_b, lora_a, alpha=lora_strength)

# Reproducible stochastic terminal-code selection for a quantized LoRA merge.
weight.addmm_(lora_b, lora_a, alpha=lora_strength, rounding_seed=seed)
```

Use `from_quantized(..., logical_dtype=...)` to construct a weight from checkpoint storage.

For a weight with shape `[out_features, in_features]`, ordinary and GELU-tanh inputs have
shape `[..., in_features]`; the SwiGLU input has shape `[..., 2 * in_features]`. The output
always has shape `[..., out_features]`.
`convrot_int8_linear(...)` applies an ordinary linear when `activation_fn` is omitted, matching
`torch.nn.functional.linear`. `activation_fn="gelu_tanh"` applies tanh-approximate GELU, while
`activation_fn="swiglu"` computes `up * silu(gate)` from `[up | gate]`. Portable paths use
PyTorch operations; optimized NVIDIA preparation uses shared Triton activation primitives and
native approximate tanh, so GELU preparation may differ from the portable path by one INT8 code
rather than being bitwise identical. Optimized Triton configurations whose power-of-two
preparation extent is at most 16,384 absorb these activations into input preparation across every
supported ConvRot group size, logical dtype, row count, and accelerator target. This selection is
measured on exact SM120 and optimistic on other targets. Larger extents materialize the activation
and retain the same semantics. Both
`F.linear` with a ConvRot INT8 weight and the explicit INT8 entry point are inference-only and
reject autograd inputs.

For compiled inference, `convrot_int8_compile_options()` installs deterministic post-AOT Inductor
rewrites. An exclusive tanh-approximate GELU or `chunk(2, dim=-1)` `[up | gate]` SwiGLU chain
feeding a ConvRot linear becomes an activated input-preparation node followed by a prepared
linear. This avoids the materialized activated input and lets its source die before the linear
output is allocated.
Separately, two or more ordinary ConvRot linears fed by the same graph value become one explicit
input preparation followed by independent prepared GEMMs at the original operation positions.
Prepared tensors are ordinary graph values—there is no hidden runtime cache—and unmatched,
eager, and training paths remain unchanged. Existing post-grad compiler passes in the supplied
options mapping are preserved. Pass the result through `torch.compile(options=...)`; PyTorch
treats `mode` and `options` as mutually exclusive, so do not also supply `mode`.

`addmm_` computes `weight = beta * weight + alpha * (mat1 @ mat2)` and requantizes
the result. It preserves the ConvRot tensor and quantized storage identities, allowing
offload integrations to keep their existing buffers. Repeated updates are lossy, so
reload a pristine base weight before changing or removing a previously merged adapter.
Passing an unsigned 64-bit `rounding_seed` stochastically selects one of the two adjacent
INT8 codes with probability proportional to distance, without changing the deterministic
row scales or consuming PyTorch's process-global random-number generator. Omitting the seed
retains nearest-integer rounding. This is an inference operation and does not support
autograd. Torch and Triton each replay for a fixed seed, device, and backend; their random
samples are not promised to match each other or different Triton versions byte-for-byte.

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
loop. The online-softmax state, denominator, and PV numerator also remain FP32. The
numerator stays in UINT8 probability-code units during the recurrence, and the common
factor of 255 is removed once in the output epilogue.

For centered V, Piper Attention uses the exact identity

```text
softmax(QK) @ V = softmax(QK) @ (V - mean_sequence(V)) + mean_sequence(V)
```

For non-causal attention, it stores only the compact FP32
`[batch, head, feature]` mean, subtracts it while quantizing V, and restores it in the
attention epilogue. This improves signed-INT8 precision when V has a large feature bias
and preserves constant V exactly. Causal attention leaves V uncentered so per-row INT8
rounding cannot make an earlier output depend on future V rows. Both paths preserve the
original K/V sequence order.

Native mixed-sign MMA is selected on the supported NVIDIA backend through the packaged
stock-Triton extension. The integer-PV benchmark retains the exact affine identity
`u @ v = (u - 128) @ v + 128 * sum(v)` as a signed-INT8 correctness control; unsupported
production targets use the portable quantized reference instead. The public optimized
dispatch supports NVIDIA SM8x and consumer Blackwell SM12x, whose Triton lowering uses
the MMAv2 instruction rewritten by the packaged extension. Exact SM120 uses packed
four-code probability conversion for D64 and non-causal D128, while causal D128 retains
the faster stock conversion. SM89 and exact SM120 have measured schedules; other
supported targets use the generic schedule. Production plan selection depends on target, head
dimension, and causal mode, not sequence length. Hopper lowers the operation through
unsupported WGMMA and therefore uses the slow portable quantized reference. Native ROCm
mixed-sign lowering remains future work.

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
is inference-only and does not support autograd. Its production execution plans are also
sequence-length invariant.

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
uv run ruff format --check .
uv run pyright
uv build
```

GPU tests use the `gpu` pytest marker. The pre-commit test hook hides CUDA so commits run
the portable suite; run `uv run pytest` directly to exercise installed GPU backends.

## Releases

Releases follow the compatibility and release policy in [VERSIONING.md](VERSIONING.md).
Distribution artifacts are built from version tags and published to PyPI by GitHub Actions
using Trusted Publishing; maintainers do not upload releases from local environments.
