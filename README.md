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
| `piper_kernels` | Public dense and sparse Piper Attention plus SageAttention2++ operators |
| `piper_kernels.attention` | Attention dispatch, portable references, and optimized backends |
| `piper_kernels.linear` | Linear operators, tensor formats, and optimized backends |
| `piper_kernels.linear.convrot` | ConvRot INT8 and NVFP4 tensors, linear operators, and compiler integrations |

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
rather than being bitwise identical. Optimized Triton preparation uses up to three equal
power-of-two chunks of at most 16,384 columns, fusing rows through 49,152 columns across every
supported ConvRot group size, logical dtype, row count, and accelerator target. This selection is
measured on exact SM120 and optimistic on other targets. Larger rows materialize the activation
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

The cross-operator ConvRot-to-sparse-Piper optimization is enabled explicitly by importing
`convrot_sparse_piper_compile_options` from
`piper_kernels.fusions.convrot_sparse_piper`. It installs the fusion pass before the ordinary
ConvRot pass. On exact SM120, it recognizes a compatible H3-style region containing three
bias-free ConvRot Q/K/V projections, D128 RMSNorm and split-half RoPE for Q/K, followed by
`sparse_piper_attention`. The rewrite shares input preparation and emits quantized Q/K/V
plus routing summaries directly, avoiding the three materialized BF16 projection outputs. Arbitrary
logical sequence lengths are written directly into internally K64-padded attention storage; only
the final projection tile is masked, and the result retains the exact logical length. It fails closed
for unsupported shapes, layouts, or parameters; the ordinary ConvRot and sparse-attention APIs
remain independent.

Because no projected activation is externally observable in the fused region, projection,
RMSNorm, and RoPE stay in FP32 until the final INT8 Q/K/V encoding. This removes otherwise
redundant FP32-to-BF16-to-FP32 round trips without materializing FP32 activation tensors.

The internal `piper_kernels.fusions.projected_qk` layer owns projection-independent RMSNorm and
RoPE. The existing Sage Q/K quantization layer owns signed-Hadamard grouped Q/K encoding shared
with dense Piper, while `piper_kernels.attention.kernels.sparse_piper` owns only sparse Piper's
tile-scaled V encoding. `piper_kernels.fusions.convrot_sage_qk` adapts ConvRot projection tiles
to those boundaries and owns ConvRot validation; the explicit sparse fusion adds routing summaries,
storage, and graph rewriting. Another projection backend can therefore compose the same pieces
without depending on ConvRot internals or adding a backend protocol to attention.

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

## NVFP4 construction

Piper's ordinary and ConvRot NVFP4 wrappers can quantize a floating-point weight without
exposing TorchAO storage construction to the caller:

```python
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

activation_quantization = QuantizeTensorToNVFP4Kwargs(
    block_size=16,
    is_swizzled_scales=True,
    use_triton_kernel=False,
    use_dynamic_per_tensor_scale=True,
)

weight = PiperNVFP4Tensor.from_hp(
    dense_weight,
    compute_per_tensor_scale=True,
    is_swizzled_scales=True,
    act_quant_kwargs=activation_quantization,
)
rotated_weight = ConvRotNVFP4Tensor.from_hp(
    dense_weight,
    group_size=64,
    compute_per_tensor_scale=True,
    is_swizzled_scales=True,
    act_quant_kwargs=activation_quantization,
)
```

For ConvRot, the global NVFP4 scale is derived after rotation. This keeps rotation and
quantization in one package-owned operation and prevents callers from accidentally scaling the
logical basis instead of the stored basis. `SUPPORTED_GROUP_SIZES` is exported from
`piper_kernels.linear.convrot` for format-policy validation.

## Piper Attention

Piper Attention is the package's key-scaled integer-PV attention algorithm:

```python
from piper_kernels import piper_attention

output = piper_attention(query, key, value, is_causal=False)
```

It follows FlashAttention's fused online-softmax structure and SageAttention's K
smoothing plus INT8 QK quantization. Before quantization, it applies the same fixed
signed, normalized Hadamard transform across each Q head and centered K head. This
orthogonal change of basis preserves their exact dot products while smoothing outliers
for the subsequent integer quantizers. Its distinct PV path quantizes each V key row
with one signed-INT8 scale, folds those scales into nonnegative probabilities, and uses
`UINT8 x INT8 -> INT32` tensor-core products. The probability multiplier remains FP32
so every finite FP16 input scale is representable without a conversion in the hot loop.
The online-softmax state, denominator, and PV numerator also remain FP32. The numerator
stays in UINT8 probability-code units during the recurrence, and the common factor of
255 is removed once in the output epilogue.

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

## Sparse Piper Attention

Sparse Piper is a separate non-causal SM120 operator for pre-tiled H3-style self-attention:

```python
from piper_kernels import SparsePiperAttention

attention = SparsePiperAttention(
    (0.2, 0.4, 0.6),
    routing="mean",
)
output = attention(
    query,
    key,
    value,
    sparse_key_blocks=1036,
    sparse_query_blocks=1024,  # optional leading routed-query K64 blocks
    block_lengths=block_lengths,  # optional valid-front padded K64 storage
)
```

Inputs use pre-tiled `[batch, sequence, heads, 128]` BF16 layout. Without `block_lengths`, every row
participates in attention and the sequence length may be arbitrary; the operator pads only its
internal quantized storage to K64. Supplying one contiguous device INT32 length in `[1, 64]` per
physical K64 block instead selects valid-front padded storage. The output retains that physical
layout so the caller can apply its existing gather; padded query rows are unspecified.
`sparse_key_blocks` is a runtime count of complete routeable physical K64 prefix tiles, so any
compact partial final tile belongs to the dense suffix. Routing defaults to FP32 min/max pooling;
passing `routing="mean"` instead scores FP32 Q64/K64 mean summaries. Both policies select the
same per-head block budget over the sparse prefix, after which every query attends to every
remaining K/V row in the same softmax. By default every query block uses that policy. Supplying
`sparse_query_blocks` makes only that many leading K64 query blocks routed; later query blocks
attend every K/V block densely. This supports packed video-first layouts followed by dense
non-video queries using one runtime scalar rather than a per-block mask. Engine owns only the
semantic per-layer ratio profile.
Each opaque attention call derives its temporary physical keep counts, packed offsets, and exact
route storage from that immutable model configuration and the current prefix length. Dynamic
compiled graphs accept changed prefix lengths and their resulting route capacities without compiling
another graph or SM120 attention kernel. Routes remain call-local because both policies depend on
the current Q/K values. Compatible ConvRot INT8, NVFP4, and ConvRot NVFP4 compiler rewrites preserve
the selected policy while producing its summaries directly from fused projections.

Sparse Piper also exposes a policy-independent coarse-attention residual:

```python
from piper_kernels import mean_pool_coarse_residual

output = mean_pool_coarse_residual(
    fine_output,
    query,
    key,
    value,
    compression_gate,
    sparse_key_blocks=sparse_key_blocks,
    coarse_key_blocks=total_key_blocks,
    coarse_scale=coarse_scale,
    block_lengths=block_lengths,
)
```

This convenience path mean-pools Q and K/V blocks, applies dense coarse attention, expands each
result over its physical K64 query block, multiplies the caller-provided gate directly without an
implicit activation, and adds it to fine attention. `coarse_key_blocks` may include blocks after the
sparse-routing prefix, including a partial compact tail; omitting it preserves the original
`sparse_key_blocks` scope. `block_lengths` is optional for compact storage and selects valid-front
internally padded storage when supplied. The `minmax_pool_coarse_residual` convenience API
derives extrema-based scores under the same layout contract, while
`coarse_attention_residual` remains available for learned or already-materialized block scores.
These composable implementations are the correctness and training contract; compatible compiled
ConvRot INT8, NVFP4, and ConvRot NVFP4 graphs fuse the shared route scores, wider coarse attention,
and gated residual, including valid-front padded storage.
When a compatible static ConvRot INT8, NVFP4, or ConvRot NVFP4 projection immediately consumes the
quantized attention result, the bounded output rewrite also supports `block_lengths` and the coarse
residual together with `sparse_query_blocks`. It passes the coarse result and compression gate into
each ranged attention launch and projects that chunk directly, so the full BF16 attention output is
not materialized.

The SM120 path writes packed UINT16 routes, pairs two logical K64 tiles in one physical K128
recurrence, and uses one centered-V INT8 scale per logical tile. Its online numerator and
pre-rounding denominator remain FP32. Unsupported devices use a slow portable implementation of
the same quantized Sparse Piper arithmetic. A separate exact-BF16 sparse reference serves as its
quality oracle; it is not the public fallback.

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

This is SageAttention2++, not a Piper Attention-specific algorithm: K is smoothed, the
same fixed signed, normalized Hadamard transform is applied to Q and centered K, Q/K are
quantized to INT8 with the canonical architecture-specific granularity, V and the
online-softmax probabilities are quantized to E4M3, each 64-key P x V tile accumulates
in FP16, and tile results are buffered in FP32. All optimized device code is Triton; the
package contains no CUDA extension. Unsupported devices use the slow portable quantized
reference.

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
