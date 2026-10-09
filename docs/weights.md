# Quantized weights

Piper weight tensors hold packed values and quantization metadata while exposing
a logical floating-point shape and dtype. Construct them offline from dense
weights, or wrap checkpoint storage, then use them with `F.linear` or an inference
module. The application owns checkpoint loading, calibration, and model quality.

| Format | Stored representation | Execution |
|---|---|---|
| `ConvRotInt8Tensor` | Grouped rotation, INT8 values, FP32 scale per output | Linear and causal Conv3D, with optimized NVIDIA/AMD kernels and portable fallback. |
| `PiperNVFP4Tensor` | Packed FP4 values, FP8 block scales, optional FP32 global scale | Piper's native linear path targets exact NVIDIA SM120. |
| `ConvRotNVFP4Tensor` | NVFP4 in a grouped rotated basis | Same native NVFP4 target. |

See [installation](../README.md#installation) for format extras and platform setup.
Weight construction and storage are separate from execution support: being able
to load a format does not imply a fast kernel on that device.

## Run a ConvRot INT8 linear

```python
import torch
import torch.nn.functional as F
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
dense_weight = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
x = torch.randn(2, 256, device=device, dtype=torch.bfloat16)
int8_weight = ConvRotInt8Tensor.from_hp(dense_weight, group_size=64)

with torch.inference_mode():
    output = F.linear(x, int8_weight)
assert output.shape == (2, 128)

# Install the same storage in a module without initializing another dense weight.
layer = torch.nn.Linear(256, 128, bias=False, device="meta")
layer.weight = torch.nn.Parameter(int8_weight, requires_grad=False)
```

Weights have shape `[out_features, in_features]`; inputs are
`[..., in_features]`. FP16, BF16, and FP32 logical dtypes are supported. Input and
weight must share device and logical dtype, except for supported autocast at the
linear boundary. An optional bias has one floating-point value per output feature
on the same device. Execution is inference-only.

ConvRot group sizes are 16, 64, or 256, and must divide the input width. The exported
`piper_kernels.weights.convrot.SUPPORTED_GROUP_SIZES` contains these choices.
Quantization rotates and allocates new storage; `dequantize(output_dtype=...)`
returns the approximate logical weight in its original basis.

To wrap existing **rotated** INT8 storage, use `from_quantized`. Unlike `from_hp`,
it does not quantize the supplied values:

```python
checkpoint_weight = ConvRotInt8Tensor.from_quantized(
    int8_weight.qdata,
    int8_weight.scale,
    group_size=int8_weight.group_size,
    logical_dtype=int8_weight.dtype,
    act_per_tensor_scale=int8_weight.act_per_tensor_scale,
)
```

`qdata` is contiguous INT8 `[out, in]`; `scale` is FP32 `[out, 1]` or `[out]`.
Contiguous storage, including mmap storage, is reused. Noncontiguous inputs are
canonicalized, which can copy. Store the group size, logical dtype, and optional
activation scale with the packed tensors; they are part of the format.

Omitting `act_per_tensor_scale` uses dynamic per-row activation scaling. A static
scale must be a finite positive FP32 scalar tensor on the weight device, calibrated
**after** any activation and rotation for the chosen group size. It moves and
serializes with the weight. Conversion/dequantization alone do not need an
activation scale. Numerical values are caller preconditions, not runtime scans;
see the [validation contract](development.md#validation-contract).

For explicit activated linears, import `convrot_int8_linear` from
`piper_kernels.linear.convrot`. Its `activation_fn="gelu_tanh"` applies
tanh-approximate GELU; `"swiglu"` consumes `[..., 2 * in_features]` ordered as
`[up | gate]` and computes `up * silu(gate)`. Omitting the argument is ordinary
linear. Fused preparation uses FP32 arithmetic and may choose neighboring INT8
codes compared with separately materialized low-precision activations.

## Construct and run NVFP4

This complete example needs an NVIDIA SM120 GPU and the NVFP4 dependencies.
The native path requires 16-value blocks, swizzled scales, and the activation
configuration shown here. It accepts FP16, BF16, or FP32 activations matching the
logical weight dtype. ConvRot input widths must also divide into rotation groups.

```python
import torch
import torch.nn.functional as F
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor

activation_quantization = QuantizeTensorToNVFP4Kwargs(
    block_size=16,
    is_swizzled_scales=True,
    use_triton_kernel=False,
    use_dynamic_per_tensor_scale=True,
)
dense_weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
options = dict(
    compute_per_tensor_scale=True,
    is_swizzled_scales=True,
    act_quant_kwargs=activation_quantization,
)
nvfp4_weight = PiperNVFP4Tensor.from_hp(dense_weight, **options)
rotated_weight = ConvRotNVFP4Tensor.from_hp(dense_weight, group_size=64, **options)
with torch.inference_mode():
    output = F.linear(x, nvfp4_weight)
    rotated_output = F.linear(x, rotated_weight)
assert output.shape == rotated_output.shape == (2, 128)
```

`compute_per_tensor_scale=True` derives the weight's global scale; for ConvRot it
does so after rotation. Do not also supply `per_tensor_scale`. Dynamic activation
scaling derives a global input scale. For calibrated static activation scaling,
set `use_dynamic_per_tensor_scale=False` and supply `act_per_tensor_scale` to the
weight constructor: a finite positive FP32 scalar on the weight device, calibrated
in the stored basis.

`PiperNVFP4Tensor.from_torchao(existing)` wraps existing TorchAO NVFP4 storage
without copying. `ConvRotNVFP4Tensor.from_torchao(existing, group_size=...)` only
attaches rotation metadata: the stored values must already be in that basis.
For packed checkpoint constructors and all stored fields, see the
[ordinary NVFP4](../src/piper_kernels/weights/nvfp4/tensor.py) and
[ConvRot NVFP4](../src/piper_kernels/weights/convrot/nvfp4/tensor.py) classes.
Piper's [NVFP4 reference](../src/piper_kernels/linear/nvfp4/reference.py) supports
independent correctness comparisons. ConvRot NVFP4 linear requires the supported
native path and raises when its execution requirements are not met.

## Run a causal Conv3D

ConvRot INT8 also represents logical `[out, in, 3, 3, 3]` weights. Execution requires
a static input scale and FP16 or FP32 `[batch, channels, time, height, width]`
activations; output is FP16. Input channels must be a power of two from 64 to 4096.

```python
import torch
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.conv3d.convrot.int8 import ConvRotInt8Conv3d

device = "cuda" if torch.cuda.is_available() else "cpu"
kernel = torch.randn(16, 64, 3, 3, 3, device=device, dtype=torch.float16)
# Illustrative scale; use a calibrated value for a real checkpoint.
input_scale = torch.tensor(0.05, device=device, dtype=torch.float32)
conv_weight = ConvRotInt8Tensor.from_hp(kernel, group_size=64, act_per_tensor_scale=input_scale)
conv = ConvRotInt8Conv3d(conv_weight, stride=(1, 1, 1), padding="reflect")
video = torch.randn(1, 64, 2, 4, 4, device=device, dtype=torch.float16)
with torch.inference_mode():
    output = conv(video)
assert output.shape == (1, 16, 2, 4, 4)
```

Temporal padding adds two zero frames before the input. Spatial padding comes
from the model architecture; do not add it twice when replacing an existing
layer. See the [Conv3D guide](../src/piper_kernels/conv3d/convrot/int8/README.md)
for padding modes, packed checkpoint layout, offload integration, and H3 compiler
integration. Matrix updates, GGUF conversion, transpose, and sharding below apply
to 2-D weights only.

## Compile a model

The operators support ordinary `torch.compile`. Optional compiler helpers enable
preparation sharing and fusion of compatible operations. For the INT8 `layer` above:

```python
from piper_kernels.linear.convrot import convrot_int8_compile_options

compiled_layer = torch.compile(layer, options=convrot_int8_compile_options())
```

Use `nvfp4_compile_options` from `piper_kernels.linear.nvfp4` for ordinary NVFP4,
or `convrot_nvfp4_compile_options` from `piper_kernels.linear.convrot.nvfp4` for
ConvRot NVFP4. These passes share compatible input preparation across linears.
The INT8 helper also absorbs supported exclusive GELU inputs. Static preparation
can be shared only when projections use the same scale graph value.

For FFNs, use the matching package under `piper_kernels.fusions`:
`{convrot_int8,nvfp4,convrot_nvfp4}_{swiglu,gelu}_ffn`. Each exports
`<package_name>_compile_options()`, installs FFN fusion before ordinary linear
rewriting, and preserves each projection's static/dynamic scale. SwiGLU matches
separate gate/value/down projections; GELU matches an exclusive
`down(gelu(up(x), approximate="tanh"))`. FFN fusions support FP16/BF16 activations.
See [attention](attention.md#compile-and-fuse-projections) for attention-region
helpers, and the [specializations](../src/piper_kernels/specializations) for
model-specific compile options.

Pass the result through `torch.compile(options=...)`; do not also pass `mode`.
Helpers preserve existing options, and unmatched patterns retain their original
operations. Static INT8 weights bypass AOTAutograd's persistent disk cache because
that cache cannot distinguish shared from independent scale tensors. Compilation,
Inductor caching, and reuse of the compiled graph still work. There is no hidden
prepared-weight cache.

## Merge an adapter

All three matrix wrappers support logical in-place updates. Continuing with the
INT8 weight:

```python
lora_a = torch.randn(4, int8_weight.shape[1], device=int8_weight.device, dtype=int8_weight.dtype)
lora_b = torch.randn(int8_weight.shape[0], 4, device=int8_weight.device, dtype=int8_weight.dtype)
with torch.no_grad():
    int8_weight.addmm_(lora_b, lora_a, alpha=0.1, rounding_seed=42)
```

`addmm_` computes `beta * weight + alpha * (mat1 @ mat2)`; `add_` computes
`weight + alpha * update` from an exact-shape dense update. Inputs must match the
weight's logical dtype and device. Updates are inference-only, require an
untransposed matrix, and requantize into the existing storage. They preserve the
wrapper, packed-data/scales storage, and activation calibration.

An optional unsigned 64-bit `rounding_seed` enables stochastic code selection
without changing scale selection or consuming PyTorch's global RNG. Omitting it
uses nearest rounding. Reproduction is specific to the backend/device; Torch and
Triton need not choose identical codes. Standard update signatures support
`torch.compile`, but the `add_` seed extension is for eager updates because of
Dynamo's built-in signature. Repeated updates are lossy: restore pristine weights
before replacing or removing a merged adapter. Callers must also respect ownership
of the storage they update, including read-only checkpoint mappings.

For custom quantizers, the [stochastic quantization package](../src/piper_kernels/stochastic_quantization/__init__.py)
exposes integer and codebook rounding; its `triton` module supplies kernel primitives.

## Convert packed GGUF storage

Both ConvRot formats expose `from_gguf(packed_bytes, quant_type=..., group_size=...)`
and `weight.copy_from_gguf_(packed_bytes, quant_type=...)`. Pass a 2-D packed byte
tensor and its GGML quantization type; the type may be omitted if the tensor carries
a `quant_type` attribute. The package does not parse GGUF files or map tensor names.

Conversion decodes, rotates, and quantizes without allocating a dense weight;
`copy_from_gguf_` refills compatible existing storage. INT8 requires a compatible
Triton accelerator; NVFP4 conversion requires exact SM120. Unsupported devices
raise rather than allocate a dense fallback. ConvRot NVFP4 accepts the same scale
and activation configuration as `from_hp`, including
`compute_per_tensor_scale=True`. See the [INT8](../src/piper_kernels/weights/convrot/int8/tensor.py)
and [NVFP4](../src/piper_kernels/weights/convrot/nvfp4/tensor.py) constructors for signatures.

## Shard a quantized weight

Use `shard_quantized_weight` to partition an already quantized full weight without
requantizing. Unlike a view, each shard owns its storage; NVFP4 block scales are
repacked with fresh padding.

```python
from piper_kernels.weights.sharding import shard_quantized_weight

row_shard = shard_quantized_weight(int8_weight, dim=0, start=0, length=64)
```

`dim=0` partitions output rows at any nonempty contiguous interval. `dim=1`
partitions input channels and must align both start and length to ConvRot groups
and, for NVFP4, 16-value quantization blocks. Inputs must be contiguous,
untransposed matrices. NVFP4 accepts flat/canonical and ordinary/swizzled scales,
but execution still requires its supported layout. Ordinary slice/narrow views
and DTensor redistribution of quantized weights are unsupported.

For DTensor, install the local quantized shard **before** `parallelize_module`.
This example assumes an initialized 1-D `mesh` and the same `full_weight` loaded on
each rank:

```python
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

dim = 0  # 0: output rows; 1: input channels
assert full_weight.shape[dim] % mesh.size() == 0
length = full_weight.shape[dim] // mesh.size()
local_weight = shard_quantized_weight(
    full_weight, dim=dim, start=mesh.get_local_rank() * length, length=length
).to(mesh.device_type)
distributed_weight = DTensor.from_local(
    local_weight,
    mesh,
    [Shard(dim)],
    run_check=False,
    shape=full_weight.shape,
    stride=full_weight.stride(),
)
out_features, in_features = full_weight.shape
distributed_layer = torch.nn.Linear(in_features, out_features, bias=False, device="meta")
distributed_layer.weight = torch.nn.Parameter(distributed_weight, requires_grad=False)
plan = ColwiseParallel() if dim == 0 else RowwiseParallel()
distributed_layer = parallelize_module(distributed_layer, mesh, plan)
```

Bias must also be a DTensor: `Shard(0)` for output-row sharding, `Replicate()` for
input-channel sharding. Output-row shards concatenate; input-channel shards sum
partial outputs and add bias once. Input partitioning may change dynamic scales
and accumulation order, so compare with tolerances against the corresponding local
quantized computation. For fused sharded attention/FFNs, keep complete heads or
intermediate-feature shards local, unwrap activations around the fused region,
and place collectives outside it; compare against the local **fused** computation.

## Storage and checkpoint compatibility

Same-shape `view`/`view_as` and matrix transposes share storage and retain format
metadata. `as_strided` supports the existing layout or its transpose with unchanged
storage offset. Other shape/layout changes are unsupported. A transposed weight
can participate in activation matrix products, but cannot be made contiguous,
updated in place, or passed as another `F.linear` weight. The DTensor `mm`/`addmm`
path supports the linear case: `alpha=beta=1` and a vector bias.

Floating-point conversions change the logical dtype while reusing packed
storage; converting to the current dtype and device returns the same weight.
Views and conversions have the same contracts under `torch.no_grad()` and
`torch.inference_mode()`, including weights created outside inference mode.

Use `load_state_dict(..., assign=True)` when preserving supplied checkpoint storage
matters. Tensor subclass checkpoints carry Python class paths: the current classes
live under `piper_kernels.weights`, and checkpoints pickled with former `linear`
class paths must be re-exported; legacy import aliases are not provided. Packed
tensor/scale layouts are unchanged. See the [changelog](../CHANGELOG.md) when
upgrading a pinned consumer.
