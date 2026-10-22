# ConvRot INT8 Conv3D integration

For a first conversion and inference example, see
[weights](../../../../../docs/weights.md). `ConvRotInt8Tensor` represents both
linear and convolution weights, including ordinary `F.conv3d`/`nn.Conv3d` calls.
Layers own bias, stride, and padding; quantization state belongs to the weight.

## Implementation boundaries

The functional and storage boundaries follow ConvRot INT8 linear's conventions:

- `_arguments.py` binds Python/ATen calls for eager execution and compiler
  rewrites, then resolves supported stride and padding geometry.
- `_functional.py` adapts tensor-subclass dispatch to typed execution, applies
  standard-call autocast, and implements the causal convenience functions.
- `dispatch.py` owns inference and storage validation before custom-op dispatch.
- `_ops.py` defines stable custom-op schemas, fake implementations, and runtime
  backend selection. `_validation.py` supplies the shared metadata checks.

Convolution custom operators retain a portable reference runtime because the
pre-grad padding pass emits them on CPU as well as GPU. Compiler integration is
loaded only when compile options are requested. Its cache keys track each of
these boundaries and the backend implementations.

## Data and execution contract

| Value | Contract |
|---|---|
| Input | FP16 or FP32 NCTHW, including noncontiguous inputs. |
| Output / optional residual | FP16. |
| Logical weight | `[out, in, 3, 3, 3]`, FP16/BF16/FP32 logical dtype. |
| Packed `qdata` | Contiguous INT8 `[out, 3, 3, 3, in]`. |
| Weight `scale` | Contiguous FP32 `[out, 1]`, one per output channel. |
| `act_per_tensor_scale` | One finite positive FP32 scalar tensor on the weight device, shared across positions, frames, and calls. Execution requires it; conversion/dequantization can omit it. |
| Input channels | Powers of two from 64 through 4096, divisible by rotation group size 16, 64, or 256. This bounds the full `27 * channels` INT32 sum. |
| Bias | Optional FP16/FP32 vector, one value per output channel; noncontiguous storage is supported. |

Standard calls use PyTorch stride and padding geometry: nonnegative symmetric
zero padding, `"valid"`, or `"same"` with stride 1. `padding=0` adds nothing.
Activations are batched NCTHW; groups=1 and dilation=1 are required. FP16 autocast
casts eligible FP32 activations/bias; BF16 autocast is unsupported. Output is
always FP16, independent of logical weight dtype.

The explicit `ConvRotInt8Conv3d`, `conv3d`, and `group_norm_silu_conv3d` APIs retain
their causal contract: two zero frames before the input, with spatial padding
`reflect`, `reflect_right` (bottom/right only), or `none`. Reflection requires
height and width greater than one. These APIs resolve that geometry into the
same kernel's explicit padding configuration; they do not define another
convolution implementation. Their FP32 inputs retain precision under autocast.

GroupNorm statistics, affine transforms, SiLU, rotation, rescaling, bias, and
residual addition use FP32 intermediates; INT8 products accumulate in INT32.
The fused path does not reproduce eliminated FP16 intermediate rounding. FP32
inputs retain normalization precision through preparation.

Quantization rotates input-channel groups independently at each kernel position
and derives one scale over each output channel's complete filter. Dequantization
applies scales and inverse rotation in FP32, restores logical layout, and casts
to the requested dtype. It is lossy. Transpose, linear execution, GGUF conversion,
matrix updates, and weight sharding remain 2-D operations.

## Checkpoints and offloading

Quantize offline with `from_hp()`, which allocates storage. Save `qdata`, `scale`,
`act_per_tensor_scale`, `group_size`, and logical dtype. The engine manifest owns
source-model fingerprints and calibration provenance; architecture supplies
stride, padding, and bias placement. The tensor flatten/unflatten protocol
exposes the storage and metadata to checkpoint and offload integrations.

Reconstruct a prequantized convolution from loaded checkpoint tensors:

```python
import torch
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.conv3d.convrot.int8 import ConvRotInt8Conv3d

weight = ConvRotInt8Tensor.from_quantized(
    loaded_qdata,
    loaded_scale,
    group_size=64,
    logical_dtype=torch.float16,
    act_per_tensor_scale=loaded_input_scale,
)
conv = ConvRotInt8Conv3d(weight, loaded_bias, stride=(1, 1, 1), padding="reflect")
```

Contiguous storage is reused, including checkpoint mappings; a flat scale can
be reshaped without copying. `from_quantized()` canonicalizes noncontiguous
storage, so converters should write the packed layout directly to preserve
mapped backing. The layer registers inference parameters `weight` and optional
`bias`. Follow the shared [checkpoint-loading guidance](../../../../../docs/weights.md#storage-and-checkpoint-compatibility)
to preserve storage. `weights_only=True` loading needs `ConvRotInt8Tensor` in
`torch.serialization.safe_globals`.

Logical dtype changes preserve INT8 data and FP32 weight/activation scales.
Device moves include every inner tensor. The logical weight is contiguous;
channels-last memory-format requests are unsupported. Convolution quantization
and dequantization combine packing and output dtype conversion into one copy.

Offload adapters must preserve all three storage fields, including the optional
activation scale. Offload validation requires canonical contiguous storage and
never repacks it. There is no prepared-weight cache: compiled calls read current
weight tensors and observe weight or activation-scale replacement.

## Padding and compilation

The internal convolution accepts six nonnegative padding amounts in `F.pad`
order (width, height, time; before/after per axis). Time boundaries supply zeros;
spatial boundaries supply zeros or reflection. Reflection amounts must be
smaller than the corresponding input dimension. Output shapes, execution-plan
selection, the portable reference, and GPU indexing use this same geometry.
GPU loads supply boundary values directly, so fused execution prepares only
original pixels and avoids materializing padded activations.

```python
from piper_kernels.conv3d.convrot.int8 import convrot_int8_conv3d_compile_options

compiled = torch.compile(model, fullgraph=True, options=convrot_int8_conv3d_compile_options())
```

The optional pre-grad pass lowers standard quantized convolution calls and
absorbs exclusive chains of zero padding and spatial reflection. It reads live
weight storage through the graph, including activation-scale replacement.
Padding fusion preserves autocast rounding. Shared pads, nonzero constant pads,
cropping, temporal reflection, and combinations of spatial zero/reflection
borders not represented by the kernel remain separate operations. Fusion uses
static shapes; ordinary compiled execution also supports dynamic shapes.
Correctness never depends on matching a padding pattern. Original floating-point
convolutions are left unchanged.

## Backend ownership

Optimized execution targets NVIDIA SM120/SM8x and RDNA4 (`gfx1200`/`gfx1201`);
other targets use the portable reference. RX 9070 XT and RTX 4070 Ti SUPER have
hardware coverage; `gfx1200` has offline compilation coverage.

[_backend.py](_backend.py) selects typed vendor entry points. Vendor policy
selects preparation, convolution, and weight loading from target/shape metadata.
[_plan.py](_plan.py) combines `PreparationSchedule` and `ConvolutionSchedule`
into a concrete `ConvolutionExecutionPlan`; [_dispatch.py](_dispatch.py) resolves
dimensions/alignment and [triton.py](triton.py) runs shared kernels.
Preparation schedules use input rows; convolution schedules use output rows
after padding and stride. The tuner substitutes a convolution tile while
retaining production preparation and recomputing descriptor eligibility.

AMD uses HIP quantization rounding and pointer weight loads. Only SM120 uses
weight descriptors; SM8x uses its own tiles to avoid the register pressure of
SM120 schedules. Moving checkpoints between supported targets needs no
activation-scale conversion. Follow the
[validation contract](../../../../../docs/development.md#validation-contract).
Use the [shared Conv3D benchmark](../../../../../benchmarks/README.md#run-the-suite)
for quantized correctness and matching FP16 performance comparisons.

## MiniMax-H3 integration

`piper_kernels.specializations.minimax_h3_vae.conv3d.P995_INPUT_SCALES` contains
candidate scales for 29 encoder convolutions. Existing selection uses group 64
for 128 input channels and 256 otherwise, excluding RGB input convolutions and
1x1 shortcuts. The engine owns layer eligibility, checkpoint compatibility,
installation, and quality validation; these constants are candidate calibration.

Install prequantized layers and compile the encoder for inference:

```python
from piper_kernels.specializations.minimax_h3_vae import (
    minimax_h3_vae_convrot_int8_conv3d_compile_options,
)

encoder.compile(
    dynamic=False,
    fullgraph=True,
    options=minimax_h3_vae_convrot_int8_conv3d_compile_options(),
)
```

Run under `torch.no_grad()` or inference mode. The pre-grad rewrite includes
the generic padding pass and matches framewise GroupNorm, SiLU, and optional
residual around standard or explicit ConvRot convolutions. It requires static shapes and leaves unsupported
or shared patterns unfused. It does not quantize floating-point Conv3D nodes.

Keep original forwards and ordinary convolution modules, installing prequantized
weights without replacing the layers. H3's explicit temporal zero padding and
spatial reflection are absorbed into the shared convolution when the graph
matches. The specialization also removes intermediate FP16 rounding, including
a Conv3D autocast conversion, when fusing normalization/SiLU. For existing causal
wrappers, retain any caller-side bottom/right reflection and select spatial
`padding="none"`; use `reflect_right` only if that caller-side pad is absent.
Encoder options are independent of, and composable with, decoder options.

The fused boundary has separate statistics, preparation, and convolution
launches, with a full INT8 activation and small FP32 statistics buffers.
Operator validation does not establish full-encoder quality, memory, or speed;
those require the actual checkpoint and engine.
