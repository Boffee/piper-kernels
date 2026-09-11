# Static-scale ConvRot INT8 Conv3D

`ConvRotInt8Tensor` represents both linear and convolution weights.
`ConvRotInt8Conv3d` is a thin inference layer consuming a convolution weight,
plus bias, stride, and spatial padding. Quantization state belongs to the weight.

- Input and output: FP16 NCTHW, including noncontiguous inputs and residuals.
- Logical weight: `[out, in, 3, 3, 3]`, with FP16, BF16, or FP32 logical dtype.
- Packed `qdata`: contiguous INT8 `[out, 3, 3, 3, in]`.
- Weight `scale`: contiguous FP32 `[out, 1]`, one per output channel.
- `act_per_tensor_scale`: one finite positive FP32 scalar tensor on the weight
  device, shared across all channels, positions, frames, and calls. Conversion
  and dequantization can omit it; convolution execution requires it.
- Input channels: powers of two from 64 through 4096, divisible by rotation
  group size (16, 64, or 256). This bounds the full `27 * channels` INT32 sum.
- Optional bias: FP16, one per output channel, including noncontiguous storage.
- Temporal padding: two zero frames before the input. Spatial padding:
  `reflect`, `reflect_right` (bottom/right only), or `none`. Reflection requires
  height and width greater than one. Strides are three positive integers;
  dilation and convolution groups are not supported.

The optimized backend targets SM120; other devices use the portable reference.
GroupNorm statistics, affine transforms, SiLU, rotation, rescaling, bias, and
residual addition use FP32 intermediates. The convolution accumulates INT8
products in INT32. Eliminated FP16 intermediate rounding is not reproduced.

## Quantize and dequantize weights

Use the same weight conversion API as linear ConvRot INT8 and ConvRot NVFP4:

```python
import torch
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

weight = ConvRotInt8Tensor.from_hp(
    dense_weight,  # [out, in, 3, 3, 3]
    group_size=64,
    act_per_tensor_scale=torch.tensor(
        calibrated_input_scale,
        dtype=torch.float32,
        device=dense_weight.device,
    ),
)
reconstructed = weight.dequantize(output_dtype=torch.float32)
```

Quantization rotates input-channel groups independently at each kernel position
and derives one weight scale over the complete filter for each output channel.
Dequantization applies the scales and inverse rotation in FP32, restores the
logical layout, and casts to the requested dtype (default: the logical dtype).
The result approximates the original weight; quantization is lossy.

Linear execution remains dynamically scaled. Passing a static activation scale
for a 2-D weight is currently unsupported. Matrix transpose, linear execution,
GGUF conversion, matrix updates, and weight sharding remain 2-D operations.

## Save offline and load without copying

Run `from_hp()` when producing the checkpoint: it allocates new storage.
Store `qdata`, `scale`, and `act_per_tensor_scale`, plus the weight's `group_size`
and logical dtype. TorchAO's `__tensor_flatten__` / `__tensor_unflatten__`
protocol exposes these tensors and metadata for checkpoint/offload integrations.
Keep source-model fingerprints and calibration provenance in the engine manifest.
Stride, padding, and bias placement come from the model architecture.

```python
from piper_kernels.conv3d.convrot.int8 import ConvRotInt8Conv3d

weight = ConvRotInt8Tensor.from_quantized(
    loaded_qdata,
    loaded_scale,
    group_size=64,
    logical_dtype=torch.float16,
    act_per_tensor_scale=loaded_activation_scale,
)
conv = ConvRotInt8Conv3d(weight, loaded_bias, stride=(1, 1, 1), padding="reflect")
output = conv(activation)
```

Contiguous checkpoint storage is reused, including mmap storage from safetensors
or `torch.load(..., mmap=True)`. A flat weight scale is reshaped without copying.
Noncontiguous storage is canonicalized by `from_quantized()`; converters should
write the packed layout directly to retain mmap backing during loading.

The layer registers ordinary inference parameters `weight` and optional `bias`.
PyTorch state dictionaries serialize the tensor subclass; load with
`load_state_dict(..., assign=True)` to preserve checkpoint storage. For
`weights_only=True`, allowlist `ConvRotInt8Tensor` with
`torch.serialization.safe_globals`. Logical dtype changes preserve INT8 weight
data and FP32 weight/activation scales. Device moves include every inner tensor.
The logical weight uses contiguous layout; channels-last memory-format requests
are unsupported. Packing and output dtype conversion are combined into one copy
when quantizing or dequantizing convolution weights.

Offload's ConvRot INT8 adapter must capture and reconstruct all three storage
fields, including optional `act_per_tensor_scale`, as its NVFP4 adapter does.
Offload validation requires canonical contiguous storage and never repacks it.
There is no separate prepared-weight cache. Compiled calls read the current
weight tensors and observe weight and activation-scale replacement.

## MiniMax-H3 integration

`piper_kernels.specializations.minimax_h3_vae.conv3d.P995_ACTIVATION_SCALES`
contains candidate activation scales for 29 encoder convolutions. Existing H3
selection uses rotation groups of 64 for 128 input channels and 256 otherwise.
RGB input convolutions and 1x1 shortcuts are excluded. These constants are
candidate calibration data; the engine owns checkpoint compatibility and quality
validation. Weight conversion uses `ConvRotInt8Tensor.from_hp()` directly.

Install `ConvRotInt8Conv3d` layers from the prequantized checkpoint, then compile
the encoder under `torch.no_grad()` or inference mode:

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

The pre-grad rewrite recognizes H3's framewise GroupNorm, SiLU, reflection
padding, and optional residual around an explicit ConvRot convolution. It needs
static shapes and leaves unsupported or shared patterns unfused. It does not
quantize arbitrary floating-point Conv3D nodes.

Keep the original model forwards. For downsamplers, leave the caller's
bottom/right reflection and install the convolution with `padding="none"`;
the compiler absorbs the pad. Use `reflect_right` directly only if the caller's
padding has been removed. Encoder options remain independent of the INT8 and
NVFP4 decoder options and can be composed with them.

The earlier experimental engine's private convolution/block/encoder wrappers
and runtime weight quantization should be replaced by this shared weight API,
the thin convolution layer, and the graph pass. Engine still owns layer
eligibility, checkpoint selection, and installation.

The fused boundary uses separate statistics, preparation, and convolution
launches, allocating a full INT8 activation and small FP32 statistics buffers.
Full-encoder quality, peak memory, and end-to-end performance still need
validation with the actual checkpoint and engine.
