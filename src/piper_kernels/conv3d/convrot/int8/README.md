# Static-scale ConvRot INT8 Conv3D

`ConvRotInt8Tensor` represents both linear and convolution weights.
`ConvRotInt8Conv3d` is a thin inference layer consuming a convolution weight,
plus bias, stride, and spatial padding. Quantization state belongs to the weight.

- Input: FP16 or FP32 NCTHW, including noncontiguous inputs. Output and residual: FP16.
  FP32 inputs retain normalization precision through INT8 preparation.
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

Optimized backends target NVIDIA SM120 and ROCm RDNA4 (`gfx1200`/`gfx1201`);
other devices use the portable reference. RX 9070 XT (`gfx1201`) has hardware
correctness/performance coverage; `gfx1200` also has offline compilation coverage.
GroupNorm statistics, affine transforms, SiLU, rotation, rescaling, bias, and
residual addition use FP32 intermediates. The convolution accumulates INT8
products in INT32. Eliminated FP16 intermediate rounding is not reproduced.

The public operations and weight format are shared. `_backend.py` selects a
vendor implementation; `_nvidia/` and `_amd/` own target support and launch policy.
Both use the common Triton kernels and launch mechanics in `triton.py`. AMD uses
HIP quantization rounding and pointer-based weight loads; NVIDIA retains its
SM120 tile policy and optional weight descriptors. No activation-scale conversion
or checkpoint migration is needed when moving between supported devices.

## ROCm validation and benchmarks

Use an existing ROCm environment; the repository's default `uv` sources select CUDA:

```shell
PYTHONPATH=src /path/to/rocm-env/bin/python -m pytest -o addopts='' tests/conv3d/convrot/int8
PYTHONPATH=src /path/to/rocm-env/bin/python benchmarks/benchmark_convrot_int8_conv3d_rocm.py
```

The hardware regressions also run through `scripts/run_rocm_regressions.py`.
They cover FP16/FP32 and noncontiguous inputs, all padding modes, channel counts
through 4096, exact INT32 accumulation, fused normalization, dynamic compilation,
and graph capture with live activation-scale changes. Offline tests check integer
matrix instructions on SM120 and both RDNA4 targets.

The benchmark reports plain and fused convolution timings against the portable
reference and standard PyTorch ROCm FP16 convolution, with cache-flushed and
graph-replay measurements. The FP16 baselines use both contiguous and
`channels_last_3d` inputs/weights. They include matching causal/reflection padding
and, for the fused comparison, eager framewise GroupNorm and SiLU. Logical FP16
weight reconstruction and initial layout conversions happen outside timing;
normalization, activation, padding, and any internal layout copies remain timed.
FP16 omits activation quantization and retains FP16 intermediate rounding, so it
is a performance baseline rather than the INT8 correctness oracle.

Repeat `--shape N,C,T,H,W,O` to select synthetic cases; `--dtype float32` selects
FP32 input for INT8/reference (the FP16 baseline still uses FP16), and `--tune`
sweeps prepared convolution tiles without changing production policy.
`--miopen-benchmark` enables vendor algorithm search before timing. The report
records that setting and `PYTORCH_MIOPEN_SUGGEST_NHWC`, which gates native
channels-last MIOpen execution in the tested PyTorch build.
`--skip-reference-timing` skips only the portable reference's timings, retaining
the correctness comparison. These measurements do not establish performance or
quality for an entire encoder or a real checkpoint.

Initial RX 9070 XT measurements (2026-09-19, FP16 inputs, graph replay, milliseconds):

| N,C,T,H,W,O | Native plain | Reference plain | Native fused | Reference fused |
| --- | ---: | ---: | ---: | ---: |
| 1,128,5,64,64,128 | 0.106 | 3.399 | 0.121 | 4.544 |
| 1,256,3,32,32,256 | 0.064 | 1.606 | 0.070 | 1.667 |
| 1,512,3,16,16,512 | 0.078 | 1.450 | 0.084 | 1.947 |

These compare with the portable quantized reference, not a vendor-tuned FP16
convolution. They use the benchmark's seeded synthetic weights and default
reflection padding, stride, and scales. Environment: Python 3.13.13,
PyTorch `2.14.0+rocm10.1.0a20260908`, HIP `7.16.26354`, Triton 3.8.0;
command: `PYTHONPATH=src /path/to/rocm-env/bin/python
benchmarks/benchmark_convrot_int8_conv3d_rocm.py --rep-ms 100`.

Standard FP16 comparison on the same RX 9070 XT/software stack (2026-09-19):
median of three separate process runs, graph replay, milliseconds, with
`--rep-ms 100 --miopen-benchmark` and `PYTORCH_MIOPEN_SUGGEST_NHWC=1`. The table
selects the faster FP16 layout per operation/shape: contiguous for C=128 and
plain C=256; native channels-last for fused C=256 and both C=512 operations.

To reproduce the FP16 comparison, run the following command in three separate
processes and take the median for each reported timing:

```shell
PYTORCH_MIOPEN_SUGGEST_NHWC=1 PYTHONPATH=src \
  /path/to/rocm-env/bin/python benchmarks/benchmark_convrot_int8_conv3d_rocm.py \
  --rep-ms 100 --miopen-benchmark --skip-reference-timing
```

| N,C,T,H,W,O | INT8 plain | FP16 plain | Plain speedup | INT8 fused | FP16 GN/SiLU/conv | Fused speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,128,5,64,64,128 | 0.107 | 0.317 | 2.97x | 0.122 | 0.405 | 3.31x |
| 1,256,3,32,32,256 | 0.065 | 0.190 | 2.90x | 0.071 | 0.237 | 3.34x |
| 1,512,3,16,16,512 | 0.079 | 0.191 | 2.43x | 0.085 | 0.219 | 2.57x |

Ratios use unrounded timings. Both paths include padding; the INT8 path includes
rotation/quantization. FP16 uses the dequantized logical filter, with no runtime
weight conversion. Three additional runs with the default MIOpen layout setting
favored contiguous FP16 for every case (plain: 0.317/0.190/0.209 ms;
GN/SiLU/conv: 0.404/0.238/0.234 ms).

MIOpen's first algorithm search logged unavailable candidate kernels in this
nightly build, but execution and correctness checks completed;
the repeat runs had no MIOpen search errors. With native channels-last enabled,
the portable FP32 reference became very slow (about 700 ms); the two repeat runs
therefore used `--skip-reference-timing`, while still checking correctness against
it. Results characterize this installed stack, not an exhaustive search over
ROCm versions or convolution implementations.

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

Linear execution uses the stored static input scale when present and dynamic
per-row scaling otherwise. Matrix transpose, linear execution, GGUF conversion,
matrix updates, and weight sharding remain 2-D operations.

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
    act_per_tensor_scale=loaded_input_scale,
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

`piper_kernels.specializations.minimax_h3_vae.conv3d.P995_INPUT_SCALES`
contains candidate input scales for 29 encoder convolutions. Existing H3
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
