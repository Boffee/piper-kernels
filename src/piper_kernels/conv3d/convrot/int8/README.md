# ConvRot INT8 Conv3D internals

For conversion, inference, and caller constraints, see
[Conv3D usage](../../../../../docs/weights.md#run-a-conv3d). This note covers
implementation boundaries, arithmetic, storage integration, and backend policy.

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
loaded only when compile options are requested.

## Arithmetic

GroupNorm statistics, affine transforms, SiLU, rotation, rescaling, bias, and
residual addition use FP32 intermediates; INT8 products accumulate in INT32.
The fused path does not reproduce eliminated FP16 intermediate rounding. FP32
inputs retain normalization precision through preparation. The input-channel
limit bounds the full `27 * channels` INT32 sum.

Quantization rotates input-channel groups independently at each kernel position
and derives one scale over each output channel's complete filter. Dequantization
applies scales and inverse rotation in FP32, restores logical layout, and casts
to the requested dtype. Convolution quantization and dequantization combine
packing and output dtype conversion into one copy. The
[weight representation](../../../weights/convrot/int8/tensor.py) owns the stored
fields and supported tensor operations.

## Storage and offload integration

Layers own bias, stride, and padding; quantization state belongs to the weight.
The tensor flatten/unflatten protocol exposes storage and metadata to checkpoint
and offload integrations. Device moves include every inner tensor; logical dtype
changes preserve INT8 data and FP32 scales. Follow the shared
[checkpoint-loading guidance](../../../../../docs/weights.md#storage-and-checkpoint-compatibility)
for preserving mapped storage.

Offload adapters must preserve `qdata`, `scale`, and the optional
`act_per_tensor_scale`. Offload validation requires canonical contiguous storage
and never repacks it. There is no prepared-weight cache: compiled calls read
current weight tensors and observe weight or activation-scale replacement.

## Padding and compilation

The internal convolution accepts six nonnegative padding amounts in `F.pad`
order (width, height, time; before/after per axis). Time boundaries supply zeros;
spatial boundaries supply zeros or reflection. Reflection amounts must be
smaller than the corresponding input dimension. Output shapes, execution-plan
selection, the portable reference, and GPU indexing use this same geometry.
GPU loads supply boundary values directly, so fused execution prepares only
original pixels and avoids materializing padded activations.

The optional [pre-grad pass](_compile.py) lowers standard quantized convolution
calls and absorbs exclusive chains of zero padding and spatial reflection. It
reads live weight storage through the graph, including activation-scale replacement.
Padding fusion preserves autocast rounding. Shared pads, nonzero constant pads,
cropping, temporal reflection, and combinations of spatial zero/reflection
borders not represented by the kernel remain separate operations. Fusion supports
static and symbolic activation shapes when the graph proves the required
reflection bounds.
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

MiniMax-H3 calibration, installation, and normalization-fusion assumptions live
with the [encoder specialization](../../../specializations/minimax_h3_vae/conv3d/README.md).
