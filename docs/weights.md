# Quantized weights

Piper weight tensors hold packed values and quantization metadata while exposing
a logical floating-point shape and dtype. Construct them offline from dense
weights, or wrap checkpoint storage, then use them with `F.linear`, supported
`F.conv3d` calls, or an inference module. The application owns checkpoint loading,
calibration, and model quality.

| Format | Stored representation | Execution |
|---|---|---|
| `ConvRotInt8Tensor` | Grouped rotation, INT8 values, FP32 scale per output | Linear and Conv3D, with optimized NVIDIA/AMD kernels and portable fallback; embedding lookups. |
| `ConvRotTrellisTensor` | Packed 2–6 bpw trellis bitstreams, FP8 block scales, FP32 row scale | Linear (`F.linear`) via Triton kernels on supported NVIDIA/AMD GPUs and CPU fallback. |
| `PiperNVFP4Tensor` | Packed FP4 values, FP8 block scales, optional FP32 global scale | Native W4A4 linear and fusions target exact NVIDIA SM120; weight-only W4A16 (`act_quant_kwargs=None`) targets NVIDIA SM75+ and supported AMD ROCm GPUs. |
| `ConvRotNVFP4Tensor` | NVFP4 in a grouped rotated basis | Same W4A4 and W4A16 GPU targets, plus a CPU reference fallback for weight-only W4A16. |

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
linear_layer = torch.nn.Linear(256, 128, bias=False, device="meta")
linear_layer.weight = torch.nn.Parameter(int8_weight, requires_grad=False)
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

This complete example supports both full W4A4 execution (requires an NVIDIA SM120
GPU and the NVFP4 dependencies) and weight-only W4A16 execution (`act_quant_kwargs=None`,
supported on NVIDIA SM75+ and Linux/Windows AMD `gfx942`, `gfx1100`, `gfx1151`,
`gfx1200`, and `gfx1201`). The native path requires 16-value blocks, swizzled
scales, and FP16, BF16, or FP32 activations matching the logical weight dtype.
ConvRot input widths must also divide into rotation groups.

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

For W4A16 weight-only execution across SM75+, supported ROCm GPUs, and CPU (where
ConvRot weight-only runs on the portable reference while ordinary NVFP4 delegates to
TorchAO or dequantized linear), construct the same tensors with `act_quant_kwargs=None`
and `act_per_tensor_scale=None`. Set `act_quant_kwargs` as shown above on SM120 when
you want W4A4 activations, including compiled FFN and sparse attention fusions.

`compute_per_tensor_scale=True` computes the weight global scale (after rotation
for ConvRot; mutually exclusive with `per_tensor_scale`). For static activation
scaling, set `use_dynamic_per_tensor_scale=False` and pass a finite positive FP32
scalar `act_per_tensor_scale` on the weight device calibrated in the stored basis;
omitting it fixes unit global scaling (`1.0`, still computing 16-value block
scales). The same scale contract applies to NVFP4/ConvRot FFN and sparse output
fusions; see [compilation](#compile-a-model) for preparation sharing.

Zero scales decode a block as zero without NaNs, and independent vs fused backends need
not agree bitwise (`use_triton_kernel=True` in `from_hp` selects TorchAO's `1/64` normal
floor instead of Piper's `1/512` E4M3 floor).

`PiperNVFP4Tensor.from_torchao(existing)` wraps TorchAO storage without copying;
`ConvRotNVFP4Tensor.from_torchao(existing, group_size=...)` attaches rotation
metadata to values already in that basis (see
[ordinary NVFP4](../src/piper_kernels/weights/nvfp4/tensor.py) and
[ConvRot NVFP4](../src/piper_kernels/weights/convrot/nvfp4/tensor.py) for packed
constructors and fields, and [NVFP4 reference](../src/piper_kernels/linear/nvfp4/reference.py)
for `linear` / `linear_w4a16` numerical checks). Native W4A4 ConvRot NVFP4 requires
SM120; weight-only and unsupported paths either use portable fallbacks or raise when
the backend constraints are unmet.

## Look up embedding rows

A 2-D ConvRot INT8 weight also serves as an `nn.Embedding` or `F.embedding` table of
shape `[num_embeddings, embedding_dim]`. A lookup selects the requested rows' INT8
values and scales, then dequantizes only those rows into the logical dtype and original
basis; the table is never decoded whole. The rotation stays within each row, so rows
decode independently.

```python
import torch
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

table = ConvRotInt8Tensor.from_hp(torch.randn(1000, 256, dtype=torch.bfloat16), group_size=256)
embedding = torch.nn.Embedding(1000, 256, device="meta")
embedding.weight = torch.nn.Parameter(table, requires_grad=False)

with torch.inference_mode():
    vectors = embedding(torch.tensor([[3, 7, 3]]))
assert vectors.shape == (1, 3, 256)
```

`padding_idx` affects only gradients, so it does not change a lookup. `max_norm`
renormalization and transposed weights raise `NotImplementedError`.

## Run a quantized MoE FFN

`piper_kernels.moe` provides offload-first single-token (`M = 1`), small-batch, and prefill
(`M > 1`) execution across
expert-major quantized slabs stored as `[num_slots, ...]`: NVFP4 W4A16 (`"nvfp4_w4a16"`),
ConvRot NVFP4 W4A16 (`"convrot_nvfp4_w4a16"`), and ConvRot INT8 W8A8 (`"convrot_int8"`). Each
contiguous slab holds `up_gate_qdata`, `up_gate_scale`, `down_qdata`, `down_scale`, and optional
per-slot `float32` NVFP4 global scales (`[num_slots]`, `[num_slots, 1]`, or
`[num_slots, 1, 1]`), so runtimes can wrap resident slabs or staging buffers zero-copy with
`MoEWeights.from_quantized`.

NVFP4 slabs use packed `uint8` nibbles and `SWIZZLE_32_4_4` `float8_e4m3fn` 16-value block scales
on widths divisible by 16 (and `group_size` for ConvRot). ConvRot INT8 slabs use `int8` values,
`float32` per-row scales (`[num_slots, out]` or `[num_slots, out, 1]`), `group_size` in
`{16, 64, 256}`, and widths up to `131_071` so INT32 accumulators never wrap; GPU INT8 kernels
target NVIDIA SM80+ and supported AMD ROCm GPUs, while earlier CUDA targets and CPU run the
portable reference.

```python
import torch
from piper_kernels.moe import MoEWeights, moe_ffn, prepare_moe_input

device = "cuda" if torch.cuda.is_available() else "cpu"
num_slots, hidden_size, intermediate_size = 4, 128, 128
up_gate_hp = torch.randn(
    num_slots, 2 * intermediate_size, hidden_size, device=device, dtype=torch.bfloat16
)
down_hp = torch.randn(
    num_slots, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16
)
hidden_states = torch.randn(1, hidden_size, device=device, dtype=torch.bfloat16)
topk_ids = torch.tensor([[0, 2, 5]], device=device, dtype=torch.int64)
topk_weights = torch.tensor([[0.5, 0.3, 0.2]], device=device, dtype=torch.bfloat16)
expert_to_slot = torch.tensor([2, -1, 0, -1, 3, 1], device=device, dtype=torch.int64)
residual_out = torch.zeros_like(hidden_states)

weights = MoEWeights.from_hp(
    up_gate_hp,  # [num_slots, 2 * I, H] ordered [gate | up] when gate_first=True
    down_hp,  # [num_slots, H, I]
    weight_format="nvfp4_w4a16",
    activation_fn="swiglu",
    gate_first=True,
)
with torch.inference_mode():
    prepared = prepare_moe_input(hidden_states, weights)  # optional reuse across waves
    output = moe_ffn(
        hidden_states,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=expert_to_slot,  # negative or out-of-range slots skip cleanly on-device
        out=residual_out,
        accumulate=True,  # add weighted slot hits into caller-owned out; all-miss leaves out intact
        prepared_input=prepared,
    )
assert output is residual_out and output.shape == (1, hidden_size)
assert torch.isfinite(output).all()
```

Set `gate_first=False` when checkpoints pack `[up | gate]` halves (matching Piper's linear SwiGLU
convention), and pass `activation_fn="gelu_tanh"` for `tanh`-GELU experts without a paired gate
projection. `prepare_moe_input` returns an independent snapshot safe to reuse across waves
even when accumulating into `out=input` (storing FP32 rotated activations for
`"convrot_nvfp4_w4a16"` across FP16, BF16, and FP32 inputs). Calling `moe_ffn`
without `out` returns a freshly allocated contiguous tensor; multi-row (`M > 1`) calls
compact and batch active expert slots on-device across NVFP4 W4A16 and ConvRot INT8 (chunked
under a bounded workspace for large prompts).
Caller-supplied `out` must match input
shape/dtype/device, have `stride(-1) == 1` with disjoint rows (`out.view(rows, H).stride(0) >= H`
when `rows > 1`), must not use negated/conjugated views or alias weight storage, and must be
writable in the active inference/autograd mode (`out is input` is supported zero-copy at `M = 1`).

## Run a Conv3D

ConvRot INT8 represents logical `[out, in, 3, 3, 3]` weights usable with `F.conv3d`
and `nn.Conv3d` for inference. Execution requires a static activation scale, FP16
or FP32 NCTHW inputs with matching optional bias, power-of-two input channels in
`[64, 4096]` divisible by `group_size`, `groups=1`, and `dilation=1`; output is
always FP16. Noncontiguous inputs and bias are supported; channels-last weight
memory formats are not.

```python
import torch
from torch.nn import functional as F
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
kernel = torch.randn(16, 64, 3, 3, 3, device=device, dtype=torch.float16)
# Illustrative scale; use a calibrated value for a real checkpoint.
input_scale = torch.tensor(0.05, device=device, dtype=torch.float32)
conv_weight = ConvRotInt8Tensor.from_hp(kernel, group_size=64, act_per_tensor_scale=input_scale)
video = torch.randn(1, 64, 2, 4, 4, device=device, dtype=torch.float16)
with torch.inference_mode():
    output = F.conv3d(video, conv_weight, padding=1)
assert output.shape == (1, 16, 2, 4, 4)

conv_layer = torch.nn.Conv3d(64, 16, 3, padding=1, bias=False, device="meta")
conv_layer.weight = torch.nn.Parameter(conv_weight, requires_grad=False)
```

For checkpoint loading, `from_quantized` takes `[out, 3, 3, 3, in]` INT8 `qdata`
and `[out]`/`[out, 1]` FP32 `scale`; `act_per_tensor_scale` can be omitted for
conversion/dequantization only. Stride accepts one or three positive integers, and
padding accepts nonnegative symmetric counts, `"valid"`, or `"same"` (stride `1`)
matching PyTorch output shapes (`padding=0` adds no hidden temporal frames).
Unbatched inputs, transposed convolutions, other kernel sizes, and BF16 autocast
are unsupported (FP16 autocast casts eligible FP32 inputs/bias while keeping FP16
output). Results approximate floating-point convolution.

For causal video convolution, pad two preceding zero frames with `F.pad` and call
`F.conv3d(..., padding=0)` (applying any spatial reflection separately). The
`ConvRotInt8Conv3d` / `conv3d` convenience APIs in `piper_kernels.conv3d.convrot.int8` add two
causal zero frames automatically, accept `padding="none"|"reflect"|"reflect_right"`
(`height, width > 1`), retain FP32 input precision under autocast, and accept an
optional FP16 residual.

Both APIs work under `torch.compile`; `compile_options(conv3d=True)`
additionally fuses explicit zero/spatial-reflection padding chains. See the
[Conv3D implementation](../src/piper_kernels/conv3d/convrot/int8/triton.py),
[padding compiler pass](../src/piper_kernels/conv3d/convrot/int8/_compile.py),
and the [MiniMax-H3 encoder specialization](../src/piper_kernels/specializations/minimax_h3_vae/conv3d/README.md).
The sections below (updates, GGUF, trellis, sharding) cover 2-D weights only.

## Compile a model

Operators work with `torch.compile` directly; pass `piper_kernels.compile_options`
through `torch.compile(options=...)` (without `mode`) to share input preparation
or fuse blocks while leaving unmatched ops unchanged:

```python
from piper_kernels import compile_options

compiled_layer = torch.compile(linear_layer, options=compile_options())
```

Repeated calls compose by unioning enabled flags into one phase-ordered compiler
pipeline (regardless of call order) while preserving unrelated Inductor options:

- **Linears (`linear=True`, default)**: shares input preparation across compatible
  ConvRot INT8, NVFP4, and ConvRot NVFP4 linears (INT8 also fuses exclusive
  `tanh`-GELU inputs) while preserving static/dynamic scales.
- **FFNs (`ffn=True`)**: fuses FP16/BF16 ConvRot INT8, NVFP4, and ConvRot NVFP4
  SwiGLU and exclusive `tanh`-GELU FFNs while preserving scales. When updating a
  residual stream, enable `residual_updates=True` (see
  [in-place residual updates](attention.md#update-a-residual-stream-in-place)).
- **Attention (`attention=True`, `attention_output=True`)**: see
  [attention](attention.md#compile-and-fuse-projections).
- **Conv3D (`conv3d=True`)**: fuses explicit zero/spatial-reflection padding
  chains before ConvRot INT8 Conv3D.

Static INT8 weights bypass AOTAutograd's persistent disk cache (Inductor caching
and graph reuse remain enabled). There is no hidden prepared-weight cache.

## Merge an adapter

All three 2-D formats support untransposed inference-only in-place updates
(`addmm_` for `beta * weight + alpha * (mat1 @ mat2)`; `add_` for
`beta * weight + alpha * update` where Piper adds optional `beta`). Updates
preserve the wrapper and activation calibration while overwriting existing
`qdata` and `scale` storage in place, so clone a weight first if other
wrappers share those checkpoint tensors and must stay unchanged. Finite real
`alpha` and `beta` default to `1.0` (supporting integers and zero coefficients,
with `alpha=0, beta=1` a no-op); inputs must share the weight's logical dtype and
device, and INT8 `addmm_` factors (`mat1`/`mat2`) aliasing `qdata` or `scale`
storage are preserved across the in-place write (copied before merging on
Triton):

```python
lora_a = torch.randn(4, int8_weight.shape[1], device=int8_weight.device, dtype=int8_weight.dtype)
lora_b = torch.randn(int8_weight.shape[0], 4, device=int8_weight.device, dtype=int8_weight.dtype)
with torch.no_grad():
    int8_weight.addmm_(lora_b, lora_a, alpha=0.1, rounding_seed=42)
```

Where supported on Triton, `addmm_` avoids a full dense update allocation, falling back
to dense PyTorch elsewhere (lower scratch does not guarantee higher speed).

Pass an unsigned 64-bit `rounding_seed` for stochastic code rounding without
touching global RNG (reproducible per backend/device only; Dynamo compiles
standard signatures without `beta`/`rounding_seed`). Repeated updates are lossy, so
restore pristine weights before unmerging and respect read-only checkpoint
buffers. Custom quantizers can reuse
[`piper_kernels.stochastic_quantization`](../src/piper_kernels/stochastic_quantization/__init__.py)
and its `.triton` module.

## Convert packed GGUF storage

Both ConvRot formats expose `from_gguf(packed_bytes, quant_type=..., group_size=...)`
and `weight.copy_from_gguf_(packed_bytes, quant_type=...)` on 2-D packed byte
tensors (omitting `quant_type` if the tensor carries that attribute). The package
does not parse GGUF files.

Conversion decodes, rotates, and quantizes without a dense temporary weight, and
`copy_from_gguf_` refills existing storage; INT8 requires Triton GPU support and
NVFP4 requires exact SM120 (unsupported devices raise). ConvRot NVFP4 accepts the
same scale and activation options as `from_hp`, including
`compute_per_tensor_scale=True`. See
[INT8](../src/piper_kernels/weights/convrot/int8/tensor.py) and
[NVFP4](../src/piper_kernels/weights/convrot/nvfp4/tensor.py) constructors.

## Encode, execute, and decode trellis-coded weights

Trellis coding packs a rotated ConvRot INT8 weight to 2–6 bits per weight for
offline export (`block_size <= 256`, independent of ConvRot `group_size`):

```python
from piper_kernels.trellis import TrellisStorage
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor

storage = TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=4.5)

# Keep packed trellis storage in memory for direct F.linear execution:
trellis_weight = ConvRotTrellisTensor.from_storage(
    storage,
    group_size=256,
    logical_dtype=weight.dtype,
)
output = torch.nn.functional.linear(input, trellis_weight)

# Or decode at load time into ordinary ConvRot INT8 storage:
int8_weight = ConvRotInt8Tensor.from_trellis(
    storage.bitstreams,  # uint8 [out, in / 256, 143]
    storage.block_scales,  # float8_e4m3fn [out, in / 256]
    storage.scale,  # float32 [out, 1]
    block_size=storage.block_size,
    group_size=256,
)
int8_weight.copy_from_trellis_(storage.bitstreams, storage.block_scales, storage.scale)
```

`ConvRotTrellisTensor` also wraps checkpoint storage zero-copy via
`ConvRotTrellisTensor.from_quantized(...)`, accepts `TrellisStorage.from_hp`
arguments directly via `ConvRotTrellisTensor.from_hp(...)`, and supports
`dequantize()`, transposed `mm`/`addmm`/`matmul`, and CPU execution; small
batches keep high-precision activations while larger batches use dynamic INT8
activation quantization on supported W8A8 backends (`F.embedding`, sharding,
FSDP2, and in-place weight updates are unsupported on `ConvRotTrellisTensor`).

A checkpoint stores `(bitstreams, block_scales, scale, block_size, group_size)`.
`TrellisStorage.from_hp` also accepts an exact `bytes_per_block` instead of
`bits_per_weight`, and `piper_kernels.trellis` exports `TrellisLayout`,
`resolve_block_size`, and `resolve_layout`. `ConvRotTrellisTensor.from_quantized`,
`ConvRotInt8Tensor.from_trellis`, and `copy_from_trellis_` accept optional
`row_widths` (`uint8 [out]`) with 1-D row-concatenated `bitstreams`
(`uint8 [total_bytes]`), where valid row widths (`2–6` code bits per weight)
and bitstream capacity are caller preconditions. `TrellisStorage.from_hp`
requires an NVIDIA SM80+ GPU and validates finite input weights. GPU decoding and
execution require a supported Triton GPU (validated on NVIDIA decode and AMD
RDNA4 decode/execution hardware, with offline compile coverage for NVIDIA direct
execution and additional AMD targets), reuse contiguous `scale` storage, and
support in-place `ConvRotInt8Tensor` refills via `copy_from_trellis_`; see
[trellis format details](../src/piper_kernels/trellis/README.md).

## Shard a quantized weight

`shard_quantized_weight` copies a contiguous untransposed 2-D shard with its own
packed storage (repacking NVFP4 padding):

```python
from piper_kernels.weights.sharding import shard_quantized_weight

row_shard = shard_quantized_weight(int8_weight, dim=0, start=0, length=64)
```

`dim=0` splits output rows at any nonempty interval; `dim=1` aligns both `start`
and `length` to ConvRot groups and 16-value NVFP4 blocks. NVFP4 accepts
flat/canonical and ordinary/swizzled scales, though execution requires its
supported layout. Direct narrow/slice
views and DTensor redistribution are unsupported. For DTensor, install the local
shard **before** `parallelize_module`:

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

Use `Shard(0)` bias for `dim=0` and `Replicate()` for `dim=1`. Sequence-sharded
(`Shard(1)`) ConvRot INT8 linears and dense (`attention_output=True`) or sparse
Piper attention fusions fold `PrepareModuleInput`/`RowwiseParallel(output_layouts=Shard(1))`
collectives automatically (and fold reduce-scatters into two-step in-place residual adds
with `residual_updates=True`); see
[sequence-parallel output collectives](../src/piper_kernels/fusions/convrot_int8_piper/README.md#sequence-parallel-output-collectives)
for alignment and fallback details. For other fused sharded blocks, keep full heads or
FFN intermediate shards local, unwrap activations around fusions, keep collectives
outside, and compare against the local **fused** output.

Contiguous untransposed 2-D `ConvRotInt8Tensor`, `PiperNVFP4Tensor`, and
`ConvRotNVFP4Tensor` weights also work with PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`)
using even `dim=0` sharding (`out_features % shard_world_size == 0`), including
`set_modules_to_forward_prefetch`, `reshard_after_forward`, `MixedPrecisionPolicy`
(updating the logical floating dtype without requantizing), `CPUOffloadPolicy` (including
`pin_memory=True`), 2-D FSDP + TP, and `compile_options(...)`. All-gather payloads stay
zero-copy for ConvRot INT8, unswizzled NVFP4, and 128-row-aligned swizzled NVFP4
(`local_rows % 128 == 0`); when swizzled `local_rows % 128 != 0`, FSDP2 hooks
unswizzle block scales for the collective and reswizzle into FSDP-managed scale storage.
For `device="meta"` initialization, install `device="meta"` quantized weight wrappers
**before** calling `fully_shard(...)` (FSDP2 detects `fsdp_pre_all_gather` during module
wrapping), then load local shards with `shard_quantized_weight(..., dim=0)`,
`DTensor.from_local(..., [Shard(0)])`, and `load_state_dict(..., assign=True)`. Under
`torch.compile(..., options=compile_options(ffn=True, residual_updates=True))`,
`fully_shard` places a graph boundary around its unshard/reshard hooks; wrap the
residual-updating FFN submodule with `fully_shard` (and commit attention's in-place
residual before entering it in hybrid TP+SP Attention + FSDP2 FFN blocks) so both
attention and FFN residual updates fold without recompiles.

## Storage and checkpoint compatibility

Same-shape `view`/`view_as`, transposes, and matching `as_strided` views share
storage, and FSDP2 setup also uses an opaque 1-D flat view handle (`view(-1)` or
`view_as` to 1-D on untransposed 2-D weights, rejecting compute ops), full-extent
`dim=0` `narrow`/`slice` aliases, even `dim=0` `split`, `new_zeros`, same-layout
`copy_`, and `is_pinned`/`pin_memory`; partial slices and other shape/layout ops
remain unsupported. Transposed weights only support
activation GEMMs (no `.contiguous()`, in-place updates, or `F.linear` reuse).
DTensor `mm`/`addmm` supports linear `alpha=beta=1` with vector bias.

Floating conversions reuse packed storage while updating the logical dtype (same
contracts under `torch.no_grad()` and `torch.inference_mode()`). Use
`load_state_dict(..., assign=True)` to keep caller-supplied storage, add
`ConvRotInt8Tensor` to `torch.serialization.safe_globals` for `weights_only=True`,
and export checkpoints with current `piper_kernels.weights` class paths (former
`piper_kernels.linear` aliases are removed; packed layouts are unchanged). See
[CHANGELOG.md](../CHANGELOG.md) when upgrading.
