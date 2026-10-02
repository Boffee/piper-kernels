# Dense Piper Attention internals

For the public API, layouts, and supported targets, see
[attention](../../../../docs/attention.md). This note covers the implementation
boundaries needed when changing a backend. Dispatch uses operand-device metadata;
unsupported targets retain the portable quantized reference. Preparation and
attention reductions are numerical work, not implicit input validation.

| Location | Responsibility |
|---|---|
| [_nvidia](./_nvidia) | Triton and `cp.async` Gluon kernels, execution plans, target/shape policy. |
| [_amd](./_amd) | RDNA4 Gluon attention and packed V preparation. |
| [_quantization.py](_quantization.py) | Shared FP32 K/V statistics and per-token V quantization. |
| [shared NVIDIA fragments](../kernels/piper/_nvidia) | Dense/sparse mixed-sign MMA, probability packing, accumulator rescaling. |
| [shared AMD fragments](../kernels/piper/_amd) | Dense/sparse WMMA layouts and arithmetic. |

## Numerical boundaries

Dense V has an FP32 scale per token; sparse V has one per K64 tile. Dense
attention therefore advances through K64 tiles while loading per-token scale
metadata. A full sparse route list does not reproduce dense quantization.
Q/K scale granularity is backend-specific. Non-causal V is centered in FP32 and
its mean restored in the output; causal V stays uncentered.

The recurrence uses signed INT8 QK, unsigned-by-signed INT8 PV, and FP32 online
softmax. Numerators remain in probability-code units until the final division
by 255 and the softmax denominator. The denominator uses unrounded FP32
probabilities. Preserve the all-zero scale guards and handle zero QK scales
before applying masked infinities.

GQA/MQA maps each query head to its KV head without repeating K/V storage.
Query windows retain global row coordinates for causal masking. Changes to
preparation, layouts, or rounding must also cover the
[projection-fusion consumers](../../fusions/convrot_int8_piper/README.md).

## NVIDIA scheduling

Full query tiles and a ragged final tile share one launch. Full tiles use
unmasked access; the tail uses masked Q pointer loads and output stores, even
when full tiles use descriptors. Optimized causal traversal starts with the
longest prefixes. Quantization and statistics use separate preparation launches.

Exact SM120 uses Triton. Its execution plan chooses descriptor loads, tiling,
pipeline stages, and causal ordering from host metadata. Descriptor setup must
be amortized by traversal work; rectangular attention uses actual K length.
Output strides matter for fused output buffers. The causal strided D64 schedule
interleaves heads across query-tile groups and bounds ragged-kernel registers
to preserve occupancy. D128 splits PV into two FP32 D64 accumulators.

Exact SM89 uses the Gluon kernel, staging Q/K/V and per-key V multipliers
through `cp.async` commit groups. It retains per-thread Q/K scales, derives the
V log-scale bound, and masks the final K tile plus the causal diagonal when
needed. D64 quantizes Q in the prologue; D128 uses a smaller query tile and a
register cap to retain occupancy. Ragged lengths reuse one compiled kernel.

The plan's `attention_kernel` explicitly selects `triton` or `gluon_async_copy`.
Capability checks belong to the plan. Exact thresholds and launch choices live
in [_nvidia/policy.py](_nvidia/policy.py); the
[offline tuner](../../../../benchmarks/tune_piper_attention.py) can compare
implementations without changing production dispatch.

## RDNA4 behavior

The Gluon implementation uses four wave32 warps and Q64/K64 tiles, with Q32/K64
scale groups. It supports causal attention and non-causal rectangular attention,
ragged tails, outer strides, and GQA/MQA. V preparation writes initialized padding directly into packed WMMA
storage, avoiding a full-size transpose/repacking allocation.

Softmax processes 16-column fragments, combining score scaling with FP32 maxima
and exponential arguments. Probability rounding still uses the logical K64
tile's maximum and each token's V scale. Interior tiles omit elementwise masks;
key tails and causal diagonals retain them. AMD uses round-to-nearest-even
probability packing. FMA/reduction reassociation and backend rounding mean
cross-backend bitwise equality is not promised.

`gfx1201` has hardware coverage on RX 9070 XT; `gfx1200` has offline compilation
coverage. This does not establish exhaustive performance tuning. Use the shared
[attention benchmark](../../../../benchmarks/README.md#attention-and-projection-fusion)
to compare complete operators and quality on the target device. Backend tests
live under `tests/attention/piper_attention`, with GQA and shared-fragment tests
alongside them. Follow the [validation contract](../../../../docs/development.md#validation-contract)
when changing dispatch, fake execution, or tensor checks.
