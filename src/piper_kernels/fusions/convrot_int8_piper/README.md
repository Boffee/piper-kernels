# ConvRot INT8 projection fusion for dense Piper

Use the opt-in compiler pass around a model that calls `piper_attention`:

```python
import torch
from piper_kernels.fusions.convrot_int8_piper import convrot_int8_piper_compile_options

compiled = torch.compile(
    model,
    fullgraph=True,
    options=convrot_int8_piper_compile_options(fuse_output=True),
)
```

The pass fuses ConvRot INT8 projection, per-head RMSNorm, split-half RoPE,
signed-Hadamard smoothing, and Q32 INT8 quantization. When K follows the same
normalized RoPE pattern and V is a plain ConvRot projection, it also fuses their
preparation. Projections sharing an input and static-scale setting reuse ConvRot
input preparation. Unsupported K/V transforms or escaping operands retain the
Q-only path.

`fuse_output=True` additionally folds Q projection, attention, and an exclusive
ConvRot INT8 output projection into a bounded pipeline. It recognizes the BHSD
to BSHD transpose followed by merging heads. Output fusion defaults to disabled;
Q/K/V preparation fusion remains enabled. This is a pipeline of GPU kernels
behind compiler boundaries, not a single kernel. The pass runs before ordinary
ConvRot INT8 optimizations and preserves caller-supplied compiler options.

## Supported inputs and numerical contract

The integration supports FP16/BF16, D64/D128, MHA/GQA, ragged sequences, and causal
or rectangular non-causal attention. Projection backends target NVIDIA SM120 and
AMD RDNA4. Unsupported targets and unmatched graph patterns retain the original
operations. RMSNorm may be affine or non-affine, RoPE may cover part or all of a
head, and projections may have bias. Output preparation supports dynamic
per-token scales and a supplied static scale.

Projection, RMSNorm, and RoPE arithmetic stay in FP32. K uses BF16 temporary
storage and FP32 tile sums of those represented values; it reduces the global
post-transform mean and centers K in FP32 before K64 quantization. Dense and
sparse Piper use the same K producer, mean reduction, and quantization path.
Sparse routing summaries retain the uncentered FP32 transform values. Q and V
retain FP32 intermediates until quantization. These rounding boundaries can
produce different results from materializing each operation in FP16/BF16 or
keeping a full FP32 K temporary. Non-causal V projects the represented input
mean before per-token quantization; causal V skips that reduction and remains
uncentered.

The internal producer boundaries are
`piper_kernels::convrot_int8_piper_project_query`,
`convrot_int8_piper_project_key`, and `convrot_int8_piper_project_value`.
`piper_attention_from_quantized_query` consumes quantized Q with floating K/V;
`piper_attention_from_quantized` consumes all prepared operands.

Q uses contiguous Q64-padded INT8 storage and FP32 Q32 scales including the
softmax scale and `log2(e)`. K/V use K64-padded storage. K has FP32 K64 scales;
V has FP32 per-token multipliers and log scales. V codes use transposed storage
on NVIDIA and packed WMMA tiles on RDNA4. NVIDIA log scales preserve the existing
FP16 rounding. Numerical contents are producer preconditions. Validation reads
host metadata only; fake execution allocates only outputs and does not select
hardware or inspect tensor contents. Transformed K must remain finite in BF16
storage; this numerical precondition is not checked at runtime.

## Shared implementation and bounded storage

Dense and sparse Piper share ConvRot projection storage validation, the
RMSNorm/RoPE projection tile, projection configuration and tile indexing, the
projected-mean kernel, output validation and chunk projector, compiler tuple
matching, and the attention-to-output stream pipeline.
`convrot_int8_centered_projection` provides BF16 storage and FP32 statistics for
projection tiles independently of their Q/K/V role. The shared
`convrot_int8_sage_qk.key` adapter owns K transforms and centered K64 encoding,
with optional FP32 sparse routing summaries. Dense
per-token V quantization remains separate from sparse V tile scales.
Both producers use typed backend methods with caller-owned buffers; target
configurations live in separate NVIDIA and RDNA4 modules. As in sparse Piper,
`_kernels.py` contains device kernels and `triton.py` owns their launchers.
Compiler matching uses optional backend selection; validated execution requires
a supported backend through `_backend.py`.

K/V stay global. Q uses one reusable buffer with global RoPE positions and causal
row coordinates. When the output width is at least the merged attention width,
chunks temporarily occupy unwritten final output rows. The shared ConvRot INT8
projector reads a batch's whole attention chunk into separate INT8 preparation
storage before overwriting those rows. Narrower outputs retain up to two
attention buffers. Other projector types must explicitly support this
read-before-write guarantee to use output storage.

Dense output fusion defaults to an 8192-row chunk cap shared with sparse Piper.
Balanced windows use the smallest 128-aligned uniform size preserving the
minimum chunk count. SM120 non-causal attention also considers windows below
a GPU scheduling-wave boundary, using the SM count and measured two resident
CTAs per SM. It chooses such a window only if it reduces total predicted waves,
including the remainder.
Causal attention and other targets use balanced windows. `query_chunk_rows` on
the internal output operator overrides the cap. Sparse Piper uses fixed
8192-row windows. This default applies to NVIDIA and AMD targets. These rules
do not depend on model identities or benchmark sequence ranges.

The final output, prepared input, and global K/V remain full size. Q and output
preparation scratch are bounded by the selected window. K preparation still
requires a global BF16 temporary plus compact FP32 mean-reduction storage.
This halves the K temporary relative to FP32, though another stage may still
determine peak allocation.

## Performance and limitations

With BF16 K storage, paired RTX 5090 H3 measurements at 32,769, 65,537, and
100,001 rows put the 8192-row cap within 0.3-1.3% of the 16384-row cap. At
100,001 rows, it saved about 352 MiB of peak extra allocation. Sparse Piper at
the same length and 20% keep with mean routing was about 1.1% slower with fixed
8192-row windows than with 16384 rows, saving about 387 MiB. These synthetic
BF16 B1/H56/D128, width-5376 measurements include the full fused pipeline;
allocation figures include output/workspace and exclude resident inputs and
weights. The 8192-row default balances the measured latency and memory costs;
explicit overrides remain available for other workloads.

The measurements below predate BF16 K temporary storage and the shared 8192-row
default. Output fusion has no automatic profitability guard. RTX 5090 synthetic
BF16 D128 benchmarks using H3 and Krea2 shapes reached roughly parity or modest speedups
against Q/K/V fusion plus a separate output projection. Controls covered nearby
head counts, batch sizes 1 and 2, FP16/BF16, causal/non-causal attention, and
aligned/ragged lengths. Outputs matched the Q/K/V-fused baseline exactly in
ordinary execution and CUDA graph replay. RDNA4 has compiler coverage; these
performance results are NVIDIA measurements.

Krea2-shaped measurements used 48 Q heads, 12 K/V heads, D128, width 6144, full
RoPE, and non-causal attention. They exclude the model's text padding mask and
sigmoid output gate, so they do not establish full-model integration. At 100001
rows, output storage reuse reduced peak extra allocation from 2.57 GiB to
2.20 GiB, versus 2.86 GiB for Q/K/V fusion with separate output projection.
At 32769 rows, the corresponding values were 1.22, 0.85, and 0.94 GiB. These
measurements exclude resident inputs and weights and include outputs/workspace.
Reuse left paired timing essentially unchanged. It does not always lower the
peak: at 16385 rows, full fusion used 0.52 GiB versus 0.47 GiB. H3's narrower
output retains separate attention buffers.

A paired chunk-cap sweep found the existing scheduler with a 4096-row cap within
about 0.7-2.1% of the 16384-row cap across the measured H3/Krea2 cases. Smaller
windows save scratch, but may not reduce peak allocation when K preparation
already dominates. Compare caps on the intended workload rather than assuming
the same tradeoff everywhere.

D64 uses descriptor loads when K traversal amortizes setup. Causal strided D64
also groups query tiles across heads and limits ragged-kernel registers to
preserve occupancy. These improvements do not remove every output-fusion gap:
measured non-causal D64 low-head GQA cases still trailed by roughly 3-10%, and
longer four-head causal GQA cases by about 13%. The bounded pipeline does not
guarantee throughput parity for every shape.

Use `benchmarks/benchmark_piper_fusion.py` for paired timings, exact-output
checks, memory measurements, and chunk-cap comparisons on your device.
