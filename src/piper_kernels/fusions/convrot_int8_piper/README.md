# ConvRot INT8 query fusion for dense Piper

Use the opt-in compiler pass around a model that calls `piper_attention`:

```python
import torch
from piper_kernels.fusions.convrot_int8_piper import convrot_int8_piper_compile_options

compiled = torch.compile(
    model,
    fullgraph=True,
    options=convrot_int8_piper_compile_options(),
)
```

The pass recognizes a ConvRot INT8 Q projection followed by per-head RMSNorm,
split-half RoPE, and the transpose into dense Piper's BHSD layout. It combines
projection, normalization, rotation, signed-Hadamard smoothing, and Q32 INT8
quantization. The resulting custom operator consumes quantized Q and floating
K/V. K retains its sequence-wide post-transform mean, and V retains dense
per-token scales and causal centering behavior.

The integration supports FP16/BF16 attention, D64/D128, grouped query heads,
ragged sequences, and causal or rectangular non-causal attention. Projection
kernels target NVIDIA SM120 and AMD RDNA4; unsupported targets or unmatched
graph patterns retain the original operations. RMSNorm may be affine or
non-affine, RoPE may cover part or all of the head, and projection bias is
supported.

As in the existing sparse projection fusions, projection, RMSNorm, and RoPE
intermediates stay in FP32 until Q quantization. Results can therefore differ
slightly from a graph that materializes each intermediate in FP16/BF16.

The compiler-visible boundaries are
`piper_kernels::convrot_int8_piper_project_query` and
`piper_kernels::piper_attention_from_quantized_query`. The latter accepts
contiguous Q64-padded INT8 storage and contiguous FP32 Q32 scale groups, which
already include the softmax scale and `log2(e)`. Fake execution validates host
metadata and constructs only output tensors; it does not select hardware or
inspect numerical tensor contents.

This pass installs before the ordinary ConvRot INT8 compiler pass and preserves
caller-supplied compiler options. K/V projection fusion and output-projection
fusion remain separate follow-up work.
