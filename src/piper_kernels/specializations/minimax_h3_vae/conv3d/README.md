# MiniMax-H3 encoder Conv3D specialization

This opt-in specialization fuses framewise normalization around prequantized H3
encoder convolutions. See [Conv3D usage](../../../../../docs/weights.md#run-a-conv3d)
for weight construction and the [shared implementation](../../../conv3d/convrot/int8/README.md)
for convolution storage, arithmetic, and backend support.

## Calibration and installation

`piper_kernels.specializations.minimax_h3_vae.conv3d.P995_INPUT_SCALES` provides
candidate encoder scales; [calibration.py](calibration.py) records their assumptions
and provenance. The application owns weight installation and checkpoint validation.

Keep original forwards and ordinary convolution modules, installing prequantized
weights without replacing the layers. H3's explicit temporal zero padding and
spatial reflection are absorbed into the shared convolution when the graph
matches. For existing causal wrappers, retain any caller-side bottom/right
reflection and select spatial `padding="none"`; use `reflect_right` only if that
caller-side pad is absent.

## Compile the encoder

After installing prequantized weights, compile the encoder for inference:

```python
from piper_kernels.specializations.minimax_h3_vae import (
    minimax_h3_vae_convrot_int8_conv3d_compile_options,
)

encoder.compile(
    dynamic=True,
    fullgraph=True,
    options=minimax_h3_vae_convrot_int8_conv3d_compile_options(),
)
```

Run under `torch.no_grad()` or inference mode. Encoder options are independent
of, and composable with, decoder options.

The helper fuses exclusive framewise GroupNorm, SiLU, and optional residual
around compatible ConvRot convolutions, including their padding. It leaves
unmatched operations unchanged and does not quantize floating-point weights.
Fusion removes intermediate FP16 rounding, including a Conv3D autocast conversion.

Batch, time, and spatial dimensions may be symbolic; channels and model
configuration remain fixed. Compatible activation sizes and tile-view strides
reuse graphs, while singleton dimensions or different layout constraints can
require separate graphs. Native kernels may still compile for new shapes.

Fusion retains a full INT8 activation buffer. Validate full-encoder quality,
memory, and latency with the actual checkpoint and application.
