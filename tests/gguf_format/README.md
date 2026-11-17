# GGUF decoder reference data

`reference.zip` contains the finite packed inputs and independently decoded
FP32 values for the ConvRot INT8 and NVFP4 numerical tests. Each entry uses the
encoding, logical shape, and CPU random seed in its name. Its two members
contain packed bytes and little-endian FP32 values in row-major order.

The input generator is shared with the synthetic packed-data tests. The
expected values come from llama.cpp's Python `gguf.dequantize` in release
0.19.0. The cases retain the earlier seeds, shapes, and encoding coverage.
The BF16 test without a global scale uses an explicit seed for both scale
layouts.

Regenerate the archive with the locked test stack and the pinned upstream
decoder in a temporary uv environment:

```shell
uv run --locked --with 'gguf==0.19.0' python tests/gguf_format/regenerate_reference.py
```
