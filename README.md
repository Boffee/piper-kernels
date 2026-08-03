# Piper Kernels

Reusable PyTorch inference operators and optimized kernels for the Piper ecosystem and
other consumers.

The package owns operator semantics, portable PyTorch references, tensor subclasses,
and optimized backends. It deliberately does not know about model repositories,
checkpoint metadata, pipeline frameworks, or device-offloading policy.

## Planned operators

| Package | Role |
|---|---|
| `piper_kernels.convrot` | Rotated INT8 W8A8 tensor and linear operator |
| `piper_kernels.attention` | Attention operators, including a future SageAttention backend |

## ConvRot

Wrap checkpoint storage without dequantizing it, then use the resulting tensor as a
normal linear weight:

```python
import torch

from piper_kernels.convrot import ConvRotInt8Tensor

weight = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=64)
output = torch.nn.functional.linear(activation, weight, bias)
```

The operator selects its Triton implementation on supported CUDA devices and otherwise
uses the portable PyTorch reference. Install the tensor format and optimized backend with
`piper-kernels[convrot,triton]`. The base package does not require TorchAO or Triton, so
future attention-only consumers do not inherit quantization-specific dependencies.

## Dependency direction

Applications such as Piper consume this package. Integrations such as torch-offload may
optionally recognize its tensor types, but `piper-kernels` does not depend on either
project.

## Development

```shell
uv sync --dev
uv run pytest
uv run ruff check .
uv run pyright
uv build
```

GPU tests use the `gpu` pytest marker. The pre-commit test hook hides CUDA so commits run
the portable suite; run `uv run pytest` directly to exercise installed GPU backends.
