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
