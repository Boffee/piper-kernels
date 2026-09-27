"""Validated NVIDIA projection and chunked output integration targets."""

from piper_kernels._triton.targets import AcceleratorTarget

# SM89 splits the fused attention output into 8,192-row query windows. Against the shared
# 4,096 rows, the H3 fusion benchmark ran 1.4-3.0% faster at 8K-150K tokens for 61-238 MiB
# more workspace (see benchmarks/README.md).
SM89_QUERY_CHUNK_ROWS = 8_192


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native sparse Piper attention targets: exact SM120 and SM89."""
    return target.is_cuda_capability(12, 0) or is_sm89(target)


def is_sm89(target: AcceleratorTarget) -> bool:
    """SM89 has its own projection kernels and output chunks; SM120 keeps the shared ones."""
    return target.is_cuda_capability(8, 9)
