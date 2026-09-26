"""Validated NVIDIA projection and chunked output integration targets."""

from piper_kernels._triton.targets import AcceleratorTarget

# Query rows per fused attention-output chunk on SM89. With the shared 4,096-row chunks, the
# compiled H3 attention block ran 3% slower at 100K tokens than without the fusion; 8,192
# rows remove that loss and keep most of the memory saving.
SM89_QUERY_CHUNK_ROWS = 8_192


def supports_target(target: AcceleratorTarget) -> bool:
    """Match the native sparse Piper attention targets: exact SM120 and SM89."""
    return target.is_cuda_capability(12, 0) or is_sm89(target)


def is_sm89(target: AcceleratorTarget) -> bool:
    """SM89 projects with its own measured configuration; SM120 keeps the original one."""
    return target.is_cuda_capability(8, 9)
