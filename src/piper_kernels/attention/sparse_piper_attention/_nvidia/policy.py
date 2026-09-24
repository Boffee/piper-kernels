"""NVIDIA SM89 and SM120 support and execution policy for sparse Piper attention.

Both targets run the same paired-K128 recurrence with mixed-sign MMAv2. SM120
loads operands with TMA; SM89 has no TMA and uses Ampere-style ``cp.async``.
"""

from piper_kernels._triton.targets import AcceleratorTarget


def uses_tensor_descriptors(target: AcceleratorTarget) -> bool:
    """Select the TMA kernel on exact NVIDIA SM120; HIP capability tuples are not CUDA."""
    return target.is_cuda_capability(12, 0)


def uses_async_copies(target: AcceleratorTarget) -> bool:
    """Select the ``cp.async`` kernel on exact NVIDIA SM89 (Ada), which lacks TMA."""
    return target.is_cuda_capability(8, 9)


def supports_target(target: AcceleratorTarget) -> bool:
    """Enable exact NVIDIA SM89 and SM120 sparse attention and its routing kernels."""
    return uses_tensor_descriptors(target) or uses_async_copies(target)


def skip_dense_routing(head_dim: int) -> bool:
    """Choose direct full-keep traversal for D64; D128 is faster with route lists.

    Measured on both SM120 and SM89, where skipping the route list is 4 to 5
    percent faster for full-keep D64 attention.
    """
    return head_dim == 64


def select_attention_schedule(
    head_dim: int,
    query_rows: int,
    key_rows: int,
    *,
    skip_dense_routing: bool,
    has_coarse_residual: bool,
    selected_key_rows: int,
) -> tuple[int, int]:
    """Select measured D64 schedules; retain Q64/four warps for D128.

    Crossover checks cover 32/56 heads at 8k, 16k, and 32k rows. Small
    query chunks and coarse residuals keep their existing query granularity.
    Very small per-head key budgets retain four warps even on long sequences.
    Selected key rows are the rounded-down head average, including dense keys.
    """
    if head_dim != 64:
        return 64, 4
    if skip_dense_routing:
        if min(query_rows, key_rows) >= 32768 and not has_coarse_residual:
            return 128, 4
        return 64, 4
    if min(query_rows, key_rows) >= 8192 and selected_key_rows >= 1024:
        return 64, 2
    return 64, 4


def use_fused_preparation(head_dim: int, sequence_length: int) -> bool:
    """Use fused Q/K summaries above the measured SM120 short-row crossover."""
    return sequence_length >= (2048 if head_dim == 64 else 1024)


# One 128-thread CTA per 21,504 registers lets three share an Ada SM.
_SM89_D64_MAX_REGISTERS = 168


def sm89_max_registers(head_dim: int) -> int | None:
    """Return the per-thread register cap for SM89's Q64 four-warp launch, if any.

    SM89's ``cp.async`` addressing lifts its D64 kernel to 179-183 registers,
    which fits two CTAs per Ada SM; capping it at 168, where SM120's kernel
    already sits, fits three. D128 is held to two CTAs by shared memory anyway.
    """
    return _SM89_D64_MAX_REGISTERS if head_dim == 64 else None
