"""SM120 support and execution policy for sparse Piper attention."""

from piper_kernels._triton.targets import AcceleratorTarget


def supports_target(target: AcceleratorTarget) -> bool:
    """Enable only exact NVIDIA SM120; HIP capability tuples are not CUDA."""
    return target.is_cuda_capability(12, 0)


def skip_dense_routing(head_dim: int) -> bool:
    """Choose direct full-keep traversal for D64; D128 is faster with route lists."""
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
