"""Metadata-only dense output windows under a caller-supplied storage cap."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._nvidia.policy import select_execution_plan

_ALIGNMENT = 128


def _balanced_query_chunk_rows(sequence: int, maximum: int) -> int:
    """Choose the smallest aligned uniform window with the minimum chunk count."""
    chunks = (sequence + maximum - 1) // maximum
    rows = (sequence + chunks - 1) // chunks
    return (rows + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _wave_query_chunk_rows(
    sequence: int,
    maximum: int,
    *,
    block_rows: int,
    parallel_heads: int,
    concurrent_blocks: int,
) -> int:
    """Avoid extra partially occupied waves without exceeding the validated cap.

    A noncausal CTA traverses all K tiles, so a few CTAs spilling into an extra
    scheduling wave can cost nearly another full traversal. Compare balanced
    windows with the largest aligned window below a whole-wave boundary. Use
    the latter only when it reduces total waves, including the final window;
    ties retain balanced storage and its minimum launch count.
    """
    balanced = _balanced_query_chunk_rows(sequence, maximum)
    if sequence <= maximum:
        return balanced
    waves = (maximum // block_rows * parallel_heads) // concurrent_blocks
    wave_rows = waves * concurrent_blocks // parallel_heads * block_rows
    wave_rows = wave_rows // _ALIGNMENT * _ALIGNMENT
    if wave_rows == 0:
        return balanced

    def total_waves(rows: int) -> int:
        full, tail = divmod(sequence, rows)

        def window_waves(length: int) -> int:
            blocks = (length + block_rows - 1) // block_rows * parallel_heads
            return (blocks + concurrent_blocks - 1) // concurrent_blocks

        return full * window_waves(rows) + window_waves(tail)

    return wave_rows if total_waves(wave_rows) < total_waves(balanced) else balanced


def select_query_chunk_rows(
    query_shape: tuple[int, int, int, int],
    device: torch.device,
    maximum: int,
    *,
    is_causal: bool,
) -> int:
    """Use calibrated occupancy only for the corresponding device/kernel plan.

    Shapes and the 128-aligned cap are already validated. This function reads
    hardware metadata only; it does not compile or launch a calibration kernel.
    Other backends and causal attention retain balanced windows.
    """
    batch, heads, sequence, head_dim = query_shape
    balanced = _balanced_query_chunk_rows(sequence, maximum)
    if sequence <= maximum or is_causal:
        return balanced
    target = AcceleratorTarget.from_device(device)
    if not target.is_nvidia_cuda:
        return balanced
    plan = select_execution_plan(
        target, head_dim=head_dim, is_causal=is_causal, query_length=sequence
    )
    if not plan.output_ctas_per_sm:
        return balanced
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    return _wave_query_chunk_rows(
        sequence,
        maximum,
        block_rows=plan.block_m,
        parallel_heads=batch * heads,
        concurrent_blocks=multiprocessors * plan.output_ctas_per_sm,
    )
