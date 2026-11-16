"""Metadata-only dense output windows under a caller-supplied storage cap."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._nvidia.policy import select_execution_plan


def _balanced_query_chunk_rows(sequence: int, maximum: int, alignment: int = 128) -> int:
    """Choose the smallest aligned uniform window with the minimum chunk count."""
    chunks = (sequence + maximum - 1) // maximum
    rows = (sequence + chunks - 1) // chunks
    return (rows + alignment - 1) // alignment * alignment


def _wave_query_chunk_rows(
    sequence: int,
    maximum: int,
    *,
    block_rows: int,
    parallel_heads: int,
    concurrent_blocks: int,
    alignment: int = 128,
) -> int:
    """Avoid extra partially occupied waves without exceeding the validated cap."""
    balanced = _balanced_query_chunk_rows(sequence, maximum, alignment)
    if sequence <= maximum or concurrent_blocks < 1:
        return balanced
    waves = (maximum // block_rows * parallel_heads) // concurrent_blocks
    wave_rows = waves * concurrent_blocks // parallel_heads * block_rows
    wave_rows = wave_rows // alignment * alignment
    if wave_rows < alignment:
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
