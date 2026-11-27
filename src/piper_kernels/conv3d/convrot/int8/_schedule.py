"""Shared Triton-free schedule dataclasses and shape helpers for ConvRot INT8 Conv3D."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

type Padding = tuple[int, int, int, int, int, int]


class ConvolutionSchedule(NamedTuple):
    """Complete immutable convolution tile and launch choices."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


class PreparationSchedule(NamedTuple):
    """Rotation/quantization tile and warp count."""

    block_m: int
    num_warps: int


@dataclass(frozen=True, slots=True)
class ConvolutionExecutionPlan:
    """Concrete preparation, convolution, and weight-load choices for one operation."""

    preparation: PreparationSchedule
    convolution: ConvolutionSchedule
    use_weight_descriptor: bool
    # Fence only LDS at barriers; an RDNA workaround in _triton.amd_compiler_workarounds.
    lds_only_fences: bool = False
    accumulator_seed: int = 0


def tile_count(rows: int, outputs: int, block_m: int, block_n: int) -> int:
    """Return the grid tile count for ``(rows, outputs)`` at ``(block_m, block_n)``."""
    return ((rows + block_m - 1) // block_m) * ((outputs + block_n - 1) // block_n)


def output_shape(
    input_shape: Sequence[int],
    output_channels: int,
    stride: Sequence[int],
    padding: Padding,
) -> tuple[int, int, int, int, int]:
    """Compute the NCTHW output shape for a 3x3x3 convolution with F.pad-order padding."""
    batch, _, frames, height, width = input_shape
    return (
        batch,
        output_channels,
        (frames + padding[4] + padding[5] - 3) // stride[0] + 1,
        (height + padding[2] + padding[3] - 3) // stride[1] + 1,
        (width + padding[0] + padding[1] - 3) // stride[2] + 1,
    )
