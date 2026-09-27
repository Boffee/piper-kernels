"""Shared schedules and execution plans for static-scale ConvRot INT8 convolutions."""

from dataclasses import dataclass
from typing import NamedTuple


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
