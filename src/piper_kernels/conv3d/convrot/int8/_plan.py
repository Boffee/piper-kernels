"""Shared launch-plan values for static-scale ConvRot INT8 convolutions."""

from dataclasses import dataclass
from typing import NamedTuple


class ConvolutionPlan(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


class PreparationPlan(NamedTuple):
    block_m: int
    num_warps: int


@dataclass(frozen=True, slots=True)
class ConvolutionExecutionPlan:
    """Concrete preparation, convolution, and weight-load choices for one operation."""

    preparation: PreparationPlan
    convolution: ConvolutionPlan
    use_weight_descriptor: bool
