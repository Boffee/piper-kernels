"""Shared launch-plan values for static-scale ConvRot INT8 convolutions."""

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
