"""Workspace and mean reduction for projections stored before global centering.

Consumers compose ConvRot projection tiles with their own transforms, then call
``store_projection_tile`` inside the producer. This layer owns BF16 storage and
FP32 statistics; it imposes no Q/K/V transforms, routing, or quantization policy.
"""

# pyright: reportCallIssue=false, reportArgumentType=false

import torch
import triton

from piper_kernels._triton import reductions
from piper_kernels._triton.runtime import device_context

from . import _kernels


def source_files() -> tuple[str, ...]:
    return tuple(
        path for path in (__file__, _kernels.__file__, reductions.__file__) if path is not None
    )


def allocate_workspace(
    input: torch.Tensor,  # noqa: A002
    shape: tuple[int, int, int, int],
    *,
    tile_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate [batch, group, row, feature] storage, tile sums, and global means.

    Callers provide a storage length divisible by their statistics tile size.
    """
    batch, groups, storage_length, features = shape
    stored = input.new_empty(shape, dtype=torch.bfloat16)
    partials = input.new_empty(
        (batch, groups, storage_length // tile_rows, features), dtype=torch.float32
    )
    mean = input.new_empty((batch, groups, features), dtype=torch.float32)
    return stored, partials, mean


def finalize_mean(
    partials: torch.Tensor,
    sequence_length: int,
    *,
    out: torch.Tensor,
) -> None:
    """Reduce represented-value sums on the partials' device.

    Zeroed internal padding contributes to the logical denominator; storage
    padding beyond ``sequence_length`` does not.
    """
    batch, groups, num_chunks, features = partials.shape
    with device_context(partials.device):
        _kernels._mean_finalize_kernel[(batch * groups, triton.cdiv(features, 64))](
            partials,
            out,
            sequence_length,
            num_chunks,
            features=features,
            block_chunks=triton.next_power_of_2(num_chunks),
            block_d=64,
            num_warps=4,
        )
