"""Launch-grid ordering shared by head-major attention preparation kernels."""

from __future__ import annotations

import torch

# CUDA caps gridDim.y at 65,535; keep larger block counts on gridDim.x.
_MAX_GRID_DIM_Y = 65_535


def select_block_grid(tensor: torch.Tensor, blocks: int) -> tuple[tuple[int, int, int], int]:
    """Order inner grid axes to follow the tighter of block and head strides."""
    batch, heads = tensor.shape[0], tensor.shape[1]
    heads_inner = int(
        heads > 1 and blocks <= _MAX_GRID_DIM_Y and tensor.stride(2) > tensor.stride(1)
    )
    grid = (heads, blocks, batch) if heads_inner else (blocks, heads, batch)
    return grid, heads_inner
