"""Metadata-only policy tests for NVFP4 GELU FFNs."""

import math
from typing import cast

import pytest
import torch

from piper_kernels.fusions.nvfp4_gelu_ffn.triton import _default_chunk_rows
from piper_kernels.weights.nvfp4 import _layout


def _prepared_bytes_per_row(features: int) -> int:
    rows = _layout.SCALE_ROW_TILE
    qdata_shape = cast(tuple[int, int], _layout.qdata_shape(rows, features))
    scale_shape = cast(tuple[int, int], _layout.scale_shape(rows, features))
    return (math.prod(qdata_shape) + math.prod(scale_shape)) // rows


@pytest.mark.parametrize(
    ("input_features", "intermediate_features", "output_features", "gated_updates"),
    [(5_120, 13_824, 5_120, False), (80, 144, 192, True)],
)
def test_default_chunk_rows_bounds_reusable_workspace(
    input_features: int,
    intermediate_features: int,
    output_features: int,
    gated_updates: bool,
) -> None:
    input = torch.empty(100_000, input_features, dtype=torch.bfloat16, device="meta")  # noqa: A001
    up = torch.empty(
        intermediate_features,
        input_features // 2,
        dtype=torch.uint8,
        device="meta",
    )
    down = torch.empty(
        output_features,
        intermediate_features // 2,
        dtype=torch.uint8,
        device="meta",
    )

    rows = _default_chunk_rows(input, up, down, gated_updates=gated_updates)
    projected_output_bytes = (
        output_features * input.element_size()
        if gated_updates and output_features > intermediate_features
        else 0
    )
    bytes_per_row = (
        _prepared_bytes_per_row(input_features)
        + intermediate_features * input.element_size()
        + _prepared_bytes_per_row(intermediate_features)
        + projected_output_bytes
    )

    assert rows % 128 == 0
    assert rows * bytes_per_row <= 512 * 1_024**2
    assert (rows + 128) * bytes_per_row > 512 * 1_024**2
