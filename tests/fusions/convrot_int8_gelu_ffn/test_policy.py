"""Metadata-only policy tests for ConvRot INT8 GELU FFNs."""

import pytest
import torch

from piper_kernels.fusions.convrot_int8_gelu_ffn.triton import _default_chunk_rows


@pytest.mark.parametrize(
    ("output_features", "adds_residual"),
    [(5_120, False), (16_384, True)],
)
def test_default_chunk_rows_bounds_reusable_workspace(
    output_features: int,
    adds_residual: bool,
) -> None:
    input = torch.empty(100_000, 5_120, dtype=torch.bfloat16, device="meta")  # noqa: A001
    up = torch.empty(13_824, 5_120, dtype=torch.int8, device="meta")
    down = torch.empty(output_features, 13_824, dtype=torch.int8, device="meta")

    rows = _default_chunk_rows(input, up, down, adds_residual=adds_residual)
    projected_output_bytes = (
        output_features * input.element_size() if adds_residual and output_features > 13_824 else 0
    )
    bytes_per_row = (
        13_824 * input.element_size() + 13_824 + torch.float32.itemsize + projected_output_bytes
    )

    assert rows % 128 == 0
    assert rows * bytes_per_row <= 1_024**3
    assert (rows + 128) * bytes_per_row > 1_024**3
