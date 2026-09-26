"""Metadata checks for dense Piper's caller-owned output storage."""

import torch


def validate_output_buffer(
    output: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Require the contiguous logical output expected by the dense kernels."""
    if (
        output.layout is not torch.strided
        or output.shape != shape
        or output.dtype is not dtype
        or output.device != device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Piper Attention output must match the query shape, dtype, and device "
            "and use contiguous strided storage"
        )
