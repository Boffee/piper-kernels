"""Activation and output dtypes supported by sparse Piper attention."""

import torch

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_output_dtype(dtype: torch.dtype) -> None:
    """Validate the floating output types supported by the kernels."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("sparse Piper output dtype must be float16, bfloat16, or float32")
