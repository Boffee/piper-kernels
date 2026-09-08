"""Hardware coverage gates for the currently validated sparse-fusion backend."""

import torch

from piper_kernels.fusions.convrot_int8_sparse_piper import _backend


def projection_available(head_dim: int = 128) -> bool:
    return (
        torch.cuda.is_available()
        and _backend.select_projection_backend(torch.empty(0, device="cuda"), head_dim=head_dim)
        is not None
    )


def output_available(head_dim: int = 128) -> bool:
    return (
        torch.cuda.is_available()
        and _backend.select_output_backend(torch.empty(0, 0, 0, head_dim, device="cuda"))
        is not None
    )
