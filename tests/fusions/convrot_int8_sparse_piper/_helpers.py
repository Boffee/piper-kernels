"""Hardware coverage gates for the currently validated sparse-fusion backend."""

from _accelerators import (
    convrot_int8_dense_projection_available,
    convrot_int8_sparse_output_available,
    convrot_int8_sparse_projection_available,
)


def projection_available(head_dim: int = 128) -> bool:
    return convrot_int8_sparse_projection_available(head_dim)


def dense_projection_available(head_dim: int = 128) -> bool:
    """Dense Piper projections cover fewer targets than the sparse fusion (not SM8x)."""
    return convrot_int8_dense_projection_available(head_dim)


def output_available(head_dim: int = 128) -> bool:
    return convrot_int8_sparse_output_available(head_dim)
