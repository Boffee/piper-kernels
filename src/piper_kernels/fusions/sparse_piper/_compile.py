"""Shared compiler validation for projected sparse Piper attention."""

from __future__ import annotations

import math

import torch
from torch._inductor.pattern_matcher import Match
from torch.fx.node import Argument

from piper_kernels.attention.sparse_piper_attention import _budget
from piper_kernels.linear import _preparation_sharing as preparation_sharing


def static_int(value: object) -> int | None:
    """Return a non-boolean static integer."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def integer_scalar_metadata(value: object) -> int | torch.SymInt | None:
    """Resolve static or symbolic integer metadata from an FX argument."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, torch.SymInt)):
        return value
    if isinstance(value, torch.fx.Node):
        metadata = value.meta.get("val")
        if isinstance(metadata, (int, torch.SymInt)) and not isinstance(metadata, bool):
            return metadata
    return None


def integer_scalar_argument(value: object) -> Argument | None:
    """Return an FX-compatible integer argument when its metadata is valid."""
    if integer_scalar_metadata(value) is None:
        return None
    return value if isinstance(value, (int, torch.SymInt, torch.fx.Node)) else None


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted > 0 else None


def source_files() -> tuple[str, ...]:
    """Return sources that affect shared sparse-attention validation."""
    return tuple(file_name for file_name in (__file__, _budget.__file__) if file_name is not None)


def valid_sparse_piper_attention(  # noqa: PLR0911
    match: Match,
    *,
    batch: int | torch.SymInt,
    sequence_length: int | torch.SymInt,
    heads: int,
    device: torch.device,
    head_dim: int,
    tile_rows: int,
) -> bool:
    """Validate the projection-independent portion of a sparse Piper match."""
    names = (
        "sparse_q_norm_weight",
        "sparse_k_norm_weight",
        "sparse_cos",
        "sparse_sin",
    )
    if any(not isinstance(match.kwargs[name], torch.fx.Node) for name in names):
        return False
    metadata = {
        name: preparation_sharing.tensor_metadata(match.kwargs[name])  # type: ignore[arg-type]
        for name in names
    }
    if any(
        value is None or value.layout is not torch.strided or not value.is_contiguous()
        for value in metadata.values()
    ):
        return False

    shape = match.kwargs["sparse_attention_shape"]
    output = preparation_sharing.tensor_metadata(match.output_node())
    if (
        output is None
        or output.ndim != 4
        or not isinstance(shape, (list, tuple))
        or len(shape) != 4
    ):
        return False
    shape_batch = integer_scalar_metadata(shape[0])
    shape_sequence_length = integer_scalar_metadata(shape[1])
    if (
        shape_batch is None
        or shape_sequence_length is None
        or preparation_sharing.dimension_key(shape_batch)
        != preparation_sharing.dimension_key(batch)
        or preparation_sharing.dimension_key(shape_sequence_length)
        != preparation_sharing.dimension_key(sequence_length)
        or static_int(shape[2]) != heads
        or static_int(shape[3]) != head_dim
        or output.dtype is not torch.bfloat16
        or output.device != device
        or preparation_sharing.dimension_key(output.shape[0])
        != preparation_sharing.dimension_key(batch)
        or preparation_sharing.dimension_key(output.shape[1])
        != preparation_sharing.dimension_key(sequence_length)
        or tuple(output.shape[2:]) != (heads, head_dim)
        or (isinstance(sequence_length, int) and sequence_length < tile_rows)
    ):
        return False

    for name in ("sparse_q_norm_weight", "sparse_k_norm_weight"):
        norm = metadata[name]
        assert norm is not None
        if (
            norm.dtype is not torch.bfloat16
            or tuple(norm.shape) != (head_dim,)
            or norm.device != device
        ):
            return False
    if any(
        _positive_float(match.kwargs[name]) is None
        for name in (
            "sparse_q_norm_epsilon",
            "sparse_k_norm_epsilon",
            "sparse_softmax_scale",
        )
    ):
        return False

    rotary_dim = integer_scalar_metadata(match.kwargs["sparse_rotary_dim"])
    half_rotary_dim = integer_scalar_metadata(match.kwargs["sparse_half_rotary_dim"])
    cos = metadata["sparse_cos"]
    sin = metadata["sparse_sin"]
    assert cos is not None
    assert sin is not None
    if (
        rotary_dim is None
        or half_rotary_dim is None
        or cos.dtype is not torch.float32
        or sin.dtype is not torch.float32
        or cos.ndim != 2
        or sin.ndim != 2
        or preparation_sharing.dimension_key(cos.shape[0])
        != preparation_sharing.dimension_key(sequence_length)
        or preparation_sharing.dimension_key(sin.shape[0])
        != preparation_sharing.dimension_key(sequence_length)
        or preparation_sharing.dimension_key(cos.shape[1])
        != preparation_sharing.dimension_key(rotary_dim)
        or preparation_sharing.dimension_key(sin.shape[1])
        != preparation_sharing.dimension_key(rotary_dim)
        or preparation_sharing.dimension_key(half_rotary_dim)
        != preparation_sharing.dimension_key((rotary_dim + 1) // 2)
        or cos.device != device
        or sin.device != device
    ):
        return False
    if isinstance(rotary_dim, int) and (
        rotary_dim < 2
        or rotary_dim > head_dim
        or rotary_dim % 2
        or not isinstance(half_rotary_dim, int)
        or half_rotary_dim != rotary_dim // 2
    ):
        return False

    sparse_key_blocks = integer_scalar_argument(match.kwargs["sparse_key_blocks"])
    static_sparse_key_blocks = static_int(sparse_key_blocks)
    if (
        sparse_key_blocks is None
        or (static_sparse_key_blocks is not None and static_sparse_key_blocks < 1)
        or (
            static_sparse_key_blocks is not None
            and isinstance(sequence_length, int)
            and static_sparse_key_blocks > sequence_length // tile_rows
        )
    ):
        return False
    ratios = match.kwargs["sparse_head_keep_ratio_units"]
    return bool(
        isinstance(ratios, (list, tuple))
        and len(ratios) == heads
        and all(
            isinstance(units, int)
            and not isinstance(units, bool)
            and 1 <= units <= _budget._RATIO_SCALE
            for units in ratios
        )
    )


__all__ = [
    "integer_scalar_argument",
    "integer_scalar_metadata",
    "source_files",
    "static_int",
    "valid_sparse_piper_attention",
]
