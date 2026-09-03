"""Shared compiler validation for projected sparse Piper attention."""

from __future__ import annotations

import math

import torch
from torch._inductor.pattern_matcher import Match
from torch.fx.experimental.symbolic_shapes import guard_or_false
from torch.fx.node import Argument

from piper_kernels.attention.sparse_piper_attention import _budget
from piper_kernels.fusions.sparse_piper import _pattern as sparse_piper_pattern
from piper_kernels.linear import _preparation_sharing as preparation_sharing

_SHAPE_ONLY_VIEW_TARGETS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
)


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


def unwrap_shape_only_views(value: object) -> torch.fx.Node | None:
    """Return the first non-view producer beneath contiguous reshape/view calls."""
    if not isinstance(value, torch.fx.Node):
        return None
    node = value
    while node.target in _SHAPE_ONLY_VIEW_TARGETS:
        if node.kwargs or len(node.args) != 2 or not isinstance(node.args[0], torch.fx.Node):
            return None
        node = node.args[0]
    return node


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted > 0 else None


def source_files() -> tuple[str, ...]:
    """Return sources that affect shared sparse-attention validation."""
    return tuple(file_name for file_name in (__file__, _budget.__file__) if file_name is not None)


def valid_sparse_piper_coarse_residual(match: Match) -> bool:
    """Validate projection-independent operands for a matched coarse residual."""
    if match.kwargs["sparse_routing_mode"] != match.kwargs["coarse_routing_mode"]:
        return False
    gate_node = match.kwargs["coarse_compression_gate"]
    if not isinstance(gate_node, torch.fx.Node):
        return False
    gate = preparation_sharing.tensor_metadata(gate_node)
    output = preparation_sharing.tensor_metadata(match.output_node())
    coarse_scale = _positive_float(match.kwargs["coarse_scale"])
    sparse_key_blocks = integer_scalar_metadata(match.kwargs["sparse_key_blocks"])
    coarse_key_blocks = integer_scalar_metadata(match.kwargs["coarse_key_blocks"])
    return bool(
        gate is not None
        and output is not None
        and gate.layout is torch.strided
        and gate.ndim == 4
        and gate.dtype is torch.bfloat16
        and gate.device == output.device
        and gate.stride(-1) == 1
        and len(gate.shape) == len(output.shape)
        and all(
            preparation_sharing.dimension_key(gate_dimension)
            == preparation_sharing.dimension_key(output_dimension)
            for gate_dimension, output_dimension in zip(gate.shape, output.shape, strict=True)
        )
        and coarse_scale is not None
        and sparse_key_blocks is not None
        and coarse_key_blocks is not None
        and guard_or_false(coarse_key_blocks >= sparse_key_blocks)
    )


def emit_quantized_sparse_piper_attention(  # noqa: PLR0913
    graph: torch.fx.Graph,
    attention_arguments: tuple[Argument, ...],
    *,
    head_keep_ratio_units: Argument,
    sparse_key_blocks: Argument,
    logical_sequence_length: Argument,
    routing_mode: Argument,
    block_lengths: Argument | None = None,
    sparse_query_blocks: Argument | None = None,
    block_mean: Argument | None = None,
    compression_gate: Argument | None = None,
    coarse_scale: Argument | None = None,
    coarse_key_blocks: Argument | None = None,
) -> torch.fx.Node:
    """Emit the shared fine or coarse quantized sparse-attention call."""
    coarse_arguments = block_mean, compression_gate, coarse_scale, coarse_key_blocks
    with_coarse = any(argument is not None for argument in coarse_arguments)
    if with_coarse:
        if any(argument is None for argument in coarse_arguments):
            raise ValueError("coarse sparse attention requires every coarse operand")
        return graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default,
            args=(
                *attention_arguments,
                block_mean,
                compression_gate,
                head_keep_ratio_units,
                sparse_key_blocks,
                logical_sequence_length,
                routing_mode,
                coarse_scale,
                block_lengths,
                coarse_key_blocks,
                sparse_query_blocks,
            ),
        )
    return graph.call_function(
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
        args=(
            *attention_arguments,
            head_keep_ratio_units,
            sparse_key_blocks,
            logical_sequence_length,
            routing_mode,
            *sparse_piper_pattern.optional_attention_layout_arguments(
                block_lengths,
                sparse_query_blocks,
            ),
        ),
    )


def valid_sparse_piper_attention(  # noqa: PLR0911, PLR0912
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
    block_lengths_node = match.kwargs.get("sparse_block_lengths")
    if block_lengths_node is not None:
        if not isinstance(block_lengths_node, torch.fx.Node):
            return False
        block_lengths = preparation_sharing.tensor_metadata(block_lengths_node)
        if (
            block_lengths is None
            or block_lengths.ndim != 1
            or block_lengths.dtype is not torch.int32
            or block_lengths.device != device
            or block_lengths.layout is not torch.strided
            or not block_lengths.is_contiguous()
            or preparation_sharing.dimension_key(block_lengths.shape[0] * tile_rows)
            != preparation_sharing.dimension_key(sequence_length)
        ):
            return False
    sparse_query_blocks_value = match.kwargs.get("sparse_query_blocks")
    if sparse_query_blocks_value is not None:
        sparse_query_blocks = integer_scalar_metadata(sparse_query_blocks_value)
        if sparse_query_blocks is None:
            return False
        static_sparse_query_blocks = static_int(sparse_query_blocks)
        if static_sparse_query_blocks is not None and (
            static_sparse_query_blocks < 0
            or (
                isinstance(sequence_length, int)
                and static_sparse_query_blocks > (sequence_length + tile_rows - 1) // tile_rows
            )
        ):
            return False
    head_keep_ratio_units = match.kwargs["sparse_head_keep_ratio_units"]
    return bool(
        isinstance(head_keep_ratio_units, (list, tuple))
        and len(head_keep_ratio_units) == heads
        and all(
            isinstance(units, int)
            and not isinstance(units, bool)
            and 1 <= units <= _budget._RATIO_SCALE
            for units in head_keep_ratio_units
        )
    )


__all__ = [
    "emit_quantized_sparse_piper_attention",
    "integer_scalar_argument",
    "integer_scalar_metadata",
    "source_files",
    "static_int",
    "valid_sparse_piper_attention",
    "valid_sparse_piper_coarse_residual",
]
