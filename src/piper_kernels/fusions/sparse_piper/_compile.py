"""Shared compiler validation for projected sparse Piper attention."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch._inductor.pattern_matcher import Match
from torch._library.fake_class_registry import FakeScriptObject
from torch.fx.experimental.symbolic_shapes import guard_or_false
from torch.fx.node import Argument

from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS
from piper_kernels.attention.sparse_piper_attention import _dtype
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile
from piper_kernels.fusions.projected_qk import _compile as projected_qk_compile
from piper_kernels.fusions.projected_qk._compile import (
    integer_scalar_argument,
    integer_scalar_metadata,
    static_int,
)
from piper_kernels.fusions.sparse_piper import _pattern as sparse_piper_pattern
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear._compile_fx import (
    ordered_tuple_output_producer as ordered_tuple_output_producer,  # noqa: PLC0414 - shared export
)

_SHAPE_ONLY_VIEW_TARGETS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
)


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


def valid_sparse_piper_coarse_residual(match: Match) -> bool:
    """Validate projection-independent operands for a matched coarse residual."""
    if match.kwargs["sparse_routing_mode"] != match.kwargs["coarse_routing_mode"]:
        return False
    gate_node = match.kwargs["coarse_gate"]
    if not isinstance(gate_node, torch.fx.Node):
        return False
    gate = preparation_sharing.tensor_metadata(gate_node)
    output = preparation_sharing.tensor_metadata(match.output_node())
    coarse_scale = projected_qk_compile.positive_float(match.kwargs["coarse_scale"])
    sparse_key_blocks = integer_scalar_metadata(match.kwargs["sparse_key_blocks"])
    coarse_key_blocks = integer_scalar_metadata(match.kwargs["coarse_key_blocks"])
    return bool(
        gate is not None
        and output is not None
        and gate.layout is torch.strided
        and gate.ndim == 4
        and gate.dtype is output.dtype
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
    ratio_profile: Argument,
    sparse_key_blocks: Argument,
    logical_sequence_length: Argument,
    routing_mode: Argument,
    block_lengths: Argument | None = None,
    sparse_query_blocks: Argument | None = None,
    coarse_gate: Argument | None = None,
    coarse_scale: Argument | None = None,
    coarse_key_blocks: Argument | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    force_tile_ids: Argument | None = None,
    force_tile_counts: Argument | None = None,
    global_row_offset: Argument | None = None,
) -> torch.fx.Node:
    """Emit the shared fine or coarse quantized sparse-attention call.

    ``global_row_offset`` places the fine call's Q rows; coarse calls take no offset.
    """
    force_arguments = (force_tile_ids, force_tile_counts) if force_tile_ids is not None else ()
    coarse_arguments = coarse_gate, coarse_scale, coarse_key_blocks
    with_coarse = any(argument is not None for argument in coarse_arguments)
    if with_coarse:
        if any(argument is None for argument in coarse_arguments):
            raise ValueError("coarse sparse attention requires every coarse operand")
        if global_row_offset is not None:
            raise ValueError("fused coarse sparse attention does not take a query offset")
        return graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default,
            args=(
                *attention_arguments,
                coarse_gate,
                ratio_profile,
                sparse_key_blocks,
                logical_sequence_length,
                routing_mode,
                coarse_scale,
                coarse_key_blocks,
                block_lengths,
                sparse_query_blocks,
                *force_arguments,
            ),
            kwargs={"output_dtype": output_dtype},
        )
    return graph.call_function(
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
        args=(
            *attention_arguments,
            ratio_profile,
            sparse_key_blocks,
            logical_sequence_length,
            routing_mode,
            *sparse_piper_pattern.optional_attention_layout_arguments(
                block_lengths,
                sparse_query_blocks,
                *force_arguments,
            ),
        ),
        kwargs={
            "output_dtype": output_dtype,
            **({} if global_row_offset is None else {"global_row_offset": global_row_offset}),
        },
    )


@contextmanager
def canonical_query_offsets(graph: torch.fx.Graph) -> Iterator[None]:
    """Pass sparse attention's ``global_row_offset`` as the keyword patterns declare.

    Tracing passes it positionally, after explicit defaults for the optional layout
    operands, which no positional pattern spells. Matching sees the defaults dropped and
    the offset named; calls that survive get their traced arguments back for lowering.
    """
    target = torch.ops.piper_kernels.sparse_piper_attention.default
    schema = target._schema.arguments
    offset_index = next(
        i for i, argument in enumerate(schema) if argument.name == "global_row_offset"
    )
    first_optional = next(i for i, argument in enumerate(schema) if argument.has_default_value())
    traced: dict[torch.fx.Node, tuple[tuple[Argument, ...], dict[str, Argument]]] = {}
    for node in graph.find_nodes(op="call_function", target=target):
        if len(node.args) != offset_index + 1:
            continue
        traced[node] = (node.args, dict(node.kwargs))
        arguments = list(node.args[:offset_index])
        while len(arguments) > first_optional and arguments[-1] is None:
            arguments.pop()
        offset = node.args[offset_index]
        node.args = tuple(arguments)
        node.kwargs = {**node.kwargs, "global_row_offset": offset}
    try:
        yield
    finally:
        live_nodes = set(graph.nodes)
        for node, (arguments, keywords) in traced.items():
            if node in live_nodes:
                node.args, node.kwargs = arguments, keywords


def attention_head_dim(match: Match) -> int | None:
    """Read the supported head width from the semantic attention output."""
    output = preparation_sharing.tensor_metadata(match.output_node())
    if output is None or output.ndim != 4:
        return None
    head_dim = static_int(output.shape[-1])
    return head_dim if head_dim in SUPPORTED_HEAD_DIMS else None


def valid_sparse_piper_attention(  # noqa: PLR0911
    match: Match,
    *,
    batch: int | torch.SymInt,
    sequence_length: int | torch.SymInt,
    heads: int,
    device: torch.device,
    head_dim: int,
    tile_rows: int,
    query_rows: int | torch.SymInt | None = None,
) -> bool:
    """Validate the projection-independent portion of a sparse Piper match.

    ``query_rows`` is Q's row count when Q holds a slice of the ``sequence_length`` K/V
    rows. Its attention shape and RoPE tables are then the ``sparse_q_*`` captures.
    """
    query_rows = sequence_length if query_rows is None else query_rows
    output = preparation_sharing.tensor_metadata(match.output_node())
    if (
        output is None
        or output.ndim != 4
        or output.dtype not in _dtype.SUPPORTED_DTYPES
        or output.device != device
        or preparation_sharing.dimension_key(output.shape[0])
        != preparation_sharing.dimension_key(batch)
        or preparation_sharing.dimension_key(output.shape[1])
        != preparation_sharing.dimension_key(query_rows)
        or tuple(output.shape[2:]) != (heads, head_dim)
        or (isinstance(sequence_length, int) and sequence_length < tile_rows)
    ):
        return False

    if not _valid_row_operands(
        match,
        batch=batch,
        sequence_length=sequence_length,
        query_rows=query_rows,
        heads=heads,
        head_dim=head_dim,
        device=device,
    ):
        return False
    if (
        any(
            not projected_qk_compile.valid_rmsnorm(
                match.kwargs.get(f"{prefix}_norm_weight"),
                match.kwargs[f"{prefix}_norm_epsilon"],
                head_dim=head_dim,
                device=device,
                supported_dtypes=_dtype.SUPPORTED_DTYPES,
            )
            for prefix in ("sparse_q", "sparse_k")
        )
        or projected_qk_compile.positive_float(match.kwargs["sparse_softmax_scale"]) is None
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
    return _valid_ratio_profile(match.kwargs["sparse_ratio_profile"], heads)


def _valid_row_operands(
    match: Match,
    *,
    batch: int | torch.SymInt,
    sequence_length: int | torch.SymInt,
    query_rows: int | torch.SymInt,
    heads: int,
    head_dim: int,
    device: torch.device,
) -> bool:
    """Check K/V's and Q's attention shapes and RoPE tables against their row counts."""
    shape = match.kwargs["sparse_attention_shape"]
    for rows, rows_shape in (
        (sequence_length, shape),
        (query_rows, match.kwargs.get("sparse_q_attention_shape", shape)),
    ):
        if not isinstance(rows_shape, (list, tuple)) or len(rows_shape) != 4:
            return False
        shape_batch = integer_scalar_metadata(rows_shape[0])
        shape_rows = integer_scalar_metadata(rows_shape[1])
        if (
            shape_batch is None
            or shape_rows is None
            or preparation_sharing.dimension_key(shape_batch)
            != preparation_sharing.dimension_key(batch)
            or preparation_sharing.dimension_key(shape_rows)
            != preparation_sharing.dimension_key(rows)
            or static_int(rows_shape[2]) != heads
            or static_int(rows_shape[3]) != head_dim
        ):
            return False
    for rows, cos, sin in (
        (sequence_length, match.kwargs["sparse_cos"], match.kwargs["sparse_sin"]),
        (
            query_rows,
            match.kwargs.get("sparse_q_cos", match.kwargs["sparse_cos"]),
            match.kwargs.get("sparse_q_sin", match.kwargs["sparse_sin"]),
        ),
    ):
        if not projected_qk_compile.valid_rope_tables(
            cos,
            sin,
            match.kwargs["sparse_rotary_dim"],
            match.kwargs["sparse_half_rotary_dim"],
            sequence_length=rows,
            head_dim=head_dim,
            device=device,
        ):
            return False
    return True


def _valid_ratio_profile(profile_node: object, heads: int) -> bool:
    if not isinstance(profile_node, torch.fx.Node):
        return False
    profile = profile_node.meta.get("val")
    if isinstance(profile, FakeScriptObject):
        profile = profile.real_obj
    # The profile's custom-class guard fixes this metadata without specializing
    # its values. Runtime and fake operators share the same opaque argument.
    return isinstance(profile, _SparsePiperRatioProfile) and profile.num_heads == heads


__all__ = [
    "attention_head_dim",
    "emit_quantized_sparse_piper_attention",
    "integer_scalar_argument",
    "integer_scalar_metadata",
    "static_int",
    "valid_sparse_piper_attention",
    "valid_sparse_piper_coarse_residual",
]
