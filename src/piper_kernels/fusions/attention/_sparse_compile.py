"""Shared imperative FX matchers, validation, and emission for sparse Piper fusions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch._library.fake_class_registry import FakeScriptObject
from torch.fx.experimental.symbolic_shapes import guard_or_false
from torch.fx.node import Argument

from piper_kernels.attention.kernels.sparse_piper import layout as sparse_layout
from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS
from piper_kernels.attention.sparse_piper_attention import _contracts as _dtype
from piper_kernels.attention.sparse_piper_attention._contracts import _SparsePiperRatioProfile
from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear._compile_fx import (
    ordered_tuple_output_producer as ordered_tuple_output_producer,  # noqa: PLC0414 - shared export
)

from . import _projected_qk_compile as projected_qk_compile
from ._projected_qk_compile import (
    integer_scalar_argument,
    integer_scalar_metadata,
    static_int,
)

_aten = torch.ops.aten
_SHAPE_ONLY_VIEW_TARGETS = (
    _aten.reshape.default,
    _aten.view.default,
)

QUANTIZED_ATTENTION_ARGUMENT_NAMES = (
    "query",
    "query_scale",
    "query_summary",
    "key",
    "key_scale",
    "key_center",
    "key_center_scale",
    "key_summary",
    "key_aux",
    "value",
    "value_scale_multiplier",
    "value_mean",
    "ratio_profile",
    "sparse_key_blocks",
    "logical_sequence_length",
    "routing_mode",
)


@dataclass(frozen=True, slots=True)
class SparsePiperProjectionMatch:
    """Imperative anchor match for projected Q/K/V feeding sparse Piper attention."""

    output_node: torch.fx.Node
    fine_node: torch.fx.Node
    coarse_node: torch.fx.Node | None
    query_projection: torch.fx.Node
    key_projection: torch.fx.Node
    value_projection: torch.fx.Node
    attention_shape: Argument
    q_norm_weight: torch.fx.Node | None
    q_norm_epsilon: Argument
    k_norm_weight: torch.fx.Node | None
    k_norm_epsilon: Argument
    cos: torch.fx.Node
    sin: torch.fx.Node
    rotary_dim: Argument | None
    half_rotary_dim: Argument
    activation_dtype: torch.dtype | None
    ratio_profile: Argument
    sparse_key_blocks: Argument
    softmax_scale: Argument
    routing_mode: Argument
    block_lengths: Argument
    sparse_query_blocks: Argument
    force_tile_ids: Argument
    force_tile_counts: Argument
    coarse_gate: Argument | None = None
    coarse_key_blocks: Argument | None = None
    coarse_scale: Argument | None = None
    coarse_routing_mode: Argument | None = None


@dataclass(frozen=True, slots=True)
class ReshapedQuantizedAttentionMatch:
    """Imperative anchor match for ``quantized_sparse_piper_attention -> reshape``."""

    reshape_node: torch.fx.Node
    attention_node: torch.fx.Node
    attention_shape: Argument
    attention_args: dict[str, Argument]
    is_coarse: bool
    force_kwargs: dict[str, Argument]

    @property
    def attention_arguments(self) -> tuple[Argument, ...]:
        return tuple(self.attention_args[name] for name in QUANTIZED_ATTENTION_ARGUMENT_NAMES)

    @property
    def bounded_arguments(self) -> tuple[Argument, ...]:
        args = self.attention_args
        return (
            args.get("block_lengths"),
            args["coarse_gate"] if self.is_coarse else None,
            args["coarse_scale"] if self.is_coarse else None,
            args["coarse_key_blocks"] if self.is_coarse else None,
            args.get("sparse_query_blocks"),
        )

    def defer_gate[GateT: tuple[Argument, ...]](
        self,
        extract_gate: Callable[[object], GateT | None],
    ) -> tuple[tuple[Argument, ...], GateT | None]:
        """Extract a prepared gate and clear the materialized gate slot when deferred."""
        args = self.bounded_arguments
        gate = extract_gate(args[1])
        return (args, None) if gate is None else ((args[0], None, *args[2:]), gate)

    def match_projected_query(self, target: object) -> dict[str, Argument] | None:
        """Recover the bound projected-Q producer matching this attention call."""
        args = self.attention_args
        return match_projected_query_producer(
            args["query"],
            args["query_scale"],
            args["query_summary"],
            target,
            routing_mode=args["routing_mode"],
            block_lengths=args.get("block_lengths"),
        )


@dataclass(frozen=True, slots=True)
class ValidatedAttentionOutput:
    """Validated format-independent metadata for quantized attention -> reshape -> linear."""

    query: torch.Tensor
    projected: torch.Tensor
    input_features: int


def same_dimension(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    """Compare two static or symbolic dimensions structurally."""
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


def optional_tensor_metadata(value: object) -> tuple[bool, torch.Tensor | None]:
    """Validate an optional FX node operand and return its fake-tensor metadata."""
    if value is None:
        return True, None
    if not isinstance(value, torch.fx.Node):
        return False, None
    metadata = preparation_sharing.tensor_metadata(value)
    return metadata is not None, metadata


def view_shape_argument(node: torch.fx.Node) -> Argument:
    """Return the bound shape/size argument of a view node or its static metadata shape."""
    bound = linear_fx.bind_arguments(node)
    if bound is not None:
        for key in ("shape", "size"):
            if key in bound:
                return bound[key]
    value = preparation_sharing.tensor_metadata(node)
    assert value is not None
    return tuple(value.shape)


def sparse_projection_metadata(
    template: torch.Tensor,
    *,
    batch: int | torch.SymInt,
    heads: int,
    storage_sequence_length: int | torch.SymInt,
    head_dim: int,
    mean_pool_routing: bool,
    value_shape: tuple[int | torch.SymInt, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Construct fake-tensor metadata tuples for fused Q, K, and V sparse projections."""
    tiles = storage_sequence_length // sparse_layout.TILE_ROWS
    query_values = (
        template.new_empty((batch, heads, storage_sequence_length, head_dim), dtype=torch.int8),
        template.new_empty(
            (batch, heads, storage_sequence_length // sparse_layout.QUERY_SCALE_ROWS),
            dtype=torch.float32,
        ),
        template.new_empty((batch, heads, tiles, head_dim), dtype=torch.float32),
    )
    key_values = (
        template.new_empty((batch, heads, storage_sequence_length, head_dim), dtype=torch.int8),
        template.new_empty((batch, heads, tiles), dtype=torch.float32),
        template.new_empty((batch, heads, tiles, 2, head_dim), dtype=torch.int8),
        template.new_empty((batch, heads, tiles, 2), dtype=torch.float32),
        template.new_empty((batch, heads, tiles, head_dim), dtype=torch.float32),
        template.new_empty(
            (batch, heads, 0 if mean_pool_routing else tiles, head_dim), dtype=torch.float32
        ),
    )
    value_values = (
        template.new_empty(value_shape, dtype=torch.int8),
        template.new_empty((batch, heads, tiles, 1), dtype=torch.float32),
        template.new_empty((batch, heads, tiles, head_dim), dtype=torch.bfloat16),
    )
    return query_values, key_values, value_values


def _optional_attention_layout_arguments[ArgumentT](
    block_lengths: ArgumentT | None,
    sparse_query_blocks: ArgumentT | None,
    force_tile_ids: ArgumentT | None = None,
    force_tile_counts: ArgumentT | None = None,
) -> tuple[ArgumentT | None, ...]:
    """Preserve positional defaults only when a later layout argument is present."""
    if force_tile_ids is not None:
        return block_lengths, sparse_query_blocks, force_tile_ids, force_tile_counts
    if sparse_query_blocks is not None:
        return block_lengths, sparse_query_blocks
    return (block_lengths,) if block_lengths is not None else ()


def _same_shape_arg(left: Argument, right: Argument) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    return len(left) == len(right) and all(
        linear_fx.same_argument(a, b) for a, b in zip(left, right, strict=True)
    )


def match_sparse_piper_projection(  # noqa: PLR0911, PLR0912
    node: torch.fx.Node,
) -> SparsePiperProjectionMatch | None:
    """Anchor-match fine or fine+coarse sparse Piper attention and its Q/K/V transforms."""
    sparse_attention_op = torch.ops.piper_kernels.sparse_piper_attention.default
    coarse_residual_op = torch.ops.piper_kernels.sparse_piper_coarse_residual.default

    coarse_node: torch.fx.Node | None = None
    coarse_bound: dict[str, Argument] | None = None
    fine_node: torch.fx.Node | None = None

    if node.op == "call_function" and node.target is sparse_attention_op:
        fine_node = node
    elif node.op == "call_function" and node.target is _aten.add.Tensor:
        add_args = linear_fx.binary_args(node, _aten.add.Tensor)
        if add_args is None:
            return None
        for fine_cand, coarse_cand in (add_args, add_args[::-1]):
            if (
                isinstance(fine_cand, torch.fx.Node)
                and isinstance(coarse_cand, torch.fx.Node)
                and fine_cand.target is sparse_attention_op
                and coarse_cand.target is coarse_residual_op
                and len(fine_cand.users) == 1
                and len(coarse_cand.users) == 1
            ):
                fine_node = fine_cand
                coarse_node = coarse_cand
                break
    if fine_node is None:
        return None

    fine_bound = linear_fx.bind_call(fine_node, sparse_attention_op)
    if fine_bound is None:
        return None
    force_tile_ids = fine_bound.get("force_tile_ids")
    force_tile_counts = fine_bound.get("force_tile_counts")
    if (force_tile_ids is None) != (force_tile_counts is None):
        return None

    query = fine_bound["query"]
    key = fine_bound["key"]
    value = fine_bound["value"]
    if (
        not isinstance(query, torch.fx.Node)
        or not isinstance(key, torch.fx.Node)
        or not isinstance(value, torch.fx.Node)
    ):
        return None

    expected_consumers = {fine_node}
    if coarse_node is not None:
        coarse_bound = linear_fx.bind_call(coarse_node, coarse_residual_op)
        if (
            coarse_bound is None
            or coarse_bound["query"] is not query
            or coarse_bound["key"] is not key
            or coarse_bound["value"] is not value
            or coarse_bound.get("block_lengths") is not fine_bound.get("block_lengths")
        ):
            return None
        expected_consumers.add(coarse_node)

    if (
        set(query.users) != expected_consumers
        or set(key.users) != expected_consumers
        or set(value.users) != expected_consumers
    ):
        return None

    expected_users = len(expected_consumers)
    q_rope = projected_qk_compile.match_normalized_rope(query, expected_users=expected_users)
    k_rope = projected_qk_compile.match_normalized_rope(key, expected_users=expected_users)
    v_reshape = linear_fx.bind_call(value, _aten.reshape.default)
    if q_rope is None or k_rope is None or v_reshape is None:
        return None
    v_proj = v_reshape["self"]
    if (
        not isinstance(v_proj, torch.fx.Node)
        or len(v_proj.users) != 1
        or q_rope.cos is not k_rope.cos
        or q_rope.sin is not k_rope.sin
        or q_rope.activation_dtype is not k_rope.activation_dtype
        or not linear_fx.same_argument(q_rope.rotary_dim, k_rope.rotary_dim)
        or not linear_fx.same_argument(q_rope.half_rotary_dim, k_rope.half_rotary_dim)
        or not _same_shape_arg(q_rope.shape, k_rope.shape)
        or not _same_shape_arg(q_rope.shape, v_reshape["shape"])
    ):
        return None

    return SparsePiperProjectionMatch(
        output_node=node,
        fine_node=fine_node,
        coarse_node=coarse_node,
        query_projection=q_rope.projection_node,
        key_projection=k_rope.projection_node,
        value_projection=v_proj,
        attention_shape=q_rope.shape,
        q_norm_weight=q_rope.norm_weight,
        q_norm_epsilon=q_rope.norm_epsilon,
        k_norm_weight=k_rope.norm_weight,
        k_norm_epsilon=k_rope.norm_epsilon,
        cos=q_rope.cos,
        sin=q_rope.sin,
        rotary_dim=q_rope.rotary_dim,
        half_rotary_dim=q_rope.half_rotary_dim,
        activation_dtype=q_rope.activation_dtype,
        ratio_profile=fine_bound["ratio_profile"],
        sparse_key_blocks=fine_bound["sparse_key_blocks"],
        softmax_scale=fine_bound["scale"],
        routing_mode=fine_bound["routing_mode"],
        block_lengths=fine_bound.get("block_lengths"),
        sparse_query_blocks=fine_bound.get("sparse_query_blocks"),
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
        coarse_gate=None if coarse_bound is None else coarse_bound["coarse_gate"],
        coarse_key_blocks=None if coarse_bound is None else coarse_bound["coarse_key_blocks"],
        coarse_scale=None if coarse_bound is None else coarse_bound["coarse_scale"],
        coarse_routing_mode=None if coarse_bound is None else coarse_bound["routing_mode"],
    )


def match_reshaped_quantized_attention(
    reshape_node: object,
) -> ReshapedQuantizedAttentionMatch | None:
    """Anchor-match ``quantized_sparse_piper_attention(_with_coarse_residual) -> reshape``."""
    if not isinstance(reshape_node, torch.fx.Node) or len(reshape_node.users) != 1:
        return None
    reshape_bound = linear_fx.bind_call(reshape_node, _aten.reshape.default)
    if reshape_bound is None:
        return None
    attention_node = reshape_bound["self"]
    quantized_attention = torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default
    coarse_attention = (
        torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
    )
    if (
        not isinstance(attention_node, torch.fx.Node)
        or attention_node.op != "call_function"
        or attention_node.target not in (quantized_attention, coarse_attention)
        or len(attention_node.users) != 1
    ):
        return None
    attention_args = linear_fx.bind_arguments(attention_node)
    if attention_args is None:
        return None
    force_tile_ids = attention_args.get("force_tile_ids")
    force_tile_counts = attention_args.get("force_tile_counts")
    if (force_tile_ids is None) != (force_tile_counts is None):
        return None
    force_kwargs: dict[str, Argument] = (
        {}
        if force_tile_ids is None
        else {"force_tile_ids": force_tile_ids, "force_tile_counts": force_tile_counts}
    )
    return ReshapedQuantizedAttentionMatch(
        reshape_node=reshape_node,
        attention_node=attention_node,
        attention_shape=reshape_bound["shape"],
        attention_args=attention_args,
        is_coarse=attention_node.target is coarse_attention,
        force_kwargs=force_kwargs,
    )


def valid_sparse_piper_coarse_residual(matched: SparsePiperProjectionMatch) -> bool:
    """Validate projection-independent operands for a matched coarse residual."""
    if matched.routing_mode != matched.coarse_routing_mode:
        return False
    gate_node = matched.coarse_gate
    if not isinstance(gate_node, torch.fx.Node):
        return False
    gate = preparation_sharing.tensor_metadata(gate_node)
    output = preparation_sharing.tensor_metadata(matched.output_node)
    coarse_scale = projected_qk_compile.positive_float(matched.coarse_scale)
    sparse_key_blocks = integer_scalar_metadata(matched.sparse_key_blocks)
    coarse_key_blocks = integer_scalar_metadata(matched.coarse_key_blocks)
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
            same_dimension(gate_dim, out_dim)
            for gate_dim, out_dim in zip(gate.shape, output.shape, strict=True)
        )
        and coarse_scale is not None
        and sparse_key_blocks is not None
        and coarse_key_blocks is not None
        and guard_or_false(coarse_key_blocks >= sparse_key_blocks)
    )


def emit_quantized_sparse_piper_attention(
    graph: torch.fx.Graph,
    matched: SparsePiperProjectionMatch,
    attention_arguments: tuple[Argument, ...],
    *,
    logical_sequence_length: Argument,
    coarse_gate: Argument | None,
    output_dtype: torch.dtype,
) -> torch.fx.Node:
    """Emit the shared fine or coarse quantized sparse-attention call."""
    force_arguments = (
        (matched.force_tile_ids, matched.force_tile_counts)
        if matched.force_tile_ids is not None
        else ()
    )
    coarse_arguments = coarse_gate, matched.coarse_scale, matched.coarse_key_blocks
    with_coarse = any(argument is not None for argument in coarse_arguments)
    if with_coarse:
        if any(argument is None for argument in coarse_arguments):
            raise ValueError("coarse sparse attention requires every coarse operand")
        return graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default,
            args=(
                *attention_arguments,
                coarse_gate,
                matched.ratio_profile,
                matched.sparse_key_blocks,
                logical_sequence_length,
                matched.routing_mode,
                matched.coarse_scale,
                matched.coarse_key_blocks,
                matched.block_lengths,
                matched.sparse_query_blocks,
                *force_arguments,
            ),
            kwargs={"output_dtype": output_dtype},
        )
    return graph.call_function(
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
        args=(
            *attention_arguments,
            matched.ratio_profile,
            matched.sparse_key_blocks,
            logical_sequence_length,
            matched.routing_mode,
            *_optional_attention_layout_arguments(
                matched.block_lengths,
                matched.sparse_query_blocks,
                *force_arguments,
            ),
        ),
        kwargs={"output_dtype": output_dtype},
    )


def attention_head_dim(output_node: torch.fx.Node) -> int | None:
    """Read the supported head width from the semantic attention output."""
    output = preparation_sharing.tensor_metadata(output_node)
    if output is None or output.ndim != 4:
        return None
    head_dim = static_int(output.shape[-1])
    return head_dim if head_dim in SUPPORTED_HEAD_DIMS else None


def valid_sparse_piper_attention(  # noqa: PLR0911
    matched: SparsePiperProjectionMatch,
    *,
    batch: int | torch.SymInt,
    sequence_length: int | torch.SymInt,
    heads: int,
    device: torch.device,
    head_dim: int,
    tile_rows: int,
) -> bool:
    """Validate the projection-independent portion of a sparse Piper match."""
    shape = matched.attention_shape
    output = preparation_sharing.tensor_metadata(matched.output_node)
    if (
        output is None
        or output.ndim != 4
        or not isinstance(shape, (list, tuple))
        or len(shape) != 4
    ):
        return False
    expected_activation_dtype = None if output.dtype is torch.float32 else output.dtype
    if matched.activation_dtype is not expected_activation_dtype:
        return False
    shape_batch = integer_scalar_metadata(shape[0])
    shape_sequence_length = integer_scalar_metadata(shape[1])
    if (
        shape_batch is None
        or shape_sequence_length is None
        or not same_dimension(shape_batch, batch)
        or not same_dimension(shape_sequence_length, sequence_length)
        or static_int(shape[2]) != heads
        or static_int(shape[3]) != head_dim
        or output.dtype not in _dtype.SUPPORTED_DTYPES
        or output.device != device
        or not same_dimension(output.shape[0], batch)
        or not same_dimension(output.shape[1], sequence_length)
        or tuple(output.shape[2:]) != (heads, head_dim)
        or (isinstance(sequence_length, int) and sequence_length < tile_rows)
    ):
        return False

    if (
        not projected_qk_compile.valid_rmsnorm(
            matched.q_norm_weight,
            matched.q_norm_epsilon,
            head_dim=head_dim,
            device=device,
            supported_dtypes=_dtype.SUPPORTED_DTYPES,
        )
        or not projected_qk_compile.valid_rmsnorm(
            matched.k_norm_weight,
            matched.k_norm_epsilon,
            head_dim=head_dim,
            device=device,
            supported_dtypes=_dtype.SUPPORTED_DTYPES,
        )
        or projected_qk_compile.positive_float(matched.softmax_scale) is None
    ):
        return False
    rotary_dim = head_dim if matched.rotary_dim is None else matched.rotary_dim
    if not projected_qk_compile.valid_rope_tables(
        matched.cos,
        matched.sin,
        rotary_dim,
        matched.half_rotary_dim,
        sequence_length=sequence_length,
        head_dim=head_dim,
        device=device,
    ):
        return False

    sparse_key_blocks = integer_scalar_argument(matched.sparse_key_blocks)
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
    block_lengths_node = matched.block_lengths
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
            or not same_dimension(block_lengths.shape[0] * tile_rows, sequence_length)
        ):
            return False
    sparse_query_blocks_value = matched.sparse_query_blocks
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
    return _valid_ratio_profile(matched.ratio_profile, heads)


def _valid_ratio_profile(profile_node: object, heads: int) -> bool:
    if not isinstance(profile_node, torch.fx.Node):
        return False
    profile = profile_node.meta.get("val")
    if isinstance(profile, FakeScriptObject):
        profile = profile.real_obj
    # The profile's custom-class guard fixes this metadata without specializing
    # its values. Runtime and fake operators share the same opaque argument.
    return isinstance(profile, _SparsePiperRatioProfile) and profile.num_heads == heads


def validate_reshaped_attention_output(
    linear_node: torch.fx.Node,
    reshaped_match: ReshapedQuantizedAttentionMatch,
) -> ValidatedAttentionOutput | None:
    """Validate format-independent metadata for one quantized attention output match."""
    query_arg = reshaped_match.attention_args["query"]
    if not isinstance(query_arg, torch.fx.Node):
        return None
    query = preparation_sharing.tensor_metadata(query_arg)
    attention = preparation_sharing.tensor_metadata(reshaped_match.attention_node)
    reshaped = preparation_sharing.tensor_metadata(reshaped_match.reshape_node)
    projected = preparation_sharing.tensor_metadata(linear_node)
    if query is None or attention is None or reshaped is None or projected is None:
        return None
    shape = reshaped_match.attention_shape
    if (
        query.ndim != 4
        or query.dtype is not torch.int8
        or attention.ndim != 4
        or attention.dtype not in _dtype.SUPPORTED_DTYPES
        or reshaped_match.attention_args.get("output_dtype", torch.bfloat16) is not attention.dtype
        or reshaped.ndim != 3
        or reshaped.dtype is not attention.dtype
        or projected.ndim != 3
        or projected.dtype is not attention.dtype
        or not isinstance(shape, (list, tuple))
        or len(shape) != 3
    ):
        return None
    batch, sequence_length = attention.shape[:2]
    heads = static_int(attention.shape[2])
    head_dim = static_int(attention.shape[3])
    if heads is None or heads < 1 or head_dim not in SUPPORTED_HEAD_DIMS:
        return None
    input_features = heads * head_dim
    if (
        not same_dimension(query.shape[0], batch)
        or query.shape[1] != heads
        or query.shape[3] != head_dim
        or not same_dimension(reshaped.shape[0], batch)
        or not same_dimension(reshaped.shape[1], sequence_length)
        or reshaped.shape[2] != input_features
        or not same_dimension(projected.shape[0], batch)
        or not same_dimension(projected.shape[1], sequence_length)
        or any(
            value.device != query.device
            or value.layout is not torch.strided
            or not value.is_contiguous()
            for value in (query, attention, reshaped, projected)
        )
    ):
        return None
    return ValidatedAttentionOutput(query, projected, input_features)


def match_projected_query_producer(
    query: object,
    query_scale: object,
    query_summary: object,
    target: object,
    *,
    routing_mode: object,
    block_lengths: object,
) -> dict[str, Argument] | None:
    """Recover one shared quantized projected-Q producer matching routing and layout."""
    producer = ordered_tuple_output_producer((query, query_scale, query_summary), target)
    query_value = (
        preparation_sharing.tensor_metadata(query) if isinstance(query, torch.fx.Node) else None
    )
    if producer is None or query_value is None or query_value.ndim != 4:
        return None
    bound = linear_fx.bind_arguments(producer)
    if bound is None:
        return None
    head_dim = bound.get("head_dim")
    if (
        (head_dim is not None and head_dim != query_value.shape[-1])
        or bound.get("routing_mode", _dtype._MINMAX_ROUTING) != routing_mode
        or bound.get("block_lengths") is not block_lengths
    ):
        return None
    return bound


def unwrap_reshaped_producer(value: object) -> torch.fx.Node | None:
    """Return the first non-view producer reached through at least one shape-only view."""
    if not isinstance(value, torch.fx.Node):
        return None
    node = value
    while node.op == "call_function" and node.target in _SHAPE_ONLY_VIEW_TARGETS:
        bound = linear_fx.bind_arguments(node)
        if bound is None or not isinstance(bound.get("self"), torch.fx.Node):
            return None
        node = bound["self"]
        assert isinstance(node, torch.fx.Node)
    return None if node is value else node


def replace_matched_nodes(
    graph: torch.fx.Graph,
    original: torch.fx.Node,
    replacement: torch.fx.Node,
    *extra_erased: torch.fx.Node,
    in_place_output: torch.fx.Node | None = None,
) -> None:
    """Rewire ``original`` to ``replacement`` (or ``in_place_output``) and erase replaced nodes."""
    if in_place_output is None:
        replacement.meta = original.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        original.replace_all_uses_with(replacement)
    else:
        replacement.meta["val"] = None
        original.replace_all_uses_with(in_place_output)
    graph.erase_node(original)
    for node in extra_erased:
        graph.erase_node(node)


def fold_matched_nodes[MatchT](
    graph: torch.fx.Graph,
    match: Callable[[torch.fx.Node], MatchT | None],
    replace: Callable[[torch.fx.Graph, MatchT], None],
) -> bool:
    """Match and replace non-overlapping FX regions, pruning dead code when modified."""
    changed = False
    for node in list(graph.nodes):
        matched = match(node)
        if matched is None:
            continue
        replace(graph, matched)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__ = [
    "QUANTIZED_ATTENTION_ARGUMENT_NAMES",
    "ReshapedQuantizedAttentionMatch",
    "SparsePiperProjectionMatch",
    "ValidatedAttentionOutput",
    "attention_head_dim",
    "emit_quantized_sparse_piper_attention",
    "fold_matched_nodes",
    "integer_scalar_argument",
    "integer_scalar_metadata",
    "match_projected_query_producer",
    "match_reshaped_quantized_attention",
    "match_sparse_piper_projection",
    "optional_tensor_metadata",
    "ordered_tuple_output_producer",
    "replace_matched_nodes",
    "same_dimension",
    "sparse_projection_metadata",
    "static_int",
    "unwrap_reshaped_producer",
    "valid_sparse_piper_attention",
    "valid_sparse_piper_coarse_residual",
    "validate_reshaped_attention_output",
    "view_shape_argument",
]
