"""Internal sparse Piper orchestration for already-quantized operands."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile

from . import _backend
from ._budget import _ResolvedRouteLayout
from ._dtype import validate_output_dtype
from ._force_tiles import validate_force_tiles
from ._interfaces import LaunchAttention
from ._key_layout import (
    SparsePiperKeyLayout,
    resolve_route_layout,
    validate_key_layout_storage,
)
from ._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
    _PreparedSparsePiperContext,
)
from ._routing import (
    packed_routes_and_coarse_from_summaries,
    packed_routes_from_summaries,
)
from ._routing_modes import (
    validate_routing_mode,
)
from .coarse import _validate_coarse_key_blocks


@dataclass(frozen=True, slots=True)
class _PreparedQuantizedSparsePiperContext:
    """Global sparse-attention state used by independently produced Q chunks."""

    route_layout: _ResolvedRouteLayout
    key_summary: torch.Tensor
    key_aux: torch.Tensor
    routing_mode: int
    launch: LaunchAttention
    pooled_value: torch.Tensor | None = None
    coarse_scale: float | None = None
    force_tile_mask: torch.Tensor | None = None

    @property
    def kernel_context(self) -> _PreparedSparsePiperContext:
        return self.launch.context


def _prepare_quantized_sparse_piper_context(  # noqa: PLR0913, PLR0917
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    logical_sequence_length: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    *,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    start_head: int = 0,
    end_head: int | None = None,
) -> _PreparedQuantizedSparsePiperContext:
    """Prepare global K/V and routing state without requiring materialized Q.

    Forced IDs/counts follow SparsePiperAttention.forward and cover every global
    query block. Build their membership once here; Q windows take row views.
    The operands hold ``ratio_profile``'s query heads ``[start_head, end_head)``,
    which keep the blocks they keep when every profile head runs together.
    A fused coarse branch must cover every complete key block, which routing scores.
    """
    backend = _backend.require_attention_backend(key)
    layout = resolve_route_layout(
        ratio_profile.units,
        key_layout,
        key.device,
        start_head=start_head,
        end_head=end_head,
    )
    validate_routing_mode(routing_mode)
    pooled_value = None
    if block_mean is None:
        if coarse_scale is not None or coarse_key_blocks is not None:
            raise ValueError("coarse output metadata requires block means")
        route_key_blocks = layout.num_key_blocks
    else:
        if coarse_scale is None or coarse_key_blocks is None:
            raise ValueError("coarse output requires a scale and a key block scope")
        _validate_coarse_key_blocks(
            layout.num_key_blocks,
            coarse_key_blocks,
            available_coarse_key_blocks=block_mean.shape[2],
        )
        route_key_blocks = coarse_key_blocks
        pooled_value = block_mean[:, :, :route_key_blocks]
    kernel_context = _prepare_sparse_piper_context_from_quantized(
        key,
        key_scale,
        value,
        value_scale_multiplier,
        value_mean,
        layout.head_keep_blocks,
        layout.route_head_offsets,
        value_format=backend.value_format,
        routes_per_query=(
            0 if backend.skip_dense_routing and layout.keeps_all_blocks else layout.routes_per_query
        ),
        logical_sequence_length=logical_sequence_length,
        block_lengths=block_lengths,
        sparse_query_blocks=sparse_query_blocks,
    )
    # After storage validation, so a wrong logical length is reported as such.
    validate_key_layout_storage(
        key_layout,
        _quantized_attention_output_sequence_length(key, logical_sequence_length, block_lengths),
    )
    validate_force_tiles(
        force_tile_ids,
        force_tile_counts,
        query_blocks=key.shape[2] // TILE_ROWS,
        device=key.device,
    )
    force_tile_mask = None
    if force_tile_ids is not None and not layout.keeps_all_blocks:
        assert force_tile_counts is not None
        force_tile_mask = _backend.build_force_tile_bitmask(
            force_tile_ids, force_tile_counts, layout.num_key_blocks
        )
    return _PreparedQuantizedSparsePiperContext(
        route_layout=layout,
        key_summary=key_summary[:, :, :route_key_blocks],
        key_aux=key_aux[:, :, :route_key_blocks],
        routing_mode=routing_mode,
        launch=backend.bind_context(kernel_context),
        pooled_value=pooled_value,
        force_tile_mask=force_tile_mask,
        coarse_scale=coarse_scale,
    )


def _prepare_quantized_sparse_piper_query(
    context: _PreparedQuantizedSparsePiperContext,
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    *,
    global_block_offset: int,
) -> tuple[_PreparedSparsePiperAttention, torch.Tensor | None]:
    """Route and prepare one compact Q chunk against global K/V state."""
    force_tile_mask = context.force_tile_mask
    if force_tile_mask is not None:
        force_tile_mask = force_tile_mask[
            global_block_offset : global_block_offset + query_summary.shape[2]
        ]
    if context.pooled_value is None:
        routed = packed_routes_from_summaries(
            query_summary,
            context.key_summary,
            context.key_aux,
            context.route_layout,
            context.routing_mode,
            skip_dense_routing=context.kernel_context.routes_per_query == 0,
            force_tile_mask=force_tile_mask,
        )
        coarse_output = None
    else:
        assert context.coarse_scale is not None
        routed_with_coarse = packed_routes_and_coarse_from_summaries(
            query_summary,
            context.key_summary,
            context.key_aux,
            context.pooled_value,
            context.route_layout,
            coarse_scale=context.coarse_scale,
            routing_mode=context.routing_mode,
            skip_dense_routing=context.kernel_context.routes_per_query == 0,
            force_tile_mask=force_tile_mask,
        )
        routed = routed_with_coarse.routes
        coarse_output = routed_with_coarse.coarse_output
    return (
        _PreparedSparsePiperAttention(
            launch=context.launch,
            query=_prepare_sparse_piper_query_from_quantized(
                query,
                query_scale,
                routed.indices,
                context.kernel_context,
                global_block_offset=global_block_offset,
            ),
        ),
        coarse_output,
    )


def _prepare_quantized_sparse_piper_attention(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    logical_sequence_length: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
    *,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> _PreparedSparsePiperAttention:
    """Build validated shared launch state for quantized sparse Piper."""
    context = _prepare_quantized_sparse_piper_context(
        key,
        key_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        ratio_profile,
        key_layout,
        logical_sequence_length,
        routing_mode,
        block_lengths,
        sparse_query_blocks=sparse_query_blocks,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    return _prepare_quantized_sparse_piper_query(
        context,
        query,
        query_scale,
        query_summary,
        global_block_offset=0,
    )[0]


def _prepare_quantized_sparse_piper_attention_with_coarse(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    block_mean: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    logical_sequence_length: int,
    routing_mode: int,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    *,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> tuple[_PreparedSparsePiperAttention, torch.Tensor]:
    """Prepare fine routes and the coarse branch over its key block scope."""
    context = _prepare_quantized_sparse_piper_context(
        key,
        key_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        ratio_profile,
        key_layout,
        logical_sequence_length,
        routing_mode,
        block_lengths,
        block_mean,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    prepared, coarse_output = _prepare_quantized_sparse_piper_query(
        context,
        query,
        query_scale,
        query_summary,
        global_block_offset=0,
    )
    assert coarse_output is not None
    return prepared, coarse_output


def _quantized_attention_output_sequence_length(
    query: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
) -> int:
    """Resolve compact logical rows or valid-front padded physical rows."""
    return query.shape[2] if block_lengths is not None else logical_sequence_length


def validate_quantized_attention_output(
    attention_storage: torch.Tensor,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None,
    *,
    key_layout: SparsePiperKeyLayout,
    output_dtype: torch.dtype,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> int:
    """Validate output, layout, and forced-route metadata, returning the output rows.

    Fake quantized and projected-output boundaries share these metadata checks.
    Runtime contexts validate the layout and forced inputs when preparing global
    routing state.
    """
    validate_output_dtype(output_dtype)
    sequence_length = _quantized_attention_output_sequence_length(
        attention_storage, logical_sequence_length, block_lengths
    )
    validate_key_layout_storage(key_layout, sequence_length)
    validate_force_tiles(
        force_tile_ids,
        force_tile_counts,
        query_blocks=(sequence_length + TILE_ROWS - 1) // TILE_ROWS,
        device=attention_storage.device,
    )
    return sequence_length


def _new_quantized_attention_output(
    query: torch.Tensor,
    sequence_length: int,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return query.new_empty(
        (query.shape[0], sequence_length, query.shape[1], query.shape[3]),
        dtype=output_dtype,
    )


def _launch_quantized_sparse_piper_attention(
    prepared: _PreparedSparsePiperAttention,
    output: torch.Tensor,
    *,
    query_block_offset: int = 0,
    query_block_count: int | None = None,
    coarse_output: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
) -> None:
    """Launch a prepared quantized sparse-Piper query range."""
    prepared.launch(
        prepared.query,
        output,
        query_block_offset=query_block_offset,
        query_block_count=query_block_count,
        coarse_output=coarse_output,
        coarse_gate=coarse_gate,
    )


@torch.library.custom_op(
    "piper_kernels::sparse_piper_attention_from_quantized",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _sparse_piper_attention_from_quantized_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    logical_sequence_length: int,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run quantized sparse Piper with compact or internally padded K64 storage.

    ``block_lengths`` is trusted layout metadata with one INT32 value per
    physical K64 block. Each value must be in ``[1, 64]``, valid tokens must
    occupy the front of the block, and the metadata becomes the source of
    token validity instead of ``logical_sequence_length``. Supplying it returns
    the full padded query storage; outputs for padded query rows are unspecified
    and must be removed by the caller's layout gather. ``sparse_query_blocks``
    optionally limits route use to the leading query blocks; the suffix is dense.
    ``key_layout`` must cover the output sequence's complete K64 blocks.
    """
    validate_output_dtype(output_dtype)
    sequence_length = _quantized_attention_output_sequence_length(
        query, logical_sequence_length, block_lengths
    )
    output = _new_quantized_attention_output(query, sequence_length, output_dtype=output_dtype)
    prepared = _prepare_quantized_sparse_piper_attention(
        query,
        query_scale,
        query_summary,
        key,
        key_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        ratio_profile,
        key_layout,
        logical_sequence_length,
        routing_mode,
        block_lengths,
        sparse_query_blocks,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    _launch_quantized_sparse_piper_attention(prepared, output.transpose(1, 2))
    return output


@torch.library.custom_op(
    "piper_kernels::sparse_piper_attention_with_coarse_residual_from_quantized",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _sparse_piper_attention_with_coarse_residual_from_quantized_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    block_mean: torch.Tensor,
    coarse_gate: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    logical_sequence_length: int,
    routing_mode: int,
    coarse_scale: float,
    block_lengths: torch.Tensor | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run quantized sparse attention plus a gated, independently scoped residual."""
    validate_output_dtype(output_dtype)
    sequence_length = _quantized_attention_output_sequence_length(
        query, logical_sequence_length, block_lengths
    )
    output = _new_quantized_attention_output(query, sequence_length, output_dtype=output_dtype)
    prepared, coarse_output = _prepare_quantized_sparse_piper_attention_with_coarse(
        query,
        query_scale,
        query_summary,
        key,
        key_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        block_mean,
        ratio_profile,
        key_layout,
        logical_sequence_length,
        routing_mode,
        coarse_scale,
        block_lengths,
        coarse_key_blocks,
        sparse_query_blocks,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    _launch_quantized_sparse_piper_attention(
        prepared,
        output.transpose(1, 2),
        coarse_output=coarse_output,
        coarse_gate=coarse_gate,
    )
    return output


@_sparse_piper_attention_from_quantized_op.register_fake
def _sparse_piper_attention_from_quantized_op_fake(
    query: torch.Tensor,
    _query_scale: torch.Tensor,
    _query_summary: torch.Tensor,
    _key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_summary: torch.Tensor,
    _key_aux: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    logical_sequence_length: int,
    _routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
    _sparse_query_blocks: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    sequence_length = validate_quantized_attention_output(
        query,
        logical_sequence_length,
        _block_lengths,
        key_layout=key_layout,
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    return _new_quantized_attention_output(query, sequence_length, output_dtype=output_dtype)


@_sparse_piper_attention_with_coarse_residual_from_quantized_op.register_fake
def _sparse_piper_attention_with_coarse_residual_from_quantized_op_fake(
    query: torch.Tensor,
    _query_scale: torch.Tensor,
    _query_summary: torch.Tensor,
    _key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_summary: torch.Tensor,
    _key_aux: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _block_mean: torch.Tensor,
    _coarse_gate: torch.Tensor,
    _ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    logical_sequence_length: int,
    _routing_mode: int,
    _coarse_scale: float,
    _block_lengths: torch.Tensor | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    sequence_length = validate_quantized_attention_output(
        query,
        logical_sequence_length,
        _block_lengths,
        key_layout=key_layout,
        output_dtype=output_dtype,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )
    return _new_quantized_attention_output(query, sequence_length, output_dtype=output_dtype)
