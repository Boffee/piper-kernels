"""Public orchestration for sparse Piper Attention."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile

from . import _backend
from ._block_layout import validate_block_lengths, validate_sparse_query_blocks
from ._budget import _RATIO_SCALE, _normalize_head_keep_ratios, _ResolvedRouteLayout
from ._dtype import SUPPORTED_DTYPES
from ._force_tiles import validate_force_tiles
from ._key_layout import (
    SparsePiperKeyLayout,
    resolve_keep_blocks,
    resolve_route_layout,
    validate_key_layout,
    validate_key_layout_storage,
)
from ._routing import packed_routes_from_sequences, packed_routes_from_summaries
from ._routing_modes import (
    _ROUTING_NAME_BY_MODE,
    routing_mode_from_name,
    validate_routing_mode,
)
from .reference import reference_sparse_piper_attention


class SparsePiperAttention(torch.nn.Module):
    """Sparse attention with an immutable per-query-head ratio profile.

    Replacing this module with another profile of the same head count and routing
    policy reuses compiled graphs, including compatible projection/output fusions.
    Ratios are runtime operator inputs; other shape and model guards still apply.
    """

    def __init__(
        self,
        head_keep_ratios: Sequence[float] | torch.Tensor,
        *,
        routing: str = "minmax",
    ) -> None:
        super().__init__()
        self._ratio_profile = _SparsePiperRatioProfile(
            _normalize_head_keep_ratios(head_keep_ratios)
        )
        self._routing_mode = routing_mode_from_name(routing)

    @property
    def head_keep_ratios(self) -> tuple[float, ...]:
        """Return the device-independent semantic ratio profile."""
        return tuple(units / _RATIO_SCALE for units in self._ratio_profile.units)

    @property
    def routing(self) -> str:
        """Return the block-routing policy selected for this module."""
        return _ROUTING_NAME_BY_MODE[self._routing_mode]

    def resolve_keep_blocks(self, key_layout: SparsePiperKeyLayout) -> tuple[tuple[int, ...], ...]:
        """Return each region's physical K64 keep budget per query head, in region order.

        Each sparse region rounds independently across heads and retains at least
        one block per head; dense regions return their block count. Forced IDs in
        each sparse region must fit every head's budget for that region.

        Call this host-only configuration query outside compiled forward calls.
        It allocates no tensors and accesses no device state.
        """
        validate_key_layout(key_layout)
        return resolve_keep_blocks(self._ratio_profile.units, key_layout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_layout: SparsePiperKeyLayout,
        sparse_query_blocks: int | None = None,
        scale: float | None = None,
        block_lengths: torch.Tensor | None = None,
        force_tile_ids: torch.Tensor | None = None,
        force_tile_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route leading query blocks over key regions with independent budgets.

        Without ``block_lengths``, Q/K/V use compact FP16, BF16, or FP32
        ``[batch, sequence, heads, head_dim]`` storage. Supplying one valid-prefix
        length per physical K64 block selects internally padded storage and
        returns that same physical layout; padded query outputs are unspecified.
        ``sparse_query_blocks`` optionally limits routing to the leading query
        blocks; all later query blocks attend every key block densely. Query heads
        may be an integer multiple of the shared K/V head count. Keep ratios and
        routes belong to query heads; K/V preparation and storage remain shared.

        ``key_layout`` must cover the sequence's complete K64 blocks exactly; a
        compact partial final block is always attended. Each sparse region keeps
        its own per-head budget from ``resolve_keep_blocks``, dense regions keep
        every block, and all retained keys share one softmax. Layouts are runtime
        configuration: see ``SparsePiperKeyLayout`` for compiled-graph reuse.

        Experimental fixed-budget forced routing takes contiguous device INT32
        ``force_tile_ids[query_blocks, capacity]`` and ``force_tile_counts[query_blocks]``.
        Each row's active prefix is shared across batches and heads. Counts must
        lie in [0, capacity]; active IDs must be unique absolute block indices in
        [0, key_layout.num_blocks). IDs in a sparse region consume that region's
        budget and must fit every head's budget for it; IDs in dense regions are
        already retained. These value requirements are caller preconditions, not
        runtime tensor-content checks. Forced tiles consume the existing budget;
        the remainder is selected by score. ID/count contents may change at fixed
        shapes without recompiling.
        """
        converted_scale = _validate_inputs(
            query,
            key,
            value,
            self._ratio_profile.num_heads,
            sparse_query_blocks=sparse_query_blocks,
            scale=scale,
            block_lengths=block_lengths,
        )
        validate_force_tiles(
            force_tile_ids,
            force_tile_counts,
            query_blocks=(query.shape[1] + _BLOCK_ROWS - 1) // _BLOCK_ROWS,
            device=query.device,
        )
        validate_key_layout(key_layout)
        return _sparse_piper_attention_op(
            query,
            key,
            value,
            self._ratio_profile,
            key_layout,
            converted_scale,
            self._routing_mode,
            block_lengths,
            sparse_query_blocks,
            force_tile_ids,
            force_tile_counts,
        )


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    *,
    sparse_query_blocks: int | None,
    scale: float | None,
    block_lengths: torch.Tensor | None,
) -> float:
    tensors = (query, key, value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must use [batch,sequence,heads,features]")
    if (
        query.shape[:2] != key.shape[:2]
        or query.shape[3] != key.shape[3]
        or key.shape != value.shape
        or query.shape[2] < 1
        or key.shape[2] < 1
        or query.shape[2] % key.shape[2]
    ):
        raise ValueError(
            "sparse Piper requires matching batch/sequence/features "
            "and query heads divisible by KV heads"
        )
    if query.dtype not in SUPPORTED_DTYPES or any(
        tensor.dtype is not query.dtype for tensor in tensors
    ):
        raise ValueError("sparse Piper Q/K/V must share float16, bfloat16, or float32 dtype")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must share a device")
    if any(tensor.layout is not torch.strided or tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must have contiguous feature dimensions")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("sparse Piper is inference-only and does not support autograd")

    _batch, sequence, heads, head_dim = query.shape
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError("sparse Piper requires head_dim=64 or 128")
    if sequence < 64:
        raise ValueError("sparse Piper requires at least 64 sequence rows")
    if num_heads != heads:
        raise ValueError("sparse Piper ratio profile must contain one value per head")
    if block_lengths is not None:
        validate_block_lengths(
            block_lengths,
            sequence_length=sequence,
            device=query.device,
            context="sparse Piper attention",
            require_contiguous=True,
            check_values=False,
        )
    validate_sparse_query_blocks(
        sparse_query_blocks,
        query_blocks=(sequence + _BLOCK_ROWS - 1) // _BLOCK_ROWS,
        context="sparse Piper",
    )
    converted_scale = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(converted_scale) or converted_scale <= 0:
        raise ValueError("sparse Piper scale must be finite and positive")
    return converted_scale


def _run_sparse_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: _ResolvedRouteLayout,
    *,
    sparse_query_blocks: int | None,
    scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute validated sparse routing outside Dynamo tracing."""
    backend = _backend.select_attention_backend(query)
    num_routed_key_blocks = layout.num_routed_key_blocks
    routed_key_rows = num_routed_key_blocks * _BLOCK_ROWS
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    routed_key = key_head_major[:, :, :routed_key_rows]
    validate_routing_mode(routing_mode)
    force_tile_mask = None
    if force_tile_ids is not None and not layout.keeps_all_blocks:
        assert force_tile_counts is not None
        force_tile_mask = _backend.build_force_tile_bitmask(
            force_tile_ids, force_tile_counts, layout.num_key_blocks
        )
    # Take min/max routing summaries from the Q/K quantization pass
    # unless this call keeps every block, which needs no summaries at all.
    fuse_summaries = (
        backend is not None
        and not layout.keeps_all_blocks
        and _backend.use_fused_operand_preparation(query_head_major, key_head_major, routing_mode)
    )
    if not fuse_summaries:
        routes = packed_routes_from_sequences(
            query_head_major,
            routed_key,
            layout,
            routing_mode,
            block_lengths,
            skip_dense_routing=backend is not None and backend.skip_dense_routing,
            force_tile_mask=force_tile_mask,
        )
        if backend is None:
            return reference_sparse_piper_attention(
                query,
                key,
                value,
                routes,
                num_routed_key_blocks=num_routed_key_blocks,
                scale=scale,
                block_lengths=block_lengths,
                sparse_query_blocks=sparse_query_blocks,
            )
        value_head_major = value.transpose(1, 2)
        output = torch.empty_like(query, memory_format=torch.contiguous_format)
        prepared = backend.prepare(
            query_head_major,
            scale,
            num_routed_key_blocks=num_routed_key_blocks,
            combined_key=key_head_major,
            combined_value=value_head_major,
            block_lengths=block_lengths,
            sparse_query_blocks=sparse_query_blocks,
        ).with_routes(
            routes.indices,
            routes.head_keep_blocks,
            routes.route_head_offsets,
            bind=backend.bind_context,
        )
    else:
        assert backend is not None
        operands = backend.prepare(
            query_head_major,
            scale,
            num_routed_key_blocks=num_routed_key_blocks,
            combined_key=key_head_major,
            combined_value=value.transpose(1, 2),
            block_lengths=block_lengths,
            sparse_query_blocks=sparse_query_blocks,
            emit_summaries=True,
        )
        assert operands.query_summary is not None
        assert operands.key_summary is not None
        assert operands.key_aux is not None
        routes = packed_routes_from_summaries(
            operands.query_summary,
            operands.key_summary,
            operands.key_aux,
            layout,
            routing_mode,
            skip_dense_routing=backend.skip_dense_routing,
            force_tile_mask=force_tile_mask,
        )
        prepared = operands.with_routes(
            routes.indices,
            routes.head_keep_blocks,
            routes.route_head_offsets,
            bind=backend.bind_context,
        )
        # Release the routing summaries, then allocate the output, so neither
        # coexists with the other or with routing's scratch storage.
        del operands
        output = torch.empty_like(query, memory_format=torch.contiguous_format)
    prepared.launch(prepared.query, output.transpose(1, 2))
    return output


@torch.library.custom_op(
    "piper_kernels::sparse_piper_attention",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _sparse_piper_attention_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_key_layout_storage(key_layout, query.shape[1])
    return _run_sparse_piper_attention(
        query,
        key,
        value,
        resolve_route_layout(ratio_profile.units, key_layout, query.device),
        sparse_query_blocks=sparse_query_blocks,
        scale=scale,
        routing_mode=routing_mode,
        block_lengths=block_lengths,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )


@_sparse_piper_attention_op.register_fake
def _sparse_piper_attention_op_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _ratio_profile: _SparsePiperRatioProfile,
    key_layout: SparsePiperKeyLayout,
    _scale: float,
    _routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
    _sparse_query_blocks: int | None = None,
    _force_tile_ids: torch.Tensor | None = None,
    _force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_key_layout_storage(key_layout, query.shape[1])
    return torch.empty_like(query, memory_format=torch.contiguous_format)
