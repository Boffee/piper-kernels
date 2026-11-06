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
from ._budget import (
    _RATIO_SCALE,
    _normalize_head_keep_ratios,
    _resolve_head_keep_blocks,
    _resolve_route_layout,
    _ResolvedRouteLayout,
)
from ._dtype import SUPPORTED_DTYPES
from ._force_tiles import validate_force_tiles
from ._regions import (
    SparsePiperKeyRegion,
    resolve_region_keep_blocks,
    resolve_region_route_layout,
    validate_key_regions,
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

    def resolve_head_keep_blocks(self, sparse_key_blocks: int) -> tuple[int, ...]:
        """Return the physical K64 keep budget for each query head, in head order.

        ``sparse_key_blocks`` must be an integer in [1, 65536]. Budgets use the
        same rounding as attention, retain at least one block per head, and
        exclude the always-dense key suffix. Each forced count must fit the
        smallest returned budget as well as the caller's ID capacity.

        Call this host-only configuration query outside compiled forward calls,
        using the current module and a concrete sparse prefix size. It allocates
        no tensors and accesses no device state.
        """
        return _resolve_head_keep_blocks(self._ratio_profile.units, sparse_key_blocks)

    def resolve_region_keep_blocks(
        self, key_regions: tuple[SparsePiperKeyRegion, ...]
    ) -> tuple[tuple[int, ...], ...]:
        """Return per-head budgets in region order without accessing device state.

        Sparse regions round independently across heads, retaining at least one
        block per head per region. Dense regions return their full block count.
        Call outside compiled forward calls. Forced IDs in each sparse region
        must fit every head's budget for that region.
        """
        validate_key_regions(key_regions)
        return resolve_region_keep_blocks(
            self._ratio_profile.units,
            tuple(region.num_blocks for region in key_regions),
            tuple(region.sparse for region in key_regions),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        sparse_key_blocks: int | None = None,
        key_regions: tuple[SparsePiperKeyRegion, ...] | None = None,
        sparse_query_blocks: int | None = None,
        scale: float | None = None,
        block_lengths: torch.Tensor | None = None,
        force_tile_ids: torch.Tensor | None = None,
        force_tile_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend with independent regional budgets or a sparse key prefix.

        Without ``block_lengths``, Q/K/V use compact FP16, BF16, or FP32
        ``[batch, sequence, heads, head_dim]`` storage. Supplying one valid-prefix
        length per physical K64 block selects internally padded storage and
        returns that same physical layout; padded query outputs are unspecified.
        ``sparse_query_blocks`` optionally limits routing to the leading query
        blocks; all later query blocks attend every key block densely. Query heads
        may be an integer multiple of the shared K/V head count. Keep ratios and
        routes belong to query heads; K/V preparation and storage remain shared.

        Supply either ``sparse_key_blocks`` for the existing prefix layout or
        experimental ``key_regions`` for a complete ordered partition of key
        storage. Regions must cover exactly the sequence's complete K64 blocks
        (at most 65536); compact partial tails require padding. Sparse regions
        apply the head ratios independently, and dense regions keep all blocks,
        within one softmax. ``resolve_region_keep_blocks`` reports their budgets.
        Region calls support standalone compilation; projection/output compiler
        fusions currently recognize only the prefix form.

        Experimental fixed-budget forced routing takes contiguous device INT32
        ``force_tile_ids[query_blocks, capacity]`` and ``force_tile_counts[query_blocks]``.
        Each row's active prefix is shared across batches and heads. Counts must
        lie in [0, capacity], and active IDs must be unique. For prefix layouts,
        IDs lie in [0, sparse_key_blocks) and the active count must fit every head's
        budget from ``resolve_head_keep_blocks``.
        These value requirements are caller preconditions, not runtime tensor-content checks.
        Forced tiles consume the existing budget; the remainder is selected by
        score. ID/count contents may change at fixed shapes without recompiling.
        Compatible projection/output fusions preserve these runtime inputs,
        including chunked query execution and the independent coarse residual.
        With ``key_regions``, IDs lie within the complete physical block count. Forced IDs
        in each sparse region consume that region's budget independently; IDs
        in dense regions are already included. These counts must fit every head's
        regional budget, as a caller precondition.
        """
        if (sparse_key_blocks is None) == (key_regions is None):
            raise ValueError("supply exactly one of sparse_key_blocks or key_regions")
        if key_regions is not None:
            sparse_key_blocks = validate_key_regions(key_regions)
        assert sparse_key_blocks is not None
        converted_scale = _validate_inputs(
            query,
            key,
            value,
            self._ratio_profile.num_heads,
            sparse_key_blocks=sparse_key_blocks,
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
        if key_regions is not None:
            if query.shape[1] != sparse_key_blocks * _BLOCK_ROWS:
                raise ValueError(
                    "key_regions must cover exactly the sequence's complete K64 blocks"
                )
            return _sparse_piper_region_attention_op(
                query,
                key,
                value,
                self._ratio_profile,
                [region.num_blocks for region in key_regions],
                [region.sparse for region in key_regions],
                converted_scale,
                self._routing_mode,
                block_lengths,
                sparse_query_blocks,
                force_tile_ids,
                force_tile_counts,
            )
        return _sparse_piper_attention_op(
            query,
            key,
            value,
            self._ratio_profile,
            sparse_key_blocks,
            converted_scale,
            self._routing_mode,
            block_lengths,
            sparse_query_blocks,
            force_tile_ids,
            force_tile_counts,
        )


def _validate_sparse_key_blocks(
    sparse_key_blocks: int,
    *,
    available_sparse_key_blocks: int,
) -> None:
    if isinstance(sparse_key_blocks, bool):
        raise TypeError("sparse_key_blocks must be an integer")
    if torch.compiler.is_compiling():
        torch._check(
            sparse_key_blocks >= 1,
            lambda: "sparse_key_blocks must be positive",
        )
        torch._check(
            sparse_key_blocks <= available_sparse_key_blocks,
            lambda: "sparse_key_blocks cannot exceed the sequence block count",
        )
        return
    if not isinstance(sparse_key_blocks, int):
        raise TypeError("sparse_key_blocks must be an integer")
    if not 1 <= sparse_key_blocks <= available_sparse_key_blocks:
        raise ValueError("sparse_key_blocks must fit the sequence block count")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    *,
    sparse_key_blocks: int,
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
    _validate_sparse_key_blocks(
        sparse_key_blocks,
        available_sparse_key_blocks=sequence // _BLOCK_ROWS,
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


def _run_sparse_piper_attention(  # noqa: PLR0913
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: _ResolvedRouteLayout,
    *,
    sparse_key_blocks: int,
    sparse_query_blocks: int | None,
    scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute validated sparse routing outside Dynamo tracing."""
    backend = _backend.select_attention_backend(query)
    sparse_key_rows = sparse_key_blocks * _BLOCK_ROWS
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    sparse_key = key_head_major[:, :, :sparse_key_rows]
    validate_routing_mode(routing_mode)
    force_tile_mask = None
    if force_tile_ids is not None and not layout.keeps_all_blocks(sparse_key_blocks):
        assert force_tile_counts is not None
        force_tile_mask = _backend.build_force_tile_bitmask(
            force_tile_ids, force_tile_counts, sparse_key_blocks
        )
    # Take min/max routing summaries from the Q/K quantization pass
    # unless this call keeps every block, which needs no summaries at all.
    fuse_summaries = (
        backend is not None
        and not layout.keeps_all_blocks(sparse_key_blocks)
        and _backend.use_fused_operand_preparation(query_head_major, key_head_major, routing_mode)
    )
    if not fuse_summaries:
        routes = packed_routes_from_sequences(
            query_head_major,
            sparse_key,
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
                sparse_key_blocks=sparse_key_blocks,
                scale=scale,
                block_lengths=block_lengths,
                sparse_query_blocks=sparse_query_blocks,
            )
        value_head_major = value.transpose(1, 2)
        output = torch.empty_like(query, memory_format=torch.contiguous_format)
        prepared = backend.prepare(
            query_head_major,
            scale,
            sparse_key_blocks=sparse_key_blocks,
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
            sparse_key_blocks=sparse_key_blocks,
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
    sparse_key_blocks: int,
    scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    layout = _resolve_route_layout(
        ratio_profile.units,
        sparse_key_blocks,
        query.device,
    )
    return _run_sparse_piper_attention(
        query,
        key,
        value,
        layout,
        sparse_key_blocks=sparse_key_blocks,
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
    _sparse_key_blocks: int,
    _scale: float,
    _routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
    _sparse_query_blocks: int | None = None,
    _force_tile_ids: torch.Tensor | None = None,
    _force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)


@torch.library.custom_op(
    "piper_kernels::sparse_piper_region_attention",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _sparse_piper_region_attention_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ratio_profile: _SparsePiperRatioProfile,
    region_blocks: list[int],
    region_sparse: list[bool],
    scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    layout = resolve_region_route_layout(
        ratio_profile.units, region_blocks, region_sparse, query.device
    )
    # The existing consumer already supports a complete absolute route list:
    # set its routeable extent to the entire sequence, leaving no implicit suffix.
    return _run_sparse_piper_attention(
        query,
        key,
        value,
        layout,
        sparse_key_blocks=sum(region_blocks),
        sparse_query_blocks=sparse_query_blocks,
        scale=scale,
        routing_mode=routing_mode,
        block_lengths=block_lengths,
        force_tile_ids=force_tile_ids,
        force_tile_counts=force_tile_counts,
    )


@_sparse_piper_region_attention_op.register_fake
def _sparse_piper_region_attention_op_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _ratio_profile: _SparsePiperRatioProfile,
    _region_blocks: list[int],
    _region_sparse: list[bool],
    _scale: float,
    _routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
    _sparse_query_blocks: int | None = None,
    _force_tile_ids: torch.Tensor | None = None,
    _force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
