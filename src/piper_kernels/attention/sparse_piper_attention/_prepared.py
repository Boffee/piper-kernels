"""Shared quantized sparse-Piper storage, validation, and launch binding; no accelerator runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import (
    QUERY_SCALE_ROWS,
    SUPPORTED_HEAD_DIMS,
    TILE_ROWS,
    ValueFormat,
)

from ._contracts import SUPPORTED_DTYPES, validate_sparse_query_blocks

if TYPE_CHECKING:
    from ._contracts import BindAttention, LaunchAttention

_DO_NOT_SPECIALIZE_ARGUMENTS = (
    "query_block_offset",
    "global_query_block_offset",
    "logical_sequence_length",
    "sparse_key_blocks",
    "sparse_query_blocks",
    "stride_rb",
    "stride_rq",
)


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperContext:
    """Sequence-global K/V and policy metadata validated or produced by preparation.

    Zero routes per query denotes canonical full keep without a route list.
    Each backend validates whether it implements that execution mode.
    Caller-supplied route lists retain their ordering and semantics. K/V tensors
    carry KV heads; keep budgets and route offsets carry query heads.

    K/V, scales, means, and policy tensors remain unchanged for the bound
    launcher's lifetime. Producers write V in ``value_format`` directly;
    binding never creates another V representation. Rebind after mutation.
    """

    key: torch.Tensor
    value: torch.Tensor
    value_format: ValueFormat
    key_scale: torch.Tensor
    key_center: torch.Tensor
    key_center_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_mean: torch.Tensor
    route_head_offsets: torch.Tensor
    head_keep_blocks: torch.Tensor
    routes_per_query: int
    block_lengths: torch.Tensor | None
    sparse_key_blocks: int
    sparse_query_blocks: int | None
    logical_sequence_length: int


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperQuery:
    """Quantized storage and routes validated against one context and global offset."""

    data: torch.Tensor
    scale: torch.Tensor
    routes: torch.Tensor
    global_block_offset: int = 0


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperAttention:
    """Sequence-global context paired with one full or local query range."""

    launch: LaunchAttention
    query: _PreparedSparsePiperQuery

    @property
    def context(self) -> _PreparedSparsePiperContext:
        """The bound launch owns the immutable sequence-global operands."""
        return self.launch.context


@dataclass(frozen=True, slots=True)
class _PreparedSparsePiperOperands:
    """Quantized Q/K/V operands before routes are known.

    ``query_summary``/``key_summary``/``key_aux`` are the min/max routing
    summaries, present only when preparation was asked to emit them.
    Key summaries cover only the sparse prefix; query summaries cover all blocks.
    """

    key: torch.Tensor
    value: torch.Tensor
    value_format: ValueFormat
    key_scale: torch.Tensor
    key_center: torch.Tensor
    key_center_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_mean: torch.Tensor
    query: torch.Tensor
    query_scale: torch.Tensor
    block_lengths: torch.Tensor | None
    sparse_key_blocks: int
    sparse_query_blocks: int | None
    logical_sequence_length: int
    query_summary: torch.Tensor | None = None
    key_summary: torch.Tensor | None = None
    key_aux: torch.Tensor | None = None

    def with_routes(
        self,
        routes: torch.Tensor,
        head_keep_blocks: torch.Tensor,
        route_head_offsets: torch.Tensor,
        *,
        bind: BindAttention,
    ) -> _PreparedSparsePiperAttention:
        """Bind selected routes to these quantized operands."""
        context = _PreparedSparsePiperContext(
            key=self.key,
            value=self.value,
            value_format=self.value_format,
            key_scale=self.key_scale,
            key_center=self.key_center,
            key_center_scale=self.key_center_scale,
            value_scale_multiplier=self.value_scale_multiplier,
            value_mean=self.value_mean,
            route_head_offsets=route_head_offsets,
            head_keep_blocks=head_keep_blocks,
            routes_per_query=routes.shape[2],
            block_lengths=self.block_lengths,
            sparse_key_blocks=self.sparse_key_blocks,
            sparse_query_blocks=self.sparse_query_blocks,
            logical_sequence_length=self.logical_sequence_length,
        )
        query = _PreparedSparsePiperQuery(
            data=self.query,
            scale=self.query_scale,
            routes=routes,
        )
        return _PreparedSparsePiperAttention(launch=bind(context), query=query)


@dataclass(frozen=True, slots=True)
class _ValidatedAttentionLaunch:
    """Backend-neutral kernel operands resolved by common launch validation."""

    batch: int
    heads: int
    head_dim: int
    query_block_offset: int
    query_block_count: int
    global_query_block_offset: int
    query_storage_sequence_length: int
    storage_sequence_length: int
    output_sequence_length: int
    logical_sequence_length: int
    sparse_key_blocks: int
    sparse_query_blocks: int
    route_strides: tuple[int, int, int]
    output_strides: tuple[int, int, int]
    coarse_strides: tuple[int, int, int]
    gate_strides: tuple[int, int, int]
    coarse_output: torch.Tensor
    coarse_gate: torch.Tensor
    block_lengths: torch.Tensor
    mask_block_lengths: bool
    mask_ragged_tail: bool
    has_dense_query_suffix: bool
    apply_coarse_residual: bool
    skip_dense_routing: bool

    @property
    def query_rows(self) -> int:
        """Return the number of output rows covered by this launch."""
        return self.query_block_count * TILE_ROWS


def _resolve_query_block_range(
    query_state: _PreparedSparsePiperQuery,
    query_block_offset: int,
    query_block_count: int | None,
) -> tuple[int, int, int]:
    """Validate a local launch range and return its size and global offset."""
    stored_query_blocks = query_state.data.shape[2] // TILE_ROWS
    if isinstance(query_block_offset, bool) or not isinstance(query_block_offset, int):
        raise TypeError("sparse Piper query block offset must be an integer")
    if query_block_count is not None and (
        isinstance(query_block_count, bool) or not isinstance(query_block_count, int)
    ):
        raise TypeError("sparse Piper query block count must be an integer or None")
    if not 0 <= query_block_offset < stored_query_blocks:
        raise ValueError("sparse Piper query block offset must fit the prepared query storage")
    resolved_query_block_count = (
        stored_query_blocks - query_block_offset if query_block_count is None else query_block_count
    )
    if (
        resolved_query_block_count < 1
        or query_block_offset + resolved_query_block_count > stored_query_blocks
    ):
        raise ValueError("sparse Piper query block range must fit the prepared query storage")
    return (
        stored_query_blocks,
        resolved_query_block_count,
        query_state.global_block_offset + query_block_offset,
    )


def validate_attention_launch(
    context: _PreparedSparsePiperContext,
    query_state: _PreparedSparsePiperQuery,
    output: torch.Tensor,
    query_block_offset: int,
    query_block_count: int | None,
    coarse_output: torch.Tensor | None,
    coarse_gate: torch.Tensor | None,
) -> _ValidatedAttentionLaunch:
    """Validate output, epilogue, and range metadata for prepared operands.

    Context and query storage were validated or produced by preparation. Only
    caller-supplied launch arguments need validation here.
    """
    query = query_state.data
    batch, heads, query_storage_sequence_length, head_dim = query.shape
    logical_sequence_length = context.logical_sequence_length
    storage_sequence_length = context.key.shape[2]
    has_block_lengths = context.block_lengths is not None
    stored_query_blocks, resolved_query_block_count, global_query_block_offset = (
        _resolve_query_block_range(
            query_state,
            query_block_offset,
            query_block_count,
        )
    )
    output_sequence_length = (
        resolved_query_block_count * TILE_ROWS
        if has_block_lengths
        else min(
            resolved_query_block_count * TILE_ROWS,
            logical_sequence_length - global_query_block_offset * TILE_ROWS,
        )
    )
    if (
        output.shape != (batch, heads, output_sequence_length, head_dim)
        or output.dtype not in SUPPORTED_DTYPES
        or output.device != query.device
        or output.stride(-1) != 1
    ):
        raise ValueError("sparse Piper output must match the query block range")
    has_coarse_residual = coarse_output is not None or coarse_gate is not None
    if (coarse_output is None) != (coarse_gate is None):
        raise ValueError("coarse output and coarse gate must be supplied together")
    if has_coarse_residual:
        assert coarse_output is not None
        assert coarse_gate is not None
        if (
            coarse_output.shape != (batch, heads, stored_query_blocks, head_dim)
            or coarse_output.dtype is not torch.float32
            or coarse_output.device != query.device
            or coarse_output.stride(-1) != 1
        ):
            raise ValueError("sparse Piper coarse output must be FP32 [batch,heads,Q64,D64/D128]")
        if (
            coarse_gate.shape != (batch, output_sequence_length, heads, head_dim)
            or coarse_gate.dtype is not output.dtype
            or coarse_gate.device != query.device
            or coarse_gate.stride(-1) != 1
        ):
            raise ValueError("sparse Piper coarse gate must match the local attention output")
    has_dense_query_suffix = context.sparse_query_blocks is not None
    sparse_query_blocks = (
        storage_sequence_length // TILE_ROWS
        if context.sparse_query_blocks is None
        else context.sparse_query_blocks
    )
    mask_ragged_tail = not has_block_lengths and logical_sequence_length != storage_sequence_length
    resolved_coarse_output = context.key_scale if coarse_output is None else coarse_output
    resolved_coarse_gate = output if coarse_gate is None else coarse_gate
    block_lengths_operand = (
        context.head_keep_blocks if context.block_lengths is None else context.block_lengths
    )
    # The sparse prefix holds complete blocks; the dense suffix visits a compact
    # ragged tile last.
    return _ValidatedAttentionLaunch(
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        query_block_offset=query_block_offset,
        query_block_count=resolved_query_block_count,
        global_query_block_offset=global_query_block_offset,
        query_storage_sequence_length=query_storage_sequence_length,
        storage_sequence_length=storage_sequence_length,
        output_sequence_length=output_sequence_length,
        logical_sequence_length=logical_sequence_length,
        sparse_key_blocks=context.sparse_key_blocks,
        sparse_query_blocks=sparse_query_blocks,
        route_strides=(
            query_state.routes.stride(0),
            query_state.routes.stride(1),
            query_state.routes.stride(2),
        ),
        output_strides=(output.stride(0), output.stride(1), output.stride(2)),
        coarse_strides=(
            (0, 0, 0)
            if coarse_output is None
            else (
                coarse_output.stride(0),
                coarse_output.stride(1),
                coarse_output.stride(2),
            )
        ),
        gate_strides=(
            (0, 0, 0)
            if coarse_gate is None
            else (coarse_gate.stride(0), coarse_gate.stride(2), coarse_gate.stride(1))
        ),
        coarse_output=resolved_coarse_output,
        coarse_gate=resolved_coarse_gate,
        block_lengths=block_lengths_operand,
        mask_block_lengths=has_block_lengths,
        mask_ragged_tail=mask_ragged_tail,
        has_dense_query_suffix=has_dense_query_suffix,
        apply_coarse_residual=has_coarse_residual,
        skip_dense_routing=context.routes_per_query == 0,
    )


@dataclass(frozen=True, slots=True)
class _ContextLauncher:
    """Validate each query-window launch against a bound context and dispatch to a backend."""

    context: _PreparedSparsePiperContext
    _dispatch: Callable[
        [
            _PreparedSparsePiperContext,
            _PreparedSparsePiperQuery,
            torch.Tensor,
            _ValidatedAttentionLaunch,
        ],
        None,
    ]

    def __call__(
        self,
        query_state: _PreparedSparsePiperQuery,
        output: torch.Tensor,
        *,
        query_block_offset: int = 0,
        query_block_count: int | None = None,
        coarse_output: torch.Tensor | None = None,
        coarse_gate: torch.Tensor | None = None,
    ) -> None:
        """Execute a query window against the bound global context."""
        launch = validate_attention_launch(
            self.context,
            query_state,
            output,
            query_block_offset,
            query_block_count,
            coarse_output,
            coarse_gate,
        )
        self._dispatch(self.context, query_state, output, launch)


def _prepare_sparse_piper_context_from_quantized(  # noqa: PLR0912, PLR0913
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    key_center_scale: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_blocks: torch.Tensor,
    route_head_offsets: torch.Tensor,
    *,
    value_format: ValueFormat,
    sparse_key_blocks: int,
    routes_per_query: int,
    logical_sequence_length: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
) -> _PreparedSparsePiperContext:
    """Validate sequence-global quantized K/V storage and routing policy.

    ``block_lengths`` opts into internally padded K64 storage. Each entry gives
    the valid prefix length in ``[1, 64]`` of one physical block and supersedes
    ``logical_sequence_length`` for masking. Without it, storage is the padded
    form of the compact logical sequence with at most one ragged tail.
    """
    if key.ndim != 4 or key.dtype is not torch.int8:
        raise ValueError(
            "quantized sparse Piper K must be [batch,heads,storage_sequence,D64/D128] INT8"
        )
    batch, heads, storage_sequence_length, head_dim = key.shape
    if (
        head_dim not in SUPPORTED_HEAD_DIMS
        or storage_sequence_length < TILE_ROWS
        or storage_sequence_length % TILE_ROWS
    ):
        raise ValueError("quantized sparse Piper requires K64-aligned D64/D128 key storage")
    if block_lengths is None:
        if (
            logical_sequence_length < TILE_ROWS
            or logical_sequence_length > storage_sequence_length
            or (logical_sequence_length + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS
            != storage_sequence_length
        ):
            raise ValueError("quantized sparse Piper storage must be the padded logical sequence")
    elif not 1 <= logical_sequence_length <= storage_sequence_length:
        raise ValueError("quantized sparse Piper logical length must fit block-length storage")
    if (
        value.shape != value_format.shape(batch, heads, storage_sequence_length, head_dim)
        or value.dtype is not torch.int8
    ):
        raise ValueError(f"quantized sparse Piper V must use {value_format.name} INT8 storage")
    tile_count = storage_sequence_length // TILE_ROWS
    if key_scale.shape != (batch, heads, tile_count):
        raise ValueError("quantized sparse Piper K scales must contain one value per K64")
    if (
        key_center.shape != (batch, heads, tile_count, 2, head_dim)
        or key_center.dtype is not torch.int8
    ):
        raise ValueError("quantized sparse Piper K centers must be [batch,heads,K64,2,D] INT8")
    if key_center_scale.shape != (batch, heads, tile_count, 2):
        raise ValueError("quantized sparse Piper K center scales must contain two values per K64")
    if value_scale_multiplier.shape != (batch, heads, tile_count, 1):
        raise ValueError("quantized sparse Piper V scales must contain one value per K64")
    if (
        value_mean.shape != (batch, heads, tile_count, head_dim)
        or value_mean.dtype is not torch.bfloat16
    ):
        raise ValueError("quantized sparse Piper V mean must be [batch,heads,K64,D64/D128] BF16")
    # Layout construction owns the value-range invariants documented
    # above. Inspecting device values here would add a validation kernel or a host
    # synchronization to every launch; only launch-critical tensor properties
    # are checked on this hot path.
    if block_lengths is not None and (
        block_lengths.shape != (tile_count,) or block_lengths.dtype is not torch.int32
    ):
        raise ValueError("quantized sparse Piper block lengths must be one INT32 value per K64")
    scales = key_scale, key_center_scale, value_scale_multiplier
    if any(scale.dtype is not torch.float32 for scale in scales):
        raise ValueError("quantized sparse Piper scales must use FP32")
    tensors = (
        key,
        key_center,
        value,
        value_mean,
        *scales,
        head_keep_blocks,
        route_head_offsets,
        *((block_lengths,) if block_lengths is not None else ()),
    )
    if any(tensor.device != key.device for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must share a device")
    if any(tensor.layout is not torch.strided or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must be contiguous strided tensors")
    # The sparse prefix holds complete blocks, so a compact ragged tile is always
    # attended after the routes and never routed itself.
    complete_blocks = (
        tile_count if block_lengths is not None else logical_sequence_length // TILE_ROWS
    )
    if not 1 <= sparse_key_blocks <= complete_blocks:
        raise ValueError("quantized sparse Piper prefix must fit the complete K64 blocks")
    total_query_blocks = (
        tile_count
        if block_lengths is not None
        else (logical_sequence_length + TILE_ROWS - 1) // TILE_ROWS
    )
    validate_sparse_query_blocks(
        sparse_query_blocks,
        query_blocks=total_query_blocks,
        context="quantized sparse Piper",
    )
    if (
        head_keep_blocks.ndim != 1
        or head_keep_blocks.numel() < 1
        or heads < 1
        or head_keep_blocks.numel() % heads
        or head_keep_blocks.dtype is not torch.int32
    ):
        raise ValueError("quantized sparse Piper head keep blocks must be one INT32 value per head")
    if (
        route_head_offsets.shape != (head_keep_blocks.numel() + 1,)
        or route_head_offsets.dtype is not torch.int32
    ):
        raise ValueError("quantized sparse Piper route offsets must be an INT32 head vector")
    if routes_per_query < 0:
        raise ValueError("route count must be nonnegative")

    return _PreparedSparsePiperContext(
        key=key,
        value=value,
        value_format=value_format,
        key_scale=key_scale,
        key_center=key_center,
        key_center_scale=key_center_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_mean=value_mean,
        route_head_offsets=route_head_offsets,
        head_keep_blocks=head_keep_blocks,
        routes_per_query=routes_per_query,
        block_lengths=block_lengths,
        sparse_key_blocks=sparse_key_blocks,
        sparse_query_blocks=sparse_query_blocks,
        logical_sequence_length=logical_sequence_length,
    )


def _prepare_sparse_piper_query_from_quantized(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    routes: torch.Tensor,
    context: _PreparedSparsePiperContext,
    *,
    global_block_offset: int = 0,
) -> _PreparedSparsePiperQuery:
    """Validate query-local quantized storage and locate it globally."""
    if query.ndim != 4 or query.dtype is not torch.int8:
        raise ValueError(
            "quantized sparse Piper Q must be [batch,heads,storage_sequence,D64/D128] INT8"
        )
    batch, heads, storage_sequence_length, head_dim = query.shape
    if (
        batch != context.key.shape[0]
        or heads < 1
        or heads != context.head_keep_blocks.numel()
        or context.key.shape[1] < 1
        or heads % context.key.shape[1]
        or head_dim != context.key.shape[-1]
        or storage_sequence_length < TILE_ROWS
        or storage_sequence_length % TILE_ROWS
    ):
        raise ValueError(
            "quantized sparse Piper requires compatible K64-aligned D64/D128 Q storage"
        )
    query_block_count = storage_sequence_length // TILE_ROWS
    total_query_blocks = context.key.shape[2] // TILE_ROWS
    if (
        isinstance(global_block_offset, bool)
        or not isinstance(global_block_offset, int)
        or global_block_offset < 0
        or global_block_offset + query_block_count > total_query_blocks
    ):
        raise ValueError("quantized sparse Piper Q storage must fit the global sequence")
    if query_scale.shape != (batch, heads, storage_sequence_length // QUERY_SCALE_ROWS):
        raise ValueError("quantized sparse Piper Q scales must contain one value per Q32")
    if routes.shape != (batch, query_block_count, context.routes_per_query):
        raise ValueError("quantized sparse Piper routes must match batch/query/packed budgets")
    if query_scale.dtype is not torch.float32:
        raise ValueError("quantized sparse Piper Q scales must use FP32")
    if routes.dtype is not torch.uint16:
        raise ValueError("quantized sparse Piper routes must use UINT16")
    tensors = query, query_scale, routes
    if any(tensor.device != context.key.device for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must share a device")
    if any(tensor.layout is not torch.strided or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quantized sparse Piper operands must be contiguous strided tensors")
    return _PreparedSparsePiperQuery(
        data=query,
        scale=query_scale,
        routes=routes,
        global_block_offset=global_block_offset,
    )
