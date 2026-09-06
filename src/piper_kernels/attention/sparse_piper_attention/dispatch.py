"""Public orchestration for sparse Piper Attention."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS as _BLOCK_ROWS

from . import _backend
from ._block_layout import validate_block_lengths, validate_sparse_query_blocks
from ._budget import (
    _RATIO_SCALE,
    _normalize_head_keep_ratios,
    _resolve_route_layout,
    _ResolvedRouteLayout,
)
from ._routing import packed_routes_from_sequences
from ._routing_modes import (
    _ROUTING_NAME_BY_MODE,
    routing_mode_from_name,
    validate_routing_mode,
)
from .reference import reference_sparse_piper_attention


class SparsePiperAttention(torch.nn.Module):
    """Sparse attention with no derived state beyond its immutable ratio profile."""

    def __init__(
        self,
        head_keep_ratios: Sequence[float] | torch.Tensor,
        *,
        routing: str = "minmax",
    ) -> None:
        super().__init__()
        self._head_keep_ratio_units = _normalize_head_keep_ratios(head_keep_ratios)
        self._routing_mode = routing_mode_from_name(routing)

    @property
    def head_keep_ratios(self) -> tuple[float, ...]:
        """Return the device-independent semantic ratio profile."""
        return tuple(units / _RATIO_SCALE for units in self._head_keep_ratio_units)

    @property
    def routing(self) -> str:
        """Return the block-routing policy selected for this module."""
        return _ROUTING_NAME_BY_MODE[self._routing_mode]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        sparse_key_blocks: int,
        sparse_query_blocks: int | None = None,
        scale: float | None = None,
        block_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route leading query blocks over a sparse K64 prefix and dense suffix.

        Without ``block_lengths``, Q/K/V use compact BF16
        ``[batch, sequence, heads, 128]`` storage. Supplying one valid-prefix
        length per physical K64 block selects internally padded storage and
        returns that same physical layout; padded query outputs are unspecified.
        ``sparse_query_blocks`` optionally limits routing to the leading query
        blocks; all later query blocks attend every key block densely.
        """
        converted_scale = _validate_inputs(
            query,
            key,
            value,
            self._head_keep_ratio_units,
            sparse_key_blocks=sparse_key_blocks,
            sparse_query_blocks=sparse_query_blocks,
            scale=scale,
            block_lengths=block_lengths,
        )
        return _sparse_piper_attention_op(
            query,
            key,
            value,
            list(self._head_keep_ratio_units),
            sparse_key_blocks,
            converted_scale,
            self._routing_mode,
            block_lengths,
            sparse_query_blocks,
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
    head_keep_ratio_units: tuple[int, ...],
    *,
    sparse_key_blocks: int,
    sparse_query_blocks: int | None,
    scale: float | None,
    block_lengths: torch.Tensor | None,
) -> float:
    tensors = (query, key, value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must use [batch,sequence,heads,features]")
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("sparse Piper requires equal Q/K/V shapes")
    if query.dtype is not torch.bfloat16 or any(
        tensor.dtype is not query.dtype for tensor in tensors
    ):
        raise ValueError("sparse Piper Q/K/V must use bfloat16")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must share a device")
    if any(tensor.layout is not torch.strided or tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("sparse Piper Q/K/V must have contiguous feature dimensions")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("sparse Piper is inference-only and does not support autograd")

    _batch, sequence, heads, head_dim = query.shape
    if head_dim != 128:
        raise ValueError("sparse Piper requires head_dim=128")
    if sequence < 64:
        raise ValueError("sparse Piper requires at least 64 sequence rows")
    if len(head_keep_ratio_units) != heads:
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


def _run_sparse_piper_attention(
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
) -> torch.Tensor:
    """Execute validated sparse routing outside Dynamo tracing."""
    sparse_key_rows = sparse_key_blocks * _BLOCK_ROWS
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    sparse_key = key_head_major[:, :, :sparse_key_rows]
    validate_routing_mode(routing_mode)
    routes = packed_routes_from_sequences(
        query_head_major,
        sparse_key,
        layout,
        routing_mode,
        block_lengths,
    )

    backend = _backend.select_attention_backend(query)
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
        routes.indices,
        routes.head_keep_blocks,
        scale,
        sparse_key_blocks=sparse_key_blocks,
        route_head_offsets=routes.route_head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
        block_lengths=block_lengths,
        sparse_query_blocks=sparse_query_blocks,
    )
    backend.launch(prepared, output.transpose(1, 2))
    return output


@torch.library.custom_op("piper_kernels::sparse_piper_attention", mutates_args=())
def _sparse_piper_attention_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    scale: float,
    routing_mode: int,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
) -> torch.Tensor:
    layout = _resolve_route_layout(
        tuple(head_keep_ratio_units),
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
    )


@_sparse_piper_attention_op.register_fake
def _sparse_piper_attention_op_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    _scale: float,
    _routing_mode: int,
    _block_lengths: torch.Tensor | None = None,
    _sparse_query_blocks: int | None = None,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
