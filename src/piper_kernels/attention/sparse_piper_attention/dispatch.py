"""Public orchestration for sparse Piper Attention."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._budget import (
    _RATIO_SCALE,
    _normalize_head_keep_ratios,
    _resolve_route_layout,
    _ResolvedRouteLayout,
)
from .dsa import packed_dsa_routes_from_layout
from .reference import reference_sparse_piper_attention

try:
    from .gluon import (
        _launch_gluon_paired_routed_piper_attention as _launch_sm120_attention,
    )
    from .triton import (
        _prepare_folded_tile_scaled_routed_piper_attention as _prepare_sm120_attention,
    )
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("triton"):
        raise
    _launch_sm120_attention = None
    _prepare_sm120_attention = None


def _supports_sm120(target: AcceleratorTarget) -> bool:
    return _launch_sm120_attention is not None and target.is_cuda_capability(12, 0)


class SparsePiperAttention(torch.nn.Module):
    """Sparse attention with no derived state beyond its immutable ratio profile."""

    def __init__(
        self,
        head_keep_ratios: Sequence[float] | torch.Tensor,
    ) -> None:
        super().__init__()
        self._head_keep_ratio_units = _normalize_head_keep_ratios(head_keep_ratios)

    @property
    def head_keep_ratios(self) -> tuple[float, ...]:
        """Return the device-independent semantic ratio profile."""
        return tuple(units / _RATIO_SCALE for units in self._head_keep_ratio_units)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        sparse_key_blocks: int,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Route every query over a sparse K/V prefix and dense suffix."""
        converted_scale = _validate_inputs(
            query,
            key,
            value,
            self._head_keep_ratio_units,
            sparse_key_blocks=sparse_key_blocks,
            scale=scale,
        )
        return _sparse_piper_attention_op(
            query,
            key,
            value,
            list(self._head_keep_ratio_units),
            sparse_key_blocks,
            converted_scale,
        )


def _validate_sparse_key_blocks(
    sparse_key_blocks: int,
    *,
    sequence_blocks: int,
) -> None:
    if isinstance(sparse_key_blocks, bool):
        raise TypeError("sparse_key_blocks must be an integer")
    if torch.compiler.is_compiling():
        torch._check(
            sparse_key_blocks >= 1,
            lambda: "sparse_key_blocks must be positive",
        )
        torch._check(
            sparse_key_blocks <= sequence_blocks,
            lambda: "sparse_key_blocks cannot exceed the sequence block count",
        )
        return
    if not isinstance(sparse_key_blocks, int):
        raise TypeError("sparse_key_blocks must be an integer")
    if not 1 <= sparse_key_blocks <= sequence_blocks:
        raise ValueError("sparse_key_blocks must fit the sequence block count")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_keep_ratio_units: tuple[int, ...],
    *,
    sparse_key_blocks: int,
    scale: float | None,
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
    if sequence < 64 or sequence % 64:
        raise ValueError("sparse Piper requires a K64-aligned sequence")
    if len(head_keep_ratio_units) != heads:
        raise ValueError("sparse Piper ratio profile must contain one value per head")
    _validate_sparse_key_blocks(
        sparse_key_blocks,
        sequence_blocks=sequence // 64,
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
    scale: float,
    target_is_sm120: bool,
) -> torch.Tensor:
    """Execute validated sparse routing outside Dynamo tracing."""
    _batch, sequence, _heads, _head_dim = query.shape
    query_block_count = sequence // 64
    sparse_key_rows = sparse_key_blocks * 64
    query_head_major = query.transpose(1, 2)
    key_head_major = key.transpose(1, 2)
    value_head_major = value.transpose(1, 2)
    query_blocks = query_head_major.unflatten(2, (query_block_count, 64))
    key_blocks = key_head_major[:, :, :sparse_key_rows].unflatten(
        2,
        (sparse_key_blocks, 64),
    )
    routes = packed_dsa_routes_from_layout(
        query_blocks,
        key_blocks,
        layout,
    )

    if not target_is_sm120:
        return reference_sparse_piper_attention(
            query,
            key,
            value,
            routes,
            sparse_key_blocks=sparse_key_blocks,
            scale=scale,
        )

    assert _launch_sm120_attention is not None
    assert _prepare_sm120_attention is not None
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    prepared = _prepare_sm120_attention(
        query_blocks,
        key_blocks,
        routes.indices,
        routes.keep_blocks,
        scale,
        route_head_offsets=routes.head_offsets,
        combined_key=key_head_major,
        combined_value=value_head_major,
        attention_output=output.transpose(1, 2),
    )
    _launch_sm120_attention(prepared)
    return output


@torch.library.custom_op("piper_kernels::sparse_piper_attention", mutates_args=())
def _sparse_piper_attention_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    scale: float,
) -> torch.Tensor:
    layout = _resolve_route_layout(
        tuple(head_keep_ratio_units),
        sparse_key_blocks,
        query.device,
    )
    target = AcceleratorTarget.from_device(query.device)
    return _run_sparse_piper_attention(
        query,
        key,
        value,
        layout,
        sparse_key_blocks=sparse_key_blocks,
        scale=scale,
        target_is_sm120=_supports_sm120(target),
    )


@_sparse_piper_attention_op.register_fake
def _sparse_piper_attention_op_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    _scale: float,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
