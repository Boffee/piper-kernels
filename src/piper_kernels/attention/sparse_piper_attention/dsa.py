"""Exact DSA routing for sparse Piper Attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._budget import _UINT16_ROUTE_CAPACITY, _ResolvedRouteLayout

try:
    from .dsa_triton import block_summaries as _sm120_block_summaries
    from .dsa_triton import tiled_radix_select_packed_routes as _sm120_select_routes
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("triton"):
        raise
    _sm120_block_summaries = None
    _sm120_select_routes = None

_QUERY_CHUNK_BLOCKS = 384


@dataclass(frozen=True, slots=True)
class PackedDsaRoutes:
    """UINT16 routes packed by physical head for every query block."""

    indices: torch.Tensor
    head_offsets: torch.Tensor
    keep_blocks: torch.Tensor


def packed_dsa_routes_from_layout(
    query_blocks: torch.Tensor,
    key_blocks: torch.Tensor,
    layout: _ResolvedRouteLayout,
) -> PackedDsaRoutes:
    """Select exact FP32 DSA routes directly into packed UINT16 storage."""
    _validate_dsa_blocks(query_blocks, key_blocks)
    heads = query_blocks.shape[1]
    key_block_count = key_blocks.shape[2]
    _validate_route_layout(layout, heads, key_block_count, query_blocks.device)

    query_summary, key_max, key_min = _block_summaries(query_blocks, key_blocks)
    return packed_dsa_routes_from_summaries(
        query_summary,
        key_max,
        key_min,
        layout,
    )


def packed_dsa_routes_from_summaries(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    layout: _ResolvedRouteLayout,
) -> PackedDsaRoutes:
    """Select routes from existing exact Q64/K64 extrema summaries."""
    _validate_dsa_summaries(query_summary, key_max, key_min)
    heads = query_summary.shape[1]
    key_block_count = key_max.shape[2]
    _validate_route_layout(layout, heads, key_block_count, query_summary.device)

    indices = torch.empty(
        (query_summary.shape[0], query_summary.shape[2], layout.routes_per_query),
        dtype=torch.uint16,
        device=query_summary.device,
    )
    if _supports_sm120_summary_selector(query_summary):
        assert _sm120_select_routes is not None

        for start in range(0, query_summary.shape[2], _QUERY_CHUNK_BLOCKS):
            stop = min(start + _QUERY_CHUNK_BLOCKS, query_summary.shape[2])
            scores = _dsa_scores(query_summary[:, :, start:stop], key_max, key_min)
            _sm120_select_routes(
                scores,
                indices,
                layout.keep_blocks,
                layout.head_offsets,
                route_query_offset=start,
            )
    else:
        _select_portable_routes(query_summary, key_max, key_min, layout, indices)
    return PackedDsaRoutes(indices, layout.head_offsets, layout.keep_blocks)


def _validate_dsa_summaries(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
) -> None:
    if query_summary.ndim != 4 or key_max.ndim != 4 or key_min.shape != key_max.shape:
        raise ValueError("DSA summaries must use rank-four Q/max/min tensors")
    if query_summary.shape[:2] != key_max.shape[:2]:
        raise ValueError("DSA summary batch/head dimensions must match")
    if query_summary.shape[-1] != key_max.shape[-1]:
        raise ValueError("DSA summary feature dimensions must match")
    if query_summary.shape[2] < 1 or key_max.shape[2] < 1:
        raise ValueError("DSA summaries must contain query and key blocks")
    if query_summary.dtype is not torch.float32 or any(
        summary.dtype is not query_summary.dtype for summary in (key_max, key_min)
    ):
        raise ValueError("DSA summaries must use FP32")
    if any(summary.device != query_summary.device for summary in (key_max, key_min)):
        raise ValueError("DSA summaries must share a device")
    if not query_summary.is_contiguous():
        raise ValueError("DSA query summaries must be contiguous")
    if any(
        summary.stride(-1) != 1 or summary.stride(-2) != summary.shape[-1]
        for summary in (key_max, key_min)
    ):
        raise ValueError("DSA key summaries must have contiguous block features")


def _block_summaries(
    query_blocks: torch.Tensor,
    key_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _supports_sm120_selector(query_blocks, key_blocks):
        assert _sm120_block_summaries is not None

        return _sm120_block_summaries(query_blocks, key_blocks)
    return _portable_block_summaries(query_blocks, key_blocks)


def _dsa_scores(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
) -> torch.Tensor:
    """Contract exact FP32 scores while keeping only two score buffers."""
    batch, heads, query_count, head_dim = query_summary.shape
    key_count = key_max.shape[2]
    flat_query = query_summary.reshape(batch * heads, query_count, head_dim)
    flat_key_max = key_max.reshape(batch * heads, key_count, head_dim)
    flat_key_min = key_min.reshape(batch * heads, key_count, head_dim)
    scores = torch.bmm(flat_query, flat_key_max.transpose(1, 2))
    minimum_scores = torch.bmm(flat_query, flat_key_min.transpose(1, 2))
    torch.maximum(scores, minimum_scores, out=scores)
    return scores.reshape(batch, heads, query_count, key_count)


def _select_portable_routes(
    query_summary: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    layout: _ResolvedRouteLayout,
    output: torch.Tensor,
) -> None:
    offsets = layout.head_offsets.detach().cpu().tolist()
    keep_values = layout.keep_blocks.detach().cpu().tolist()
    for start in range(0, query_summary.shape[2], _QUERY_CHUNK_BLOCKS):
        stop = min(start + _QUERY_CHUNK_BLOCKS, query_summary.shape[2])
        scores = _dsa_scores(query_summary[:, :, start:stop], key_max, key_min)
        for head, count in enumerate(keep_values):
            selected = torch.argsort(
                scores[:, head],
                dim=-1,
                descending=True,
                stable=True,
            )[..., :count]
            selected = selected.sort(dim=-1).values.to(torch.uint16)
            output[:, start:stop, offsets[head] : offsets[head + 1]] = selected


def _portable_block_summaries(
    query_blocks: torch.Tensor,
    key_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = query_blocks.float()
    key = key_blocks.float()
    query_max, query_min = query.amax(dim=-2), query.amin(dim=-2)
    key_max, key_min = key.amax(dim=-2), key.amin(dim=-2)
    return query_max + query_min, key_max, key_min


def _supports_sm120_selector(query_blocks: torch.Tensor, key_blocks: torch.Tensor) -> bool:
    target = AcceleratorTarget.from_device(query_blocks.device)
    return (
        _sm120_block_summaries is not None
        and _sm120_select_routes is not None
        and target.is_cuda_capability(12, 0)
        and query_blocks.shape[-1] == 128
        and key_blocks.shape[-1] == 128
        and query_blocks.shape[-2] == 64
        and key_blocks.shape[-2] == 64
        and query_blocks.stride(-1) == 1
        and key_blocks.stride(-1) == 1
        and query_blocks.dtype in (torch.bfloat16, torch.float16)
        and key_blocks.dtype == query_blocks.dtype
    )


def _supports_sm120_summary_selector(query_summary: torch.Tensor) -> bool:
    target = AcceleratorTarget.from_device(query_summary.device)
    return _sm120_select_routes is not None and target.is_cuda_capability(12, 0)


def _validate_dsa_blocks(query_blocks: torch.Tensor, key_blocks: torch.Tensor) -> None:
    if query_blocks.ndim != 5 or key_blocks.ndim != 5:
        raise ValueError("DSA query and key blocks must be rank-five tensors")
    if query_blocks.shape[:2] != key_blocks.shape[:2]:
        raise ValueError("DSA query and key batch/head dimensions must match")
    if query_blocks.shape[-1] != key_blocks.shape[-1]:
        raise ValueError("DSA query and key feature dimensions must match")
    if query_blocks.shape[2] < 1 or key_blocks.shape[2] < 1:
        raise ValueError("DSA requires nonempty query and key block dimensions")
    if query_blocks.shape[-2] < 1 or key_blocks.shape[-2] < 1:
        raise ValueError("DSA blocks must contain at least one row")
    if query_blocks.device != key_blocks.device:
        raise ValueError("DSA query and key blocks must share a device")
    if not query_blocks.is_floating_point() or not key_blocks.is_floating_point():
        raise TypeError("DSA query and key blocks must be floating-point tensors")


def _validate_route_layout(
    layout: _ResolvedRouteLayout,
    heads: int,
    key_blocks: int,
    device: torch.device,
) -> None:
    if not 1 <= key_blocks <= _UINT16_ROUTE_CAPACITY:
        raise ValueError("DSA requires between 1 and 65,536 sparse key blocks")
    if layout.keep_blocks.shape != (heads,) or layout.head_offsets.shape != (heads + 1,):
        raise ValueError("DSA route layout does not match the attention head count")
    if layout.keep_blocks.device != device or layout.head_offsets.device != device:
        raise ValueError("DSA route layout must share the attention device")
