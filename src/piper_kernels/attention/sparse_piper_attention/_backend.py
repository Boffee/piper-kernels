"""Operation-specific sparse-Piper selection using the operands' device."""

from dataclasses import replace
from functools import partial

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import SUPPORTED_HEAD_DIMS, ValueFormat

from ._contracts import (
    _MEAN_ROUTING,
    SUPPORTED_DTYPES,
    AttentionBackend,
    MinmaxScores,
    MinmaxScoreSchedule,
    SelectRoutes,
    SequenceSummaries,
    build_portable_force_tile_bitmask,
)
from ._nvidia import policy as nvidia_policy

_RDNA4_MINMAX_SMALL_QUERY_SCHEDULE = MinmaxScoreSchedule(32, 64, 16, lds_only_fences=True)
_RDNA4_MINMAX_LARGE_QUERY_SCHEDULE = MinmaxScoreSchedule(64, 32, 16, lds_only_fences=True)
_RDNA4_MINMAX_LARGE_QUERY_THRESHOLD = 128


def supports_amd_target(target: AcceleratorTarget) -> bool:
    """Return whether target runs the wave32 RDNA4 sparse-attention kernels."""
    return target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201")


def select_amd_minmax_score_schedule(query_blocks: int) -> MinmaxScoreSchedule:
    """Return the measured FP32 min/max routing-score tile schedule for RDNA4.

    All RDNA4 schedules enable LDS-only fences to omit global cache invalidations
    at shared-memory barriers without changing output bits; 64x32x16 wins from
    128 query blocks upward while 32x64x16 avoids idle waves on shorter chunks.
    """
    if query_blocks >= _RDNA4_MINMAX_LARGE_QUERY_THRESHOLD:
        return _RDNA4_MINMAX_LARGE_QUERY_SCHEDULE
    return _RDNA4_MINMAX_SMALL_QUERY_SCHEDULE


try:
    from . import triton as preparation
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("triton"):
        raise
    preparation = None

try:
    from ._nvidia import gluon as nvidia_gluon
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("triton"):
        raise
    nvidia_gluon = None

try:
    from ._amd import gluon as amd_gluon
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("triton"):
        raise
    amd_gluon = None

_nvidia_tma_attention = (
    AttentionBackend(
        prepare=preparation._prepare_sparse_piper_operands,
        value_format=ValueFormat.TRANSPOSED,
        bind=nvidia_gluon.bind_tma_context,
    )
    if preparation is not None and nvidia_gluon is not None
    else None
)
_nvidia_tma_attention_skip_dense_routing = (
    replace(_nvidia_tma_attention, skip_dense_routing=True)
    if _nvidia_tma_attention is not None
    else None
)
_nvidia_async_copy_attention = (
    AttentionBackend(
        prepare=preparation._prepare_sparse_piper_operands,
        value_format=ValueFormat.TRANSPOSED,
        bind=nvidia_gluon.bind_async_copy_context,
    )
    if preparation is not None and nvidia_gluon is not None
    else None
)
_nvidia_async_copy_attention_skip_dense_routing = (
    replace(_nvidia_async_copy_attention, skip_dense_routing=True)
    if _nvidia_async_copy_attention is not None
    else None
)
_amd_attention = (
    AttentionBackend(
        prepare=partial(
            preparation._prepare_sparse_piper_operands,
            value_format=ValueFormat.AMD_WMMA,
            launch_kernels=amd_gluon.launch_operand_preparation,
        ),
        value_format=ValueFormat.AMD_WMMA,
        bind=amd_gluon.bind_context,
    )
    if preparation is not None and amd_gluon is not None
    else None
)


try:
    from . import _routes_triton as _route_backend
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("triton"):
        raise
    _route_backend = None

try:
    from . import _summaries_triton as _summary_backend
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("triton"):
        raise
    _summary_backend = None


try:
    from . import _scores_triton as _score_backend
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("triton"):
        raise
    _score_backend = None


def select_attention_backend(
    query: torch.Tensor,
    *,
    target: AcceleratorTarget | None = None,
) -> AttentionBackend | None:
    """Return native execution or let the caller use the quantized reference."""
    if (
        _nvidia_tma_attention is None
        and _nvidia_async_copy_attention is None
        and _amd_attention is None
    ):
        return None
    target = AcceleratorTarget.from_device(query.device) if target is None else target
    # Device probes and pre-projection activations do not have an attention head axis.
    head_dim = query.shape[-1] if query.ndim == 4 else 128
    if nvidia_policy.uses_tensor_descriptors(target):
        backend, full_keep_backend = _nvidia_tma_attention, _nvidia_tma_attention_skip_dense_routing
    elif nvidia_policy.uses_async_copies(target):
        backend, full_keep_backend = (
            _nvidia_async_copy_attention,
            _nvidia_async_copy_attention_skip_dense_routing,
        )
    else:
        return (
            _amd_attention
            if supports_amd_target(target) and head_dim in SUPPORTED_HEAD_DIMS
            else None
        )
    if backend is not None and nvidia_policy.skip_dense_routing(head_dim):
        return full_keep_backend
    return backend


def require_attention_backend(query: torch.Tensor) -> AttentionBackend:
    """Resolve execution for an already-quantized internal operator."""
    backend = select_attention_backend(query)
    if backend is None:
        raise RuntimeError(
            f"quantized-input sparse Piper implementation is unavailable on {query.device}"
        )
    return backend


def select_route_selector(
    routes: torch.Tensor,
    *,
    target: AcceleratorTarget | None = None,
) -> SelectRoutes | None:
    """Select route acceleration independently of the attention kernel."""
    if _route_backend is None:
        return None
    target = AcceleratorTarget.from_device(routes.device) if target is None else target
    if not (nvidia_policy.supports_target(target) or supports_amd_target(target)):
        return None
    return _route_backend.tiled_radix_select_packed_routes


def select_minmax_scores(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
    *,
    target: AcceleratorTarget | None = None,
) -> MinmaxScores | None:
    """Use tiled FP32 scoring for D128 min/max summaries on RDNA4 and SM120."""
    if _score_backend is None:
        return None
    tensors = query_summary, key_primary, key_aux
    if any(
        tensor.ndim != 4
        or tensor.dtype is not torch.float32
        or tensor.device != query_summary.device
        or tensor.stride(-1) != 1
        for tensor in tensors
    ):
        return None
    if not (
        query_summary.shape[-1] == 128
        and query_summary.shape[0] > 0
        and query_summary.shape[1] > 0
        and query_summary.shape[2] > 0
        and key_primary.shape[2] > 0
        and key_primary.shape == key_aux.shape
        and query_summary.shape[0] == key_primary.shape[0]
        and key_primary.shape[1] > 0
        and query_summary.shape[1] % key_primary.shape[1] == 0
        and key_primary.shape[-1] == 128
    ):
        return None
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        return None
    target = AcceleratorTarget.from_device(query_summary.device) if target is None else target
    if supports_amd_target(target):
        schedule = select_amd_minmax_score_schedule(query_summary.shape[2])
    elif nvidia_policy.supports_minmax_scores(target):
        schedule = nvidia_policy.select_sm120_minmax_score_schedule(query_summary.shape[2])
    else:
        return None
    return partial(_score_backend.minmax_scores, schedule=schedule)


def fill_full_keep_routes(routes: torch.Tensor, sparse_key_blocks: int) -> None:
    """Materialize the known all-block route list on the selected device."""
    if _route_backend is not None and select_route_selector(routes) is not None:
        _route_backend.fill_full_keep_routes(routes, sparse_key_blocks)
    else:
        blocks = torch.arange(sparse_key_blocks, device=routes.device, dtype=torch.int32)
        routes.unflatten(-1, (-1, sparse_key_blocks)).copy_(blocks.to(torch.uint16))


def use_fused_operand_preparation(
    query: torch.Tensor,
    key: torch.Tensor,
    routing_mode: int,
    *,
    target: AcceleratorTarget | None = None,
) -> bool:
    """Fuse min/max summaries into Q/K quantization on native attention backends.

    The caller has already selected the attention backend and its preparation.
    This only chooses whether that preparation should also emit summaries.
    Mean routing keeps its existing reduction order: a different FP32 sum can
    change routes tied within a ULP. SM120 also keeps its faster short-row path.
    """
    if preparation is None or routing_mode == _MEAN_ROUTING:
        return False
    if not (
        query.device == key.device
        and query.shape[-1] in SUPPORTED_HEAD_DIMS
        and key.shape[-1] == query.shape[-1]
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and query.dtype in SUPPORTED_DTYPES
        and key.dtype == query.dtype
    ):
        return False
    target = AcceleratorTarget.from_device(query.device) if target is None else target
    if nvidia_policy.uses_tensor_descriptors(target):
        return nvidia_policy.use_sm120_fused_preparation(query.shape[-1], query.shape[2])
    return nvidia_policy.uses_async_copies(target) or (
        amd_gluon is not None and supports_amd_target(target)
    )


def select_sequence_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
    block_lengths: torch.Tensor | None = None,
    *,
    target: AcceleratorTarget | None = None,
) -> SequenceSummaries | None:
    """Select native Q/K summaries for a supported device, layout, and dtype."""
    if _summary_backend is None:
        return None
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad):
        return None
    if not (
        query.device == key.device
        and query.shape[-1] in SUPPORTED_HEAD_DIMS
        and key.shape[-1] == query.shape[-1]
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and query.dtype in SUPPORTED_DTYPES
        and key.dtype == query.dtype
        and (
            block_lengths is None
            or (
                block_lengths.ndim == 1
                and block_lengths.dtype is torch.int32
                and block_lengths.device == query.device
                and block_lengths.is_contiguous()
            )
        )
    ):
        return None
    target = AcceleratorTarget.from_device(query.device) if target is None else target
    if not (nvidia_policy.supports_target(target) or supports_amd_target(target)):
        return None
    return _summary_backend.sequence_block_summaries


def build_force_tile_bitmask(
    ids: torch.Tensor, counts: torch.Tensor, sparse_key_blocks: int
) -> torch.Tensor:
    """Build call-local membership once, then share it across heads and score chunks."""
    if _route_backend is not None and select_route_selector(ids) is not None:
        return _route_backend.build_force_tile_bitmask(ids, counts, sparse_key_blocks)
    return build_portable_force_tile_bitmask(ids, counts, sparse_key_blocks)
