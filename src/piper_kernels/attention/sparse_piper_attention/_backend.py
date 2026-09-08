"""Operation-specific sparse-Piper selection using the operands' device."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import HEAD_DIM

from ._amd import policy as amd_policy
from ._interfaces import AttentionBackend, MinmaxScores, SelectRoutes, SequenceSummaries
from ._nvidia import policy as nvidia_policy

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

_nvidia_attention = (
    AttentionBackend(
        prepare=preparation._prepare_sparse_piper_attention,
        launch=nvidia_gluon._launch_sparse_piper_attention,
    )
    if preparation is not None and nvidia_gluon is not None
    else None
)
_amd_attention = (
    AttentionBackend(
        prepare=preparation._prepare_sparse_piper_attention,
        launch=amd_gluon._launch_sparse_piper_attention,
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


def select_attention_backend(query: torch.Tensor) -> AttentionBackend | None:
    """Return native execution or let the caller use the quantized reference."""
    if _nvidia_attention is None and _amd_attention is None:
        return None
    target = AcceleratorTarget.from_device(query.device)
    if nvidia_policy.supports_target(target):
        return _nvidia_attention
    return _amd_attention if amd_policy.supports_target(target) else None


def require_attention_backend(query: torch.Tensor) -> AttentionBackend:
    """Resolve execution for an already-quantized internal operator."""
    backend = select_attention_backend(query)
    if backend is None:
        raise RuntimeError(
            f"quantized-input sparse Piper implementation is unavailable on {query.device}"
        )
    return backend


def select_route_selector(routes: torch.Tensor) -> SelectRoutes | None:
    """Select route acceleration independently of the attention kernel."""
    if _route_backend is None:
        return None
    target = AcceleratorTarget.from_device(routes.device)
    if not (nvidia_policy.supports_target(target) or amd_policy.supports_target(target)):
        return None
    return _route_backend.tiled_radix_select_packed_routes


def select_minmax_scores(
    query_summary: torch.Tensor,
    key_primary: torch.Tensor,
    key_aux: torch.Tensor,
) -> MinmaxScores | None:
    """Use the measured small-query FP32 scoring kernel on RDNA4 only."""
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
        query_summary.shape[-1] == HEAD_DIM
        and query_summary.shape[0] > 0
        and query_summary.shape[1] > 0
        and 1 <= query_summary.shape[2] <= 64
        and key_primary.shape[2] > 0
        and key_primary.shape == key_aux.shape
        and query_summary.shape[:2] == key_primary.shape[:2]
        and key_primary.shape[-1] == HEAD_DIM
    ):
        return None
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        return None
    target = AcceleratorTarget.from_device(query_summary.device)
    return _score_backend.minmax_scores if amd_policy.supports_target(target) else None


def fill_full_keep_routes(routes: torch.Tensor, sparse_key_blocks: int) -> None:
    """Materialize the known all-block route list on the selected device."""
    if _route_backend is not None and select_route_selector(routes) is not None:
        _route_backend.fill_full_keep_routes(routes, sparse_key_blocks)
    else:
        blocks = torch.arange(sparse_key_blocks, device=routes.device, dtype=torch.int32)
        routes.unflatten(-1, (-1, sparse_key_blocks)).copy_(blocks.to(torch.uint16))


def select_sequence_summaries(query: torch.Tensor, key: torch.Tensor) -> SequenceSummaries | None:
    """Preserve the existing summary kernel's device and tensor constraints."""
    if _summary_backend is None:
        return None
    if not (
        nvidia_policy.supports_target(AcceleratorTarget.from_device(query.device))
        and query.device == key.device
        and query.shape[-1] == HEAD_DIM
        and key.shape[-1] == HEAD_DIM
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype
    ):
        return None
    return _summary_backend.sequence_block_summaries
