"""Operation-specific sparse-Piper selection using the operands' device."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import HEAD_DIM

from ._interfaces import AttentionBackend, SelectRoutes, SequenceSummaries
from ._nvidia import policy as nvidia_policy

try:
    from . import triton as preparation
    from ._nvidia import gluon
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("triton"):
        raise
    _nvidia_attention = None
else:
    _nvidia_attention = AttentionBackend(
        prepare=preparation._prepare_sparse_piper_attention,
        launch=gluon._launch_sparse_piper_attention,
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


def select_attention_backend(query: torch.Tensor) -> AttentionBackend | None:
    """Return native execution or let the caller use the quantized reference."""
    if _nvidia_attention is None:
        return None
    target = AcceleratorTarget.from_device(query.device)
    return _nvidia_attention if nvidia_policy.supports_target(target) else None


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
    if not nvidia_policy.supports_target(AcceleratorTarget.from_device(routes.device)):
        return None
    return _route_backend.tiled_radix_select_packed_routes


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
