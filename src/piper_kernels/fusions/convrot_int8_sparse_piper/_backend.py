"""Operation-specific selection for the validated sparse-fusion integrations."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import _backend as attention_backend
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8._interfaces import LinearBackend

from . import _interfaces
from ._interfaces import ProjectionBackend
from ._nvidia import policy as nvidia_policy

try:
    from ._nvidia import triton as _nvidia_projection
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    _nvidia_projection = None


def source_files() -> tuple[str, ...]:
    """Include selection and execution policy in the compiler-pass cache key."""
    return tuple(
        path
        for path in (
            __file__,
            _interfaces.__file__,
            nvidia_policy.__file__,
            None if _nvidia_projection is None else _nvidia_projection.__file__,
            linear_backend.__file__,
            attention_backend.__file__,
        )
        if path is not None
    )


def select_projection_backend(input: torch.Tensor) -> ProjectionBackend | None:  # noqa: A002
    """Select fused Q/K/V execution, independently of standalone linear support."""
    if _nvidia_projection is None:
        return None
    target = AcceleratorTarget.from_device(input.device)
    return _nvidia_projection if nvidia_policy.supports_target(target) else None


def require_projection_backend(input: torch.Tensor) -> ProjectionBackend:  # noqa: A002
    backend = select_projection_backend(input)
    if backend is None:
        raise ValueError(f"ConvRot INT8 sparse projections are unavailable on {input.device}")
    return backend


def select_output_backend(input: torch.Tensor) -> LinearBackend | None:  # noqa: A002
    """Select an independently validated attention-to-output integration.

    Standalone AMD attention and linear support does not yet imply validation of
    the chunked multi-stream fusion. Keep that integration disabled in this pass.
    """
    target = AcceleratorTarget.from_device(input.device)
    if not nvidia_policy.supports_target(target):
        return None
    if attention_backend.select_attention_backend(input) is None:
        return None
    return linear_backend.select_linear_backend(input)


def require_output_backend(input: torch.Tensor) -> LinearBackend:  # noqa: A002
    backend = select_output_backend(input)
    if backend is None:
        raise ValueError(f"ConvRot INT8 sparse output fusion is unavailable on {input.device}")
    return backend
