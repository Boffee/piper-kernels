"""Select dense Piper execution from operand-device metadata."""

import sys
from typing import Literal, Protocol

import torch

from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from ._nvidia import policy as nvidia_policy


class AttentionImplementation(Protocol):
    """Callable contract for dense Piper Attention native backends."""

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        is_causal: bool,
        *,
        window_left: int | None = None,
        window_right: int | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


try:
    from ._nvidia.triton import _run_piper_attention as nvidia_attention
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    nvidia_attention = None

try:
    from ._amd.gluon import run_attention as amd_attention
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    amd_attention = None


def supports_amd_target(target: AcceleratorTarget) -> bool:
    """Return whether target uses the four-wave Q64/K64 ROCm RDNA4 dense Piper kernel."""
    return sys.platform in ("linux", "win32") and target.is_hip_architecture(*GFX120X_ARCHITECTURES)


def kv_chunk_layout(target: AcceleratorTarget) -> tuple[int, bool, bool]:
    """Return ``(key_scales, grouped_qk, pack_wmma)`` from target ISA capabilities."""
    if target.is_hip_architecture(*GFX120X_ARCHITECTURES):
        return 16, True, True
    if nvidia_policy.supports_target(target):
        grouped_qk = nvidia_policy.groups_qk_scales(target)
        return (16 if grouped_qk else 1024), grouped_qk, False
    return 16, True, False


def prefill_backend(target: AcceleratorTarget) -> Literal["amd", "nvidia"] | None:
    """Return the native quantized-prefill backend name for ``target``, if any."""
    if nvidia_policy.supports_target(target):
        return "nvidia"
    if supports_amd_target(target):
        return "amd"
    return None


def select_backend(target: AcceleratorTarget) -> AttentionImplementation | None:
    """Return a native implementation or preserve the portable fallback."""
    if nvidia_policy.supports_target(target):
        return nvidia_attention
    if supports_amd_target(target):
        return amd_attention
    return None
