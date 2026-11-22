"""Select dense Piper execution from operand-device metadata."""

import sys
from collections.abc import Callable

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from ._nvidia import policy as nvidia_policy

AttentionImplementation = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float, bool], torch.Tensor
]

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
    return (
        sys.platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    )


def select_backend(target: AcceleratorTarget) -> AttentionImplementation | None:
    """Return a native implementation or preserve the portable fallback."""
    if nvidia_policy.supports_target(target):
        return nvidia_attention
    if supports_amd_target(target):
        return amd_attention
    return None
