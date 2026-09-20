"""Select dense Piper execution from operand-device metadata."""

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


def select_backend(target: AcceleratorTarget) -> AttentionImplementation | None:
    """Return a native implementation or preserve the portable fallback."""
    if nvidia_policy.supports_target(target):
        return nvidia_attention
    return None
