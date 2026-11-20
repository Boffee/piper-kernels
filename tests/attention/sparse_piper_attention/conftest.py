"""Fixtures shared by sparse Piper attention tests."""

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._amd import policy as amd_policy
from piper_kernels.attention.sparse_piper_attention._nvidia import policy as nvidia_policy


@pytest.fixture
def require_native_routing() -> None:
    """Skip GPU tests unless a native SM8x, SM120, or RDNA4 router is present."""
    if not torch.cuda.is_available():
        pytest.skip("requires native SM8x, SM120, or RDNA4 routing")
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not (nvidia_policy.supports_target(target) or amd_policy.supports_target(target)):
        pytest.skip("requires native SM8x, SM120, or RDNA4 routing")
