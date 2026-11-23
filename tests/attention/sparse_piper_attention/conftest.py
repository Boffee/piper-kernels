"""Fixtures shared by sparse Piper attention tests."""

import pytest
from _accelerators import exact_sm120_available, rdna4_available, sm8x_available


@pytest.fixture
def require_native_routing() -> None:
    """Skip GPU tests unless a native SM8x, SM120, or RDNA4 router is present."""
    if not (exact_sm120_available() or sm8x_available() or rdna4_available()):
        pytest.skip("requires native SM8x, SM120, or RDNA4 routing")
