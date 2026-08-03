"""Package-boundary smoke tests."""

import piper_kernels
import piper_kernels.attention
import piper_kernels.convrot


def test_public_packages_import() -> None:
    assert piper_kernels.__name__ == "piper_kernels"
    assert piper_kernels.attention.__name__ == "piper_kernels.attention"
    assert piper_kernels.convrot.__name__ == "piper_kernels.convrot"
