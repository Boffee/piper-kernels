"""Conservative support for ROCm fused Q/K/V projection on RDNA4."""

import sys

from piper_kernels._triton.targets import AcceleratorTarget


def supports_target(target: AcceleratorTarget) -> bool:
    return (
        sys.platform == "linux"
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    )
