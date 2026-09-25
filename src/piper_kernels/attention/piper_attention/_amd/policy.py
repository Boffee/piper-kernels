"""Conservative four-wave Q64/K64 dense Piper policy for ROCm RDNA4."""

import sys

from piper_kernels._triton.targets import AcceleratorTarget


def supports_target(target: AcceleratorTarget) -> bool:
    return (
        sys.platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    )
