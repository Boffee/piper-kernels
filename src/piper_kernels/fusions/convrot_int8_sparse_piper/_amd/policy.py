"""Validated ROCm fused sparse projection and output integrations on RDNA4."""

import sys

from piper_kernels._triton.targets import AcceleratorTarget


def supports_head_dim(head_dim: int) -> bool:
    """Head widths handled by the RDNA4 projection schedules."""
    return head_dim in (64, 128)


def supports_target(target: AcceleratorTarget) -> bool:
    return (
        sys.platform == "linux"
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    )
