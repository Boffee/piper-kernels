"""Targets supported by the wave32 RDNA4 sparse-attention kernel."""

from piper_kernels._triton.targets import AcceleratorTarget


def supports_target(target: AcceleratorTarget) -> bool:
    return target.is_amd_hip and target.is_architecture("gfx1200", "gfx1201")
