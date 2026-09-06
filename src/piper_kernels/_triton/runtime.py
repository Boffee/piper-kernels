"""Device ownership for host-side accelerator probing and kernel execution.

Capabilities belong to the requested device. Each host launcher establishes
that device for Triton, which reads its device and stream from runtime state.
The framework context restores the caller's device, including on exceptions;
it leaves each device's current stream unchanged.
"""

from contextlib import AbstractContextManager, nullcontext

import torch


def device_context(device: torch.device) -> AbstractContextManager:
    """Use the framework's device guard without requiring Triton or naming a vendor."""
    if device.type in ("cpu", "meta"):
        return nullcontext()
    device_module = torch.get_device_module(device)
    guard = getattr(device_module, "device", None)
    if guard is None:
        raise RuntimeError(f"No device context is available for {device.type}")
    return guard(device)


def supports_device(device: torch.device) -> bool:
    """Probe the requested Triton device, independently of the caller's current GPU."""
    if device.type in ("cpu", "meta"):
        return False
    try:
        from triton.runtime import driver  # noqa: PLC0415 - optional dependency

        active = driver.active
        if getattr(active.get_active_torch_device(), "type", None) != device.type:
            return False
        with device_context(device):
            active_device = active.get_active_torch_device()
            if device.index is not None and getattr(active_device, "index", None) != device.index:
                return False
            target = active.get_current_target()
            backend = getattr(target, "backend", None)
            architecture = getattr(target, "arch", None)
            # Compiler baseline, independent of any tuned matrix-operation policy.
            return isinstance(backend, str) and (
                backend != "cuda" or (isinstance(architecture, int) and architecture >= 70)
            )
    except (RuntimeError, ImportError, AttributeError):
        # Only probing can fail closed. Never catch/retry a kernel failure after
        # an in-place operation may already have mutated its output.
        return False
