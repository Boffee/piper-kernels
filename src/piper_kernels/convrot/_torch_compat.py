"""Compatibility helpers for optional private PyTorch tracing APIs."""

try:
    from torch._guards import detect_fake_mode as _detect_fake_mode
except ImportError:

    def is_fake_mode_active() -> bool:
        """Conservatively disable caching when fake-mode detection is unavailable."""
        return True

else:

    def is_fake_mode_active() -> bool:
        """Return whether PyTorch is currently executing under a fake tensor mode."""
        return _detect_fake_mode() is not None
