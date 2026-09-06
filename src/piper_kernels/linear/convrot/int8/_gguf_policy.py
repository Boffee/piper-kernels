"""Select GGUF conversion schedules independently of tuned INT8 matrix support."""

from piper_kernels._triton.targets import AcceleratorTarget

from ._plan import fused_preparation_chunks

_DEFAULT_FUSED_MAX_WIDTH = 8192


def select_conversion_chunks(target: AcceleratorTarget, in_features: int) -> tuple[int, int] | None:
    """Return fused ``(chunk_count, chunk_size)``, or None for bounded tiled execution."""
    # Fusion wins for moderate gfx1201 widths. Keep the same conservative
    # default for other Triton targets without imposing a GPU-model allowlist.
    if target.is_nvidia_cuda or in_features <= _DEFAULT_FUSED_MAX_WIDTH:
        return fused_preparation_chunks(in_features)
    return None
