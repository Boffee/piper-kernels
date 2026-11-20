"""Minimal Triton compiled-artifact lookup for GPU tests."""

from __future__ import annotations

from collections.abc import Mapping


def compiled_artifact(
    jit_kernel: object,
    artifact: str,
    *,
    specialization_index: int | None = None,
) -> str | bytes:
    """Return one TTIR/TTGIR/LLVM/PTX/cubin artifact from a compiled Triton JIT cache."""
    device_caches = getattr(jit_kernel, "device_caches", None)
    if not isinstance(device_caches, Mapping):
        raise RuntimeError("JIT object does not expose Triton's device_caches mapping")

    specializations: list[object] = []
    for _device_index, device_cache in sorted(device_caches.items()):
        if not isinstance(device_cache, tuple) or not device_cache:
            raise RuntimeError("unsupported Triton device_cache layout")
        kernel_cache = device_cache[0]
        if not isinstance(kernel_cache, Mapping):
            raise RuntimeError("Triton compiled-specialization cache is not a mapping")
        specializations.extend(
            compiled
            for _key, compiled in sorted(kernel_cache.items(), key=lambda item: str(item[0]))
        )
    if not specializations:
        raise RuntimeError("no compiled Triton specializations found")

    if specialization_index is None:
        if len(specializations) != 1:
            raise RuntimeError(
                f"expected one compiled specialization, found {len(specializations)}; "
                "select specialization_index explicitly"
            )
        specialization_index = 0
    compiled = specializations[specialization_index]
    assembly = getattr(compiled, "asm", None)
    if not isinstance(assembly, Mapping):
        raise RuntimeError("compiled Triton kernel does not expose an asm mapping")
    value = assembly.get(artifact)
    if not isinstance(value, (str, bytes)):
        available = ", ".join(sorted(assembly)) or "none"
        raise RuntimeError(
            f"compiled specialization has no {artifact!r} artifact; available: {available}"
        )
    return value
