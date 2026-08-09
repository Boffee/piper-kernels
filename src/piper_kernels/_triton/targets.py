"""Backend-aware hardware facts required by Triton kernels."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AcceleratorTarget:
    """Describe accelerator facts using compiler backend names such as CUDA/HIP."""

    backend: str
    cuda_capability: tuple[int, int] | None = None

    @classmethod
    def from_device(cls, device: torch.device) -> "AcceleratorTarget":
        """Resolve ``device`` without requiring Triton to be installed."""
        if device.type != "cuda":
            return cls(backend=device.type)
        if getattr(torch.version, "hip", None) is not None:
            return cls(backend="hip")
        return cls(
            backend="cuda",
            cuda_capability=torch.cuda.get_device_capability(device),
        )

    @classmethod
    def from_compiler_target(cls, target: object) -> "AcceleratorTarget":
        """Normalize a compiler target exposing ``backend`` and integer ``arch``."""
        backend = getattr(target, "backend", None)
        if not isinstance(backend, str):
            raise TypeError("compiler target must expose a string backend")
        architecture = getattr(target, "arch", None)
        cuda_capability = (
            (architecture // 10, architecture % 10)
            if backend == "cuda" and isinstance(architecture, int)
            else None
        )
        return cls(backend=backend, cuda_capability=cuda_capability)

    @property
    def is_nvidia_cuda(self) -> bool:
        """Return whether this target uses NVIDIA's CUDA backend."""
        return self.backend == "cuda"

    def is_cuda_capability(self, major: int, minor: int | None = None) -> bool:
        """Match a CUDA compute-capability family or exact target."""
        if not self.is_nvidia_cuda or self.cuda_capability is None:
            return False
        target_major, target_minor = self.cuda_capability
        return target_major == major and (minor is None or target_minor == minor)

    def cuda_capability_at_least(self, major: int, minor: int = 0) -> bool:
        """Return whether this is a CUDA target at or above a capability."""
        return (
            self.is_nvidia_cuda
            and self.cuda_capability is not None
            and self.cuda_capability >= (major, minor)
        )

    @property
    def supports_uint8_int8_mma(self) -> bool:
        """Return whether Piper's NVIDIA UINT8-by-INT8 MMAv2 lowering is supported."""
        # The packaged compiler hook rewrites the m16n8k32 MMAv2 lowering used by
        # consumer SM8x and SM12x. Hopper lowers this dot through WGMMA instead.
        return self.is_cuda_capability(8) or self.is_cuda_capability(12)

    @property
    def supports_fp8_fp16_mma(self) -> bool:
        """Return whether FP8 tensor-core inputs with FP16 accumulation are available."""
        return self.cuda_capability_at_least(8, 9)
