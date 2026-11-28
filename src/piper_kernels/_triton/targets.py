"""Backend-aware hardware facts required by Triton kernels."""

import sys
from dataclasses import dataclass

import torch

GFX120X_ARCHITECTURES = ("gfx1200", "gfx1201")
_AMD_W4A16_ARCHITECTURES = ("gfx942", "gfx1100", "gfx1151", *GFX120X_ARCHITECTURES)

RDNA4_INT8_ACCUMULATOR_SEED = 0x155555
RDNA4_INT8_ACCUMULATOR_SEED_MIN_REDUCTION = 4096
# Two output workgroups per WGP on 32-WGP RDNA4.
RDNA4_MIN_FULL_GRID_TILES = 64


def _normalize_architecture(architecture: object | None) -> str | None:
    """Return a canonical architecture name without optional feature suffixes."""
    if architecture is None:
        return None
    normalized = str(architecture).strip().lower().partition(":")[0]
    return normalized or None


@dataclass(frozen=True, slots=True)
class AcceleratorTarget:
    """Describe accelerator facts using compiler backend names such as CUDA/HIP."""

    backend: str
    architecture: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", self.backend.strip().lower())
        object.__setattr__(
            self,
            "architecture",
            _normalize_architecture(self.architecture),
        )

    @classmethod
    def from_device(cls, device: torch.device) -> "AcceleratorTarget":
        """Resolve ``device`` without requiring Triton to be installed."""
        if device.type != "cuda":
            return cls(backend=device.type)
        if getattr(torch.version, "hip", None) is not None:
            properties = torch.cuda.get_device_properties(device)
            return cls(
                backend="hip",
                architecture=getattr(properties, "gcnArchName", None),
            )
        major, minor = torch.cuda.get_device_capability(device)
        return cls(backend="cuda", architecture=f"sm{major}{minor}")

    @classmethod
    def from_compiler_target(cls, target: object) -> "AcceleratorTarget":
        """Normalize a compiler target exposing ``backend`` and ``arch``."""
        backend = getattr(target, "backend", None)
        if not isinstance(backend, str):
            raise TypeError("compiler target must expose a string backend")
        backend = backend.strip().lower()
        architecture = getattr(target, "arch", None)
        if backend == "cuda":
            architecture = f"sm{architecture}" if isinstance(architecture, int) else None
        elif not isinstance(architecture, str):
            architecture = None
        return cls(backend=backend, architecture=architecture)

    @property
    def is_nvidia_cuda(self) -> bool:
        """Return whether this target uses NVIDIA's CUDA backend."""
        return self.backend == "cuda"

    @property
    def is_amd_hip(self) -> bool:
        """Return whether this target uses AMD's HIP backend."""
        return self.backend == "hip"

    def is_architecture(self, *architectures: str) -> bool:
        """Match one of the exact canonical architecture names."""
        return self.architecture is not None and self.architecture in (
            _normalize_architecture(architecture) for architecture in architectures
        )

    def is_hip_architecture(self, *architectures: str) -> bool:
        """Match one of the exact canonical AMD HIP architecture names."""
        return self.is_amd_hip and self.is_architecture(*architectures)

    @property
    def cuda_capability(self) -> tuple[int, int] | None:
        """Return the CUDA compute capability encoded by an ``smXX`` architecture."""
        if not self.is_nvidia_cuda or self.architecture is None:
            return None
        encoded = self.architecture.removeprefix("sm")
        if encoded == self.architecture or not encoded.isdecimal():
            return None
        capability = int(encoded)
        return divmod(capability, 10)

    def is_cuda_capability(self, major: int, minor: int | None = None) -> bool:
        """Match a CUDA compute-capability family or exact target."""
        capability = self.cuda_capability
        if capability is None:
            return False
        target_major, target_minor = capability
        return target_major == major and (minor is None or target_minor == minor)

    def cuda_capability_at_least(self, major: int, minor: int = 0) -> bool:
        """Return whether this is a CUDA target at or above a capability."""
        capability = self.cuda_capability
        return capability is not None and capability >= (major, minor)

    @property
    def supports_uint8_int8_mma(self) -> bool:
        """Return whether Piper Attention's NVIDIA UINT8-by-INT8 MMAv2 lowering is supported."""
        # The packaged compiler hook rewrites the m16n8k32 MMAv2 lowering used by
        # consumer SM8x and SM12x. Hopper lowers this dot through WGMMA instead.
        return self.is_cuda_capability(8) or self.is_cuda_capability(12)

    @property
    def supports_fp8_fp16_mma(self) -> bool:
        """Return whether FP8 tensor-core inputs with FP16 accumulation are available."""
        return self.cuda_capability_at_least(8, 9)

    @property
    def supports_nvfp4_w4a16(self) -> bool:
        """Return whether NVFP4 W4A16 execution is supported on this target."""
        return (self.is_nvidia_cuda and self.cuda_capability_at_least(7, 5)) or (
            sys.platform in ("linux", "win32")
            and self.is_hip_architecture(*_AMD_W4A16_ARCHITECTURES)
        )


def rdna4_int8_accumulator_seed(
    target: AcceleratorTarget, reduction_width: int, output_tiles: int
) -> int:
    """Return the ``0x155555`` zero-crossing seed for long INT8 reductions on gfx1201.

    For reductions of 4,096+ INT8 elements across a full grid (>= 64 output tiles) on
    ``gfx1201``, seeding INT32 accumulators with ``0x155555`` centers partial sums away
    from the signed zero crossing, sustaining higher matrix clock frequency on a stock
    RX 9070 XT. Kernels subtract the seed in INT32 before scaling; each INT8 product
    contributes at most ``128 * 128``, bounding the overflow-safe reduction width.
    """
    seed = RDNA4_INT8_ACCUMULATOR_SEED
    if (
        target.is_hip_architecture("gfx1201")
        and output_tiles >= RDNA4_MIN_FULL_GRID_TILES
        and RDNA4_INT8_ACCUMULATOR_SEED_MIN_REDUCTION
        <= reduction_width
        <= (2**31 - 1 - seed) // (128 * 128)
    ):
        return seed
    return 0
