"""Hardware feature predicates required by Triton kernels."""

import torch


def is_nvidia_cuda(device: torch.device) -> bool:
    """Return whether ``device`` belongs to an NVIDIA CUDA runtime."""
    return device.type == "cuda" and getattr(torch.version, "hip", None) is None


def supports_uint8_int8_mma(device: torch.device) -> bool:
    """Return whether Piper's NVIDIA UINT8-by-INT8 MMAv2 lowering is supported."""
    if not is_nvidia_cuda(device):
        return False
    # The packaged compiler hook rewrites the m16n8k32 MMAv2 lowering used by
    # consumer SM8x and SM12x. Hopper lowers this dot through WGMMA instead.
    return torch.cuda.get_device_capability(device)[0] in (8, 12)


def supports_fp8_fp16_mma(device: torch.device) -> bool:
    """Return whether FP8 tensor-core inputs with FP16 accumulation are available."""
    return is_nvidia_cuda(device) and torch.cuda.get_device_capability(device) >= (8, 9)
