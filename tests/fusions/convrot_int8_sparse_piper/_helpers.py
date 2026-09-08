"""Hardware coverage gates for the currently validated sparse-fusion backend."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget


def exact_sm120_available() -> bool:
    return torch.cuda.is_available() and AcceleratorTarget.from_device(
        torch.device("cuda")
    ).is_cuda_capability(12, 0)
