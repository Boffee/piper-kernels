"""NVIDIA policy bindings for shared ConvRot INT8 convolution launchers."""

import torch

from piper_kernels._triton.targets import AcceleratorTarget

from .. import triton as convolution
from . import policy


def _policy(input: torch.Tensor) -> policy.NvidiaConvolutionPolicy:  # noqa: A002
    return policy.select_policy(AcceleratorTarget.from_device(input.device))


def conv3d(input, *args, **kwargs):  # noqa: A002
    """Launch with the SM8x or SM120 policy of the input's device."""
    return convolution.conv3d(
        input, *args, policy=_policy(input), accelerator_backend="cuda", **kwargs
    )


def group_norm_silu_conv3d(input, *args, **kwargs):  # noqa: A002
    """Launch with the SM8x or SM120 policy of the input's device."""
    return convolution.group_norm_silu_conv3d(
        input, *args, policy=_policy(input), accelerator_backend="cuda", **kwargs
    )
