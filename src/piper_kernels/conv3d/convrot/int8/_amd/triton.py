"""RDNA4 policy bindings for shared ConvRot INT8 convolution launchers."""

from functools import partial

from .. import triton as convolution
from . import policy

conv3d = partial(convolution.conv3d, policy=policy, accelerator_backend="hip")
group_norm_silu_conv3d = partial(
    convolution.group_norm_silu_conv3d, policy=policy, accelerator_backend="hip"
)
