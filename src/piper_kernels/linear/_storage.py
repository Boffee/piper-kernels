"""Storage identity for sharing prepared projection inputs."""

import torch


def same_tensor_storage(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(
        left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype is right.dtype
        and left.device == right.device
        and left.storage_offset() == right.storage_offset()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )
