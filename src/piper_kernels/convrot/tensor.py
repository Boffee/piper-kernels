"""Tensor subclass representing rotated INT8 W8A8 weights."""

from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torchao.utils import TorchAOBaseTensor

from ._dispatch import _linear
from ._reference import rotate_groups, validate_storage


class ConvRotInt8Tensor(TorchAOBaseTensor):
    """INT8 rotated weight with per-output scale and logical floating dtype."""

    tensor_data_names: ClassVar[list[str]] = ["qdata", "scale"]
    tensor_attribute_names: ClassVar[list[str]] = ["group_size", "dtype"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ConvRotInt8Tensor":
        validate_storage(qdata, scale, group_size, dtype)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            qdata.shape,
            device=qdata.device,
            dtype=dtype,
            requires_grad=False,
        )

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if self.dtype is not dtype:
            raise RuntimeError(f"ConvRot wrapper dtype mismatch: {self.dtype} != {dtype}")
        self.qdata = qdata
        self.scale = scale
        self.group_size = group_size

    @classmethod
    def from_packed(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        *,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ConvRotInt8Tensor":
        """Reconstruct a ConvRot weight from its stored tensors and metadata."""
        return cls(
            qdata.contiguous(),
            scale.reshape(-1, 1).contiguous(),
            group_size,
            dtype,
        )

    def dequantize(self) -> torch.Tensor:
        """Recover the logical weight in the unrotated basis."""
        rotated = self.qdata.to(self.dtype) * self.scale.to(self.dtype)
        return rotate_groups(rotated, self.group_size)

    def _stable_hash_for_caching(self) -> str:
        """Return a metadata fingerprint for AOTAutograd's cross-process cache."""
        return repr(
            (
                type(self).__qualname__,
                tuple(self.shape),
                self.stride(),
                str(self.device),
                str(self.dtype),
                self.group_size,
                tuple(self.qdata.shape),
                self.qdata.stride(),
                tuple(self.scale.shape),
                self.scale.stride(),
            )
        )


@ConvRotInt8Tensor.implements(torch.ops.aten.linear.default)
@ConvRotInt8Tensor.implements_torch_function(torch.nn.functional.linear)
def _convrot_linear_dispatch(
    _func: Callable[..., torch.Tensor],
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> torch.Tensor:
    activation = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    if not isinstance(activation, torch.Tensor) or not isinstance(weight, ConvRotInt8Tensor):
        raise TypeError(
            "ConvRot linear dispatch requires a tensor input and ConvRotInt8Tensor weight"
        )
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(f"ConvRot linear bias must be a tensor or None, got {type(bias).__name__}")
    return _linear(
        activation,
        weight.qdata,
        weight.scale,
        weight.group_size,
        bias,
    )
