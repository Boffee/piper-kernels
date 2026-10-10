"""ConvRot INT8 projection operands for fusion tests."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Linear:
    qdata: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None
    group_size: int = 256

    def arguments(self) -> tuple[object, ...]:
        return self.qdata, self.scale, self.bias, self.group_size


def make_linear(
    out_features: int,
    in_features: int,
    bias_dtype: torch.dtype | None,
    group_size: int = 256,
) -> Linear:
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    bias = (
        torch.randn(out_features, dtype=bias_dtype, device="cuda")
        if bias_dtype is not None
        else None
    )
    return Linear(qdata, scale, bias, group_size)
