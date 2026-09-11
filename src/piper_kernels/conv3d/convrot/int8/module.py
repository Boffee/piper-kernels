"""Causal convolutions consuming the shared ConvRot INT8 weight representation."""

import torch

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

from ._ops import SpatialPadding, _padding_flags, _weight_storage, conv3d
from ._validation import _validate_config, _validate_weight


class ConvRotInt8Conv3d(torch.nn.Module):
    """Inference-only causal 3x3x3 layer with FP16/FP32 activations and FP16 output.

    Create or load ``weight`` through ``ConvRotInt8Tensor.from_hp`` or
    ``from_quantized``. Its FP32 ``act_per_tensor_scale`` must be present.
    Parameters retain the supplied storage, including mmap backing. For state
    dictionaries, use ``load_state_dict(..., assign=True)`` to retain that storage.

    Two zero frames precede the input. ``padding`` controls spatial reflection.
    Stride and padding come from the model architecture, as with ``nn.Conv3d``;
    all quantization state is serialized with the weight.
    """

    weight: ConvRotInt8Tensor
    bias: torch.Tensor | None
    padding: SpatialPadding

    def __init__(
        self,
        weight: ConvRotInt8Tensor,
        bias: torch.Tensor | None = None,
        *,
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: SpatialPadding,
    ) -> None:
        super().__init__()
        qdata, scale, group_size, _ = _weight_storage(weight)
        _validate_weight(qdata, scale, bias, group_size)
        symmetric, right = _padding_flags(padding)
        _validate_config(list(stride), symmetric, right)
        self.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
        self.register_parameter(
            "bias", None if bias is None else torch.nn.Parameter(bias, requires_grad=False)
        )
        self.stride = stride
        self.padding = padding

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return conv3d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            residual=residual,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.weight.shape[1]}, {self.weight.shape[0]}, "
            f"stride={self.stride}, padding={self.padding!r}"
        )
