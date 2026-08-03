"""Package-boundary smoke tests."""

import piper_kernels
import piper_kernels.attention
import piper_kernels.convrot
from piper_kernels.convrot import (
    ConvRotInt8Tensor,
    int8_convrot_linear,
    to_convrot_int8_tensor,
)


def test_public_packages_import() -> None:
    assert piper_kernels.__name__ == "piper_kernels"
    assert piper_kernels.attention.__name__ == "piper_kernels.attention"
    assert piper_kernels.convrot.__name__ == "piper_kernels.convrot"
    assert piper_kernels.convrot.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.convrot.int8_convrot_linear is int8_convrot_linear
    assert piper_kernels.convrot.to_convrot_int8_tensor is to_convrot_int8_tensor
