"""Package-boundary smoke tests."""

import piper_kernels
import piper_kernels.attention
import piper_kernels.attention.backends
import piper_kernels.convrot
import piper_kernels.convrot.int8
from piper_kernels.attention import sage_attention_2pp
from piper_kernels.convrot import ConvRotInt8Tensor


def test_public_packages_import() -> None:
    assert piper_kernels.__name__ == "piper_kernels"
    assert piper_kernels.attention.__name__ == "piper_kernels.attention"
    assert piper_kernels.attention.backends.__name__ == "piper_kernels.attention.backends"
    assert piper_kernels.attention.sage_attention_2pp is sage_attention_2pp
    assert piper_kernels.attention.__all__ == ["sage_attention_2pp"]
    assert piper_kernels.convrot.__name__ == "piper_kernels.convrot"
    assert piper_kernels.convrot.int8.__name__ == "piper_kernels.convrot.int8"
    assert piper_kernels.convrot.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.convrot.int8.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.convrot.__all__ == ["ConvRotInt8Tensor"]
    assert piper_kernels.convrot.int8.__all__ == ["ConvRotInt8Tensor"]
