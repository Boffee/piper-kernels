"""Package-boundary smoke tests."""

import piper_kernels
import piper_kernels.attention
import piper_kernels.attention.piper_attention
import piper_kernels.attention.sage_attention_2pp
import piper_kernels.convrot
import piper_kernels.convrot.int8
from piper_kernels import piper_attention, sage_attention_2pp
from piper_kernels.convrot import ConvRotInt8Tensor, convrot_linear


def test_public_packages_import() -> None:
    assert piper_kernels.__name__ == "piper_kernels"
    assert piper_kernels.piper_attention is piper_attention
    assert piper_kernels.sage_attention_2pp is sage_attention_2pp
    assert piper_kernels.__all__ == ["piper_attention", "sage_attention_2pp"]
    assert piper_kernels.attention.__name__ == "piper_kernels.attention"
    assert piper_kernels.attention.piper_attention.__name__ == (
        "piper_kernels.attention.piper_attention"
    )
    assert piper_kernels.attention.sage_attention_2pp.__name__ == (
        "piper_kernels.attention.sage_attention_2pp"
    )
    assert piper_kernels.convrot.__name__ == "piper_kernels.convrot"
    assert piper_kernels.convrot.int8.__name__ == "piper_kernels.convrot.int8"
    assert piper_kernels.convrot.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.convrot.int8.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.convrot.convrot_linear is convrot_linear
    assert convrot_linear.__module__ == "piper_kernels.convrot.functional"
    assert not hasattr(piper_kernels.convrot.int8, "convrot_linear")
    assert not hasattr(piper_kernels.convrot, "linear_input_act")
    assert not hasattr(piper_kernels.convrot.int8, "linear_input_act")
    assert piper_kernels.convrot.__all__ == ["ConvRotInt8Tensor", "convrot_linear"]
    assert piper_kernels.convrot.int8.__all__ == ["ConvRotInt8Tensor"]
