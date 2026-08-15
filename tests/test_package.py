"""Package-boundary smoke tests."""

import importlib

import pytest

import piper_kernels
import piper_kernels.attention
import piper_kernels.attention.piper_attention
import piper_kernels.attention.sage_attention_2pp
import piper_kernels.linear
import piper_kernels.linear.convrot
import piper_kernels.linear.convrot.int8
from piper_kernels import piper_attention, sage_attention_2pp
from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_linear


def test_removed_convrot_root_package_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module("piper_kernels.convrot")

    assert error.value.name == "piper_kernels.convrot"


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
    assert piper_kernels.linear.__name__ == "piper_kernels.linear"
    assert piper_kernels.linear.convrot.__name__ == "piper_kernels.linear.convrot"
    assert piper_kernels.linear.convrot.int8.__name__ == "piper_kernels.linear.convrot.int8"
    assert piper_kernels.linear.convrot.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.linear.convrot.int8.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.linear.convrot.convrot_linear is convrot_linear
    assert convrot_linear.__module__ == "piper_kernels.linear.convrot.functional"
    assert not hasattr(piper_kernels.linear.convrot.int8, "convrot_linear")
    assert not hasattr(piper_kernels.linear.convrot, "linear_input_act")
    assert not hasattr(piper_kernels.linear.convrot.int8, "linear_input_act")
    assert piper_kernels.linear.convrot.__all__ == ["ConvRotInt8Tensor", "convrot_linear"]
    assert piper_kernels.linear.convrot.int8.__all__ == ["ConvRotInt8Tensor"]
