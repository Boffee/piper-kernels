"""Package-boundary smoke tests."""

import importlib
import subprocess
import sys

import pytest

import piper_kernels
import piper_kernels.attention
import piper_kernels.attention.piper_attention
import piper_kernels.attention.sage_attention_2pp
import piper_kernels.attention.sparse_piper_attention
import piper_kernels.linear
import piper_kernels.linear.convrot
import piper_kernels.linear.convrot.int8
from piper_kernels import (
    SparsePiperAttentionPlan,
    piper_attention,
    prepare_sparse_piper_attention_plan,
    sage_attention_2pp,
    sparse_piper_attention,
)
from piper_kernels.linear.convrot import (
    ConvRotInt8Tensor,
    convrot_int8_compile_options,
    convrot_int8_linear,
)


def test_removed_convrot_root_package_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module("piper_kernels.convrot")

    assert error.value.name == "piper_kernels.convrot"


def test_convrot_compiler_integration_is_loaded_lazily() -> None:
    script = """
import sys
import piper_kernels.linear.convrot as convrot

assert "piper_kernels.linear.convrot.int8._compile" not in sys.modules
assert "piper_kernels.linear._preparation_sharing" not in sys.modules
convrot.convrot_int8_compile_options()
assert "piper_kernels.linear.convrot.int8._compile" in sys.modules
assert "piper_kernels.linear._preparation_sharing" in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_packages_import() -> None:
    assert piper_kernels.__name__ == "piper_kernels"
    assert piper_kernels.piper_attention is piper_attention
    assert piper_kernels.sage_attention_2pp is sage_attention_2pp
    assert piper_kernels.sparse_piper_attention is sparse_piper_attention
    assert piper_kernels.prepare_sparse_piper_attention_plan is prepare_sparse_piper_attention_plan
    assert piper_kernels.SparsePiperAttentionPlan is SparsePiperAttentionPlan
    assert piper_kernels.__all__ == [
        "SparsePiperAttentionPlan",
        "piper_attention",
        "prepare_sparse_piper_attention_plan",
        "sage_attention_2pp",
        "sparse_piper_attention",
    ]
    assert piper_kernels.attention.__name__ == "piper_kernels.attention"
    assert piper_kernels.attention.piper_attention.__name__ == (
        "piper_kernels.attention.piper_attention"
    )
    assert piper_kernels.attention.sage_attention_2pp.__name__ == (
        "piper_kernels.attention.sage_attention_2pp"
    )
    assert piper_kernels.attention.sparse_piper_attention.__name__ == (
        "piper_kernels.attention.sparse_piper_attention"
    )
    assert piper_kernels.linear.__name__ == "piper_kernels.linear"
    assert piper_kernels.linear.convrot.__name__ == "piper_kernels.linear.convrot"
    assert piper_kernels.linear.convrot.int8.__name__ == "piper_kernels.linear.convrot.int8"
    assert piper_kernels.linear.convrot.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.linear.convrot.int8.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.linear.convrot.convrot_int8_compile_options is convrot_int8_compile_options
    assert piper_kernels.linear.convrot.convrot_int8_linear is convrot_int8_linear
    assert convrot_int8_linear.__module__ == "piper_kernels.linear.convrot.int8.tensor"
    assert not hasattr(piper_kernels.linear.convrot, "convrot_linear")
    assert not hasattr(piper_kernels.linear.convrot, "convrot_compile_options")
    assert not hasattr(piper_kernels.linear.convrot.int8, "convrot_int8_linear")
    assert not hasattr(piper_kernels.linear.convrot, "linear_input_act")
    assert not hasattr(piper_kernels.linear.convrot.int8, "linear_input_act")
    assert piper_kernels.linear.convrot.__all__ == [
        "ConvRotInt8Tensor",
        "convrot_int8_compile_options",
        "convrot_int8_linear",
    ]
    assert piper_kernels.linear.convrot.int8.__all__ == ["ConvRotInt8Tensor"]
