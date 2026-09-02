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
import piper_kernels.fusions
import piper_kernels.fusions.convrot_nvfp4_sparse_piper
import piper_kernels.fusions.convrot_sparse_piper
import piper_kernels.linear
import piper_kernels.linear.convrot
import piper_kernels.linear.convrot.int8
import piper_kernels.linear.convrot.nvfp4
from piper_kernels import (
    SparsePiperAttention,
    apply_coarse_attention_residual,
    coarse_attention,
    coarse_attention_residual,
    mean_pool_block_values,
    mean_pool_coarse_residual,
    minmax_pool_coarse_residual,
    piper_attention,
    sage_attention_2pp,
)
from piper_kernels.fusions.convrot_nvfp4_sparse_piper import (
    convrot_nvfp4_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_sparse_piper import (
    convrot_sparse_piper_compile_options,
)
from piper_kernels.linear.convrot import (
    SUPPORTED_GROUP_SIZES,
    ConvRotInt8Tensor,
    convrot_int8_compile_options,
    convrot_int8_linear,
)
from piper_kernels.linear.convrot.nvfp4 import (
    ConvRotNVFP4Tensor,
    convrot_nvfp4_compile_options,
    convrot_nvfp4_linear,
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
assert "piper_kernels.fusions.convrot_sparse_piper._compile" not in sys.modules
assert "piper_kernels.attention.sparse_piper_attention._quantized_dispatch" not in sys.modules
convrot.convrot_int8_compile_options()
assert "piper_kernels.linear.convrot.int8._compile" in sys.modules
assert "piper_kernels.linear._preparation_sharing" in sys.modules
assert "piper_kernels.fusions.convrot_sparse_piper._compile" not in sys.modules
assert "piper_kernels.attention.sparse_piper_attention._quantized_dispatch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_convrot_nvfp4_compiler_integration_is_loaded_lazily() -> None:
    script = """
import sys
import piper_kernels.linear.convrot.nvfp4 as convrot_nvfp4

assert "piper_kernels.linear.convrot.nvfp4._compile" not in sys.modules
assert "piper_kernels.linear._preparation_sharing" not in sys.modules
convrot_nvfp4.convrot_nvfp4_compile_options()
assert "piper_kernels.linear.convrot.nvfp4._compile" in sys.modules
assert "piper_kernels.linear._preparation_sharing" in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_sparse_attention_does_not_load_convrot() -> None:
    script = """
import sys
import piper_kernels.attention.sparse_piper_attention

assert "piper_kernels.linear.convrot" not in sys.modules
assert "piper_kernels.linear.convrot.int8.triton" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_convrot_sage_qk_does_not_load_sparse_piper_fusion() -> None:
    script = """
import sys
import piper_kernels.fusions.convrot_sage_qk.triton

assert "piper_kernels.fusions.convrot_sparse_piper._compile" not in sys.modules
assert "piper_kernels.fusions.convrot_sparse_piper.query" not in sys.modules
assert "piper_kernels.fusions.convrot_sparse_piper.key" not in sys.modules
assert "piper_kernels.attention.sparse_piper_attention._quantized_dispatch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_convrot_sparse_piper_compiler_integration_is_loaded_lazily() -> None:
    script = """
import sys
from piper_kernels.fusions import convrot_sparse_piper

assert "piper_kernels.fusions.convrot_sparse_piper._compile" not in sys.modules
assert "piper_kernels.fusions.convrot_sparse_piper._output_compile" not in sys.modules
assert "piper_kernels.fusions.convrot_sparse_piper.output" not in sys.modules
convrot_sparse_piper.convrot_sparse_piper_compile_options()
assert "piper_kernels.fusions.convrot_sparse_piper._compile" in sys.modules
assert "piper_kernels.fusions.convrot_sparse_piper._output_compile" in sys.modules
assert "piper_kernels.fusions.convrot_sparse_piper.output" in sys.modules
assert "piper_kernels.linear.convrot.int8._compile" in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_convrot_nvfp4_sparse_piper_compiler_integration_is_loaded_lazily() -> None:
    script = """
import sys
from piper_kernels.fusions import convrot_nvfp4_sparse_piper

assert "piper_kernels.fusions.convrot_nvfp4_sparse_piper._compile" not in sys.modules
assert "piper_kernels.fusions.convrot_nvfp4_sparse_piper._output_compile" not in sys.modules
assert "piper_kernels.fusions.convrot_nvfp4_sparse_piper.output" not in sys.modules
convrot_nvfp4_sparse_piper.convrot_nvfp4_sparse_piper_compile_options()
assert "piper_kernels.fusions.convrot_nvfp4_sparse_piper._compile" in sys.modules
assert "piper_kernels.fusions.convrot_nvfp4_sparse_piper._output_compile" in sys.modules
assert "piper_kernels.fusions.convrot_nvfp4_sparse_piper.output" in sys.modules
assert "piper_kernels.linear.convrot.nvfp4._compile" in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_packages_import() -> None:
    assert piper_kernels.__name__ == "piper_kernels"
    assert piper_kernels.piper_attention is piper_attention
    assert piper_kernels.sage_attention_2pp is sage_attention_2pp
    assert piper_kernels.SparsePiperAttention is SparsePiperAttention
    assert piper_kernels.apply_coarse_attention_residual is apply_coarse_attention_residual
    assert piper_kernels.coarse_attention is coarse_attention
    assert piper_kernels.coarse_attention_residual is coarse_attention_residual
    assert piper_kernels.minmax_pool_coarse_residual is minmax_pool_coarse_residual
    assert piper_kernels.mean_pool_block_values is mean_pool_block_values
    assert piper_kernels.mean_pool_coarse_residual is mean_pool_coarse_residual
    assert piper_kernels.__all__ == [
        "SparsePiperAttention",
        "apply_coarse_attention_residual",
        "coarse_attention",
        "coarse_attention_residual",
        "mean_pool_block_values",
        "mean_pool_coarse_residual",
        "minmax_pool_coarse_residual",
        "piper_attention",
        "sage_attention_2pp",
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
    assert piper_kernels.fusions.__name__ == "piper_kernels.fusions"
    assert piper_kernels.fusions.convrot_nvfp4_sparse_piper.__name__ == (
        "piper_kernels.fusions.convrot_nvfp4_sparse_piper"
    )
    assert (
        piper_kernels.fusions.convrot_nvfp4_sparse_piper.convrot_nvfp4_sparse_piper_compile_options
        is convrot_nvfp4_sparse_piper_compile_options
    )
    assert piper_kernels.fusions.convrot_sparse_piper.__name__ == (
        "piper_kernels.fusions.convrot_sparse_piper"
    )
    assert (
        piper_kernels.fusions.convrot_sparse_piper.convrot_sparse_piper_compile_options
        is convrot_sparse_piper_compile_options
    )
    assert piper_kernels.linear.__name__ == "piper_kernels.linear"
    assert piper_kernels.linear.convrot.__name__ == "piper_kernels.linear.convrot"
    assert piper_kernels.linear.convrot.int8.__name__ == "piper_kernels.linear.convrot.int8"
    assert piper_kernels.linear.convrot.nvfp4.__name__ == "piper_kernels.linear.convrot.nvfp4"
    assert piper_kernels.linear.convrot.SUPPORTED_GROUP_SIZES is SUPPORTED_GROUP_SIZES
    assert piper_kernels.linear.convrot.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.linear.convrot.int8.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.linear.convrot.nvfp4.ConvRotNVFP4Tensor is ConvRotNVFP4Tensor
    assert (
        piper_kernels.linear.convrot.nvfp4.convrot_nvfp4_compile_options
        is convrot_nvfp4_compile_options
    )
    assert piper_kernels.linear.convrot.convrot_int8_compile_options is convrot_int8_compile_options
    assert piper_kernels.linear.convrot.convrot_int8_linear is convrot_int8_linear
    assert piper_kernels.linear.convrot.nvfp4.convrot_nvfp4_linear is convrot_nvfp4_linear
    assert convrot_int8_linear.__module__ == "piper_kernels.linear.convrot.int8.tensor"
    assert not hasattr(piper_kernels.linear.convrot, "convrot_linear")
    assert not hasattr(piper_kernels.linear.convrot, "convrot_compile_options")
    assert not hasattr(piper_kernels.linear.convrot.int8, "convrot_int8_linear")
    assert not hasattr(piper_kernels.linear.convrot, "linear_input_act")
    assert not hasattr(piper_kernels.linear.convrot.int8, "linear_input_act")
    assert piper_kernels.linear.convrot.__all__ == [
        "SUPPORTED_GROUP_SIZES",
        "ConvRotInt8Tensor",
        "convrot_int8_compile_options",
        "convrot_int8_linear",
    ]
    assert piper_kernels.linear.convrot.int8.__all__ == ["ConvRotInt8Tensor"]
    assert piper_kernels.linear.convrot.nvfp4.__all__ == [
        "ConvRotNVFP4Tensor",
        "convrot_nvfp4_compile_options",
        "convrot_nvfp4_linear",
        "dynamic_scale",
        "prepare_dynamic",
        "prepare_static",
        "prepare_static_out",
    ]
