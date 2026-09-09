"""Compile the complete fused RDNA4 kernel without requiring an AMD GPU."""

import re
import sys

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels.attention.sparse_piper_attention._amd.gluon import _sparse_piper_attention_kernel

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ROCm support is Linux-only")


def _compile_attention(
    architecture,
    has_lengths,
    has_dense_queries,
    has_coarse,
    storage_length=256,
    head_dim=128,
    output_dtype="bf16",
):
    constants = {
        "query_storage_length": storage_length,
        "storage_length": storage_length,
        "heads": 2,
        "head_dim": head_dim,
        "logical_length": storage_length,
        "sparse_blocks": 3,
        "sparse_query_blocks": 2,
        "stride_rb": 16,
        "stride_rq": 4,
        "stride_ob": 65536,
        "stride_oh": 32768,
        "stride_on": head_dim,
        "stride_cb": 1024,
        "stride_ch": 512,
        "stride_cq": head_dim,
        "stride_gb": 65536,
        "stride_gh": head_dim,
        "stride_gn": 2 * head_dim,
        "has_lengths": has_lengths,
        "has_dense_queries": has_dense_queries,
        "has_coarse": has_coarse,
    }
    signature = {
        name: "i32" for name in _sparse_piper_attention_kernel.arg_names if name not in constants
    }
    signature.update(
        {
            "query_ptr": "*i8",
            "key_ptr": "*i8",
            "value_ptr": "*i8",
            "query_scale_ptr": "*fp32",
            "parameters_ptr": "*fp32",
            "mean_ptr": "*fp32",
            "coarse_ptr": "*fp32",
            "gate_ptr": f"*{output_dtype}",
            "lengths_ptr": "*i32",
            "routes_ptr": "*u16",
            "keep_ptr": "*i32",
            "route_offsets_ptr": "*i32",
            "output_ptr": f"*{output_dtype}",
        }
    )
    return triton.compile(
        GluonASTSource(_sparse_piper_attention_kernel, signature, constexprs=constants),
        target=GPUTarget("hip", architecture, 32),
        options={
            "num_warps": 4,
            "num_stages": 1,
            "llvm_fn_attrs": (("target-features", "+cumode"),),
        },
    )


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("has_lengths", [False, True])
@pytest.mark.parametrize("has_dense_queries", [False, True])
@pytest.mark.parametrize("has_coarse", [False, True])
def test_fused_amd_compilation(architecture, has_lengths, has_dense_queries, has_coarse, head_dim):
    compiled = _compile_attention(
        architecture, has_lengths, has_dense_queries, has_coarse, head_dim=head_dim
    )
    assert compiled.asm["hsaco"]
    matrix_instructions = [
        line.replace(" ", "")
        for line in compiled.asm["amdgcn"].splitlines()
        if "v_wmma_i32_16x16x16_iu8" in line
    ]
    assert any("neg_lo:[1,1,0]" in line for line in matrix_instructions)  # signed QK
    assert any("neg_lo:[1,0,0]" in line for line in matrix_instructions)  # unsigned P, signed V
    assert all("clamp" not in line for line in matrix_instructions)
    # Keep rotated QK, online softmax, and the optional coarse epilogue in FP32.
    assert "tt.fp_to_fp" not in compiled.asm["ttgir"]
    assert compiled.asm["ttgir"].count("arith.truncf") == 1  # final BF16 store only


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("has_coarse", [False, True])
def test_output_dtype_amd_compilation(architecture, dtype, head_dim, has_coarse):
    compiled = _compile_attention(
        architecture, True, True, has_coarse, head_dim=head_dim, output_dtype=dtype
    )
    assert compiled.asm["hsaco"]
    assert f'!tt.ptr<{dtype.replace("fp", "f")}> loc("output_ptr"' in compiled.asm["ttgir"]
    assert "tt.fp_to_fp" not in compiled.asm["ttgir"]
    assert compiled.asm["ttgir"].count("arith.truncf") == (1 if dtype == "fp16" else 0)


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_dense_suffix_and_query_word_offsets_can_use_wide_addressing(architecture, head_dim):
    # Compile only: a small UINT16 sparse prefix does not bound dense storage.
    # This also exceeds the signed-32-bit Q uint64-word-offset range.
    compiled = _compile_attention(
        architecture, False, False, False, (1 << 31) // (head_dim // 8) + 64, head_dim
    )
    assert compiled.asm["hsaco"]
    assert re.search(r"arith\.(?:muli|shli) .*: tensor<1x128xi64", compiled.asm["ttgir"])
