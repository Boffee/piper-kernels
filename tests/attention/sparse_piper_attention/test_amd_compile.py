"""Compile the complete fused RDNA4 kernel without requiring an AMD GPU."""

import re
import sys

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels.attention.sparse_piper_attention._amd.gluon import (
    _requires_64bit_context_offsets,
    _requires_64bit_query_offsets,
    _sparse_piper_attention_kernel,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ROCm support is Linux-only")


def _compile_attention(
    architecture,
    mask_block_lengths,
    has_dense_query_suffix,
    apply_coarse_residual,
    storage_sequence_length=256,
    head_dim=128,
    output_dtype="bf16",
    head_groups=1,
):
    constants = {
        "head_dim": head_dim,
        "head_groups": head_groups,
        "use_64bit_query_offsets": _requires_64bit_query_offsets(
            storage_sequence_length,
            head_dim,
        ),
        "use_64bit_context_offsets": _requires_64bit_context_offsets(
            storage_sequence_length,
            head_dim,
        ),
        "mask_block_lengths": mask_block_lengths,
        "has_dense_query_suffix": has_dense_query_suffix,
        "apply_coarse_residual": apply_coarse_residual,
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
            "value_mean_ptr": "*fp32",
            "coarse_output_ptr": "*fp32",
            "coarse_gate_ptr": f"*{output_dtype}",
            "block_lengths_ptr": "*i32",
            "routes_ptr": "*u16",
            "head_keep_blocks_ptr": "*i32",
            "route_head_offsets_ptr": "*i32",
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
@pytest.mark.parametrize("mask_block_lengths", [False, True])
@pytest.mark.parametrize("has_dense_query_suffix", [False, True])
@pytest.mark.parametrize("apply_coarse_residual", [False, True])
def test_fused_amd_compilation(
    architecture,
    mask_block_lengths,
    has_dense_query_suffix,
    apply_coarse_residual,
    head_dim,
):
    compiled = _compile_attention(
        architecture,
        mask_block_lengths,
        has_dense_query_suffix,
        apply_coarse_residual,
        head_dim=head_dim,
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
@pytest.mark.parametrize("apply_coarse_residual", [False, True])
def test_output_dtype_amd_compilation(architecture, dtype, head_dim, apply_coarse_residual):
    compiled = _compile_attention(
        architecture,
        True,
        True,
        apply_coarse_residual,
        head_dim=head_dim,
        output_dtype=dtype,
    )
    assert compiled.asm["hsaco"]
    assert f'!tt.ptr<{dtype.replace("fp", "f")}> loc("output_ptr"' in compiled.asm["ttgir"]
    assert "tt.fp_to_fp" not in compiled.asm["ttgir"]
    assert compiled.asm["ttgir"].count("arith.truncf") == (1 if dtype == "fp16" else 0)


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_runtime_lengths_support_wide_query_and_context_offsets(architecture, head_dim):
    # Compile only: a small UINT16 sparse prefix does not bound dense storage.
    # This also exceeds the signed-32-bit Q uint64-word-offset range.
    storage_sequence_length = (1 << 31) // (head_dim // 8) + 64
    compiled = _compile_attention(
        architecture,
        False,
        False,
        False,
        storage_sequence_length=storage_sequence_length,
        head_dim=head_dim,
    )
    assert compiled.asm["hsaco"]
    assert re.search(r"arith\.(?:muli|shli) .*: tensor<1x128xi64", compiled.asm["ttgir"])


@pytest.mark.parametrize("head_groups", [3, 4])
def test_gqa_amd_compilation(head_groups):
    compiled = _compile_attention("gfx1201", False, True, False, head_groups=head_groups)
    assert compiled.asm["hsaco"]
