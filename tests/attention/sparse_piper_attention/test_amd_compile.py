"""Compile the complete fused RDNA4 kernel without requiring an AMD GPU."""

import re
import sys

import pytest
import triton
from _accelerators import rocm_fences_only_lds
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels.attention.sparse_piper_attention._amd import gluon as amd_gluon

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ROCm support is Linux-only")


def _compile_attention(
    architecture,
    mask_block_lengths,
    has_dense_query_suffix,
    apply_coarse_residual,
    storage_sequence_length=256,
    head_dim=128,
    output_dtype="bf16",
    query_heads_per_kv_head=1,
):
    kernel = amd_gluon._sparse_piper_attention_kernel
    constants = {
        "head_dim": head_dim,
        "query_heads_per_kv_head": query_heads_per_kv_head,
        "use_64bit_query_offsets": amd_gluon._requires_64bit_query_offsets(
            storage_sequence_length,
            head_dim,
        ),
        "use_64bit_context_offsets": amd_gluon._requires_64bit_context_offsets(
            storage_sequence_length,
            head_dim,
        ),
        "mask_block_lengths": mask_block_lengths,
        "has_dense_query_suffix": has_dense_query_suffix,
        "apply_coarse_residual": apply_coarse_residual,
    }
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        {
            "query_ptr": "*i8",
            "key_ptr": "*i8",
            "value_ptr": "*i8",
            "query_scale_ptr": "*fp32",
            "key_center_ptr": "*i8",
            "key_center_scale_ptr": "*fp32",
            "parameters_ptr": "*fp32",
            "value_mean_ptr": "*bf16",
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
        GluonASTSource(kernel, signature, constexprs=constants),
        target=GPUTarget("hip", architecture, 32),
        options={
            "num_warps": 4,
            "num_stages": 1,
            "llvm_fn_attrs": amd_gluon._LLVM_FN_ATTRS,
        },
    )


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("mask_block_lengths", [False, True])
@pytest.mark.parametrize("has_dense_query_suffix", [False, True])
@pytest.mark.parametrize("apply_coarse_residual", [False, True])
def test_fused_amd_compilation(
    mask_block_lengths,
    has_dense_query_suffix,
    apply_coarse_residual,
    head_dim,
):
    compiled = _compile_attention(
        "gfx1201",
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
    assert any(",0neg_lo:[1,0,0]" in line for line in matrix_instructions)  # zero-init K128 PV
    assert all("0x3e22f983neg_lo:[1,0,0]" not in line for line in matrix_instructions)
    assert all("clamp" not in line for line in matrix_instructions)
    # Keep rotated QK, online softmax, and the optional coarse epilogue in FP32.
    assert "tt.fp_to_fp" not in compiled.asm["ttgir"]
    assert compiled.asm["ttgir"].count("arith.truncf") == 1  # final BF16 store only
    if rocm_fences_only_lds():
        assert "global_inv" not in compiled.asm["amdgcn"]
    assert "ds_bpermute" not in compiled.asm["amdgcn"]
    assert re.search(r"amdhsa_private_segment_fixed_size\s+0\b", compiled.asm["amdgcn"])
    assert compiled.metadata.shared <= 6208


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("apply_coarse_residual", [False, True])
def test_output_dtype_amd_compilation(dtype, head_dim, apply_coarse_residual):
    compiled = _compile_attention(
        "gfx1201",
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
    assert re.search(r"amdhsa_private_segment_fixed_size\s+0\b", compiled.asm["amdgcn"])


@pytest.mark.parametrize("head_dim", [64, 128])
def test_runtime_lengths_support_wide_query_and_context_offsets(head_dim):
    # Compile only: a small UINT16 sparse prefix does not bound dense storage.
    # This also exceeds the signed-32-bit Q uint64-word-offset range.
    storage_sequence_length = (1 << 31) // (head_dim // 8) + 64
    compiled = _compile_attention(
        "gfx1201",
        False,
        False,
        False,
        storage_sequence_length=storage_sequence_length,
        head_dim=head_dim,
    )
    assert compiled.asm["hsaco"]
    assert re.search(r"arith\.(?:muli|shli) .*: tensor<1x128xi64", compiled.asm["ttgir"])


@pytest.mark.parametrize("query_heads_per_kv_head", [3, 4])
def test_gqa_amd_compilation(query_heads_per_kv_head):
    compiled = _compile_attention(
        "gfx1201", False, True, False, query_heads_per_kv_head=query_heads_per_kv_head
    )
    assert compiled.asm["hsaco"]


def test_prepare_parameters_compiles_with_runtime_count():
    """Keep count non-constexpr so parameter folding does not recompile per shape."""
    kernel = amd_gluon._prepare_parameters
    assert "count" in kernel.do_not_specialize
    signature = {
        "key_scale_ptr": "*fp32",
        "multiplier_ptr": "*fp32",
        "table_ptr": "*fp32",
        "count": "i32",
    }
    compiled = triton.compile(
        GluonASTSource(kernel, signature, constexprs={}),
        target=GPUTarget("hip", "gfx1201", 32),
        options={"num_warps": 4},
    )
    assert compiled.asm["hsaco"]


def _compile_preparation(
    architecture,
    kind,
    *,
    head_dim=128,
    input_dtype="bf16",
    emit_summaries=False,
    mask_block_lengths=False,
    mask_rows=False,
    mask_heads=False,
    aligned_16b=True,
):
    kernels = {
        "query": amd_gluon._prepare_query_kernel,
        "key": amd_gluon._prepare_key_kernel,
        "value": amd_gluon._prepare_value_kernel,
    }
    kernel = kernels[kind]
    constants = {
        "head_dim": head_dim,
        "mask_block_lengths": mask_block_lengths,
        "mask_rows": mask_rows,
        "mask_heads": mask_heads,
        "aligned_16b": aligned_16b,
    }
    if "emit_summaries" in kernel.arg_names:
        constants["emit_summaries"] = emit_summaries

    pointer_types = {
        "query_ptr": f"*{input_dtype}",
        "key_ptr": f"*{input_dtype}",
        "value_ptr": f"*{input_dtype}",
        "query_int8_ptr": "*i8",
        "key_int8_ptr": "*i8",
        "key_center_ptr": "*i8",
        "value_int8_ptr": "*i8",
        "block_lengths_ptr": "*i32",
        "value_mean_ptr": "*bf16",
    }
    signature = {}
    for name in kernel.arg_names:
        if name in constants:
            continue
        if name in pointer_types:
            signature[name] = pointer_types[name]
        elif name.endswith("_ptr"):
            signature[name] = "*fp32"
        elif name == "softmax_scale":
            signature[name] = "fp32"
        else:
            signature[name] = "i32"

    input_ptr_name = f"{kind}_ptr"
    attrs = {
        (kernel.arg_names.index(name),): [["tt.divisibility", 16]]
        for name, arg_type in signature.items()
        if arg_type.startswith("*") and (aligned_16b or name != input_ptr_name)
    }
    return kernel, triton.compile(
        GluonASTSource(kernel, signature, constexprs=constants, attrs=attrs),
        target=GPUTarget("hip", architecture, 32),
        options={
            "num_warps": 8,
            "num_stages": 1,
            "llvm_fn_attrs": amd_gluon.amd_quantization.PREP_LLVM_FN_ATTRS,
        },
    )


@pytest.mark.parametrize(
    ("kind", "emit_summaries"),
    [
        ("query", False),
        ("query", True),
        ("key", False),
        ("key", True),
        ("value", False),
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("input_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("mask_block_lengths", [False, True])
def test_vectorized_preparation_compiles_for_rdna4(
    kind,
    emit_summaries,
    head_dim,
    input_dtype,
    mask_block_lengths,
):
    kernel, compiled = _compile_preparation(
        "gfx1201",
        kind,
        head_dim=head_dim,
        input_dtype=input_dtype,
        emit_summaries=emit_summaries,
        mask_block_lengths=mask_block_lengths,
        mask_rows=mask_block_lengths,
        mask_heads=False,
        aligned_16b=True,
    )
    assert compiled.asm["hsaco"]
    assert compiled.metadata.global_scratch_size == 0
    assert re.search(r"amdhsa_private_segment_fixed_size\s+0\b", compiled.asm["amdgcn"])
    assert "global_load_b128" in compiled.asm["amdgcn"]
    if rocm_fences_only_lds():
        assert "global_inv" not in compiled.asm["amdgcn"]
    if kind in ("query", "key"):
        assert "ds_swizzle_b32" in compiled.asm["amdgcn"]
        assert "s_wait_dscnt" in compiled.asm["amdgcn"]
    expected_runtime_args = (
        "logical_sequence_length",
        "storage_tiles",
        "stride_b",
        "stride_h",
        "stride_n",
        "heads",
        "heads_inner",
    )
    for name in expected_runtime_args:
        assert name in kernel.do_not_specialize
    if kind == "query":
        assert "%softmax_scale: f32" in compiled.asm["ttgir"]
    elif kind == "key":
        assert "sparse_key_blocks" in kernel.do_not_specialize


@pytest.mark.parametrize("kind", ["query", "key", "value"])
@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
def test_vectorized_preparation_odd_heads_unaligned_and_fp32_smoke(kind, architecture):
    _kernel, compiled = _compile_preparation(
        architecture,
        kind,
        head_dim=64,
        input_dtype="fp32",
        emit_summaries=(kind != "value"),
        mask_block_lengths=False,
        mask_rows=True,
        mask_heads=True,
        aligned_16b=False,
    )
    assert compiled.asm["hsaco"]
    assert compiled.metadata.global_scratch_size == 0
    assert re.search(r"amdhsa_private_segment_fixed_size\s+0\b", compiled.asm["amdgcn"])
    if rocm_fences_only_lds():
        assert "global_inv" not in compiled.asm["amdgcn"]
