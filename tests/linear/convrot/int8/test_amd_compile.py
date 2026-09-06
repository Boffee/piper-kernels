"""Offline AMD matrix-instruction coverage; runtime tests live in test_amd."""

import sys

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._amd import policy
from piper_kernels.linear.convrot.int8._amd import triton as amd
from piper_kernels.linear.convrot.int8._kernels import triton as kernels

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ROCm support is Linux-only")


@pytest.mark.parametrize("architecture", ["gfx942", "gfx1100", "gfx1151", "gfx1200", "gfx1201"])
def test_amd_paired_projection_compiles_to_matrix_instructions(architecture):
    plan = policy.select_execution_plan(AcceleratorTarget("hip", architecture), in_features=512)
    group_m = (
        amd._LARGE_MATMUL_GROUP_M_TILES
        if plan.matmul_block_m == 128 and plan.matmul_block_n == 256
        else 0
    )
    source = ASTSource(
        kernels.int8_matmul_kernel,
        {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "output_ptr": "*fp16",
            "input_scale_ptr": "*fp32",
            "weight_scale_ptr": "*fp32",
            "bias_ptr": "*fp16",
            "second_weight_ptr": "*i8",
            "second_scale_ptr": "*fp32",
            "second_bias_ptr": "*fp32",
            "m": "i32",
            "n": "i32",
            "k": "i32",
            "output_row_stride": "i32",
            "row_block_offset": "i32",
        },
        constexprs={
            "block_m": plan.matmul_block_m,
            "block_n": plan.matmul_block_n,
            "block_k": plan.matmul_block_k,
            "has_bias": True,
            "paired": True,
            "second_has_bias": True,
            "aligned_tiles": False,
            "group_m": group_m,
        },
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", architecture, 64 if architecture == "gfx942" else 32),
        options={
            "num_warps": plan.matmul_num_warps,
            "num_stages": plan.matmul_num_stages,
            **amd._amd_matmul_compiler_options(AcceleratorTarget("hip", architecture)),
        },
    )
    assert ("v_mfma_i32" if architecture == "gfx942" else "v_wmma_i32") in compiled.asm["amdgcn"]


@pytest.mark.parametrize("architecture", ["gfx942", "gfx1100", "gfx1151", "gfx1200", "gfx1201"])
@pytest.mark.parametrize("width", [5376, 9216, 12288, 14336, 16384])
def test_amd_activated_preparation_compiles(architecture, width):
    chunked = width != 16384
    plan = policy.select_execution_plan(AcceleratorTarget("hip", architecture), in_features=width)
    constants = {
        "group_size": 256,
        "inverse_sqrt_group": 256**-0.5,
        "activation_fn": "swiglu" if chunked else "gelu_tanh",
        "accelerator_backend": "hip",
    }
    if chunked:
        constants.update(
            zip(("block0", "block1", "block2"), policy.preparation_blocks(width), strict=True)
        )
    else:
        constants["block_size"] = width
    source = ASTSource(
        amd.rotate_quantize_rows_chunked_kernel if chunked else amd.rotate_quantize_rows_kernel,
        {
            "x_ptr": "*bf16" if chunked else "*fp16",
            "q_ptr": "*i8",
            "scale_ptr": "*fp32",
            "row_width": "i32",
        },
        constexprs=constants,
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", architecture, 64 if architecture == "gfx942" else 32),
        options={"num_warps": plan.fused_num_warps},
    )
    assert compiled.asm["hsaco"]
    # BF16 inputs are widened on load; rotated/activated values stay FP32
    # until the terminal integer conversion, on every supported AMD target.
    assert "arith.truncf" not in compiled.asm["ttir"]
