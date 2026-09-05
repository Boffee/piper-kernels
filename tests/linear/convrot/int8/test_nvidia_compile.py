"""Offline compilation checks for kernels moved behind the NVIDIA implementation."""

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._nvidia import policy
from piper_kernels.linear.convrot.int8._nvidia import triton as nvidia


@pytest.mark.parametrize("architecture", [75, 89, 120])
def test_nvidia_preparation_compiles_without_a_device(architecture):
    target = AcceleratorTarget("cuda", f"sm{architecture}")
    plan = policy.select_execution_plan(target, in_features=5376)
    chunk_count, chunk_size = policy.select_fused_preparation_chunks(target, 5376)
    source = ASTSource(
        nvidia.rotate_quantize_rows_kernel,
        {"x_ptr": "*fp16", "q_ptr": "*i8", "scale_ptr": "*fp32", "row_width": "i32"},
        constexprs={
            "chunk_size": chunk_size,
            "chunk_count": chunk_count,
            "group_size": 256,
            "inverse_sqrt_group": 256**-0.5,
            "logical_dtype_code": 1,
            "activation_fn": "gelu_tanh",
            "accelerator_backend": "cuda",
            "gguf_quant_type": -1,
        },
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": plan.fused_num_warps},
    )
    assert compiled.asm["cubin"]


@pytest.mark.parametrize(
    "architecture",
    [
        pytest.param(
            75,
            marks=pytest.mark.xfail(
                triton.__version__ == "3.7.1",
                reason="Upstream SM75 INT8 dot lowering fails in Triton 3.7.1 (arith.extf on INT8)",
                raises=RuntimeError,
                strict=True,
            ),
        ),
        89,
        120,
    ],
)
def test_nvidia_paired_projection_compiles_to_matrix_instructions(architecture):
    plan = policy.select_execution_plan(
        AcceleratorTarget("cuda", f"sm{architecture}"), in_features=512
    )
    source = ASTSource(
        nvidia._int8_matmul_kernel,
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
            "group_m": 16,
        },
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": plan.matmul_num_warps, "num_stages": plan.matmul_num_stages},
    )
    assert "mma.sync" in compiled.asm["ptx"]
