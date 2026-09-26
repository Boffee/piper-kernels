"""Offline compilation checks for kernels moved behind the NVIDIA implementation."""

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton import convrot_int8 as kernels_weights
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._kernels import triton as kernels
from piper_kernels.linear.convrot.int8._nvidia import gluon, policy
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks


@pytest.mark.parametrize("architecture", [75, 89, 120])
@pytest.mark.parametrize("static", [False, True])
def test_nvidia_preparation_compiles_without_a_device(architecture, static):
    target = AcceleratorTarget("cuda", f"sm{architecture}")
    plan = policy.select_execution_plan(target, in_features=5376)
    chunk_count, chunk_size = fused_preparation_chunks(5376)
    source = ASTSource(
        kernels_weights.rotate_quantize_rows_kernel,
        {
            "x_ptr": "*fp16",
            "q_ptr": "*i8",
            "scale_ptr": "*fp32",
            "row_width": "i32",
            **({"static_scale_ptr": "*fp32"} if static else {}),
        },
        constexprs={
            "chunk_size": chunk_size,
            "chunk_count": chunk_count,
            "group_size": 256,
            "inverse_sqrt_group": 256**-0.5,
            "activation_fn": "gelu_tanh",
            "accelerator_backend": "cuda",
            "gguf_quant_type": -1,
            **({} if static else {"static_scale_ptr": None}),
        },
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": plan.fused_num_warps},
    )
    assert compiled.asm["cubin"]

    if static:
        assert "tt.reduce" not in compiled.asm["ttir"]


@pytest.mark.parametrize(
    "architecture",
    [
        pytest.param(
            75,
            marks=pytest.mark.xfail(
                triton.__version__ in ("3.7.1", "3.8.0"),
                reason="Upstream Triton 3.7.1/3.8.0 SM75 INT8 lowering fails (arith.extf on INT8)",
                raises=RuntimeError,
                strict=True,
            ),
        ),
        89,
        120,
    ],
)
@pytest.mark.parametrize("aligned_nk", [False, True])
def test_nvidia_paired_projection_compiles_to_matrix_instructions(architecture, aligned_nk):
    plan = policy.select_execution_plan(
        AcceleratorTarget("cuda", f"sm{architecture}"), in_features=512
    )
    compiled = _compile_int8_matmul(
        plan,
        architecture,
        output="*fp16",
        biases=("*fp16", "*fp32"),
        paired=True,
        aligned_m=False,
        aligned_nk=aligned_nk,
        group_m=16,
    )
    assert "mma.sync" in compiled.asm["ptx"]


# SM86/SM89 cap one block's dynamic shared memory at 99 KiB; SM80/SM87 allow more.
_SM8X_SHARED_MEMORY_LIMIT = 99 * 1024


@pytest.mark.parametrize("architecture", [80, 86, 89])
@pytest.mark.parametrize(
    ("rows", "out_features"), [(1, 4096), (256, 1024), (8192, 128), (2048, 1024), (8192, 4096)]
)
@pytest.mark.parametrize("aligned", [False, True])
def test_sm8x_schedules_compile_within_consumer_shared_memory(
    architecture, rows, out_features, aligned
):
    plan = policy.select_execution_plan(
        AcceleratorTarget("cuda", f"sm{architecture}"),
        in_features=5376,
        rows=rows,
        out_features=out_features,
    )
    assert isinstance(plan, policy.Sm8xExecutionPlan)
    compiled = _compile_int8_matmul(
        plan,
        architecture,
        output="*bf16",
        biases=("*bf16", "*bf16"),
        paired=False,
        # SM8x launches branch on ragged M per tile instead of specializing on it.
        aligned_m=False,
        aligned_nk=aligned,
        group_m=plan.matmul_group_m,
    )
    assert "mma.sync" in compiled.asm["ptx"]
    assert compiled.metadata.shared <= _SM8X_SHARED_MEMORY_LIMIT


def _compile_int8_matmul(
    plan, architecture, *, output, biases, paired, aligned_m, aligned_nk, group_m
):
    """Compile the kernel an NVIDIA plan launches: Gluon for SM8x Gluon plans, else Triton.

    ``aligned_m`` and ``group_m`` apply to Triton only; Gluon plans use their own grouping
    and branch on ragged M at run time.
    """
    signature = {
        "input_ptr": "*i8",
        "weight_ptr": "*i8",
        "output_ptr": output,
        "input_scale_ptr": "*fp32",
        "weight_scale_ptr": "*fp32",
        "bias_ptr": biases[0],
        "second_weight_ptr": "*i8",
        "second_scale_ptr": "*fp32",
        "second_bias_ptr": biases[1],
        "m": "i32",
        "n": "i32",
        "k": "i32",
        "output_row_stride": "i32",
    }
    flags = {"has_bias": True, "paired": paired, "second_has_bias": paired}
    target = GPUTarget("cuda", architecture, 32)
    if isinstance(plan, policy.Sm8xExecutionPlan) and plan.matmul_kernel == "gluon":
        # Runtime specialization marks 16-byte-aligned INT8 operands and K, which the
        # launcher requires before selecting the Gluon GEMM.
        kernel = gluon.int8_matmul_gluon_kernel
        aligned = ("input_ptr", "weight_ptr", "second_weight_ptr", "k")
        source = GluonASTSource(
            kernel,
            signature,
            attrs={(kernel.arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned},
            constexprs={
                "block_m": plan.matmul_block_m,
                "block_n": plan.matmul_block_n,
                "block_k": plan.matmul_block_k,
                "stages": plan.matmul_num_stages,
                "group_m": plan.matmul_group_m,
                "warps_m": plan.matmul_num_warps // gluon.WARPS_N,
                "whole_k_tiles": aligned_nk,
                **flags,
            },
        )
        return triton.compile(source, target=target, options={"num_warps": plan.matmul_num_warps})
    source = ASTSource(
        kernels.int8_matmul_kernel,
        signature,
        constexprs={
            "block_m": plan.matmul_block_m,
            "block_n": plan.matmul_block_n,
            "block_k": plan.matmul_block_k,
            "aligned_m": aligned_m,
            "aligned_nk": aligned_nk,
            "group_m": group_m,
            "explicit_bias_fma": isinstance(plan, policy.Sm8xExecutionPlan),
            **flags,
        },
    )
    return triton.compile(
        source,
        target=target,
        options={"num_warps": plan.matmul_num_warps, "num_stages": plan.matmul_num_stages},
    )
