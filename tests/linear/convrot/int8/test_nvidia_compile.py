"""Offline compilation checks for kernels moved behind the NVIDIA implementation."""

import pytest
import triton
from _nvidia_resources import compile_with_spill_stores
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import triton as kernels
from piper_kernels.linear.convrot.int8._nvidia import gluon_async_copy as gluon
from piper_kernels.linear.convrot.int8._nvidia import gluon_tma, policy
from piper_kernels.linear.convrot.int8._nvidia.policy import NvidiaExecutionPlan
from piper_kernels.weights.convrot.int8 import triton as kernels_weights
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
                triton.__version__ == "3.9.0",
                reason="Upstream Triton 3.9.0 SM75 INT8 lowering fails (arith.extf on INT8)",
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


@pytest.mark.parametrize("aligned_nk", [False, True])
def test_sm120_baseline_triton_paired_projection_compiles_to_matrix_instructions(aligned_nk):
    plan = policy.baseline_execution_plan(in_features=512)
    compiled = _compile_int8_matmul(
        plan,
        120,
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


@pytest.mark.parametrize("architecture", [80, 86, 89, 120])
@pytest.mark.parametrize(
    ("rows", "out_features"), [(1, 4096), (256, 1024), (8192, 128), (2048, 1024), (8192, 4096)]
)
@pytest.mark.parametrize("aligned", [False, True])
def test_sm8x_schedules_compile_within_consumer_shared_memory(
    architecture, rows, out_features, aligned
):
    plan = policy.select_execution_plan(
        AcceleratorTarget("cuda", "sm89"),
        in_features=5376,
        rows=rows,
        out_features=out_features,
    )
    assert isinstance(plan, NvidiaExecutionPlan)
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


_TMA_A_DESCRIPTOR = (
    f"tensordesc<i8[128,64],{gl.NVMMASharedLayout.get_default_for([128, 64], gl.int8)!r}>"
)
_TMA_B_DESCRIPTOR = (
    f"tensordesc<i8[256,64],{gl.NVMMASharedLayout.get_default_for([256, 64], gl.int8)!r}>"
)
_SM120_SHARED_MEMORY_LIMIT = 99 * 1024


def _tma_matmul_source(
    *,
    output: str,
    bias: str,
    paired: bool,
    stages: int,
    tail_refills: int,
    group_m: int,
    divisible_stride: bool,
):
    kernel = gluon_tma._int8_tma_matmul_kernel
    signature = {
        "input_descriptor": _TMA_A_DESCRIPTOR,
        "weight_descriptor": _TMA_B_DESCRIPTOR,
        "second_weight_descriptor": _TMA_B_DESCRIPTOR,
        "output_ptr": output,
        "input_scale_ptr": "*fp32",
        "weight_scale_ptr": "*fp32",
        "bias_ptr": bias,
        "second_scale_ptr": "*fp32",
        "second_bias_ptr": "*fp32",
        "m": "i32",
        "n": "i32",
        "output_row_stride": "i32",
        "full_groups": "i32",
    }
    aligned = ["output_ptr", "input_scale_ptr", "weight_scale_ptr", "bias_ptr"]
    aligned += ["second_scale_ptr", "second_bias_ptr"]
    if divisible_stride:
        aligned.append("output_row_stride")
    return GluonASTSource(
        kernel,
        signature,
        attrs={(kernel.arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned},
        constexprs={
            "stages": stages,
            "tail_refills": tail_refills,
            "group_m": group_m,
            "has_bias": True,
            "paired": paired,
            "second_has_bias": paired,
        },
    )


def _assert_sm120_tma_zero_spills(source: GluonASTSource) -> None:
    compiled, spill_stores = compile_with_spill_stores(
        source,
        GPUTarget("cuda", 120, 32),
        {"num_warps": 8},
    )
    assert compiled.asm["cubin"]
    assert "cp.async.bulk.tensor" in compiled.asm["ptx"]
    assert "mma.sync" in compiled.asm["ptx"]
    assert compiled.metadata.shared <= _SM120_SHARED_MEMORY_LIMIT
    assert spill_stores == 0


@pytest.mark.parametrize("output", ["*fp16", "*bf16", "*fp32"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize(("stages", "tail_refills"), [(1, 0), (2, 0), (3, 0), (3, 1), (3, 2)])
def test_sm120_tma_plain_and_paired_kernels_compile_without_spills(
    output, paired, stages, tail_refills
):
    _assert_sm120_tma_zero_spills(
        _tma_matmul_source(
            output=output,
            bias="*fp32",
            paired=paired,
            stages=stages,
            tail_refills=tail_refills,
            group_m=16,
            divisible_stride=True,
        )
    )


@pytest.mark.parametrize("output", ["*fp16", "*bf16", "*fp32"])
@pytest.mark.parametrize("paired", [False, True])
def test_sm120_tma_ragged_stride_compiles_without_spills(output, paired):
    _assert_sm120_tma_zero_spills(
        _tma_matmul_source(
            output=output,
            bias="*fp32",
            paired=paired,
            stages=3,
            tail_refills=1,
            group_m=16,
            divisible_stride=False,
        )
    )


def _tma_residual_source(
    *,
    output: str,
    has_bias: bool,
    reusable: bool,
    python_indexing: bool,
    stages: int,
    tail_refills: int,
):
    kernel = gluon_tma._int8_tma_residual_matmul_kernel
    signature = {
        "input_descriptor": _TMA_A_DESCRIPTOR,
        "weight_descriptor": _TMA_B_DESCRIPTOR,
        "output_ptr": output,
        "input_scale_ptr": "*fp32",
        "weight_scale_ptr": "*fp32",
        "bias_ptr": "*fp32" if has_bias else output,
        "m": "i32",
        "n": "i32",
        "full_groups": "i32",
        "residual_base_ptr": output,
        "residual_gate_ptr": output,
        "residual_indices_ptr": "*i64",
        "residual_reusable_gate_ptr": output,
        "residual_row_offset": "i32",
        "residual_gate_rows": "i32",
        "residual_reusable_gate_rows": "i32",
    }
    aligned = (
        "output_ptr",
        "input_scale_ptr",
        "weight_scale_ptr",
        "bias_ptr",
        "residual_base_ptr",
        "residual_gate_ptr",
        "residual_indices_ptr",
        "residual_reusable_gate_ptr",
    )
    return GluonASTSource(
        kernel,
        signature,
        attrs={(kernel.arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned},
        constexprs={
            "stages": stages,
            "tail_refills": tail_refills,
            "group_m": 16,
            "has_bias": has_bias,
            "residual_gate_row_stride": 5376,
            "residual_reusable_gate_row_stride": 5376 if reusable else 0,
            "residual_has_reusable_update": reusable,
            "residual_python_indexing": python_indexing,
        },
    )


@pytest.mark.parametrize("output", ["*fp16", "*bf16"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("reusable", [False, True])
@pytest.mark.parametrize("python_indexing", [False, True])
def test_sm120_tma_residual_kernel_compiles_without_spills(
    output, has_bias, reusable, python_indexing
):
    _assert_sm120_tma_zero_spills(
        _tma_residual_source(
            output=output,
            has_bias=has_bias,
            reusable=reusable,
            python_indexing=python_indexing,
            stages=3,
            tail_refills=1,
        )
    )


@pytest.mark.parametrize("output", ["*fp16", "*bf16"])
@pytest.mark.parametrize(("stages", "tail_refills"), [(1, 0), (2, 0), (3, 0), (3, 2)])
def test_sm120_tma_residual_schedules_compile_without_spills(output, stages, tail_refills):
    _assert_sm120_tma_zero_spills(
        _tma_residual_source(
            output=output,
            has_bias=True,
            reusable=True,
            python_indexing=True,
            stages=stages,
            tail_refills=tail_refills,
        )
    )


def _tma_swiglu_source(
    *,
    output: str,
    has_bias: bool,
    second_has_bias: bool,
    stages: int,
    tail_refills: int,
    divisible_stride: bool,
):
    kernel = gluon_tma._int8_tma_swiglu_matmul_kernel
    signature = {
        "input_descriptor": _TMA_A_DESCRIPTOR,
        "weight_descriptor": _TMA_A_DESCRIPTOR,
        "second_weight_descriptor": _TMA_A_DESCRIPTOR,
        "output_ptr": output,
        "input_scale_ptr": "*fp32",
        "weight_scale_ptr": "*fp32",
        "bias_ptr": "*fp32" if has_bias else output,
        "second_scale_ptr": "*fp32",
        "second_bias_ptr": "*fp16" if second_has_bias else output,
        "m": "i32",
        "n": "i32",
        "output_row_stride": "i32",
        "full_groups": "i32",
    }
    aligned = [
        "output_ptr",
        "input_scale_ptr",
        "weight_scale_ptr",
        "bias_ptr",
        "second_scale_ptr",
        "second_bias_ptr",
    ]
    if divisible_stride:
        aligned.append("output_row_stride")
    return GluonASTSource(
        kernel,
        signature,
        attrs={(kernel.arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned},
        constexprs={
            "stages": stages,
            "tail_refills": tail_refills,
            "group_m": 16,
            "has_bias": has_bias,
            "second_has_bias": second_has_bias,
        },
    )


@pytest.mark.parametrize("output", ["*fp16", "*bf16"])
@pytest.mark.parametrize(
    "biases",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["no-bias", "up-bias", "gate-bias", "both-biases"],
)
def test_sm120_tma_swiglu_kernel_compiles_without_spills(output, biases):
    has_bias, second_has_bias = biases
    _assert_sm120_tma_zero_spills(
        _tma_swiglu_source(
            output=output,
            has_bias=has_bias,
            second_has_bias=second_has_bias,
            stages=3,
            tail_refills=1,
            divisible_stride=True,
        )
    )


@pytest.mark.parametrize("output", ["*fp16", "*bf16"])
@pytest.mark.parametrize(("stages", "tail_refills"), [(1, 0), (2, 0), (3, 0), (3, 2)])
def test_sm120_tma_swiglu_schedules_compile_without_spills(output, stages, tail_refills):
    _assert_sm120_tma_zero_spills(
        _tma_swiglu_source(
            output=output,
            has_bias=True,
            second_has_bias=True,
            stages=stages,
            tail_refills=tail_refills,
            divisible_stride=True,
        )
    )


@pytest.mark.parametrize("output", ["*fp16", "*bf16"])
def test_sm120_tma_swiglu_ragged_stride_compiles_without_spills(output):
    _assert_sm120_tma_zero_spills(
        _tma_swiglu_source(
            output=output,
            has_bias=True,
            second_has_bias=True,
            stages=3,
            tail_refills=1,
            divisible_stride=False,
        )
    )


def _async_copy_swiglu_source(
    *,
    block_m: int,
    stages: int,
    output: str,
    has_bias: bool,
    second_has_bias: bool,
    whole_k_tiles: bool,
    divisible_stride: bool,
    group_m: int = 8,
):
    kernel = gluon._int8_swiglu_matmul_kernel
    signature = {
        "input_ptr": "*i8",
        "weight_ptr": "*i8",
        "second_weight_ptr": "*i8",
        "output_ptr": output,
        "input_scale_ptr": "*fp32",
        "weight_scale_ptr": "*fp32",
        "bias_ptr": "*fp32" if has_bias else output,
        "second_scale_ptr": "*fp32",
        "second_bias_ptr": "*fp16" if second_has_bias else output,
        "m": "i32",
        "n": "i32",
        "k": "i32",
        "output_row_stride": "i32",
    }
    aligned = ["input_ptr", "weight_ptr", "second_weight_ptr", "k"]
    if divisible_stride:
        aligned.append("output_row_stride")
    return GluonASTSource(
        kernel,
        signature,
        attrs={(kernel.arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned},
        constexprs={
            "block_m": block_m,
            "block_k": 64,
            "stages": stages,
            "group_m": group_m,
            "has_bias": has_bias,
            "second_has_bias": second_has_bias,
            "whole_k_tiles": whole_k_tiles,
        },
    )


def _assert_async_copy_swiglu_zero_spills(architecture: int, source: GluonASTSource) -> None:
    compiled, spill_stores = compile_with_spill_stores(
        source,
        GPUTarget("cuda", architecture, 32),
        {"num_warps": gluon._SWIGLU_NUM_WARPS},
    )
    assert compiled.asm["cubin"]
    assert "mma.sync" in compiled.asm["ptx"]
    assert compiled.metadata.shared <= _SM8X_SHARED_MEMORY_LIMIT
    assert spill_stores == 0


@pytest.mark.parametrize("architecture", [80, 86, 89, 120])
@pytest.mark.parametrize(("block_m", "stages"), [(128, 3), (128, 4), (256, 3), (256, 4)])
@pytest.mark.parametrize("output", ["*fp16", "*bf16"])
def test_async_copy_swiglu_schedules_compile_without_spills(architecture, block_m, stages, output):
    _assert_async_copy_swiglu_zero_spills(
        architecture,
        _async_copy_swiglu_source(
            block_m=block_m,
            stages=stages,
            output=output,
            has_bias=True,
            second_has_bias=True,
            whole_k_tiles=True,
            divisible_stride=True,
        ),
    )


@pytest.mark.parametrize("architecture", [86, 89, 120])
@pytest.mark.parametrize(("block_m", "stages"), [(128, 3), (256, 4)])
@pytest.mark.parametrize(
    "biases",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["no-bias", "up-bias", "gate-bias", "both-biases"],
)
@pytest.mark.parametrize("whole_k_tiles", [True, False], ids=["whole-k", "tail-k"])
@pytest.mark.parametrize("divisible_stride", [True, False], ids=["aligned-stride", "ragged-stride"])
def test_async_copy_swiglu_variants_compile_without_spills(
    architecture, block_m, stages, biases, whole_k_tiles, divisible_stride
):
    has_bias, second_has_bias = biases
    _assert_async_copy_swiglu_zero_spills(
        architecture,
        _async_copy_swiglu_source(
            block_m=block_m,
            stages=stages,
            output="*bf16",
            has_bias=has_bias,
            second_has_bias=second_has_bias,
            whole_k_tiles=whole_k_tiles,
            divisible_stride=divisible_stride,
        ),
    )


@pytest.mark.parametrize("architecture", [86, 89, 120])
@pytest.mark.parametrize(("block_m", "stages"), [(128, 3), (256, 4)])
def test_async_copy_swiglu_fp16_tail_ragged_ungrouped_compiles_without_spills(
    architecture, block_m, stages
):
    _assert_async_copy_swiglu_zero_spills(
        architecture,
        _async_copy_swiglu_source(
            block_m=block_m,
            stages=stages,
            output="*fp16",
            has_bias=True,
            second_has_bias=True,
            whole_k_tiles=False,
            divisible_stride=False,
            group_m=0,
        ),
    )


def _compile_int8_matmul(
    plan, architecture, *, output, biases, paired, aligned_m, aligned_nk, group_m
):
    """Compile the kernel an NVIDIA plan launches: Gluon for SM8x/SM120 Gluon plans, else Triton.

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
    if plan.matmul_kernel == "gluon_tma":
        source = _tma_matmul_source(
            output=output,
            bias=biases[0],
            paired=paired,
            stages=plan.matmul_num_stages,
            tail_refills=0,
            group_m=plan.matmul_group_m,
            divisible_stride=aligned_nk,
        )
        # Keep biases[1] for mixed bias dtype coverage on second_bias_ptr.
        source.signature["second_bias_ptr"] = biases[1]
        compiled = triton.compile(
            source, target=target, options={"num_warps": plan.matmul_num_warps}
        )
        assert "cp.async.bulk.tensor" in compiled.asm["ptx"]
        return compiled
    if plan.matmul_kernel == "gluon_async_copy":
        # Runtime specialization marks 16-byte-aligned INT8 operands and K, which the
        # launcher requires before selecting the Gluon GEMM.
        kernel = gluon._int8_matmul_kernel
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
                "warps_m": plan.matmul_num_warps // gluon._WARPS_N,
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
            "explicit_bias_fma": plan.matmul_explicit_bias_fma,
            "per_tile_tail": bool(group_m) or not plan.matmul_specialize_m,
            "accumulator_seed": 0,
            **flags,
        },
    )
    return triton.compile(
        source,
        target=target,
        options={"num_warps": plan.matmul_num_warps, "num_stages": plan.matmul_num_stages},
    )


def _splitk_attrs(out_features: int, in_features: int, *, paired: bool) -> tuple[dict, dict]:
    matmul_divisible = [0, 1, 2, 3]
    if out_features % 16 == 0:
        matmul_divisible.append(5)
    if in_features % 16 == 0:
        matmul_divisible.append(6)
    reduce_divisible = [0, 1, 2, 3, 4, 5, 6]
    if out_features % 16 == 0:
        reduce_divisible.append(8)
    if (out_features * (2 if paired else 1)) % 16 == 0:
        reduce_divisible.append(10)
    return (
        {(index,): [["tt.divisibility", 16]] for index in matmul_divisible},
        {(index,): [["tt.divisibility", 16]] for index in reduce_divisible},
    )


def _compile_nvidia_splitk_kernels(
    architecture: int,
    *,
    rows: int,
    out_features: int,
    in_features: int,
    output_ty: str = "*bf16",
    bias_ty: str | None = None,
    paired: bool = False,
    second_bias_ty: str | None = None,
    num_sms: int = 128,
) -> dict[str, tuple[triton.compiler.CompiledKernel, int]]:
    target = GPUTarget("cuda", architecture, 32)
    accel_target = AcceleratorTarget.from_compiler_target(target)
    plan = policy.select_execution_plan(
        accel_target, in_features=in_features, rows=rows, out_features=out_features
    )
    if plan.matmul_kernel in ("gluon_async_copy", "gluon_tma"):
        plan = policy.grouped_triton_plan(plan)
    split_k = kernels.select_int8_split_k(
        num_sms,
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        block_m=plan.matmul_block_m,
        block_n=plan.matmul_block_n,
        block_k=plan.matmul_block_k,
        paired=paired,
        group_m=plan.matmul_group_m,
    )
    if split_k <= 1:
        return {}
    matmul_attrs, reduce_attrs = _splitk_attrs(out_features, in_features, paired=paired)
    matmul_compiled = compile_with_spill_stores(
        ASTSource(
            kernels.int8_matmul_splitk_kernel,
            {
                "input_ptr": "*i8",
                "weight_ptr": "*i8",
                "second_weight_ptr": "*i8",
                "partial_ptr": "*i32",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "split_k": "i32",
            },
            attrs=matmul_attrs,
            constexprs={
                "block_m": plan.matmul_block_m,
                "block_n": plan.matmul_block_n,
                "block_k": plan.matmul_block_k,
                "paired": paired,
                "aligned_nk": (
                    out_features % plan.matmul_block_n == 0
                    and in_features % plan.matmul_block_k == 0
                ),
                "accumulator_seed": 0,
            },
        ),
        target,
        {"num_warps": plan.matmul_num_warps, "num_stages": plan.matmul_num_stages},
    )
    has_bias = bias_ty is not None
    second_has_bias = paired and second_bias_ty is not None
    second_bias_ptr = (
        second_bias_ty if second_has_bias and second_bias_ty is not None else output_ty
    )
    reduce_compiled = compile_with_spill_stores(
        ASTSource(
            kernels.int8_splitk_reduce_kernel,
            {
                "partial_ptr": "*i32",
                "input_scale_ptr": "*fp32",
                "weight_scale_ptr": "*fp32",
                "bias_ptr": output_ty if bias_ty is None else bias_ty,
                "second_scale_ptr": "*fp32",
                "second_bias_ptr": second_bias_ptr,
                "output_ptr": output_ty,
                "m": "i32",
                "n": "i32",
                "split_k": "i32",
                "output_row_stride": "i32",
            },
            attrs=reduce_attrs,
            constexprs={
                "block_n": kernels.INT8_SPLITK_REDUCE_BLOCK_N,
                "has_bias": has_bias,
                "paired": paired,
                "second_has_bias": second_has_bias,
                "explicit_bias_fma": plan.matmul_explicit_bias_fma,
            },
        ),
        target,
        {"num_warps": 4},
    )
    return {"splitk_matmul": matmul_compiled, "splitk_reduce": reduce_compiled}


@pytest.mark.parametrize("architecture", [80, 86, 89, 120])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize(
    ("rows", "out_features", "in_features", "bias_ty", "second_bias_ty"),
    [
        (1, 512, 16384, None, None),
        (1, 515, 16392, "*fp16", "*fp32"),
    ],
)
def test_nvidia_splitk_kernels_compile_within_resources(
    architecture: int,
    paired: bool,
    rows: int,
    out_features: int,
    in_features: int,
    bias_ty: str | None,
    second_bias_ty: str | None,
) -> None:
    compiled = _compile_nvidia_splitk_kernels(
        architecture,
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        bias_ty=bias_ty,
        paired=paired,
        second_bias_ty=second_bias_ty if paired else None,
        num_sms=128,
    )
    assert set(compiled) == {"splitk_matmul", "splitk_reduce"}
    matmul_kernel, matmul_spills = compiled["splitk_matmul"]
    _reduce_kernel, reduce_spills = compiled["splitk_reduce"]
    matmul_ptx = matmul_kernel.asm["ptx"]
    assert "mma.sync" in matmul_ptx
    if out_features % 16 == 0 and in_features % 16 == 0:
        assert "cp.async" in matmul_ptx
    assert matmul_kernel.metadata.shared <= 99 * 1024
    assert matmul_spills == 0
    assert reduce_spills == 0


def test_sm120_square_decode_compiles_splitk() -> None:
    compiled = _compile_nvidia_splitk_kernels(
        120,
        rows=1,
        out_features=5120,
        in_features=5120,
        num_sms=170,
    )
    assert set(compiled) == {"splitk_matmul", "splitk_reduce"}
    matmul_kernel, matmul_spills = compiled["splitk_matmul"]
    _reduce_kernel, reduce_spills = compiled["splitk_reduce"]
    matmul_ptx = matmul_kernel.asm["ptx"]
    assert "mma.sync" in matmul_ptx
    assert "cp.async" in matmul_ptx
    assert matmul_spills == 0
    assert reduce_spills == 0


@pytest.mark.parametrize(
    ("architecture", "rows", "out_features", "in_features", "num_sms"),
    [
        (75, 1, 512, 16384, 128),
        (120, 32, 512, 16384, 170),
        (120, 64, 512, 16384, 170),
        (120, 1, 512, 512, 170),
        (120, 1, 2, 4_194_304, 170),
    ],
)
def test_nvidia_non_splitk_workloads_skip_splitk_compilation(
    architecture: int,
    rows: int,
    out_features: int,
    in_features: int,
    num_sms: int,
) -> None:
    assert (
        _compile_nvidia_splitk_kernels(
            architecture,
            rows=rows,
            out_features=out_features,
            in_features=in_features,
            num_sms=num_sms,
        )
        == {}
    )
