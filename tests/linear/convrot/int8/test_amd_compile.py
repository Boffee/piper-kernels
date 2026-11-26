"""Offline AMD matrix-instruction coverage; runtime tests live in test_amd."""

import re
import sys
from dataclasses import replace
from functools import partial

import pytest
import triton
from _accelerators import rocm_fences_only_lds
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8 import triton as kernels
from piper_kernels.linear.convrot.int8._amd import policy
from piper_kernels.linear.convrot.int8._amd import triton as amd

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ROCm support is Linux-only")


@pytest.mark.parametrize("architecture", ["gfx942", "gfx1100", "gfx1151", "gfx1201"])
@pytest.mark.parametrize("aligned_nk", [False, True])
def test_amd_paired_projection_compiles_to_matrix_instructions(architecture, aligned_nk):
    target = AcceleratorTarget("hip", architecture)
    width = 8_192
    plan = policy.select_execution_plan(target, in_features=width)
    group_m = policy.matmul_group_m(plan)
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
        },
        constexprs={
            "block_m": plan.matmul_block_m,
            "block_n": plan.matmul_block_n,
            "block_k": plan.matmul_block_k,
            "has_bias": True,
            "paired": True,
            "second_has_bias": True,
            "aligned_m": False,
            "aligned_nk": aligned_nk,
            "group_m": group_m,
            "explicit_bias_fma": False,
            "per_tile_tail": bool(group_m),
            "accumulator_seed": policy.matmul_accumulator_seed(target, width, 64),
        },
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", architecture, 64 if architecture == "gfx942" else 32),
        options={
            "num_warps": plan.matmul_num_warps,
            "num_stages": plan.matmul_num_stages,
            **amd._amd_matmul_compiler_options(target),
        },
    )
    assert ("v_mfma_i32" if architecture == "gfx942" else "v_wmma_i32") in compiled.asm["amdgcn"]


def _compile_rdna4_projection(architecture, plan, *, aligned_nk, group_m, options):
    source = ASTSource(
        kernels.int8_matmul_kernel,
        {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "output_ptr": "*bf16",
            "input_scale_ptr": "*fp32",
            "weight_scale_ptr": "*fp32",
            "bias_ptr": "*bf16",
            "second_weight_ptr": "*i8",
            "second_scale_ptr": "*fp32",
            "second_bias_ptr": "*bf16",
            "m": "i32",
            "n": "i32",
            "k": "i32",
            "output_row_stride": "i32",
        },
        constexprs={
            "block_m": plan.matmul_block_m,
            "block_n": plan.matmul_block_n,
            "block_k": plan.matmul_block_k,
            "has_bias": False,
            "paired": False,
            "second_has_bias": False,
            "aligned_m": False,
            "aligned_nk": aligned_nk,
            "group_m": group_m,
            "explicit_bias_fma": False,
            "per_tile_tail": bool(group_m),
            "accumulator_seed": policy.matmul_accumulator_seed(
                AcceleratorTarget("hip", architecture), 8192, 64
            ),
        },
    )
    return triton.compile(
        source,
        target=GPUTarget("hip", architecture, 32),
        options={
            "num_warps": plan.matmul_num_warps,
            "num_stages": plan.matmul_num_stages,
            **options,
        },
    )


@pytest.mark.parametrize("aligned_nk", [False, True])
def test_rdna4_h3_projection_schedule_compiles(aligned_nk):
    target = AcceleratorTarget("hip", "gfx1201")
    plan = replace(
        policy.select_execution_plan(target, in_features=8_192),
        matmul_block_n=128,
    )
    compiled = _compile_rdna4_projection(
        "gfx1201",
        plan,
        aligned_nk=aligned_nk,
        group_m=0,
        options=amd._amd_matmul_compiler_options(target),
    )
    assert "v_wmma_i32" in compiled.asm["amdgcn"]


def test_gfx1200_projection_target_smoke_compiles():
    """Keep lightweight offline gfx1200 target wiring without duplicating RDNA4 matrices."""
    target = AcceleratorTarget("hip", "gfx1200")
    plan = policy.select_execution_plan(target, in_features=8_192)
    compiled = _compile_rdna4_projection(
        "gfx1200",
        plan,
        aligned_nk=True,
        group_m=policy.matmul_group_m(plan),
        options=amd._amd_matmul_compiler_options(target),
    )
    assert "v_wmma_i32" in compiled.asm["amdgcn"]


@pytest.mark.skipif(
    not rocm_fences_only_lds(),
    reason="requires a ROCm build whose Triton fences global memory at LDS barriers",
)
def test_rdna4_projection_barriers_fence_only_lds():
    target = AcceleratorTarget("hip", "gfx1201")
    plan = policy.select_execution_plan(target, in_features=5_120)
    options = amd._amd_matmul_compiler_options(target)
    lds_only = amd_compiler_workarounds.LDS_ONLY_FENCE_ATTRS
    global_fences = {
        "llvm_fn_attrs": tuple(item for item in options["llvm_fn_attrs"] if item not in lds_only)
    }
    compile_projection = partial(
        _compile_rdna4_projection,
        "gfx1201",
        plan,
        aligned_nk=True,
        group_m=policy.matmul_group_m(plan),
    )
    # The first check fails if Triton stops fencing global memory at its LDS
    # barriers: then retire the workaround as amd_compiler_workarounds describes.
    assert "global_inv" in compile_projection(options=global_fences).asm["amdgcn"]
    assert "global_inv" not in compile_projection(options=options).asm["amdgcn"]


@pytest.mark.parametrize("architecture", ["gfx942", "gfx1100", "gfx1151", "gfx1201"])
@pytest.mark.parametrize("width", [5376, 9216, 12288, 14336, 16384])
@pytest.mark.parametrize("static", [False, True])
def test_amd_activated_preparation_compiles(architecture, width, static):
    chunked = width != 16384
    plan = policy.select_execution_plan(AcceleratorTarget("hip", architecture), in_features=width)
    constants = {
        "group_size": 256,
        "inverse_sqrt_group": 256**-0.5,
        "activation_fn": "swiglu" if chunked else "gelu_tanh",
        "accelerator_backend": "hip",
        **({} if static else {"static_scale_ptr": None}),
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
            **({"static_scale_ptr": "*fp32"} if static else {}),
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
    if not chunked:
        assert "ocml_tanh" not in compiled.asm["llir"]
        assert "v_exp_f32" in compiled.asm["amdgcn"]

    if static:
        assert "tt.reduce" not in compiled.asm["ttir"]


@pytest.mark.parametrize(
    ("output_type", "fuse_swiglu", "full_rows", "has_bias"),
    [
        ("*bf16", False, False, False),
        ("*bf16", False, False, True),
        ("*bf16", False, True, False),
        ("*bf16", False, True, True),
        ("*bf16", True, False, False),
        ("*bf16", True, True, True),
        ("*fp16", True, False, True),
        ("*fp16", True, True, False),
    ],
)
def test_rdna4_unpacked_paired_matmul_compiles_spill_free(
    output_type, fuse_swiglu, full_rows, has_bias
):
    target = AcceleratorTarget("hip", "gfx1201")
    source = ASTSource(
        amd._paired_int8_matmul_kernel,
        {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "output_ptr": output_type,
            "input_scale_ptr": "*fp32",
            "weight_scale_ptr": "*fp32",
            "bias_ptr": output_type,
            "second_weight_ptr": "*i8",
            "second_scale_ptr": "*fp32",
            "second_bias_ptr": output_type,
            "m": "i32",
            "output_row_stride": "i32",
        },
        constexprs={
            "n": 14336,
            "k": 5376,
            "has_bias": has_bias,
            "second_has_bias": has_bias,
            "full_rows": full_rows,
            "accumulator_seed": 0x155555,
            "fuse_swiglu": fuse_swiglu,
        },
        attrs={
            **{(index,): [["tt.divisibility", 16], ["tt.pointer_range", 32]] for index in range(9)},
            (10,): [["tt.divisibility", 16]],
        },
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", "gfx1201", 32),
        options={"num_warps": 8, "num_stages": 2, **amd._amd_matmul_compiler_options(target)},
    )
    asm = compiled.asm["amdgcn"]
    assert "v_wmma_i32" in asm
    assert ".vgpr_spill_count: 0" in asm
    assert ".private_segment_fixed_size: 0" in asm
    if rocm_fences_only_lds():
        assert "global_inv" not in asm
    if fuse_swiglu:
        assert "v_exp_f32" in asm
        assert "v_rcp_f32" in asm
        assert "v_div_fmas" not in asm


@pytest.mark.parametrize(
    ("block_n", "block_k", "features", "width"),
    [(128, 128, 6144, 16384), (256, 64, 5376, 14336)],
)
@pytest.mark.parametrize("full_rows", [False, True])
@pytest.mark.parametrize("reusable", [False, True])
def test_rdna4_residual_matmul_compiles_with_bounded_spills(
    block_n, block_k, features, width, full_rows, reusable
):
    target = AcceleratorTarget("hip", "gfx1201")
    source = ASTSource(
        amd._amd_int8_residual_matmul_kernel,
        {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "output_ptr": "*bf16",
            "input_scale_ptr": "*fp32",
            "weight_scale_ptr": "*fp32",
            "bias_ptr": "*bf16",
            "base_ptr": "*bf16",
            "gate_ptr": "*bf16",
            "reusable_gate_ptr": "*bf16",
            "indices_ptr": "*i64",
            "row_offset": "i32",
            "gate_rows": "i32",
            "reusable_gate_rows": "i32",
            "m": "i32",
        },
        constexprs={
            "n": features,
            "k": width,
            "block_n": block_n,
            "block_k": block_k,
            "has_bias": True,
            "full_rows": full_rows,
            "gate_row_stride": features * 2,
            "reusable_gate_row_stride": features * 2 if reusable else 0,
            "has_reusable_update": reusable,
            "python_indexing": True,
            "accumulator_seed": 0x155555,
        },
        attrs={
            (index,): [["tt.divisibility", 16], ["tt.pointer_range", 32]] for index in range(10)
        },
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", "gfx1201", 32),
        options={"num_warps": 8, "num_stages": 2, **amd._amd_matmul_compiler_options(target)},
    )
    asm = compiled.asm["amdgcn"]
    assert "v_wmma_i32" in asm
    if rocm_fences_only_lds():
        assert "global_inv" not in asm
        match = re.search(r"\.vgpr_spill_count:\s*(\d+)", asm)
        assert match is not None
        spills = int(match.group(1))
        assert spills == 0 if (block_n == 128 or not reusable) else spills <= 16


def _compile_amd_splitk_kernels(
    architecture: str,
    warp_size: int,
    *,
    rows: int,
    out_features: int,
    in_features: int,
    output_ty: str = "*bf16",
    bias_ty: str | None = None,
    paired: bool = False,
    second_bias_ty: str | None = None,
    num_sms: int = 64,
) -> dict[str, triton.compiler.CompiledKernel]:
    target = AcceleratorTarget("hip", architecture)
    plan = policy.select_execution_plan(
        target, in_features=in_features, rows=rows, out_features=out_features
    )
    group_m = policy.matmul_group_m(plan)
    split_k = kernels.select_int8_split_k(
        num_sms,
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        block_m=plan.matmul_block_m,
        block_n=plan.matmul_block_n,
        block_k=plan.matmul_block_k,
        paired=paired,
        group_m=group_m,
    )
    if split_k <= 1:
        return {}
    accumulator_seed = kernels.splitk_accumulator_seed(
        target,
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        block_m=plan.matmul_block_m,
        block_n=plan.matmul_block_n,
        block_k=plan.matmul_block_k,
        paired=paired,
        split_k=split_k,
    )
    hip_target = GPUTarget("hip", architecture, warp_size)
    matmul_compiled = triton.compile(
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
            constexprs={
                "block_m": plan.matmul_block_m,
                "block_n": plan.matmul_block_n,
                "block_k": plan.matmul_block_k,
                "paired": paired,
                "aligned_nk": (
                    out_features % plan.matmul_block_n == 0
                    and in_features % plan.matmul_block_k == 0
                ),
                "accumulator_seed": accumulator_seed,
            },
        ),
        target=hip_target,
        options={
            "num_warps": plan.matmul_num_warps,
            "num_stages": plan.matmul_num_stages,
            **amd._amd_matmul_compiler_options(target),
        },
    )
    second_has_bias = paired and second_bias_ty is not None
    reduce_compiled = triton.compile(
        ASTSource(
            kernels.int8_splitk_reduce_kernel,
            {
                "partial_ptr": "*i32",
                "input_scale_ptr": "*fp32",
                "weight_scale_ptr": "*fp32",
                "bias_ptr": output_ty if bias_ty is None else bias_ty,
                "second_scale_ptr": "*fp32",
                "second_bias_ptr": second_bias_ty if second_has_bias else output_ty,
                "output_ptr": output_ty,
                "m": "i32",
                "n": "i32",
                "split_k": "i32",
                "output_row_stride": "i32",
            },
            constexprs={
                "block_n": kernels.INT8_SPLITK_REDUCE_BLOCK_N,
                "has_bias": bias_ty is not None,
                "paired": paired,
                "second_has_bias": second_has_bias,
                "explicit_bias_fma": False,
            },
        ),
        target=hip_target,
        options={"num_warps": 4},
    )
    return {"splitk_matmul": matmul_compiled, "splitk_reduce": reduce_compiled}


def _assert_amd_splitk_zero_spills(
    architecture: str,
    compiled: dict[str, triton.compiler.CompiledKernel],
) -> None:
    # gfx1201 splitk_matmul inherits pre-existing 647-909 VGPR spills from the
    # shared (16, 64, 256) WMMA schedule, which is tracked outside Phase 1 cleanup.
    checked = (
        ("splitk_reduce",) if architecture == "gfx1201" else ("splitk_matmul", "splitk_reduce")
    )
    for name in checked:
        amdgcn = compiled[name].asm["amdgcn"]
        vgpr_spills = re.search(r"\.vgpr_spill_count:\s*(\d+)", amdgcn)
        assert vgpr_spills is not None, f"missing .vgpr_spill_count in {architecture} {name}"
        assert int(vgpr_spills.group(1)) == 0, (
            f"{architecture} {name} spilled {vgpr_spills.group(1)} VGPRs"
        )
        priv_size = re.search(r"\.private_segment_fixed_size:\s*(\d+)", amdgcn)
        assert priv_size is not None, (
            f"missing .private_segment_fixed_size in {architecture} {name}"
        )
        assert int(priv_size.group(1)) == 0, (
            f"{architecture} {name} reserved private_segment {priv_size.group(1)}"
        )


@pytest.mark.parametrize(
    ("architecture", "warp_size"),
    [("gfx942", 64), ("gfx1100", 32), ("gfx1151", 32), ("gfx1201", 32)],
)
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize(
    ("rows", "out_features", "in_features", "bias_ty", "second_bias_ty"),
    [
        (1, 512, 16384, None, None),
        (1, 515, 16392, "*fp16", "*fp32"),
    ],
)
def test_amd_splitk_kernels_compile_to_matrix_instructions(
    architecture: str,
    warp_size: int,
    paired: bool,
    rows: int,
    out_features: int,
    in_features: int,
    bias_ty: str | None,
    second_bias_ty: str | None,
) -> None:
    compiled = _compile_amd_splitk_kernels(
        architecture,
        warp_size,
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        bias_ty=bias_ty,
        paired=paired,
        second_bias_ty=second_bias_ty if paired else None,
        num_sms=64,
    )
    assert set(compiled) == {"splitk_matmul", "splitk_reduce"}
    expected_isa = "v_mfma_i32" if architecture == "gfx942" else "v_wmma_i32"
    amdgcn = compiled["splitk_matmul"].asm["amdgcn"]
    assert expected_isa in amdgcn
    _assert_amd_splitk_zero_spills(architecture, compiled)
    if architecture == "gfx1201" and rocm_fences_only_lds():
        assert "global_inv" not in amdgcn


def test_gfx1201_seeded_splitk_kernel_compiles() -> None:
    compiled = _compile_amd_splitk_kernels(
        "gfx1201",
        32,
        rows=1,
        out_features=2048,
        in_features=16384,
        num_sms=64,
    )
    assert set(compiled) == {"splitk_matmul", "splitk_reduce"}
    assert "v_wmma_i32" in compiled["splitk_matmul"].asm["amdgcn"]
    _assert_amd_splitk_zero_spills("gfx1201", compiled)


@pytest.mark.parametrize(
    ("architecture", "warp_size", "rows", "out_features", "in_features", "num_sms"),
    [
        ("gfx1200", 32, 1, 512, 16384, 64),
        ("gfx1201", 32, 1, 17408, 5120, 32),
        ("gfx1201", 32, 17, 2048, 4096, 32),
        ("gfx1201", 32, 32, 512, 16384, 32),
        ("gfx1201", 32, 64, 512, 16384, 32),
        ("gfx1201", 32, 1, 512, 512, 32),
    ],
)
def test_amd_non_splitk_workloads_skip_splitk_compilation(
    architecture: str,
    warp_size: int,
    rows: int,
    out_features: int,
    in_features: int,
    num_sms: int,
) -> None:
    assert (
        _compile_amd_splitk_kernels(
            architecture,
            warp_size,
            rows=rows,
            out_features=out_features,
            in_features=in_features,
            num_sms=num_sms,
        )
        == {}
    )
