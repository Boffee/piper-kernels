"""Generic primitives compile without the tuned GEMM architecture allowlist."""

import sys

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton import convrot_int8 as kernels_weights
from piper_kernels._triton.convrot import rotate_groups_kernel
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.gguf import GGUFQuantizationType
from piper_kernels.weights.convrot.int8._policy import select_addmm_plan

_TARGETS = [
    GPUTarget("cuda", 70, 32),
    GPUTarget("cuda", 75, 32),
    *[
        pytest.param(
            GPUTarget("hip", architecture, 64 if architecture == "gfx942" else 32),
            marks=pytest.mark.skipif(sys.platform != "linux", reason="HIP compiler is Linux-only"),
        )
        for architecture in ("gfx1030", "gfx1036", "gfx942", "gfx1201")
    ],
]


@pytest.mark.parametrize(
    "target",
    [
        *[GPUTarget("cuda", architecture, 32) for architecture in (80, 89, 120)],
        *[
            pytest.param(
                GPUTarget("hip", architecture, 64 if architecture == "gfx942" else 32),
                marks=pytest.mark.skipif(
                    sys.platform != "linux", reason="HIP compiler is Linux-only"
                ),
            )
            for architecture in ("gfx1030", "gfx1036", "gfx1100", "gfx942", "gfx1201")
        ],
    ],
)
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize(
    ("group_size", "width", "rank", "write_maxima", "stochastic"),
    [
        (64, 320, 37, True, True),
        (64, 320, 64, False, False),
        (256, 256, 16, False, True),
        (256, 768, 37, False, True),
    ],
)
def test_fused_lora_merge_compiles(
    target, dtype, group_size, width, rank, write_maxima, stochastic
):
    plan = select_addmm_plan(AcceleratorTarget.from_compiler_target(target), group_size)
    assert plan is not None
    compiled = triton.compile(
        ASTSource(
            kernels_weights.merge_lora_tiles_kernel,
            {
                "q_ptr": "*i8",
                "scale_ptr": "*fp32",
                "new_scale_ptr": "*fp32",
                "mat1_ptr": f"*{dtype}",
                "mat2_ptr": f"*{dtype}",
                "maxima_ptr": "*fp32",
                "rows_count": "i32",
                "beta": "fp32",
                "alpha": "fp32",
                "rounding_seed": "i64",
            },
            constexprs={
                "row_width": width,
                "rank": rank,
                "stride_a_row": rank,
                "stride_a_col": 1,
                "stride_b_row": width,
                "stride_b_col": 1,
                "group_size": group_size,
                "matmul_dtype": {"fp16": tl.float16, "bf16": tl.bfloat16, "fp32": tl.float32}[
                    dtype
                ],
                "has_base": True,
                "stochastic": stochastic,
                "write_maxima": write_maxima,
                "single_tile": width <= plan.block_n,
                "accelerator_backend": target.backend,
                "block_m": plan.block_m,
                "block_n": plan.block_n,
                "block_k": plan.block_k,
            },
        ),
        target=target,
        options={"num_warps": plan.num_warps, "num_stages": plan.num_stages},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]
    assert compiled.metadata.shared <= 65536


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize(
    "operation", ["rotate", "quantize", "quantize_static", "update", "update_wide"]
)
def test_generic_primitives_compile(target, dtype, operation):
    if operation == "rotate":
        kernel = rotate_groups_kernel
        signature = {
            "input_ptr": f"*{dtype}",
            "output_ptr": f"*{dtype}",
            "row_width": "i32",
            "groups_per_row": "i32",
        }
        constants = {"group_size": 256, "inverse_sqrt_group": 256**-0.5}
    else:
        constants = {
            "block_size": 16384 if operation == "update_wide" else 1024,
            "reciprocal_scale": target.backend == "hip",
            "accelerator_backend": target.backend,
        }
        if operation in ("quantize", "quantize_static"):
            kernel = kernels_weights.quantize_rows_kernel
            signature = {
                "x_ptr": f"*{dtype}",
                "q_ptr": "*i8",
                "scale_ptr": "*fp32",
                "row_width": "i32",
            }
            if operation == "quantize_static":
                signature["static_scale_ptr"] = "*fp32"
            else:
                constants["static_scale_ptr"] = None
        else:
            kernel = kernels_weights.requantize_update_rows_kernel
            signature = {
                "q_ptr": "*i8",
                "scale_ptr": "*fp32",
                "update_ptr": f"*{dtype}",
                "row_width": "i32",
                "stride_q_row": "i32",
                "stride_q_col": "i32",
                "stride_scale_row": "i32",
                "stride_update_row": "i32",
                "stride_update_col": "i32",
                "beta": "fp32",
                "alpha": "fp32",
                "rounding_seed": "i64",
                "start_row": "i64",
            }
            constants.update({"has_base": True, "has_update": True, "stochastic": True})
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs=constants),
        target=target,
        options={"num_warps": 8 if operation.startswith("update") else 4},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
@pytest.mark.parametrize("write_maxima", [True, False])
def test_generic_gguf_tiles_compile(target, quant_type, write_maxima):
    compiled = triton.compile(
        ASTSource(
            kernels_weights.convert_gguf_tiles_kernel,
            {
                "data_ptr": "*u8",
                "q_ptr": "*i8",
                "scale_ptr": "*fp32",
                "maxima_ptr": "*fp32",
                "row_width": "i32",
                "tiles_per_row": "i32",
            },
            constexprs={
                "block_size": 1024,
                "group_size": 256,
                "quant_type": int(quant_type),
                "write_maxima": write_maxima,
                "accelerator_backend": target.backend,
            },
        ),
        target=target,
        options={"num_warps": 4},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]


@pytest.mark.parametrize("target", _TARGETS)
def test_generic_gguf_scale_reduction_compiles(target):
    compiled = triton.compile(
        ASTSource(
            kernels_weights.reduce_row_scales_kernel,
            {"maxima_ptr": "*fp32", "scale_ptr": "*fp32", "tiles_per_row": "i32"},
            constexprs={"block_size": 128, "reciprocal_scale": target.backend == "hip"},
        ),
        target=target,
        options={"num_warps": 4},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
def test_shared_fused_gguf_compiles(target, quant_type):
    compiled = triton.compile(
        ASTSource(
            kernels_weights.rotate_quantize_rows_kernel,
            {"x_ptr": "*u8", "q_ptr": "*i8", "scale_ptr": "*fp32", "row_width": "i32"},
            constexprs={
                "chunk_size": 2048,
                "chunk_count": 3,
                "group_size": 256,
                "inverse_sqrt_group": 256**-0.5,
                "activation_fn": None,
                "accelerator_backend": target.backend,
                "gguf_quant_type": int(quant_type),
                "static_scale_ptr": None,
            },
        ),
        target=target,
        options={"num_warps": 4},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]
