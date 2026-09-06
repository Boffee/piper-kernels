"""Generic primitives compile without the tuned GEMM architecture allowlist."""

import sys

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels.gguf import GGUFQuantizationType
from piper_kernels.linear.convrot.int8._kernels import triton as kernels
from piper_kernels.linear.convrot.triton import rotate_groups_kernel

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


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize(("dtype", "dtype_code"), [("fp16", 1), ("bf16", 2), ("fp32", 0)])
@pytest.mark.parametrize("operation", ["rotate", "quantize", "update"])
def test_generic_primitives_compile(target, dtype, dtype_code, operation):
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
            "block_size": 1024,
            "logical_dtype_code": dtype_code,
            "reciprocal_scale": target.backend == "hip",
        }
        if operation == "quantize":
            kernel = kernels.quantize_rows_kernel
            signature = {
                "x_ptr": f"*{dtype}",
                "q_ptr": "*i8",
                "scale_ptr": "*fp32",
                "row_width": "i32",
            }
        else:
            kernel = kernels.requantize_update_rows_kernel
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
            }
            constants.update({"has_base": True, "has_update": True, "stochastic": True})
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs=constants),
        target=target,
        options={"num_warps": 8 if operation == "update" else 4},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
@pytest.mark.parametrize("write_maxima", [True, False])
def test_generic_gguf_tiles_compile(target, quant_type, write_maxima):
    compiled = triton.compile(
        ASTSource(
            kernels.convert_gguf_tiles_kernel,
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
                "logical_dtype_code": 2,
                "quant_type": int(quant_type),
                "write_maxima": write_maxima,
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
            kernels.gguf_row_scales_kernel,
            {"maxima_ptr": "*fp32", "scale_ptr": "*fp32", "tiles_per_row": "i32"},
            constexprs={"block_size": 128, "reciprocal_scale": target.backend == "hip"},
        ),
        target=target,
        options={"num_warps": 4},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
@pytest.mark.parametrize("dtype_code", [0, 1, 2])
def test_shared_fused_gguf_compiles(target, quant_type, dtype_code):
    compiled = triton.compile(
        ASTSource(
            kernels.rotate_quantize_rows_kernel,
            {"x_ptr": "*u8", "q_ptr": "*i8", "scale_ptr": "*fp32", "row_width": "i32"},
            constexprs={
                "chunk_size": 2048,
                "chunk_count": 3,
                "group_size": 256,
                "inverse_sqrt_group": 256**-0.5,
                "logical_dtype_code": dtype_code,
                "activation_fn": None,
                "accelerator_backend": target.backend,
                "gguf_quant_type": int(quant_type),
            },
        ),
        target=target,
        options={"num_warps": 4},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]
