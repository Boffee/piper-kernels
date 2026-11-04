"""Offline coverage of dense RDNA4 signed QK and mixed-sign PV lowering."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels.attention.piper_attention._amd import gluon as amd_gluon

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ROCm support is Linux-only")

_PINNED_LOAD_BARRIER = "sched.barrier(i32 1)"


def _compile_dense_kernel(
    architecture="gfx1201",
    head_dim=128,
    causal=False,
    dtype="bf16",
    wide=False,
    *,
    llvm_fn_attrs=amd_gluon._LLVM_FN_ATTRS,
):
    constants = {
        "head_groups": 3,
        "head_dim": head_dim,
        "is_causal": causal,
        "wide_query_offsets": wide,
        "wide_context_offsets": wide,
    }
    kernel = amd_gluon._dense_piper_kernel
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        dict.fromkeys(
            (
                "query_scale_ptr",
                "key_scale_ptr",
                "multiplier_ptr",
                "log_scale_ptr",
                "value_mean_ptr",
            ),
            "*fp32",
        )
    )
    signature.update(query_ptr="*i8", key_ptr="*i8", value_ptr="*i8", output_ptr=f"*{dtype}")
    if wide:
        signature.update(
            dict.fromkeys(
                ("query_start", "global_query_start", "stride_ob", "stride_oh", "stride_om"),
                "i64",
            )
        )
    return triton.compile(
        GluonASTSource(kernel, signature, constexprs=constants),
        target=GPUTarget("hip", architecture, 32),
        options={"num_warps": 4, "num_stages": 1, "llvm_fn_attrs": llvm_fn_attrs},
    )


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
def test_dense_amd_compilation(architecture, head_dim, causal, dtype, wide):
    compiled = _compile_dense_kernel(architecture, head_dim, causal, dtype, wide)
    assert compiled.asm["hsaco"]
    instructions = [
        line.replace(" ", "")
        for line in compiled.asm["amdgcn"].splitlines()
        if "v_wmma_i32_16x16x16_iu8" in line
    ]
    assert any("neg_lo:[1,1,0]" in line for line in instructions)
    assert any("neg_lo:[1,0,0]" in line for line in instructions)
    assert all("clamp" not in line for line in instructions)
    assert compiled.asm["ttgir"].count("arith.truncf") == 1  # final output only


def _compile_llvm_ir_with_scalar_fma(scalar_fma: bool, head_dim: int = 128) -> str:
    """Compile in a fresh process: Triton keys kernels on globals at first compile."""
    tests_root = Path(__file__).resolve().parents[2]
    script = f"""
import sys
from triton.experimental.gluon import language as gl
from piper_kernels.attention.kernels.piper._amd import fragments

fragments._SCALAR_FMA = gl.constexpr({scalar_fma!r})
sys.path.insert(0, {str(tests_root)!r})
from attention.piper_attention.test_amd_compile import _compile_dense_kernel

sys.stdout.write(_compile_dense_kernel(head_dim={head_dim}).asm["llir"])
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("head_dim", [64, 128])
def test_dense_amd_scalar_fma_emits_no_vector_fma(head_dim):
    llvm_ir = _compile_llvm_ir_with_scalar_fma(True, head_dim)
    # RDNA4 has no packed FP32 FMA, so vector FMAs would only add register pressure.
    assert "@llvm.fma.v" not in llvm_ir
    assert "v_fma_f32 $0, $1, 0x4c800000, $2" in llvm_ir  # QK score recovery


@pytest.mark.skipif(
    not amd_compiler_workarounds.USE_SCALAR_FMA,
    reason="the installed Triton keeps these FMAs scalar",
)
def test_dense_amd_plain_fma_is_still_vectorized():
    # When this fails, the installed Triton no longer vectorizes these FMAs:
    # retire USE_SCALAR_FMA as amd_compiler_workarounds describes.
    assert "@llvm.fma.v" in _compile_llvm_ir_with_scalar_fma(False)


@pytest.mark.skipif(
    torch.version.hip is None or not amd_compiler_workarounds.UNPINNED_WARP_PIPELINE_ATTRS,
    reason="requires a ROCm build whose Triton pins warp-pipeline memory operations",
)
def test_dense_amd_launch_attributes_unpin_warp_pipeline_loads():
    pinned = _compile_dense_kernel(llvm_fn_attrs=(("target-features", "+cumode"),))
    unpinned = _compile_dense_kernel()
    # The first check fails if Triton changes how it pins loads, which would
    # otherwise silently disable the workaround, or stops pinning them: then
    # retire the workaround as amd_compiler_workarounds describes.
    assert _PINNED_LOAD_BARRIER in pinned.asm["llir"]
    assert _PINNED_LOAD_BARRIER not in unpinned.asm["llir"]
    assert "sched_barrier mask(0x00000001)" not in unpinned.asm["amdgcn"]
