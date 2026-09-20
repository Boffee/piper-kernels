"""Offline coverage of dense RDNA4 signed QK and mixed-sign PV lowering."""

import sys

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels.attention.piper_attention._amd.gluon import _dense_piper_kernel

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ROCm support is Linux-only")


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
def test_dense_amd_compilation(architecture, head_dim, causal, dtype, wide):
    constants = {
        "head_groups": 3,
        "head_dim": head_dim,
        "is_causal": causal,
        "wide_query_offsets": wide,
        "wide_context_offsets": wide,
    }
    signature = {name: "i32" for name in _dense_piper_kernel.arg_names if name not in constants}
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
    compiled = triton.compile(
        GluonASTSource(_dense_piper_kernel, signature, constexprs=constants),
        target=GPUTarget("hip", architecture, 32),
        options={
            "num_warps": 4,
            "num_stages": 1,
            "llvm_fn_attrs": (("target-features", "+cumode"),),
        },
    )
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
