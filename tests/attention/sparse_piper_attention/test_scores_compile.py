"""Compile scoring on both RDNA4 targets and check precision and addressing."""

import re
import sys

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels.attention.sparse_piper_attention._scores_triton import (
    _BLOCK_K,
    _BLOCK_M,
    _BLOCK_N,
    _minmax_scores_kernel,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires AMD compiler tooling")


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("has_score_scale", [False, True])
@pytest.mark.parametrize("specialization", ["generic", "h3_full_chunk", "h3_tail_chunk"])
def test_minmax_scoring_stays_fp32_without_spills(architecture, has_score_scale, specialization):
    constants = {
        "block_m": _BLOCK_M,
        "block_n": _BLOCK_N,
        "block_k": _BLOCK_K,
        "has_score_scale": has_score_scale,
    }
    signature = {name: "i32" for name in _minmax_scores_kernel.arg_names if name not in constants}
    signature.update({name: "*fp32" for name in signature if name.endswith("_ptr")})
    signature["score_scale"] = "fp32"
    attrs = {}
    if specialization != "generic":
        # H3 D128 prefix views retain aligned pointers and strides. Q64 also
        # specializes on divisibility; the Q27 tail, K1562, and H56 do not.
        aligned = [name for name in signature if name.endswith(("_ptr", "_stride"))]
        if specialization == "h3_full_chunk":
            aligned.append("query_blocks")
        attrs = {
            (_minmax_scores_kernel.arg_names.index(name),): [["tt.divisibility", 16]]
            for name in aligned
        }
    compiled = triton.compile(
        ASTSource(_minmax_scores_kernel, signature, constexprs=constants, attrs=attrs),
        target=GPUTarget("hip", architecture, 32),
        options={"num_warps": 4, "num_stages": 1},
    )
    assert compiled.asm["hsaco"]
    assert "arith.truncf" not in compiled.asm["ttgir"]
    assert "tt.fp_to_fp" not in compiled.asm["ttgir"]
    assert "bf16" not in compiled.asm["ttgir"]
    assert "f16" not in compiled.asm["ttgir"]
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", compiled.asm["amdgcn"])
    assert re.search(r"tensor<[^>]*xi64", compiled.asm["ttgir"])
