"""Compile minmax scoring offline across RDNA4 and SM120 and check precision and spills."""

import re
import sys

import pytest
import triton
from _nvidia_resources import stack_frame_bytes
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels.attention.sparse_piper_attention._amd.policy import (
    select_minmax_score_schedule as select_amd_minmax_score_schedule,
)
from piper_kernels.attention.sparse_piper_attention._nvidia.policy import (
    select_sm120_minmax_score_schedule,
)
from piper_kernels.attention.sparse_piper_attention._scores_triton import _minmax_scores_kernel

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux compiler tooling")


def _compile_minmax_scores(
    target: GPUTarget,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    has_score_scale: bool,
    specialization: str,
    query_heads_per_kv_head: int,
):
    constants = {
        "query_heads_per_kv_head": query_heads_per_kv_head,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
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
        target=target,
        options={"num_warps": 4, "num_stages": 1},
    )
    assert "arith.truncf" not in compiled.asm["ttgir"]
    assert "tt.fp_to_fp" not in compiled.asm["ttgir"]
    assert "bf16" not in compiled.asm["ttgir"]
    assert "f16" not in compiled.asm["ttgir"]
    assert re.search(r"tensor<[^>]*xi64", compiled.asm["ttgir"])
    return compiled


@pytest.mark.parametrize("query_heads_per_kv_head", [1, 3])
@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("has_score_scale", [False, True])
@pytest.mark.parametrize("specialization", ["generic", "h3_full_chunk", "h3_tail_chunk"])
def test_minmax_scoring_stays_fp32_without_spills(
    architecture, has_score_scale, specialization, query_heads_per_kv_head
):
    schedule = select_amd_minmax_score_schedule()
    compiled = _compile_minmax_scores(
        GPUTarget("hip", architecture, 32),
        block_m=schedule.block_m,
        block_n=schedule.block_n,
        block_k=schedule.block_k,
        has_score_scale=has_score_scale,
        specialization=specialization,
        query_heads_per_kv_head=query_heads_per_kv_head,
    )
    assert compiled.asm["hsaco"]
    scratch = re.search(r"\.amdhsa_private_segment_fixed_size\s+(\d+)\b", compiled.asm["amdgcn"])
    assert scratch
    if specialization == "generic" and triton.__version__ == "3.9.0":
        # With identical TTGIR, Triton 3.9's AMDGPU backend spills 64 bytes when
        # no pointer or stride is 16-aligned. Launches pass aligned allocations
        # and D128 strides, which select the spill-free specializations.
        assert int(scratch.group(1)) <= 64
    else:
        assert int(scratch.group(1)) == 0


@pytest.mark.parametrize("query_blocks", [64, 128])
@pytest.mark.parametrize("query_heads_per_kv_head", [1, 3])
@pytest.mark.parametrize("has_score_scale", [False, True])
@pytest.mark.parametrize("specialization", ["generic", "h3_full_chunk", "h3_tail_chunk"])
def test_sm120_minmax_scoring_stays_fp32_without_spills(
    query_blocks, has_score_scale, specialization, query_heads_per_kv_head
):
    schedule = select_sm120_minmax_score_schedule(query_blocks)
    compiled = _compile_minmax_scores(
        GPUTarget("cuda", 120, 32),
        block_m=schedule.block_m,
        block_n=schedule.block_n,
        block_k=schedule.block_k,
        has_score_scale=has_score_scale,
        specialization=specialization,
        query_heads_per_kv_head=query_heads_per_kv_head,
    )
    assert compiled.asm["cubin"]
    assert "mma.sync" not in compiled.asm["ptx"]
    assert stack_frame_bytes(compiled.asm["cubin"]) == 0
