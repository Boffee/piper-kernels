"""Offline coverage of dense RDNA4 signed QK and mixed-sign PV lowering."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton
from _accelerators import rocm_fences_only_lds
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels.attention.piper_attention._amd import gluon as amd_gluon
from piper_kernels.attention.piper_attention.triton import compile_kv_decode_kernels

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
        "query_heads_per_kv_head": 3,
        "head_dim": head_dim,
        "is_causal": causal,
        "wide_query_offsets": wide,
        "wide_context_offsets": wide,
        "lse_ptr": None,
        "key_lengths_ptr": None,
        "store_lse": False,
        "has_key_lengths": False,
        "offset_causal_from_key_length": False,
    }
    kernel = amd_gluon._dense_piper_kernel
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature["lse_ptr"] = "constexpr"
    signature["key_lengths_ptr"] = "constexpr"
    signature.update(
        dict.fromkeys(
            (
                "query_scale_ptr",
                "key_scale_ptr",
                "key_center_ptr",
                "multiplier_ptr",
                "log_scale_ptr",
                "value_mean_ptr",
            ),
            "*fp32",
        )
    )
    signature.update(
        query_ptr="*i8",
        key_ptr="*i8",
        value_ptr="*i8",
        output_ptr=f"*{dtype}",
    )
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


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
def test_dense_amd_compilation(head_dim, causal, dtype, wide):
    compiled = _compile_dense_kernel("gfx1201", head_dim, causal, dtype, wide)
    assert compiled.asm["hsaco"]
    instructions = [
        line.replace(" ", "")
        for line in compiled.asm["amdgcn"].splitlines()
        if "v_wmma_i32_16x16x16_iu8" in line
    ]
    assert any("0x3e22f983neg_lo:[1,1,0]" in line for line in instructions)
    assert any("0x3e22f983neg_lo:[1,0,0]" in line for line in instructions)
    assert all(",0neg_lo:[1,0,0]" not in line for line in instructions)
    assert all("clamp" not in line for line in instructions)
    assert compiled.asm["ttgir"].count("arith.truncf") == 1  # final output only
    if rocm_fences_only_lds():
        assert "global_inv" not in compiled.asm["amdgcn"]
    # Non-ROCm builds leave Triton 3.9's warp-pipeline load barriers in place.
    if torch.version.hip is not None or not amd_compiler_workarounds.UNPINNED_WARP_PIPELINE_ATTRS:
        assert "global_load_b64" not in compiled.asm["amdgcn"]
    assert compiled.asm["amdgcn"].count("v_div_scale_f32") == 2
    assert re.search(r"amdhsa_private_segment_fixed_size\s+0\b", compiled.asm["amdgcn"])


def test_dense_amd_gfx1200_smoke_compilation():
    """Keep lightweight Gluon gfx1200 target coverage without doubling the variant matrix."""
    compiled = _compile_dense_kernel("gfx1200", head_dim=64)
    assert compiled.asm["hsaco"]
    assert "v_wmma_i32_16x16x16_iu8" in compiled.asm["amdgcn"]


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
    assert "v_fma_f32 $0, $1, $2, $3" in llvm_ir  # block_max score scaling


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
    pinned = _compile_dense_kernel(llvm_fn_attrs=(("target-features", "-cumode"),))
    unpinned = _compile_dense_kernel()
    # The first check fails if Triton changes how it pins loads, which would
    # otherwise silently disable the workaround, or stops pinning them: then
    # retire the workaround as amd_compiler_workarounds describes.
    assert _PINNED_LOAD_BARRIER in pinned.asm["llir"]
    assert _PINNED_LOAD_BARRIER not in unpinned.asm["llir"]
    assert "sched_barrier mask(0x00000001)" not in unpinned.asm["amdgcn"]


def _assert_zero_spills_amdgcn(compiled) -> str:
    assert compiled.asm["hsaco"]
    assert compiled.metadata.global_scratch_size == 0
    amdgcn = compiled.asm["amdgcn"]
    if rocm_fences_only_lds():
        assert "global_inv" not in amdgcn
    vgpr_spills = re.search(r"vgpr_spill[^:]*:\s*(\d+)", amdgcn)
    sgpr_spills = re.search(r"sgpr_spill[^:]*:\s*(\d+)", amdgcn)
    priv_size = re.search(r"amdhsa_private_segment_fixed_size\s+(\d+)", amdgcn)
    assert vgpr_spills is not None
    assert int(vgpr_spills.group(1)) == 0
    assert sgpr_spills is not None
    assert int(sgpr_spills.group(1)) == 0
    assert priv_size is not None
    assert int(priv_size.group(1)) == 0
    return amdgcn


def _assert_zero_spill_amd_prefill(compiled) -> None:
    amdgcn = _assert_zero_spills_amdgcn(compiled)
    instructions = [
        line.replace(" ", "") for line in amdgcn.splitlines() if "v_wmma_i32_16x16x16_iu8" in line
    ]
    assert any("0x3e22f983neg_lo:[1,1,0]" in line for line in instructions)
    assert any("0x3e22f983neg_lo:[1,0,0]" in line for line in instructions)
    assert all(",0neg_lo:[1,0,0]" not in line for line in instructions)
    assert all("clamp" not in line for line in instructions)
    if torch.version.hip is not None or not amd_compiler_workarounds.UNPINNED_WARP_PIPELINE_ATTRS:
        assert "global_load_b64" not in amdgcn


def _assert_zero_spill_amd_decode(compiled, *, expected_wmma: str | None = None) -> None:
    amdgcn = _assert_zero_spills_amdgcn(compiled)
    if expected_wmma is not None:
        assert expected_wmma in amdgcn


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("has_chunk_map", "is_causal"), [(False, False), (True, True)])
@pytest.mark.parametrize("has_chunk_splits", [False, True])
@pytest.mark.parametrize("per_row_query_scale", [False, True])
def test_amd_prefill_compilation_zero_spills(
    head_dim: int,
    dtype: torch.dtype,
    has_chunk_map: bool,
    is_causal: bool,
    has_chunk_splits: bool,
    per_row_query_scale: bool,
) -> None:
    frozen, tail = amd_gluon.compile_amd_prefill_kernels(
        GPUTarget("hip", "gfx1201", 32),
        dtype=dtype,
        head_dim=head_dim,
        has_chunk_map=has_chunk_map,
        is_causal=is_causal,
        has_chunk_splits=has_chunk_splits,
        per_row_query_scale=per_row_query_scale,
    )
    _assert_zero_spill_amd_prefill(frozen)
    _assert_zero_spill_amd_prefill(tail)


def test_amd_prefill_gfx1200_smoke() -> None:
    for wide_pool in (False, True):
        for has_chunk_splits in (False, True):
            frozen, tail = amd_gluon.compile_amd_prefill_kernels(
                GPUTarget("hip", "gfx1200", 32),
                dtype=torch.bfloat16,
                head_dim=128,
                has_chunk_map=True,
                is_causal=True,
                wide_pool=wide_pool,
                has_chunk_splits=has_chunk_splits,
            )
            _assert_zero_spill_amd_prefill(frozen)
            _assert_zero_spill_amd_prefill(tail)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("wide_pool", [False, True])
@pytest.mark.parametrize("contiguous_inputs", [True, False])
def test_amd_kv_decode_compilation_zero_spills(
    head_dim: int,
    dtype: torch.dtype,
    wide_pool: bool,
    contiguous_inputs: bool,
) -> None:
    frozen, tail, reduce_k = compile_kv_decode_kernels(
        GPUTarget("hip", "gfx1201", 32),
        dtype=dtype,
        head_dim=head_dim,
        wide_pool=wide_pool,
        contiguous_inputs=contiguous_inputs,
    )
    tail_wmma = "v_wmma_f32_16x16x16_f16" if dtype is torch.float16 else "v_wmma_f32_16x16x16_bf16"
    _assert_zero_spill_amd_prefill(frozen)
    _assert_zero_spill_amd_decode(tail, expected_wmma=tail_wmma)
    _assert_zero_spill_amd_decode(reduce_k)


def test_amd_kv_decode_gfx1200_smoke() -> None:
    for head_dim in (64, 128):
        for wide_pool in (False, True):
            for reduce_vectorize_splits in (False, True):
                frozen, tail, reduce_k = compile_kv_decode_kernels(
                    GPUTarget("hip", "gfx1200", 32),
                    dtype=torch.bfloat16,
                    head_dim=head_dim,
                    wide_pool=wide_pool,
                    reduce_vectorize_splits=reduce_vectorize_splits,
                )
                _assert_zero_spill_amd_prefill(frozen)
                _assert_zero_spill_amd_decode(tail, expected_wmma="v_wmma_f32_16x16x16_bf16")
                _assert_zero_spill_amd_decode(reduce_k)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("contiguous_inputs", [True, False])
def test_amd_kv_decode_vectorized_reduce_zero_spills(
    head_dim: int,
    dtype: torch.dtype,
    contiguous_inputs: bool,
) -> None:
    _frozen, _tail, reduce_k = compile_kv_decode_kernels(
        GPUTarget("hip", "gfx1201", 32),
        dtype=dtype,
        head_dim=head_dim,
        contiguous_inputs=contiguous_inputs,
        reduce_vectorize_splits=True,
    )
    _assert_zero_spill_amd_decode(reduce_k)
