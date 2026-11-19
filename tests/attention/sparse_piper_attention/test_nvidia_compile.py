"""Offline compilation and register-pressure checks for NVIDIA sparse attention."""

import sys

import pytest
import triton
from _nvidia_resources import (
    ATTENTION_STACK_BUDGET_BYTES,
    compile_with_spill_stores,
    stack_frame_bytes,
)
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton.mixed_int8 import _MixedInt8StageHook
from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS
from piper_kernels.attention.sparse_piper_attention._nvidia import (
    gluon_async_copy,
    gluon_tma,
    policy,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="requires Linux NVIDIA compiler tooling"
)

# SM120 has 101,376 B of shared memory per SM; keeping <= 50,688 B allows 2 CTAs/SM.
_SM120_TWO_CTA_SHARED_MEMORY_LIMIT = 50_688


def _tensor_descriptor_type(rows: int, cols: int) -> str:
    layout = gl.NVMMASharedLayout.get_default_for([rows, cols], gl.int8)
    return f"tensordesc<i8[{rows},{cols}],{layout!r}>"


def _common_pointer_signature() -> dict[str, str]:
    return {
        "query_scale_ptr": "*fp32",
        "key_scale_ptr": "*fp32",
        "key_center_ptr": "*i8",
        "key_center_scale_ptr": "*fp32",
        "value_scale_multiplier_ptr": "*fp32",
        "value_mean_ptr": "*bf16",
        "coarse_output_ptr": "*fp32",
        "coarse_gate_ptr": "*bf16",
        "block_lengths_ptr": "*i32",
        "routes_ptr": "*u16",
        "head_keep_blocks_ptr": "*i32",
        "route_head_offsets_ptr": "*i32",
        "output_ptr": "*bf16",
    }


def _aligned_attrs(kernel, signature: dict[str, str]) -> dict[tuple[int], list[list[object]]]:
    aligned = [
        name
        for name, dtype in signature.items()
        if dtype.startswith("*")
        or name in ("query_storage_sequence_length", "storage_sequence_length")
    ]
    return {(kernel.arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned}


@pytest.mark.parametrize(
    (
        "mask_block_lengths",
        "mask_ragged_tail",
        "has_dense_query_suffix",
        "apply_coarse_residual",
        "mask_output_tail",
        "query_heads_per_kv_head",
        "max_spill_bytes",
    ),
    [
        (False, False, False, False, False, 1, 24),
        (False, False, False, False, False, 4, 24),
        (False, True, True, True, True, 1, 0),
        (True, False, True, True, False, 1, 0),
    ],
)
def test_sm120_d128_recurrence_keeps_two_cta_smem_and_low_spills(
    monkeypatch,
    mask_block_lengths,
    mask_ragged_tail,
    has_dense_query_suffix,
    apply_coarse_residual,
    mask_output_tail,
    query_heads_per_kv_head,
    max_spill_bytes,
):
    kernel = gluon_tma._sparse_piper_attention_kernel
    constants = {
        "query_heads_per_kv_head": query_heads_per_kv_head,
        "head_dim": 128,
        "mask_block_lengths": mask_block_lengths,
        "mask_ragged_tail": mask_ragged_tail,
        "has_dense_query_suffix": has_dense_query_suffix,
        "apply_coarse_residual": apply_coarse_residual,
        "block_m": TILE_ROWS,
        "mma_warps": 4,
        "skip_dense_routing": False,
        "mask_output_tail": mask_output_tail,
    }
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        query_desc=_tensor_descriptor_type(TILE_ROWS, 128),
        key_desc=_tensor_descriptor_type(TILE_ROWS, 128),
        value_desc=_tensor_descriptor_type(128, TILE_ROWS),
        **_common_pointer_signature(),
    )
    source = GluonASTSource(
        kernel, signature, constexprs=constants, attrs=_aligned_attrs(kernel, signature)
    )
    monkeypatch.setattr(
        triton.knobs.runtime, "add_stages_inspection_hook", _MixedInt8StageHook(None)
    )
    compiled, spill_store_bytes = compile_with_spill_stores(
        source, GPUTarget("cuda", 120, 32), {"num_warps": 4, "num_stages": 1}
    )
    assert compiled.asm["cubin"]
    assert "cp.async.bulk.tensor" in compiled.asm["ptx"]
    assert ".s32.s8.s8.s32" in compiled.asm["ptx"]
    assert ".s32.u8.s8.s32" in compiled.asm["ptx"]
    assert "piper_attention_u8s8_dot_marker" not in compiled.asm["ptx"]
    # Confirm D128 PV uses rescale_packed's zero-rescale skip instead of unconditional FMA.
    assert "PIPER_RESCALE_DONE" in compiled.asm["ptx"]
    assert compiled.metadata.shared <= _SM120_TWO_CTA_SHARED_MEMORY_LIMIT
    assert spill_store_bytes <= max_spill_bytes
    assert stack_frame_bytes(compiled.asm["cubin"]) <= max_spill_bytes


@pytest.mark.parametrize(
    ("block_m", "mma_warps", "skip_dense_routing"),
    [(64, 4, False), (64, 2, False), (128, 4, True)],
)
def test_sm120_d64_schedules_compile_within_budgets(
    monkeypatch, block_m, mma_warps, skip_dense_routing
):
    kernel = gluon_tma._sparse_piper_attention_kernel
    constants = {
        "query_heads_per_kv_head": 1,
        "head_dim": 64,
        "mask_block_lengths": False,
        "mask_ragged_tail": False,
        "has_dense_query_suffix": False,
        "apply_coarse_residual": False,
        "block_m": block_m,
        "mma_warps": mma_warps,
        "skip_dense_routing": skip_dense_routing,
        "mask_output_tail": False,
    }
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        query_desc=_tensor_descriptor_type(block_m, 64),
        key_desc=_tensor_descriptor_type(TILE_ROWS, 64),
        value_desc=_tensor_descriptor_type(64, TILE_ROWS),
        **_common_pointer_signature(),
    )
    source = GluonASTSource(
        kernel, signature, constexprs=constants, attrs=_aligned_attrs(kernel, signature)
    )
    monkeypatch.setattr(
        triton.knobs.runtime, "add_stages_inspection_hook", _MixedInt8StageHook(None)
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", 120, 32),
        options={"num_warps": mma_warps, "num_stages": 1},
    )
    assert compiled.asm["cubin"]
    assert "cp.async.bulk.tensor" in compiled.asm["ptx"]
    assert ".s32.s8.s8.s32" in compiled.asm["ptx"]
    assert ".s32.u8.s8.s32" in compiled.asm["ptx"]
    assert "piper_attention_u8s8_dot_marker" not in compiled.asm["ptx"]
    assert "PIPER_RESCALE_DONE" in compiled.asm["ptx"]
    assert compiled.metadata.shared <= _SM120_TWO_CTA_SHARED_MEMORY_LIMIT
    assert stack_frame_bytes(compiled.asm["cubin"]) <= ATTENTION_STACK_BUDGET_BYTES


@pytest.mark.parametrize("architecture", [80, 89])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sm8x_async_copy_kernels_compile_within_stack_budget(monkeypatch, architecture, head_dim):
    kernel = gluon_async_copy._sparse_piper_attention_kernel
    constants = {
        "query_heads_per_kv_head": 3,
        "head_dim": head_dim,
        "mask_block_lengths": False,
        "mask_ragged_tail": False,
        "has_dense_query_suffix": True,
        "apply_coarse_residual": True,
        "skip_dense_routing": False,
        "mask_output_tail": True,
    }
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        query_ptr="*i8",
        key_ptr="*i8",
        value_ptr="*i8",
        **_common_pointer_signature(),
    )
    source = GluonASTSource(
        kernel, signature, constexprs=constants, attrs=_aligned_attrs(kernel, signature)
    )
    max_registers = policy.select_sm8x_max_registers(head_dim)
    options: dict[str, object] = {"num_warps": 4, "num_stages": 1}
    if max_registers is not None:
        options["maxnreg"] = max_registers
    monkeypatch.setattr(
        triton.knobs.runtime, "add_stages_inspection_hook", _MixedInt8StageHook(None)
    )
    compiled = triton.compile(source, target=GPUTarget("cuda", architecture, 32), options=options)
    assert compiled.asm["cubin"]
    assert "cp.async" in compiled.asm["ptx"]
    assert ".s32.s8.s8.s32" in compiled.asm["ptx"]
    assert ".s32.u8.s8.s32" in compiled.asm["ptx"]
    assert "piper_attention_u8s8_dot_marker" not in compiled.asm["ptx"]
    assert stack_frame_bytes(compiled.asm["cubin"]) <= ATTENTION_STACK_BUDGET_BYTES
