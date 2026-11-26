"""Offline smoke coverage after separating NVIDIA dense Piper execution."""

import re
import sys
from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
import triton
from _nvidia_resources import (
    ATTENTION_STACK_BUDGET_BYTES,
    compile_with_spill_stores,
    stack_frame_bytes,
)
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton.mixed_int8 import _MixedInt8StageHook
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _quantization
from piper_kernels.attention.piper_attention._nvidia.gluon_async_copy import (
    _dense_piper_attention_kernel,
)
from piper_kernels.attention.piper_attention._nvidia.policy import select_execution_plan
from piper_kernels.attention.piper_attention._nvidia.triton import _piper_attention_kernel
from piper_kernels.attention.piper_attention.triton import (
    compile_kv_decode_kernels,
    compile_kv_prefill_kernels,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="offline compiler coverage on Linux"
)


@pytest.mark.parametrize("architecture", [89, 120])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("aligned_queries", [False, True])
def test_nvidia_pointer_kernel_retains_signed_and_mixed_mma(
    monkeypatch, architecture, head_dim, causal, aligned_queries
):
    plan = replace(
        select_execution_plan(
            AcceleratorTarget("cuda", f"sm{architecture}"),
            head_dim=head_dim,
            is_causal=causal,
            query_length=128 if aligned_queries else 129,
        ),
        use_tensor_descriptors=False,
    )
    # Launch options and plan fields for other kernels or launch paths are not
    # Triton kernel parameters.
    constants = {
        name: value
        for name, value in plan.as_dict().items()
        if name in _piper_attention_kernel.arg_names
    }
    constants.update(
        query_heads_per_kv_head=3,
        head_dim=head_dim,
        is_causal=causal,
        block_n=64,
        aligned_queries=aligned_queries,
        unmasked_key_tiles=not causal,
        use_query_tensor_descriptor=False,
        full_query=False,
        contiguous_output=False,
        query_group_size=8 if architecture == 120 and head_dim == 64 and causal else 0,
    )
    signature = {name: "i32" for name in _piper_attention_kernel.arg_names if name not in constants}
    signature.update(
        query_ptr="*i8",
        query_descriptor="*i8",
        key_ptr="*i8",
        value_ptr="*i8",
        query_scale_ptr="*fp32",
        key_scale_ptr="*fp32",
        key_center_ptr="*fp32",
        value_scale_multiplier_ptr="*fp32",
        value_log_scale_ptr="*fp16",
        value_mean_ptr="*fp32",
        output_ptr="*fp16",
    )
    # Install only the compiler-stage rewrite; no device/driver probe is needed.
    monkeypatch.setattr(
        triton.knobs.runtime, "add_stages_inspection_hook", _MixedInt8StageHook(None)
    )
    compiled = triton.compile(
        ASTSource(_piper_attention_kernel, signature, constexprs=constants),
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": plan.num_warps, "num_stages": plan.num_stages},
    )
    assert compiled.asm["cubin"]
    assert ".s32.s8.s8.s32" in compiled.asm["ptx"]
    assert ".s32.u8.s8.s32" in compiled.asm["ptx"]
    assert "piper_attention_u8s8_dot_marker" not in compiled.asm["ptx"]


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("full_launch", [False, True])
@pytest.mark.parametrize("quantized_query", [False, True])
def test_sm89_gluon_kernel_retains_signed_and_mixed_mma(
    monkeypatch, head_dim, causal, full_launch, quantized_query
):
    plan = select_execution_plan(
        AcceleratorTarget("cuda", "sm89"),
        head_dim=head_dim,
        is_causal=causal,
        query_length=129,
        quantized_query=quantized_query,
    )
    assert plan.attention_kernel == "gluon_async_copy"
    constants = {
        "query_heads_per_kv_head": 3,
        "head_dim": head_dim,
        "is_causal": causal,
        "block_q": plan.block_m,
        "quantize_query": plan.fuse_query_quantization,
        "full_query": full_launch,
        "contiguous_output": full_launch,
        # Quantized producers supply K64-padded K/V metadata.
        "padded_kv": quantized_query,
    }
    if plan.fuse_query_quantization:
        constants["query_scale_ptr"] = None
    signature = {
        name: "i32" for name in _dense_piper_attention_kernel.arg_names if name not in constants
    }
    signature.update(
        query_ptr="*bf16" if plan.fuse_query_quantization else "*i8",
        key_ptr="*i8",
        value_ptr="*i8",
        query_scale_ptr="constexpr" if plan.fuse_query_quantization else "*fp32",
        key_scale_ptr="*fp32",
        key_center_ptr="*fp32",
        multiplier_ptr="*fp32",
        value_mean_ptr="*fp32",
        output_ptr="*bf16",
        softmax_scale="fp32",
    )
    # A launch specializes pointers, K64/Q64-padded storage lengths, and Q strides
    # as 16-byte aligned, which every cp.async copy requires. Window coordinates and
    # output strides stay unspecialized.
    aligned = [
        name
        for name in signature
        if name not in constants
        and (name.endswith(("_ptr", "_storage")) or name.startswith("stride_q"))
    ]
    attrs = {
        (_dense_piper_attention_kernel.arg_names.index(name),): [["tt.divisibility", 16]]
        for name in aligned
    }
    options = {"num_warps": plan.num_warps, "num_stages": plan.num_stages}
    if plan.max_registers is not None:
        options["maxnreg"] = plan.max_registers
    monkeypatch.setattr(
        triton.knobs.runtime, "add_stages_inspection_hook", _MixedInt8StageHook(None)
    )
    compiled = triton.compile(
        GluonASTSource(_dense_piper_attention_kernel, signature, constexprs=constants, attrs=attrs),
        target=GPUTarget("cuda", 89, 32),
        options=options,
    )
    assert compiled.asm["cubin"]
    assert ".s32.s8.s8.s32" in compiled.asm["ptx"]
    assert ".s32.u8.s8.s32" in compiled.asm["ptx"]
    assert "piper_attention_u8s8_dot_marker" not in compiled.asm["ptx"]
    # This is the exact SM89 production kernel, unlike the pointer variants above.
    assert stack_frame_bytes(compiled.asm["cubin"]) <= ATTENTION_STACK_BUDGET_BYTES


@pytest.mark.parametrize("architecture", [89, 120])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_nvidia_value_preparation_accepts_runtime_head_count(architecture, head_dim):
    constants = {
        "store_log_scale": True,
        "scale_by_p_range": True,
        "pack_wmma": False,
        "tile_layout": False,
        "scatter_slots": False,
        "head_dim": head_dim,
        "has_key_lengths": False,
    }
    kernel = _quantization._quantize_value_k1024_kernel
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        value_ptr="*fp16",
        output_ptr="*i8",
        scale_multiplier_ptr="*fp32",
        log_scale_ptr="*fp16",
        value_mean_ptr="*fp32",
        slot_ids_ptr="*i8",
    )
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs=constants),
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": 4},
    )
    assert compiled.asm["cubin"]


_KV_COMPILE_TARGETS = (
    GPUTarget("cuda", 75, 32),
    GPUTarget("cuda", 80, 32),
    GPUTarget("cuda", 89, 32),
    GPUTarget("cuda", 120, 32),
    GPUTarget("hip", "gfx942", 64),
    GPUTarget("hip", "gfx1100", 32),
    GPUTarget("hip", "gfx1151", 32),
    GPUTarget("hip", "gfx1201", 32),
)


def _assert_zero_spill_kernels(
    target: GPUTarget,
    expected_count: int,
    compile_call: Callable[
        [Callable[[object, object, dict[str, object]], object] | None], tuple[object, ...]
    ],
    *,
    check_sgpr_spills: bool = True,
) -> None:
    if target.backend == "cuda":
        spill_counts: list[int] = []

        def _compile_cuda(source: object, tgt: object, options: dict[str, object]) -> object:
            compiled, spills = compile_with_spill_stores(source, tgt, options)  # type: ignore[arg-type]
            assert compiled.asm.get("cubin")
            spill_counts.append(spills)
            return compiled

        compile_call(_compile_cuda)
        assert spill_counts == [0] * expected_count
        return

    compiled_kernels = compile_call(None)
    assert len(compiled_kernels) == expected_count
    for compiled in compiled_kernels:
        assert compiled.asm.get("hsaco")  # type: ignore[attr-defined]
        assert compiled.metadata.global_scratch_size == 0  # type: ignore[attr-defined]
        amdgcn = compiled.asm.get("amdgcn", "")  # type: ignore[attr-defined]
        assert amdgcn
        vgpr_spills = re.search(r"vgpr_spill[^:]*:\s*(\d+)", amdgcn)
        sgpr_spills = re.search(r"sgpr_spill[^:]*:\s*(\d+)", amdgcn)
        priv_size = re.search(r"\.private_segment_fixed_size:\s*(\d+)", amdgcn)
        assert vgpr_spills is not None or priv_size is not None
        if vgpr_spills is not None:
            assert int(vgpr_spills.group(1)) == 0
        if check_sgpr_spills:
            assert sgpr_spills is not None
            assert int(sgpr_spills.group(1)) == 0
        if priv_size is not None:
            assert int(priv_size.group(1)) == 0


@pytest.mark.parametrize("target", _KV_COMPILE_TARGETS, ids=lambda t: f"{t.backend}-{t.arch}")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_kv_freeze_8_target_compile_zero_spills(
    target: GPUTarget,
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    # Pre-existing _quantize_value_k1024_kernel spills SGPRs into VGPRs on HIP
    # (gfx942/1100/1151 at D=128, gfx1201 at D=64/128) with zero VGPR or scratch spills.
    _assert_zero_spill_kernels(
        target,
        2,
        lambda compile_fn: _quantization.compile_kv_freeze_kernels(
            target, dtype=dtype, head_dim=head_dim, compile_fn=compile_fn
        ),
        check_sgpr_spills=False,
    )


@pytest.mark.parametrize("target", _KV_COMPILE_TARGETS, ids=lambda t: f"{t.backend}-{t.arch}")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    ("has_chunk_map", "is_causal", "contiguous_inputs"),
    [(True, True, True), (True, True, False), (False, False, True)],
)
def test_kv_decode_8_target_compile_zero_spills(
    target: GPUTarget,
    dtype: torch.dtype,
    head_dim: int,
    has_chunk_map: bool,
    is_causal: bool,
    contiguous_inputs: bool,
) -> None:
    _assert_zero_spill_kernels(
        target,
        3,
        lambda compile_fn: compile_kv_decode_kernels(
            target,
            dtype=dtype,
            head_dim=head_dim,
            has_chunk_map=has_chunk_map,
            is_causal=is_causal,
            contiguous_inputs=contiguous_inputs,
            compile_fn=compile_fn,
        ),
    )


_FALLBACK_PREFILL_TARGETS = (
    GPUTarget("cuda", 75, 32),
    GPUTarget("hip", "gfx942", 64),
    GPUTarget("hip", "gfx1100", 32),
    GPUTarget("hip", "gfx1151", 32),
)
_NVIDIA_PREFILL_TARGETS = (
    GPUTarget("cuda", 80, 32),
    GPUTarget("cuda", 86, 32),
    GPUTarget("cuda", 89, 32),
    GPUTarget("cuda", 120, 32),
)
_KV_PREFILL_TARGETS = (*_NVIDIA_PREFILL_TARGETS, GPUTarget("hip", "gfx1201", 32))


@pytest.mark.parametrize("target", _FALLBACK_PREFILL_TARGETS, ids=lambda t: f"{t.backend}-{t.arch}")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_kv_prefill_fallback_targets_merge_zero_spills(
    target: GPUTarget,
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    _assert_zero_spill_kernels(
        target,
        1,
        lambda compile_fn: compile_kv_prefill_kernels(
            target,
            dtype=dtype,
            head_dim=head_dim,
            compile_fn=compile_fn,
        ),
        check_sgpr_spills=True,
    )


@pytest.mark.parametrize("target", _NVIDIA_PREFILL_TARGETS, ids=lambda t: f"{t.backend}-{t.arch}")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(("has_chunk_map", "is_causal"), [(False, False), (True, True)])
def test_kv_prefill_native_cuda_compile_zero_spills(
    monkeypatch: pytest.MonkeyPatch,
    target: GPUTarget,
    dtype: torch.dtype,
    head_dim: int,
    has_chunk_map: bool,
    is_causal: bool,
) -> None:
    monkeypatch.setattr(
        triton.knobs.runtime, "add_stages_inspection_hook", _MixedInt8StageHook(None)
    )
    compiled_out: list[object] = []

    def _compile_and_record(
        compile_fn: Callable[[object, object, dict[str, object]], object] | None,
    ) -> tuple[object, ...]:
        res = compile_kv_prefill_kernels(
            target,
            dtype=dtype,
            head_dim=head_dim,
            has_chunk_map=has_chunk_map,
            is_causal=is_causal,
            compile_fn=compile_fn,
        )
        compiled_out.extend(res)
        return res

    _assert_zero_spill_kernels(
        target,
        3,
        _compile_and_record,
        check_sgpr_spills=True,
    )
    for compiled in compiled_out[1:]:
        ptx = compiled.asm["ptx"]  # type: ignore[attr-defined]
        assert ".s32.s8.s8.s32" in ptx
        assert ".s32.u8.s8.s32" in ptx
        assert "piper_attention_u8s8_dot_marker" not in ptx


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_kv_prefill_amd_dispatch_compile_smoke(
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    _assert_zero_spill_kernels(
        GPUTarget("hip", "gfx1201", 32),
        3,
        lambda compile_fn: compile_kv_prefill_kernels(
            GPUTarget("hip", "gfx1201", 32),
            dtype=dtype,
            head_dim=head_dim,
            compile_fn=compile_fn,
        ),
        check_sgpr_spills=True,
    )


@pytest.mark.parametrize("target", _KV_PREFILL_TARGETS, ids=lambda t: f"{t.backend}-{t.arch}")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_kv_tail_quantize_prefill_compile_zero_spills(
    target: GPUTarget,
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    _assert_zero_spill_kernels(
        target,
        2,
        lambda compile_fn: _quantization.compile_kv_tail_quantize_kernels(
            target, dtype=dtype, head_dim=head_dim, compile_fn=compile_fn
        ),
    )
