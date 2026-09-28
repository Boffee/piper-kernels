"""Offline smoke coverage after separating NVIDIA dense Piper execution."""

import sys
from dataclasses import replace

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from piper_kernels._triton.mixed_int8 import _MixedInt8StageHook
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._nvidia.gluon_async_copy import (
    _dense_piper_attention_kernel,
)
from piper_kernels.attention.piper_attention._nvidia.policy import select_execution_plan
from piper_kernels.attention.piper_attention._nvidia.triton import (
    _piper_attention_kernel,
    _quantize_value_per_key_kernel,
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
        head_groups=3,
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
        "head_groups": 3,
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


@pytest.mark.parametrize("architecture", [89, 120])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_nvidia_value_preparation_accepts_runtime_head_count(architecture, head_dim, causal):
    constants = {
        "is_causal": causal,
        "store_log_scale": True,
        "head_dim": head_dim,
        "block_n": 64,
    }
    signature = {
        name: "i32" for name in _quantize_value_per_key_kernel.arg_names if name not in constants
    }
    signature.update(
        value_ptr="*fp16",
        value_mean_ptr="*fp32",
        scale_multiplier_ptr="*fp32",
        log_scale_ptr="*fp16",
        output_ptr="*i8",
    )
    compiled = triton.compile(
        ASTSource(_quantize_value_per_key_kernel, signature, constexprs=constants),
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": 4},
    )
    assert compiled.asm["cubin"]
