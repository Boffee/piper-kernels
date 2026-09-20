"""Offline smoke coverage after separating NVIDIA dense Piper execution."""

import sys
from dataclasses import replace

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.mixed_int8 import _MixedInt8StageHook
from piper_kernels._triton.targets import AcceleratorTarget
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
def test_nvidia_pointer_kernel_retains_signed_and_mixed_mma(
    monkeypatch, architecture, head_dim, causal
):
    plan = replace(
        select_execution_plan(
            AcceleratorTarget("cuda", f"sm{architecture}"), head_dim=head_dim, is_causal=causal
        ),
        use_tensor_descriptors=False,
    )
    constants = plan.as_dict()
    constants.pop("num_warps")
    constants.pop("num_stages")
    constants.update(
        head_groups=3,
        head_dim=head_dim,
        is_causal=causal,
        block_n=64,
        unmasked_query_tiles=True,
        unmasked_key_tiles=not causal,
        use_query_tensor_descriptor=False,
    )
    signature = {name: "i32" for name in _piper_attention_kernel.arg_names if name not in constants}
    signature.update(
        query_ptr="*i8",
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
