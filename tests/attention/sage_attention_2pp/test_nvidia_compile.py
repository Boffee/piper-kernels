"""Offline NVIDIA compilation of production SageAttention2++ schedules."""

import sys

import pytest
import triton
from _nvidia_resources import ATTENTION_STACK_BUDGET_BYTES, stack_frame_bytes
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage_attention_2pp._policy import select_execution_plan
from piper_kernels.attention.sage_attention_2pp.triton import _BLOCK_N, _sage_attention_2pp_kernel

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="offline compiler coverage on Linux"
)


@pytest.mark.parametrize("architecture", [89, 120])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_production_schedule_stays_within_stack_budget(architecture, head_dim, causal):
    kernel = _sage_attention_2pp_kernel
    block_n = int(_BLOCK_N)
    plan = select_execution_plan(
        AcceleratorTarget("cuda", f"sm{architecture}"),
        candidate_block_m=128,
        head_dim=head_dim,
        is_causal=causal,
    )
    # Launch options are plan fields but not kernel parameters.
    constants = {name: value for name, value in plan.as_dict().items() if name in kernel.arg_names}
    constants.update(is_causal=causal, heads=32, head_dim=head_dim, block_n=block_n)
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        query_ptr="*i8",
        key_ptr="*i8",
        value_ptr="*fp8e4nv",
        query_scale_ptr="*fp32",
        key_scale_ptr="*fp32",
        value_scale_ptr="*fp32",
        output_ptr="*bf16",
    )
    if plan.use_tensor_descriptors:
        signature.update(
            key_ptr=f"tensordesc<i8[1, {block_n}, {head_dim}]>",
            value_ptr=f"tensordesc<fp8e4nv[1, {head_dim}, {block_n}]>",
        )
    # Launches on 16-element-aligned sequences specialize every pointer, length,
    # and stride this way.
    attrs = {
        (kernel.arg_names.index(name),): [["tt.divisibility", 16]]
        for name, kind in signature.items()
        if not kind.startswith("tensordesc")
    }
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs=constants, attrs=attrs),
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": plan.num_warps, "num_stages": plan.num_stages},
    )

    assert stack_frame_bytes(compiled.asm["cubin"]) <= ATTENTION_STACK_BUDGET_BYTES
