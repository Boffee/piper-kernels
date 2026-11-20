"""Compile tiled radix route selection offline across NVIDIA and AMD targets."""

import re
import sys

import pytest
import triton
from _nvidia_resources import stack_frame_bytes
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels.attention.sparse_piper_attention._routes_triton import (
    _SELECTOR_TILE,
    _SINGLE_LOAD_BLOCK_SIZES,
    _single_load_radix_select_packed_routes_kernel,
    _tiled_radix_select_packed_routes_kernel,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux compiler tooling")


def _compile_route_kernel(kernel, target: GPUTarget, constants: dict[str, object]) -> None:
    signature = {name: "i32" for name in kernel.arg_names if name not in constants}
    signature.update(
        scores_ptr="*fp32",
        routes_ptr="*u16",
        head_keep_blocks_ptr="*i32",
        route_head_offsets_ptr="*i32",
        force_mask_ptr="*i32",
    )
    compiled = triton.compile(
        ASTSource(kernel, signature, constexprs=constants),
        target=target,
        options={"num_warps": 4, "num_stages": 1},
    )
    if target.backend == "cuda":
        assert compiled.asm["cubin"]
        assert stack_frame_bytes(compiled.asm["cubin"]) == 0
    else:
        assert compiled.asm["hsaco"]
        scratch = re.search(
            r"\.amdhsa_private_segment_fixed_size\s+(\d+)\b", compiled.asm["amdgcn"]
        )
        assert scratch is not None
        assert int(scratch.group(1)) == 0


@pytest.mark.parametrize("has_force", [False, True])
@pytest.mark.parametrize(
    "target",
    [
        GPUTarget("cuda", 120, 32),
        GPUTarget("cuda", 89, 32),
        GPUTarget("hip", "gfx1201", 32),
    ],
)
def test_radix_route_selection_compiles_for_supported_targets(target, has_force):
    _compile_route_kernel(
        _tiled_radix_select_packed_routes_kernel,
        target,
        {"selector_tile": _SELECTOR_TILE, "has_force": has_force},
    )


@pytest.mark.parametrize("block_k", _SINGLE_LOAD_BLOCK_SIZES)
@pytest.mark.parametrize("has_force", [False, True])
@pytest.mark.parametrize(
    "target",
    [
        GPUTarget("cuda", 120, 32),
        GPUTarget("cuda", 89, 32),
        GPUTarget("hip", "gfx1201", 32),
    ],
)
def test_single_load_radix_route_selection_compiles_for_supported_targets(
    target, block_k, has_force
):
    _compile_route_kernel(
        _single_load_radix_select_packed_routes_kernel,
        target,
        {"block_k": block_k, "has_force": has_force},
    )
