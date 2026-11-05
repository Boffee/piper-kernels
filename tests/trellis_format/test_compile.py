"""The trellis decode kernel compiles for every accelerator family that loads weights."""

import sys

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels.trellis.triton import decode_int8_kernel

_HIP = pytest.mark.skipif(sys.platform != "linux", reason="HIP compiler is Linux-only")


@pytest.mark.parametrize(
    "target",
    [
        *[GPUTarget("cuda", architecture, 32) for architecture in (70, 80, 120)],
        pytest.param(GPUTarget("hip", "gfx942", 64), marks=_HIP),
        pytest.param(GPUTarget("hip", "gfx1201", 32), marks=_HIP),
    ],
    ids=lambda target: f"{target.backend}-{target.arch}",
)
def test_decode_kernel_compiles(target: GPUTarget) -> None:
    compiled = triton.compile(
        ASTSource(
            decode_int8_kernel,
            {
                "bitstreams": "*u8",
                "block_scales": "*u8",
                "qdata": "*i8",
                "num_blocks": "i32",
                "block_size": "i32",
                "block_bits": "i32",
                "step_bits": "i32",
                "num_wide_steps": "i32",
                "bytes_per_block": "i32",
            },
            constexprs={"tile_steps": 256, "tile_blocks": 8},
        ),
        target=target,
        options={"num_warps": 4},
    )
    assert compiled.asm["hsaco" if target.backend == "hip" else "ptx"]
