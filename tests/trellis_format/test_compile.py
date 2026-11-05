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


_NVIDIA_TARGETS = [GPUTarget("cuda", architecture, 32) for architecture in (70, 80, 120)]


@pytest.mark.parametrize("target", _NVIDIA_TARGETS, ids=lambda target: f"cuda-{target.arch}")
def test_encoder_step_loop_kernels_compile(target: GPUTarget) -> None:
    from piper_kernels.trellis._encoder import _squared_error_kernel  # noqa: PLC0415
    from piper_kernels.trellis._viterbi import _step_kernel, _traceback_kernel  # noqa: PLC0415

    options = {"num_warps": 4, "enable_fp_fusion": False}
    sources = [
        ASTSource(
            _step_kernel,
            {
                **dict.fromkeys(
                    ("m_ptr", "c_ptr", "w_ptr", "p1_ptr", "p2_ptr", "out_ptr"), "*fp32"
                ),
                "bp_ptr": "*u8",
                "rows": "i32",
                "mod": "i32",
            },
            constexprs={
                "int8": True,
                "state_bits": 16,
                "e_bits": 4,
                "a_bits": 5,
                "ap_bits": 4,
                "bp_width": 4096,
                "block": 4,
                "positions": 128,
            },
        ),
        ASTSource(
            _traceback_kernel,
            {
                "bp_ptr": "*u8",
                "pick_ptr": "*u8",
                "j_ptr": "*i64",
                "widths_ptr": "*i64",
                "states_ptr": "*i64",
                "chosen_ptr": "*i64",
                "batch": "i32",
                "rows": "i32",
            },
            constexprs={
                "length": 256,
                "picked": True,
                "state_bits": 16,
                "width": 4096,
                "block": 64,
            },
        ),
        ASTSource(
            _squared_error_kernel,
            {
                **dict.fromkeys(("w_ptr", "entries_ptr", "factor_ptr", "p2_ptr"), "*fp32"),
                "out_ptr": "*fp64",
                "total": "i32",
                "length": "i32",
            },
            constexprs={"int8": True, "double": True, "block": 1024},
        ),
    ]
    for source in sources:
        assert triton.compile(source, target=target, options=options).asm["ptx"]


@pytest.mark.parametrize("architecture", [80, 120])
@pytest.mark.parametrize("step_widths", [(3, 4), (4, 5), (5, 6), (6,)], ids=str)
def test_encoder_persistent_forward_compiles(
    architecture: int, step_widths: tuple[int, ...]
) -> None:
    from triton.experimental.gluon._runtime import GluonASTSource  # noqa: PLC0415

    from piper_kernels.trellis import _gluon_forward  # noqa: PLC0415

    widths = step_widths * (256 // len(step_widths))
    arguments = _gluon_forward._compile_args(*_gluon_forward._specialization(16, widths, int8=True))
    num_warps = arguments.pop("num_warps")
    arguments.pop("enable_fp_fusion")
    signature = {
        **dict.fromkeys(("w_ptr", "p1_ptr", "p2_ptr"), "*fp32"),
        "start_ptr": "*i64",
        "order_ptr": "*i32",
        "kinds_ptr": "*i32",
        "m_out_ptr": "*fp32",
        "bp_ptr": "*u8",
        **dict.fromkeys(("rows", "mod", "length", "outputs", "k0", "k1", "constrained"), "i32"),
    }
    compiled = triton.compile(
        GluonASTSource(_gluon_forward._forward_kernel, signature, constexprs=arguments),
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": num_warps, "enable_fp_fusion": False},
    )
    assert compiled.asm["ptx"]
    assert compiled.metadata.shared <= _gluon_forward.SHARED_NEED
