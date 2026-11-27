"""Compile coverage for trellis decode and encode kernels across target families."""

import re
import sys

import pytest
import triton
from _nvidia_resources import compile_with_spill_stores
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels.trellis.triton import decode_int8_kernel

_HIP = pytest.mark.skipif(sys.platform != "linux", reason="HIP compiler is Linux-only")


@pytest.mark.parametrize("exact_pow2", [True, False], ids=["pow2", "nonpow2"])
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable_rows"])
@pytest.mark.parametrize(
    "target",
    [
        *[GPUTarget("cuda", architecture, 32) for architecture in (70, 80, 120)],
        pytest.param(GPUTarget("hip", "gfx942", 64), marks=_HIP),
        pytest.param(GPUTarget("hip", "gfx1201", 32), marks=_HIP),
    ],
    ids=lambda target: f"{target.backend}-{target.arch}",
)
def test_decode_kernel_compiles(target: GPUTarget, variable_rows: bool, exact_pow2: bool) -> None:
    signature = {
        "bitstreams": "*u8",
        "block_scales": "*u8",
        "qdata": "*i8",
        "num_blocks": "i32",
        "blocks_per_row": "i32",
        "bitstreams_bytes": "i64",
        "block_size": "i32",
        "block_bits": "i32",
        "step_bits": "i32",
        "num_wide_steps": "i32",
        "bytes_per_block": "i32",
    }
    constexprs: dict[str, object] = {
        "tile_steps": 256,
        "tile_blocks": 4,
        "variable_rows": variable_rows,
        "exact_pow2": exact_pow2,
        "use_sm120_3word": False,
    }
    if variable_rows:
        signature["row_widths"] = "*u8"
        signature["row_offsets"] = "*i64"
    else:
        constexprs["row_widths"] = None
        constexprs["row_offsets"] = None
    if target.backend == "hip":
        compiled = triton.compile(
            ASTSource(decode_int8_kernel, signature, constexprs=constexprs),
            target=target,
            options={"num_warps": 2},
        )
        assert compiled.asm["hsaco"]
        amdgcn = compiled.asm["amdgcn"]
        vgpr_spills = re.search(r"\.vgpr_spill_count:\s*(\d+)", amdgcn)
        priv_size = re.search(r"\.private_segment_fixed_size:\s*(\d+)", amdgcn)
        assert vgpr_spills is not None
        assert int(vgpr_spills.group(1)) == 0
        assert priv_size is not None
        assert int(priv_size.group(1)) == 0
        return
    compiled, spills = compile_with_spill_stores(
        ASTSource(decode_int8_kernel, signature, constexprs=constexprs),
        target=target,
        options={"num_warps": 2},
    )
    assert compiled.asm["ptx"]
    assert spills == 0


@pytest.mark.parametrize("tile_steps", [64, 128, 256], ids=lambda v: f"tile{v}")
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable_rows"])
@pytest.mark.parametrize("exact_pow2", [True, False], ids=["pow2", "nonpow2"])
def test_decode_kernel_sm120_3word_compiles(
    tile_steps: int, variable_rows: bool, exact_pow2: bool
) -> None:
    signature = {
        "bitstreams": "*u8",
        "block_scales": "*u8",
        "qdata": "*i8",
        "num_blocks": "i32",
        "blocks_per_row": "i32",
        "bitstreams_bytes": "i64",
        "block_size": "i32",
        "block_bits": "i32",
        "step_bits": "i32",
        "num_wide_steps": "i32",
        "bytes_per_block": "i32",
    }
    constexprs: dict[str, object] = {
        "tile_steps": tile_steps,
        "tile_blocks": 4,
        "variable_rows": variable_rows,
        "exact_pow2": exact_pow2,
        "use_sm120_3word": True,
    }
    if variable_rows:
        signature["row_widths"] = "*u8"
        signature["row_offsets"] = "*i64"
    else:
        constexprs["row_widths"] = None
        constexprs["row_offsets"] = None
    compiled, spills = compile_with_spill_stores(
        ASTSource(decode_int8_kernel, signature, constexprs=constexprs),
        target=GPUTarget("cuda", 120, 32),
        options={"num_warps": 2},
    )
    ptx = compiled.asm["ptx"]
    assert ptx
    assert "dp4a.s32.u32" in ptx
    assert spills == 0


_NVIDIA_SM80_TARGETS = [GPUTarget("cuda", architecture, 32) for architecture in (80, 86, 89, 120)]


@pytest.mark.parametrize("target", _NVIDIA_SM80_TARGETS, ids=lambda target: f"cuda-{target.arch}")
def test_encoder_triton_kernels_compile(target: GPUTarget) -> None:
    from piper_kernels.trellis import _encoder, _gluon_forward  # noqa: PLC0415

    sources = [
        (
            ASTSource(
                _encoder._estimate_block_scales_kernel,
                {
                    "w_ptr": "*fp32",
                    "out_scale_ptr": "*fp32",
                    "out_cap_amax_ptr": "*fp32",
                    "num_blocks": "i32",
                    "c_q": "fp32",
                },
                constexprs={"block_size": 256, "tile_t": 256},
            ),
            {"num_warps": 4},
        ),
        (
            ASTSource(
                _encoder._score_row_multipliers_kernel,
                {
                    "w_ptr": "*fp32",
                    "sc_ptr": "*fp32",
                    "eff_s_ptr": "*fp32",
                    "out_err_ptr": "*fp32",
                    "num_blocks": "i32",
                    "blocks_per_row": "i32",
                    "c_q": "fp32",
                },
                constexprs={"block_size": 256, "tile_t": 256, "num_mults": 29},
            ),
            {"num_warps": 4},
        ),
        (
            ASTSource(
                _encoder._refit_fp8_kernel,
                {
                    "w_ptr": "*fp32",
                    "states_ptr": "*i64",
                    "p2_ptr": "*fp32",
                    "window_ptr": "*fp32",
                    "out_scale_ptr": "*fp32",
                    "num_blocks": "i32",
                },
                constexprs={"block_size": 256, "tile_t": 256, "num_cands": 17},
            ),
            {"num_warps": 4, "enable_fp_fusion": False},
        ),
        (
            ASTSource(
                _gluon_forward._traceback_kernel,
                {
                    "bp_pair_ptr": "*u16",
                    "slot_ptr": "*i64",
                    "j_ptr": "*i64",
                    "widths_ptr": "*i64",
                    "states_ptr": "*i64",
                    "batch": "i32",
                    "num_pairs": "i32",
                },
                constexprs={
                    "length": 256,
                    "trace_steps": 256,
                    "state_bits": 16,
                    "width": 4096,
                    "lb": 4,
                    "block": 64,
                },
            ),
            {"num_warps": 2},
        ),
    ]
    for source, options in sources:
        compiled, spills = compile_with_spill_stores(source, target=target, options=options)
        assert compiled.asm["ptx"]
        assert spills == 0


@pytest.mark.parametrize("architecture", [80, 86, 89, 120])
@pytest.mark.parametrize(("state_bits", "cost_only"), [(16, False), (12, True)])
@pytest.mark.parametrize("step_widths", [(2,), (2, 3), (3, 4), (4, 5), (5, 6), (6,)], ids=str)
def test_encoder_persistent_forward_compiles(
    architecture: int, state_bits: int, cost_only: bool, step_widths: tuple[int, ...]
) -> None:
    from triton.experimental.gluon._runtime import GluonASTSource  # noqa: PLC0415

    from piper_kernels.trellis import _gluon_forward  # noqa: PLC0415

    widths = step_widths * (256 // len(step_widths))
    arguments = _gluon_forward._compile_args(
        *_gluon_forward._specialization(state_bits, widths, cost_only=cost_only)
    )
    num_warps = arguments.pop("num_warps")
    signature = {
        "w_pair_ptr": "*u32",
        "alpha_cta_ptr": "*fp32",
        "start_pair_ptr": "*i32",
        "kinds_ptr": "*i32",
        "out_ptr": "*u32",
        "bp_pair_ptr": "*u16",
        **dict.fromkeys(
            ("num_pairs", "length", "outputs", "k0", "constrained", "write_bp_from"), "i32"
        ),
    }
    kernel = _gluon_forward._forward_kernel
    aligned = (
        "w_pair_ptr",
        "alpha_cta_ptr",
        "start_pair_ptr",
        "kinds_ptr",
        "out_ptr",
        "bp_pair_ptr",
    )
    compiled, spills = compile_with_spill_stores(
        GluonASTSource(
            kernel,
            signature,
            attrs={(kernel.arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned},
            constexprs=arguments,
        ),
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": num_warps},
    )
    assert compiled.asm["ptx"]
    assert spills == 0
    assert compiled.metadata.shared <= _gluon_forward.SHARED_NEED
