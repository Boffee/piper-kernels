"""Offline matrix-instruction and preparation coverage for ConvRot INT8 Conv3D."""

import re
import sys

import pytest
import triton
from _accelerators import rocm_fences_only_lds
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.amd_compiler_workarounds import lds_only_fence_options
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.conv3d.convrot.int8 import _policy
from piper_kernels.conv3d.convrot.int8 import triton as kernels


def _compile_conv3d(
    target,
    *,
    channels,
    outputs,
    rows,
    has_bias,
    has_residual,
    convolution_schedule=None,
):
    plan = _policy.select_execution_plan(
        AcceleratorTarget.from_compiler_target(target),
        channels=channels,
        outputs=outputs,
        input_rows=rows,
        output_rows=rows,
        output_height=17,
        weight_aligned=True,
        group_norm=False,
        convolution_schedule=convolution_schedule,
    )
    schedule = plan.convolution
    # Cover ordinary valid convolution as well as causal spatial reflection
    # on every accepted compiler target without multiplying the tile matrix.
    causal = rows == 16384
    source = ASTSource(
        kernels._conv3d_kernel,
        {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "weight_scale_ptr": "*fp32",
            "bias_ptr": "*fp16",
            "residual_ptr": "*fp16",
            "output_ptr": "*fp16",
            "output_rows": "i32",
            "output_channels": "i32",
            "input_scale": "*fp32",
            **{
                f"residual_stride_{axis}": "i64"
                for axis in (
                    "batch",
                    "channel",
                    "frame",
                    "height",
                    "width",
                )
            },
        },
        constexprs={
            "input_channels": channels,
            "input_frames": 3,
            "input_height": 17,
            "input_width": 19,
            "output_frames": 3 if causal else 1,
            "output_height": 17 if causal else 15,
            "output_width": 19 if causal else 17,
            "stride_frames": 1,
            "stride_height": 1,
            "stride_width": 1,
            "block_m": schedule.block_m,
            "block_n": schedule.block_n,
            "block_k": schedule.block_k,
            "bias_stride": 2,
            "has_bias": has_bias,
            "has_residual": has_residual,
            "padding_front": 2 if causal else 0,
            "padding_top": 1 if causal else 0,
            "padding_left": 1 if causal else 0,
            "spatial_reflection": causal,
            "use_weight_descriptor": False,
            "loop_num_stages": schedule.num_stages,
            "accumulator_seed": plan.accumulator_seed,
        },
    )
    compiled = triton.compile(
        source,
        target=target,
        options={
            "num_warps": schedule.num_warps,
            "num_stages": schedule.num_stages,
            **lds_only_fence_options(plan.lds_only_fences),
        },
    )
    return plan, compiled


@pytest.mark.parametrize("channels", [64, 128, 256, 512, 1024, 4096])
@pytest.mark.parametrize("rows", [64, 16384])
@pytest.mark.parametrize(
    "target",
    [
        GPUTarget("cuda", 120, 32),
        GPUTarget("cuda", 89, 32),
        pytest.param(
            GPUTarget("hip", "gfx1201", 32),
            marks=pytest.mark.skipif(sys.platform != "linux", reason="ROCm requires Linux"),
        ),
    ],
)
def test_convolution_compiles_to_integer_matrix_instructions(channels, rows, target):
    _, compiled = _compile_conv3d(
        target,
        channels=channels,
        outputs=128,
        rows=rows,
        has_bias=True,
        has_residual=True,
    )
    if target.backend == "hip":
        assert "v_wmma_i32" in compiled.asm["amdgcn"]
        if rocm_fences_only_lds():
            assert "global_inv" not in compiled.asm["amdgcn"]
    else:
        assert "mma.sync" in compiled.asm["ptx"]


@pytest.mark.skipif(sys.platform != "linux", reason="ROCm requires Linux")
def test_amd_seeded_convolution_compile_invariants():
    target = GPUTarget("hip", "gfx1201", 32)
    wide_plan, wide_compiled = _compile_conv3d(
        target,
        channels=256,
        outputs=256,
        rows=16384,
        has_bias=False,
        has_residual=False,
        convolution_schedule=_policy.ConvolutionSchedule(128, 256, 64, 8, 2),
    )
    assert wide_plan.accumulator_seed == 0x155555
    assert "v_wmma_i32" in wide_compiled.asm["amdgcn"]
    assert "0x155555" in wide_compiled.asm["amdgcn"]
    scratch_match = re.search(
        r"amdhsa_private_segment_fixed_size\s+(\d+)\b", wide_compiled.asm["amdgcn"]
    )
    assert scratch_match is not None
    assert int(scratch_match.group(1)) <= 768
    if rocm_fences_only_lds():
        assert "global_inv" not in wide_compiled.asm["amdgcn"]

    medium_plan, medium_compiled = _compile_conv3d(
        target,
        channels=256,
        outputs=128,
        rows=16384,
        has_bias=False,
        has_residual=False,
        convolution_schedule=_policy.ConvolutionSchedule(128, 128, 64, 8, 2),
    )
    assert medium_plan.accumulator_seed == 0x155555
    assert "v_wmma_i32" in medium_compiled.asm["amdgcn"]
    assert "0x155555" in medium_compiled.asm["amdgcn"]
    assert re.search(r"amdhsa_private_segment_fixed_size\s+0\b", medium_compiled.asm["amdgcn"])
    if rocm_fences_only_lds():
        assert "global_inv" not in medium_compiled.asm["amdgcn"]

    _, default_compiled = _compile_conv3d(
        target,
        channels=256,
        outputs=256,
        rows=16384,
        has_bias=False,
        has_residual=False,
    )
    assert "v_wmma_i32" in default_compiled.asm["amdgcn"]
    default_scratch = re.search(
        r"amdhsa_private_segment_fixed_size\s+(\d+)\b", default_compiled.asm["amdgcn"]
    )
    assert default_scratch is not None
    assert int(default_scratch.group(1)) <= 768
    if rocm_fences_only_lds():
        assert "global_inv" not in default_compiled.asm["amdgcn"]


@pytest.mark.skipif(sys.platform != "linux", reason="ROCm requires Linux")
@pytest.mark.parametrize("channels", [64, 256, 4096])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("fused", [False, True])
def test_amd_preparation_compiles_without_intermediate_rounding(channels, dtype, fused):
    schedule = _policy._amd_preparation_schedule(channels)
    signature = {
        "input_ptr": f"*{dtype}",
        "output_ptr": "*i8",
        "token_count": "i32",
        "input_scale": "*fp32",
        **{f"stride_{axis}": "i64" for axis in ("batch", "channel", "frame", "height", "width")},
    }
    constants = {
        "token_volume": 3 * 17 * 19,
        "input_height": 17,
        "input_width": 19,
        "channels": channels,
        "group_size": 64,
        "accelerator_backend": "hip",
        "block_m": schedule.block_m,
    }
    if fused:
        signature.update(
            dict.fromkeys(("affine_weight_ptr", "affine_bias_ptr", "mean_ptr", "rstd_ptr"), "*fp32")
        )
        constants.update(
            {
                "affine_weight_stride": 2,
                "affine_bias_stride": 2,
                "frames": 3,
                "groups": 32,
                "channels_per_group": channels // 32,
            }
        )
    source = ASTSource(
        kernels._prepare_group_norm_silu_kernel if fused else kernels._prepare_channelwise_kernel,
        signature,
        constexprs=constants,
    )
    compiled = triton.compile(
        source, target=GPUTarget("hip", "gfx1201", 32), options={"num_warps": schedule.num_warps}
    )
    assert compiled.asm["hsaco"]
    assert "arith.truncf" not in compiled.asm["ttir"]
