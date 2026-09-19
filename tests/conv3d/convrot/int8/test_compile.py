"""Offline matrix-instruction coverage for the shared convolution kernel."""

import sys

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels.conv3d.convrot.int8 import triton as kernels
from piper_kernels.conv3d.convrot.int8._amd import policy as amd_policy
from piper_kernels.conv3d.convrot.int8._nvidia import policy as nvidia_policy


@pytest.mark.parametrize("channels", [64, 128, 256, 512, 1024, 4096])
@pytest.mark.parametrize("rows", [64, 16384])
@pytest.mark.parametrize(
    "target",
    [
        GPUTarget("cuda", 120, 32),
        pytest.param(
            GPUTarget("hip", "gfx1200", 32),
            marks=pytest.mark.skipif(sys.platform != "linux", reason="ROCm requires Linux"),
        ),
        pytest.param(
            GPUTarget("hip", "gfx1201", 32),
            marks=pytest.mark.skipif(sys.platform != "linux", reason="ROCm requires Linux"),
        ),
    ],
)
def test_convolution_compiles_to_integer_matrix_instructions(channels, rows, target):
    policy = amd_policy if target.backend == "hip" else nvidia_policy
    plan = policy.convolution_plan(channels, 128, rows)
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
            "output_frames": 3,
            "output_height": 17,
            "output_width": 19,
            "stride_frames": 1,
            "stride_height": 1,
            "stride_width": 1,
            "block_m": plan.block_m,
            "block_n": plan.block_n,
            "block_k": plan.block_k,
            "bias_stride": 2,
            "has_bias": True,
            "has_residual": True,
            "symmetric_spatial_padding": True,
            "right_spatial_padding": False,
            "use_weight_descriptor": False,
            "loop_num_stages": plan.num_stages,
        },
    )
    compiled = triton.compile(
        source,
        target=target,
        options={"num_warps": plan.num_warps, "num_stages": plan.num_stages},
    )
    if target.backend == "hip":
        assert "v_wmma_i32" in compiled.asm["amdgcn"]
    else:
        assert "mma.sync" in compiled.asm["ptx"]


@pytest.mark.skipif(sys.platform != "linux", reason="ROCm requires Linux")
@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("channels", [64, 256, 4096])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("fused", [False, True])
def test_amd_preparation_compiles_without_intermediate_rounding(
    architecture, channels, dtype, fused
):
    plan = amd_policy.preparation_plan(channels, 1024, group_norm=fused)
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
        "block_m": plan.block_m,
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
        source, target=GPUTarget("hip", architecture, 32), options={"num_warps": plan.num_warps}
    )
    assert compiled.asm["hsaco"]
    assert "arith.truncf" not in compiled.asm["ttir"]
