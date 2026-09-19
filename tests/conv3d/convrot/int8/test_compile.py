"""Offline matrix-instruction coverage for the shared convolution kernel."""

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels.conv3d.convrot.int8 import triton as kernels
from piper_kernels.conv3d.convrot.int8._nvidia import policy


@pytest.mark.parametrize("channels", [64, 128, 256, 512, 1024, 4096])
def test_nvidia_convolution_compiles_to_integer_matrix_instructions(channels):
    plan = policy.convolution_plan(channels, 128, 1024)
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
        target=GPUTarget("cuda", 120, 32),
        options={"num_warps": plan.num_warps, "num_stages": plan.num_stages},
    )
    assert "mma.sync" in compiled.asm["ptx"]
