"""RDNA4 launch policy for static-scale ConvRot INT8 convolutions."""

# Shared ConvolutionPolicy signatures include dimensions not used by every target.
# ruff: noqa: ARG001

import sys

from piper_kernels._triton.targets import AcceleratorTarget

from .._plan import ConvolutionExecutionPlan, ConvolutionPlan, PreparationPlan


def supports_target(target: AcceleratorTarget) -> bool:
    """gfx1201 has runtime coverage; gfx1200 also has offline compilation coverage."""
    return (
        sys.platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    )


def _convolution_plan(channels: int, outputs: int, rows: int) -> ConvolutionPlan:
    # RX 9070 XT synthetic H3-style sweeps: larger row tiles amortize the
    # implicit 3-D gather at high resolution; smaller tiles keep enough work
    # in flight for deeper, low-resolution convolutions.
    if channels <= 128 and outputs >= 128 and rows >= 8192:
        return ConvolutionPlan(128, 128, 128, 8, 2)
    if channels <= 256 and outputs >= 128 and rows >= 1024:
        return ConvolutionPlan(64, 128, 128, 4, 2)
    if outputs >= 64 and rows >= 128:
        return ConvolutionPlan(64, 64, 64, 4, 2)
    return ConvolutionPlan(32, 64, 64, 4, 2)


def preparation_plan(channels: int, rows: int, *, group_norm: bool) -> PreparationPlan:
    # Bound the FP32 rotation tile independently of the convolution output width.
    return PreparationPlan(max(1, min(16, 4096 // channels)), 4)


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    channels: int,
    outputs: int,
    input_rows: int,
    output_rows: int,
    output_height: int,
    weight_aligned: bool,
    group_norm: bool,
    convolution_plan: ConvolutionPlan | None = None,
) -> ConvolutionExecutionPlan:
    """Resolve the RDNA4 schedules and pointer loads before shared execution."""
    if not supports_target(target):
        raise ValueError(f"ConvRot INT8 convolution has no AMD policy for {target}")
    return ConvolutionExecutionPlan(
        preparation=preparation_plan(channels, input_rows, group_norm=group_norm),
        convolution=(
            _convolution_plan(channels, outputs, output_rows)
            if convolution_plan is None
            else convolution_plan
        ),
        use_weight_descriptor=False,
    )
