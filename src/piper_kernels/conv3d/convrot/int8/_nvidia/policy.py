"""SM120 launch policy for static-scale ConvRot INT8 causal convolutions."""

from piper_kernels._triton.targets import AcceleratorTarget

from .._plan import ConvolutionPlan, PreparationPlan


def supports_target(target: AcceleratorTarget) -> bool:
    return target.is_cuda_capability(12, 0)


def use_weight_descriptor(
    channels: int, outputs: int, height: int, block_n: int, *, aligned: bool
) -> bool:
    return (
        aligned and channels == 128 and outputs % block_n == 0 and (height >= 256 or outputs > 128)
    )


def convolution_plan(channels: int, outputs: int, rows: int) -> ConvolutionPlan:
    """Select SM120 convolution tiles from logical channel counts and output rows."""
    plan = ConvolutionPlan(64, 128, 128, 4, 3)
    if channels == 128:
        plan = ConvolutionPlan(128, 128, 64, 4, 3)
    elif channels == 256 and rows >= 200_000:
        plan = ConvolutionPlan(128, 128, 64, 4, 4)
    elif channels == 256 and rows >= 30_000:
        plan = ConvolutionPlan(128, 128, 64, 4, 2)
    elif channels == 256 and outputs == 256:
        plan = ConvolutionPlan(64, 128, 128, 4, 3)
    elif channels == 256 or (channels == 512 and rows >= 5_000):
        plan = ConvolutionPlan(128, 128, 128, 8, 3)
    elif channels == 512 and outputs > 512:
        plan = ConvolutionPlan(64, 128, 128, 4, 3)
    elif channels == 512:
        plan = ConvolutionPlan(32, 128, 256, 8, 3)
    elif outputs <= 64:
        plan = ConvolutionPlan(64, 64, 64, 4, 3)
    return plan


def preparation_plan(channels: int, rows: int, *, group_norm: bool) -> PreparationPlan:
    """Select rotation/quantization tiles independently of convolution outputs."""
    plan = PreparationPlan(8, 8)
    if channels == 128:
        plan = PreparationPlan(64, 4)
    elif channels == 256:
        if group_norm and rows >= 200_000:
            plan = PreparationPlan(32, 4)
        elif group_norm:
            plan = PreparationPlan(16, 4)
        elif rows >= 200_000:
            plan = PreparationPlan(64, 4)
        else:
            plan = PreparationPlan(32, 8)
    elif channels == 512 and group_norm and rows >= 5_000:
        plan = PreparationPlan(16, 8)
    elif channels == 512 and not group_norm and rows >= 5_000:
        plan = PreparationPlan(32, 8)
    return plan
