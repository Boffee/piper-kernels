"""NVIDIA SM8x and SM120 launch policy for static-scale ConvRot INT8 causal convolutions.

Both targets run the shared Triton kernels. SM120 retains its measured tiles and may
load aligned 128-channel weights through tensor descriptors. SM8x has no TMA and loads
weights through pointers; SM120's 128x128 four-warp tile spills registers there, so SM8x
selects its own tiles, measured on SM89.
"""

from dataclasses import dataclass

from piper_kernels._triton.targets import AcceleratorTarget

from .._plan import ConvolutionPlan, PreparationPlan

# Below 32 row tiles of 64, 64x128 tiles leave SM8x GPUs underfilled.
_SM8X_WIDE_MIN_ROWS = 2048

_SM8X_WIDE = ConvolutionPlan(64, 128, 128, 4, 3)
_SM8X_SHORT = ConvolutionPlan(64, 64, 128, 4, 3)
_SM8X_NARROW_OUTPUT = ConvolutionPlan(32, 32, 128, 2, 3)


def supports_target(target: AcceleratorTarget) -> bool:
    """Enable exact SM120 and the SM8x family (SM80, SM86, SM87, SM89)."""
    return target.is_cuda_capability(12, 0) or target.is_cuda_capability(8)


def _sm120_use_weight_descriptor(
    channels: int, outputs: int, height: int, block_n: int, *, aligned: bool
) -> bool:
    return (
        aligned and channels == 128 and outputs % block_n == 0 and (height >= 256 or outputs > 128)
    )


def _sm120_convolution_plan(channels: int, outputs: int, rows: int) -> ConvolutionPlan:
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


def _sm8x_convolution_plan(outputs: int, rows: int) -> ConvolutionPlan:
    """Select one of three measured SM8x tiles from output width and rows."""
    if outputs <= 64:
        return _SM8X_NARROW_OUTPUT
    return _SM8X_WIDE if rows >= _SM8X_WIDE_MIN_ROWS else _SM8X_SHORT


def _preparation_plan(channels: int, rows: int, *, group_norm: bool) -> PreparationPlan:
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


@dataclass(frozen=True, slots=True)
class NvidiaConvolutionPolicy:
    """Launch policy for one NVIDIA target family, consumed by the shared launchers."""

    sm8x: bool

    def convolution_plan(self, channels: int, outputs: int, rows: int) -> ConvolutionPlan:
        if self.sm8x:
            return _sm8x_convolution_plan(outputs, rows)
        return _sm120_convolution_plan(channels, outputs, rows)

    def preparation_plan(self, channels: int, rows: int, *, group_norm: bool) -> PreparationPlan:
        # Preparation is bandwidth-bound; no SM89 tile beat SM120's.
        return _preparation_plan(channels, rows, group_norm=group_norm)

    def use_weight_descriptor(
        self, channels: int, outputs: int, height: int, block_n: int, *, aligned: bool
    ) -> bool:
        return not self.sm8x and _sm120_use_weight_descriptor(
            channels, outputs, height, block_n, aligned=aligned
        )


_SM120_POLICY = NvidiaConvolutionPolicy(sm8x=False)
_SM8X_POLICY = NvidiaConvolutionPolicy(sm8x=True)


def select_policy(target: AcceleratorTarget) -> NvidiaConvolutionPolicy:
    """Select SM8x or SM120 launch policy for a supported target."""
    if not supports_target(target):
        raise ValueError(f"ConvRot INT8 convolution has no NVIDIA policy for {target}")
    return _SM8X_POLICY if target.is_cuda_capability(8) else _SM120_POLICY
