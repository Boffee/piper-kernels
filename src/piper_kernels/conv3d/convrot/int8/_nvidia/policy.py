"""Measured NVIDIA schedules for static-scale ConvRot INT8 causal convolutions.

SM120 and SM8x run the shared Triton kernels with independently selected convolution
schedules. Preparation is shared. Selection resolves the target into concrete launch
choices, including whether to use an aligned weight descriptor.
"""

from piper_kernels._triton.targets import AcceleratorTarget

from .._plan import ConvolutionExecutionPlan, ConvolutionPlan, PreparationPlan

# Historical SM120 crossovers, retained along with their channel/output predicates.
_SM120_C256_LARGE_MIN_ROWS = 200_000
_SM120_C256_MEDIUM_MIN_ROWS = 30_000
_SM120_C512_LARGE_MIN_ROWS = 5_000

# SM89 measurements favor narrower tiles below 32 row tiles of 64.
_SM8X_WIDE_MIN_ROWS = 2048

_BASE_CONVOLUTION = ConvolutionPlan(64, 128, 128, 4, 3)
_SM120_K64 = ConvolutionPlan(128, 128, 64, 4, 3)
_SM120_K64_DEEP_PIPELINE = ConvolutionPlan(128, 128, 64, 4, 4)
_SM120_K64_SHORT_PIPELINE = ConvolutionPlan(128, 128, 64, 4, 2)
_SM120_LARGE_M = ConvolutionPlan(128, 128, 128, 8, 3)
_SM120_LARGE_K = ConvolutionPlan(32, 128, 256, 8, 3)
_SM120_NARROW_OUTPUT = ConvolutionPlan(64, 64, 64, 4, 3)
_SM8X_SHORT = ConvolutionPlan(64, 64, 128, 4, 3)
_SM8X_NARROW_OUTPUT = ConvolutionPlan(32, 32, 128, 2, 3)

_PREPARE_DEFAULT = PreparationPlan(8, 8)
_PREPARE_WIDE = PreparationPlan(64, 4)
_PREPARE_MEDIUM = PreparationPlan(32, 4)
_PREPARE_SHORT = PreparationPlan(16, 4)
_PREPARE_MEDIUM_8_WARPS = PreparationPlan(32, 8)
_PREPARE_SHORT_8_WARPS = PreparationPlan(16, 8)


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
    """Select a complete historical SM120 convolution schedule."""
    if channels == 128:
        return _SM120_K64
    if channels == 256:
        if rows >= _SM120_C256_MEDIUM_MIN_ROWS:
            return (
                _SM120_K64_DEEP_PIPELINE
                if rows >= _SM120_C256_LARGE_MIN_ROWS
                else _SM120_K64_SHORT_PIPELINE
            )
        return _BASE_CONVOLUTION if outputs == 256 else _SM120_LARGE_M
    if channels == 512:
        if rows >= _SM120_C512_LARGE_MIN_ROWS:
            return _SM120_LARGE_M
        return _BASE_CONVOLUTION if outputs > 512 else _SM120_LARGE_K
    return _SM120_NARROW_OUTPUT if outputs <= 64 else _BASE_CONVOLUTION


def _sm8x_convolution_plan(outputs: int, rows: int) -> ConvolutionPlan:
    """Select one of three measured SM8x tiles from output width and rows."""
    if outputs <= 64:
        return _SM8X_NARROW_OUTPUT
    return _BASE_CONVOLUTION if rows >= _SM8X_WIDE_MIN_ROWS else _SM8X_SHORT


def preparation_plan(channels: int, rows: int, *, group_norm: bool) -> PreparationPlan:
    """Select preparation independently of output dimensions for both target families.

    Preparation is bandwidth-bound; no measured SM89 tile beat SM120's schedule.
    """
    if channels == 128:
        return _PREPARE_WIDE
    if channels == 256:
        if group_norm:
            return _PREPARE_MEDIUM if rows >= _SM120_C256_LARGE_MIN_ROWS else _PREPARE_SHORT
        return _PREPARE_WIDE if rows >= _SM120_C256_LARGE_MIN_ROWS else _PREPARE_MEDIUM_8_WARPS
    if channels == 512 and rows >= _SM120_C512_LARGE_MIN_ROWS:
        return _PREPARE_SHORT_8_WARPS if group_norm else _PREPARE_MEDIUM_8_WARPS
    return _PREPARE_DEFAULT


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
    """Resolve preparation, convolution, and descriptor choices once before execution.

    An explicit convolution tile lets tuning reuse production preparation and
    recompute descriptor eligibility for the candidate's column width.
    """
    if not supports_target(target):
        raise ValueError(f"ConvRot INT8 convolution has no NVIDIA policy for {target}")
    sm8x = target.is_cuda_capability(8)
    if convolution_plan is None:
        convolution_plan = (
            _sm8x_convolution_plan(outputs, output_rows)
            if sm8x
            else _sm120_convolution_plan(channels, outputs, output_rows)
        )
    return ConvolutionExecutionPlan(
        preparation=preparation_plan(channels, input_rows, group_norm=group_norm),
        convolution=convolution_plan,
        use_weight_descriptor=not sm8x
        and _sm120_use_weight_descriptor(
            channels, outputs, output_height, convolution_plan.block_n, aligned=weight_aligned
        ),
    )
