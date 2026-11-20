"""Schedules, execution plans, and target policy for ConvRot INT8 Conv3D.

Kept Triton-free so CPU imports, validation, and offline compilation/benchmark
planning never depend on Triton. SM120, SM8x, and RDNA4 share the kernels in
``triton.py`` while selecting independently measured launch schedules.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from piper_kernels._triton.targets import AcceleratorTarget

type Padding = tuple[int, int, int, int, int, int]

_SM120_C256_LARGE_MIN_ROWS = 200_000
_SM120_C256_MEDIUM_MIN_ROWS = 30_000
_SM120_C512_LARGE_MIN_ROWS = 5_000

# SM89 measurements favor narrower tiles below 32 row tiles of 64.
_SM8X_WIDE_MIN_ROWS = 2048


class ConvolutionSchedule(NamedTuple):
    """Complete immutable convolution tile and launch choices."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


class PreparationSchedule(NamedTuple):
    """Rotation/quantization tile and warp count."""

    block_m: int
    num_warps: int


@dataclass(frozen=True, slots=True)
class ConvolutionExecutionPlan:
    """Concrete preparation, convolution, and weight-load choices for one operation."""

    preparation: PreparationSchedule
    convolution: ConvolutionSchedule
    use_weight_descriptor: bool
    # Fence only LDS at barriers; an RDNA workaround in _triton.amd_compiler_workarounds.
    lds_only_fences: bool = False


_BASE_CONVOLUTION = ConvolutionSchedule(64, 128, 128, 4, 3)
_SM120_K64 = ConvolutionSchedule(128, 128, 64, 4, 3)
_SM120_K64_DEEP_PIPELINE = ConvolutionSchedule(128, 128, 64, 4, 4)
_SM120_K64_SHORT_PIPELINE = ConvolutionSchedule(128, 128, 64, 4, 2)
_SM120_LARGE_M = ConvolutionSchedule(128, 128, 128, 8, 3)
_SM120_LARGE_K = ConvolutionSchedule(32, 128, 256, 8, 3)
_SM120_NARROW_OUTPUT = ConvolutionSchedule(64, 64, 64, 4, 3)
_SM8X_SHORT = ConvolutionSchedule(64, 64, 128, 4, 3)
_SM8X_NARROW_OUTPUT = ConvolutionSchedule(32, 32, 128, 2, 3)

_PREPARE_DEFAULT = PreparationSchedule(8, 8)
_PREPARE_WIDE = PreparationSchedule(64, 4)
_PREPARE_MEDIUM = PreparationSchedule(32, 4)
_PREPARE_SHORT = PreparationSchedule(16, 4)
_PREPARE_MEDIUM_8_WARPS = PreparationSchedule(32, 8)
_PREPARE_SHORT_8_WARPS = PreparationSchedule(16, 8)


def output_shape(
    input_shape: Sequence[int],
    output_channels: int,
    stride: Sequence[int],
    padding: Padding,
) -> tuple[int, int, int, int, int]:
    """Compute the NCTHW output shape for a 3x3x3 convolution with F.pad-order padding."""
    batch, _, frames, height, width = input_shape
    return (
        batch,
        output_channels,
        (frames + padding[4] + padding[5] - 3) // stride[0] + 1,
        (height + padding[2] + padding[3] - 3) // stride[1] + 1,
        (width + padding[0] + padding[1] - 3) // stride[2] + 1,
    )


def _supports_nvidia(target: AcceleratorTarget) -> bool:
    """Enable exact SM120 and the SM8x family (SM80, SM86, SM87, SM89)."""
    return target.is_cuda_capability(12, 0) or target.is_cuda_capability(8)


def _supports_amd(target: AcceleratorTarget) -> bool:
    """gfx1201 has runtime coverage; gfx1200 also has offline compilation coverage."""
    return (
        sys.platform in ("linux", "win32")
        and target.is_amd_hip
        and target.is_architecture("gfx1200", "gfx1201")
    )


def supports_target(target: AcceleratorTarget) -> bool:
    """Return whether ``target`` has a tuned native Conv3D schedule."""
    return _supports_nvidia(target) or _supports_amd(target)


def _sm120_use_weight_descriptor(
    channels: int, outputs: int, height: int, block_n: int, *, aligned: bool
) -> bool:
    return (
        aligned and channels == 128 and outputs % block_n == 0 and (height >= 256 or outputs > 128)
    )


def _sm120_convolution_schedule(channels: int, outputs: int, rows: int) -> ConvolutionSchedule:
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


def _sm8x_convolution_schedule(outputs: int, rows: int) -> ConvolutionSchedule:
    if outputs <= 64:
        return _SM8X_NARROW_OUTPUT
    return _BASE_CONVOLUTION if rows >= _SM8X_WIDE_MIN_ROWS else _SM8X_SHORT


def _nvidia_preparation_schedule(
    channels: int, rows: int, *, group_norm: bool
) -> PreparationSchedule:
    # Preparation is bandwidth-bound; no measured SM89 tile beat SM120's schedule.
    if channels == 128:
        return _PREPARE_WIDE
    if channels == 256:
        if group_norm:
            return _PREPARE_MEDIUM if rows >= _SM120_C256_LARGE_MIN_ROWS else _PREPARE_SHORT
        return _PREPARE_WIDE if rows >= _SM120_C256_LARGE_MIN_ROWS else _PREPARE_MEDIUM_8_WARPS
    if channels == 512 and rows >= _SM120_C512_LARGE_MIN_ROWS:
        return _PREPARE_SHORT_8_WARPS if group_norm else _PREPARE_MEDIUM_8_WARPS
    return _PREPARE_DEFAULT


def _amd_convolution_schedule(channels: int, outputs: int, rows: int) -> ConvolutionSchedule:
    # RX 9070 XT synthetic H3-style sweeps: larger row tiles amortize the
    # implicit 3-D gather at high resolution; smaller tiles keep enough work
    # in flight for deeper, low-resolution convolutions.
    if channels <= 128 and outputs >= 128 and rows >= 8192:
        return ConvolutionSchedule(128, 128, 128, 8, 2)
    if channels <= 256 and outputs >= 128 and rows >= 1024:
        return ConvolutionSchedule(64, 128, 128, 4, 2)
    if outputs >= 64 and rows >= 128:
        return ConvolutionSchedule(64, 64, 64, 4, 2)
    return ConvolutionSchedule(32, 64, 64, 4, 2)


def _amd_preparation_schedule(channels: int) -> PreparationSchedule:
    # Bound the FP32 rotation tile independently of the convolution output width.
    return PreparationSchedule(max(1, min(16, 4096 // channels)), 4)


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
    convolution_schedule: ConvolutionSchedule | None = None,
) -> ConvolutionExecutionPlan:
    """Resolve preparation, convolution, and weight-load choices once before launch.

    An explicit ``convolution_schedule`` lets tuning reuse production preparation
    and recompute weight-descriptor eligibility for the candidate's column width.
    """
    if _supports_nvidia(target):
        sm8x = target.is_cuda_capability(8)
        if convolution_schedule is None:
            convolution_schedule = (
                _sm8x_convolution_schedule(outputs, output_rows)
                if sm8x
                else _sm120_convolution_schedule(channels, outputs, output_rows)
            )
        return ConvolutionExecutionPlan(
            preparation=_nvidia_preparation_schedule(channels, input_rows, group_norm=group_norm),
            convolution=convolution_schedule,
            use_weight_descriptor=not sm8x
            and _sm120_use_weight_descriptor(
                channels,
                outputs,
                output_height,
                convolution_schedule.block_n,
                aligned=weight_aligned,
            ),
        )
    if _supports_amd(target):
        return ConvolutionExecutionPlan(
            preparation=_amd_preparation_schedule(channels),
            convolution=(
                _amd_convolution_schedule(channels, outputs, output_rows)
                if convolution_schedule is None
                else convolution_schedule
            ),
            use_weight_descriptor=False,
            lds_only_fences=True,
        )
    raise ValueError(f"ConvRot INT8 convolution has no native policy for {target}")


def default_execution_plan(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    group_norm: bool,
    padding: Padding,
    target: AcceleratorTarget | None = None,
    convolution_schedule: ConvolutionSchedule | None = None,
) -> ConvolutionExecutionPlan:
    """Resolve the production plan from validated operand metadata."""
    target = AcceleratorTarget.from_device(input.device) if target is None else target
    batch, channels, frames, height, width = input.shape
    outputs = weight_qdata.shape[0]
    shape = output_shape(input.shape, outputs, stride, padding)
    return select_execution_plan(
        target,
        channels=channels,
        outputs=outputs,
        input_rows=batch * frames * height * width,
        output_rows=batch * math.prod(shape[2:]),
        output_height=shape[3],
        weight_aligned=weight_qdata.data_ptr() % 16 == 0,
        group_norm=group_norm,
        convolution_schedule=convolution_schedule,
    )
