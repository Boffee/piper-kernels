"""Host-side specialization policy tests for INT8 ConvRot preparation."""

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.convrot.int8._policy import PreparationPlan, select_preparation_plan

_SM89 = AcceleratorTarget(backend="cuda", architecture="sm89")
_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")
_SM121 = AcceleratorTarget(backend="cuda", architecture="sm121")
_HIP = AcceleratorTarget(backend="hip", architecture="gfx1200")


@pytest.mark.parametrize(
    ("target", "expected_fusion"),
    [
        (_SM89, False),
        (_SM120, True),
        (_SM121, False),
        (_HIP, False),
    ],
)
def test_preparation_plan_keeps_sm120_tuning_target_exact(
    target: AcceleratorTarget,
    expected_fusion: bool,
) -> None:
    plan = select_preparation_plan(
        target,
        rows=512,
        in_features=512,
        group_size=256,
        dtype=torch.float16,
    )

    assert plan == PreparationPlan(
        block_size=512,
        fuse_rotation_quantization=expected_fusion,
        fused_num_warps=4,
    )


@pytest.mark.parametrize(
    ("rows", "in_features", "group_size", "dtype", "block_size", "expected"),
    [
        (512, 512, 256, torch.float16, 512, True),
        (512, 14_336, 256, torch.bfloat16, 16_384, True),
        (511, 512, 256, torch.float16, 512, False),
        (512, 0, 256, torch.float16, 128, False),
        (512, 512, 64, torch.float16, 512, False),
        (512, 512, 256, torch.float32, 512, False),
        (512, 16_640, 256, torch.bfloat16, 32_768, False),
    ],
)
def test_preparation_plan_centralizes_fusion_boundaries(
    rows: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    block_size: int,
    expected: bool,
) -> None:
    plan = select_preparation_plan(
        _SM120,
        rows=rows,
        in_features=in_features,
        group_size=group_size,
        dtype=dtype,
    )

    assert plan.block_size == block_size
    assert plan.fuse_rotation_quantization is expected


@pytest.mark.parametrize(
    ("target", "rows", "in_features", "swiglu", "expected_warps"),
    [
        (_SM120, 511, 14_336, True, 8),
        (_SM120, 8191, 14_336, True, 8),
        (_SM120, 8192, 14_336, True, 16),
        (_SM120, 8192, 14_336, False, 4),
        (_SM120, 8192, 7168, True, 4),
        (_SM121, 8192, 14_336, True, 4),
        (_HIP, 8192, 14_336, True, 4),
    ],
)
def test_preparation_plan_centralizes_fused_launch_schedule(
    target: AcceleratorTarget,
    rows: int,
    in_features: int,
    swiglu: bool,
    expected_warps: int,
) -> None:
    plan = select_preparation_plan(
        target,
        rows=rows,
        in_features=in_features,
        group_size=256,
        dtype=torch.bfloat16,
        swiglu=swiglu,
    )

    assert plan.fused_num_warps == expected_warps
