"""Host-side execution-plan policy tests for INT8 ConvRot."""

from dataclasses import replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._policy import (
    ConvRotInt8LinearExecutionPlan,
    select_execution_plan,
)

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
def test_execution_plan_keeps_sm120_tuning_target_exact(
    target: AcceleratorTarget,
    expected_fusion: bool,
) -> None:
    plan = select_execution_plan(
        target,
        rows=512,
        out_features=96,
        in_features=512,
        group_size=256,
        dtype=torch.float16,
    )

    assert plan.fuse_rotation_quantization is expected_fusion
    assert plan.fused_num_warps == 4


@pytest.mark.parametrize(
    (
        "rows",
        "in_features",
        "group_size",
        "dtype",
        "expected_fusion",
    ),
    [
        (512, 512, 256, torch.float16, True),
        (512, 14_336, 256, torch.bfloat16, True),
        (511, 512, 256, torch.float16, False),
        (512, 512, 64, torch.float16, False),
        (512, 512, 256, torch.float32, False),
        (512, 16_640, 256, torch.bfloat16, False),
    ],
)
def test_execution_plan_centralizes_fusion_boundaries(
    rows: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    expected_fusion: bool,
) -> None:
    plan = select_execution_plan(
        _SM120,
        rows=rows,
        out_features=96,
        in_features=in_features,
        group_size=group_size,
        dtype=dtype,
    )

    assert plan.fuse_rotation_quantization is expected_fusion


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
def test_execution_plan_centralizes_fused_launch_schedule(
    target: AcceleratorTarget,
    rows: int,
    in_features: int,
    swiglu: bool,
    expected_warps: int,
) -> None:
    plan = select_execution_plan(
        target,
        rows=rows,
        out_features=96,
        in_features=in_features,
        group_size=256,
        dtype=torch.bfloat16,
        swiglu=swiglu,
    )

    assert plan.fused_num_warps == expected_warps


def test_preparation_schedule_is_independent_of_output_width() -> None:
    plans = [
        select_execution_plan(
            _SM120,
            rows=8192,
            out_features=out_features,
            in_features=6144,
            group_size=256,
            dtype=torch.bfloat16,
        )
        for out_features in (0, 1, 4096, 16_384)
    ]

    preparation_schedules = {
        (
            plan.fuse_rotation_quantization,
            plan.fused_num_warps,
            plan.rotation_num_warps,
            plan.quantization_num_warps,
        )
        for plan in plans
    }
    assert len(preparation_schedules) == 1


@pytest.mark.parametrize(
    (
        "target",
        "rows",
        "out_features",
        "expected_block_m",
        "expected_block_n",
        "expected_block_k",
        "expected_warps",
    ),
    [
        (_SM120, 1, 96, 32, 64, 32, 4),
        (_SM120, 63, 127, 32, 64, 32, 4),
        (_SM120, 64, 128, 64, 128, 32, 4),
        (_SM120, 511, 21_504, 64, 128, 32, 4),
        (_SM120, 512, 96, 128, 256, 128, 8),
        (_SM120, 37_710, 21_504, 128, 256, 128, 8),
        (_SM121, 37_710, 21_504, 64, 128, 32, 4),
        (_SM89, 37_710, 21_504, 64, 128, 32, 4),
    ],
)
def test_execution_plan_selects_exact_sm120_large_matmul_schedule(
    target: AcceleratorTarget,
    rows: int,
    out_features: int,
    expected_block_m: int,
    expected_block_n: int,
    expected_block_k: int,
    expected_warps: int,
) -> None:
    plan = select_execution_plan(
        target,
        rows=rows,
        out_features=out_features,
        in_features=512,
        group_size=256,
        dtype=torch.bfloat16,
    )

    assert plan.matmul_block_m == expected_block_m
    assert plan.matmul_block_n == expected_block_n
    assert plan.matmul_block_k == expected_block_k
    assert plan.matmul_num_warps == expected_warps
    assert plan.matmul_num_stages == 3


def test_execution_plan_serializes_flat_tuning_fields() -> None:
    plan = select_execution_plan(
        _SM120,
        rows=512,
        out_features=96,
        in_features=512,
        group_size=256,
        dtype=torch.float16,
    )

    assert plan.as_dict() == {
        "fuse_rotation_quantization": True,
        "fused_num_warps": 4,
        "rotation_num_warps": 4,
        "quantization_num_warps": 8,
        "matmul_block_m": 128,
        "matmul_block_n": 256,
        "matmul_block_k": 128,
        "matmul_num_warps": 8,
        "matmul_num_stages": 3,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"matmul_block_m": 8},
        {"matmul_block_n": 512},
        {"matmul_block_k": 16},
        {"matmul_num_warps": 16},
        {"matmul_num_stages": 5},
    ],
)
def test_execution_plan_rejects_invalid_matmul_launch_choices(
    changes: dict[str, int],
) -> None:
    plan = ConvRotInt8LinearExecutionPlan(
        fuse_rotation_quantization=True,
        fused_num_warps=4,
        rotation_num_warps=4,
        quantization_num_warps=8,
        matmul_block_m=64,
        matmul_block_n=128,
        matmul_block_k=32,
    )

    with pytest.raises(ValueError, match="ConvRot"):
        replace(plan, **changes)


def test_execution_plan_rejects_invalid_fused_warp_count() -> None:
    plan = select_execution_plan(
        _SM120,
        rows=512,
        out_features=96,
        in_features=512,
        group_size=256,
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="ConvRot"):
        replace(plan, fused_num_warps=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"rotation_num_warps": 16},
        {"quantization_num_warps": 16},
    ],
)
def test_execution_plan_rejects_invalid_split_launch_choices(
    changes: dict[str, int],
) -> None:
    plan = select_execution_plan(
        _SM120,
        rows=512,
        out_features=96,
        in_features=512,
        group_size=256,
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="ConvRot"):
        replace(plan, **changes)
