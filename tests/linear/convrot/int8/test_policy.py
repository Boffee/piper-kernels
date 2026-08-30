"""Host-side execution-plan policy tests for INT8 ConvRot."""

from dataclasses import replace

import pytest

from piper_kernels.linear.convrot.int8._policy import (
    LinearExecutionPlan,
    select_execution_plan,
    select_fused_preparation_chunks,
)


@pytest.mark.parametrize(
    ("in_features", "expected_fusion", "expected_fused_warps"),
    [
        (512, True, 4),
        (5_376, True, 2),
        (14_336, True, 4),
        (16_640, True, 4),
        (28_672, True, 8),
        (49_152, True, 8),
        (49_408, False, 4),
    ],
)
def test_execution_plan_centralizes_fusion_extent(
    in_features: int,
    expected_fusion: bool,
    expected_fused_warps: int,
) -> None:
    plan = select_execution_plan(in_features=in_features)

    assert plan.fuse_rotation_quantization is expected_fusion
    assert plan.fused_num_warps == expected_fused_warps


@pytest.mark.parametrize(
    ("in_features", "expected_chunks"),
    [
        (512, (1, 512)),
        (4_096, (1, 4_096)),
        (4_097, (3, 2_048)),
        (5_376, (3, 2_048)),
        (6_144, (3, 2_048)),
        (6_145, (1, 8_192)),
        (7_168, (1, 8_192)),
        (8_192, (1, 8_192)),
        (8_193, (3, 4_096)),
        (9_728, (3, 4_096)),
        (12_288, (3, 4_096)),
        (12_289, (2, 8_192)),
        (14_336, (2, 8_192)),
        (16_384, (2, 8_192)),
        (16_385, (3, 8_192)),
        (16_640, (3, 8_192)),
        (24_576, (3, 8_192)),
        (24_577, (2, 16_384)),
        (28_672, (2, 16_384)),
        (32_768, (2, 16_384)),
        (32_769, (3, 16_384)),
        (40_960, (3, 16_384)),
        (49_152, (3, 16_384)),
        (49_153, None),
    ],
)
def test_fused_preparation_selects_low_padding_equal_chunks(
    in_features: int,
    expected_chunks: tuple[int, int] | None,
) -> None:
    assert select_fused_preparation_chunks(in_features) == expected_chunks


def test_execution_plan_uses_uniform_mid_size_fused_launch_schedule() -> None:
    plans = [select_execution_plan(in_features=in_features) for in_features in (7168, 14_336)]

    assert {plan.fused_num_warps for plan in plans} == {4}


def test_execution_plan_selects_uniform_matmul_schedule() -> None:
    plan = select_execution_plan(in_features=512)

    assert plan.matmul_block_m == 128
    assert plan.matmul_block_n == 256
    assert plan.matmul_block_k == 128
    assert plan.matmul_num_warps == 8
    assert plan.matmul_num_stages == 3


def test_execution_plan_serializes_flat_tuning_fields() -> None:
    plan = select_execution_plan(in_features=512)

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
    plan = LinearExecutionPlan(
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
    plan = select_execution_plan(in_features=512)

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
    plan = select_execution_plan(in_features=512)

    with pytest.raises(ValueError, match="ConvRot"):
        replace(plan, **changes)
