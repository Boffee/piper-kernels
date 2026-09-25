"""Host-side execution-plan policy tests for INT8 ConvRot."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._nvidia import policy as nvidia_policy
from piper_kernels.linear.convrot.int8._nvidia.policy import (
    NvidiaExecutionPlan,
    select_execution_plan,
)
from piper_kernels.linear.convrot.int8._plan import LinearExecutionPlan
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks

_SM120 = AcceleratorTarget("cuda", "sm120")


def test_shared_plan_does_not_impose_nvidia_warp_limits():
    shared = LinearExecutionPlan(
        fuse_rotation_quantization=True,
        fused_num_warps=32,
        rotation_num_warps=4,
        quantization_num_warps=4,
        matmul_block_m=64,
        matmul_block_n=64,
        matmul_block_k=64,
    )
    assert shared.fused_num_warps == 32
    with pytest.raises(ValueError, match="fused preparation num_warps"):
        NvidiaExecutionPlan(**shared.as_dict())


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_shared_plan_rejects_invalid_launch_dimensions(value):
    with pytest.raises(ValueError, match="positive integer"):
        LinearExecutionPlan(
            fuse_rotation_quantization=True,
            fused_num_warps=value,
            rotation_num_warps=4,
            quantization_num_warps=4,
            matmul_block_m=64,
            matmul_block_n=64,
            matmul_block_k=64,
        )


@pytest.mark.parametrize("architecture", ["sm75", "sm80", "sm89", "sm90", "sm100", "sm121"])
@pytest.mark.parametrize("in_features", [512, 5_376, 14_336, 28_672, 49_152, 49_408])
def test_supported_nvidia_targets_keep_the_existing_schedule(architecture, in_features):
    target = AcceleratorTarget("cuda", architecture)

    assert select_execution_plan(target, in_features=in_features) == select_execution_plan(
        _SM120, in_features=in_features
    )


@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("cpu"),
        AcceleratorTarget("meta"),
        AcceleratorTarget("hip", "gfx1201"),
        AcceleratorTarget("hip", "gfx942"),
        AcceleratorTarget("cuda", "sm70"),
        AcceleratorTarget("cuda"),
    ],
)
def test_execution_planning_rejects_unsupported_targets(target):
    with pytest.raises(ValueError, match="no optimized policy"):
        select_execution_plan(target, in_features=512)


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
    plan = select_execution_plan(_SM120, in_features=in_features)

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
    assert fused_preparation_chunks(in_features) == expected_chunks


def test_execution_plan_uses_uniform_mid_size_fused_launch_schedule() -> None:
    plans = [
        select_execution_plan(_SM120, in_features=in_features) for in_features in (7168, 14_336)
    ]

    assert {plan.fused_num_warps for plan in plans} == {4}


def test_execution_plan_selects_uniform_matmul_schedule() -> None:
    plan = select_execution_plan(_SM120, in_features=512)

    assert plan.matmul_block_m == 128
    assert plan.matmul_block_n == 256
    assert plan.matmul_block_k == 128
    assert plan.matmul_num_warps == 8
    assert plan.matmul_num_stages == 3


@pytest.mark.parametrize(
    ("rows", "k", "n", "block_m"),
    [
        (1, 5376, 14336, 32),
        (32, 5376, 14336, 32),
        (33, 5376, 14336, 64),
        (128, 1024, 2048, 32),
        (128, 1024, 3072, 64),
        (256, 1024, 1024, 32),
        (257, 1024, 1024, 64),
        (512, 1024, 1024, 64),
        (512, 5376, 14336, 128),
        (2048, 3072, 1024, 64),
        (2049, 3072, 1024, 64),
        (2303, 3072, 1024, 64),
        (2304, 3072, 1024, 128),
        (1025, 1024, 2048, 64),
        (2560, 3072, 1024, 128),
        (2048, 1024, 3072, 128),
        (3072, 1024, 3072, 128),
        (128, 96, 5376, 64),
        (512, 96, 5376, 128),
        (16384, 5376, 96, 128),
        (32768, 5376, 96, 128),
        (100000, 96, 5376, 128),
        (4096, 2048, 16, 32),
        (4097, 2048, 16, 64),
        (100000, 2048, 16, 64),
        (100000, 5376, 64, 64),
        (100000, 5376, 65, 128),
    ],
)
def test_shape_aware_schedule_preserves_preparation(rows, k, n, block_m):
    previous = select_execution_plan(_SM120, in_features=k)
    actual = select_execution_plan(_SM120, in_features=k, rows=rows, out_features=n)
    assert actual.matmul_block_m == block_m
    assert (
        replace(
            actual,
            matmul_block_m=previous.matmul_block_m,
            matmul_block_n=previous.matmul_block_n,
            matmul_num_warps=previous.matmul_num_warps,
            matmul_num_stages=previous.matmul_num_stages,
        )
        == previous
    )


@pytest.mark.parametrize("out_features", [16, 64, 65, 257, 1024, 3072, 5376, 14336])
def test_shape_aware_schedule_is_monotonic_in_rows(out_features):
    blocks = [
        select_execution_plan(
            _SM120,
            in_features=2048,
            out_features=out_features,
            rows=rows,
        ).matmul_block_m
        for rows in (*range(1, 4097), 8192, 16384, 32768, 100000)
    ]
    assert blocks == sorted(blocks)
    assert set(blocks) <= {32, 64, 128}


@pytest.mark.parametrize("architecture", ["sm75", "sm80", "sm89", "sm90", "sm100", "sm121"])
def test_shape_aware_schedule_keeps_unmeasured_targets_unchanged(architecture):
    target = AcceleratorTarget("cuda", architecture)
    assert select_execution_plan(
        target, in_features=3072, rows=128, out_features=2048
    ) == select_execution_plan(target, in_features=3072)


@pytest.mark.parametrize(("rows", "out_features"), [(None, None), (128, 2048)])
def test_architecture_policy_owns_preparation_and_matmul(monkeypatch, rows, out_features):
    base = select_execution_plan(AcceleratorTarget("cuda", "sm89"), in_features=5376)
    expected = replace(
        base,
        fuse_rotation_quantization=False,
        fused_num_warps=16,
        rotation_num_warps=8,
        quantization_num_warps=2,
        matmul_block_m=16,
        matmul_block_n=128,
        matmul_block_k=64,
        matmul_num_warps=4,
        matmul_num_stages=2,
    )
    architecture_policy = Mock(return_value=expected)
    monkeypatch.setattr(nvidia_policy, "_sm120_execution_plan", architecture_policy)

    actual = select_execution_plan(_SM120, in_features=5376, rows=rows, out_features=out_features)

    assert actual is expected
    architecture_policy.assert_called_once_with(
        in_features=5376, rows=rows, out_features=out_features
    )


def test_execution_plan_serializes_flat_tuning_fields() -> None:
    plan = select_execution_plan(_SM120, in_features=512)

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
    plan = NvidiaExecutionPlan(
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
    plan = select_execution_plan(_SM120, in_features=512)

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
    plan = select_execution_plan(_SM120, in_features=512)

    with pytest.raises(ValueError, match="ConvRot"):
        replace(plan, **changes)
