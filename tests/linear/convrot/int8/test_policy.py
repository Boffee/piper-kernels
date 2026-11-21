"""Host-side execution-plan policy tests for INT8 ConvRot."""

from dataclasses import fields, replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._nvidia import policy as nvidia_policy
from piper_kernels.linear.convrot.int8._nvidia.policy import (
    NvidiaExecutionPlan,
    select_execution_plan,
)
from piper_kernels.linear.convrot.int8._plan import LinearExecutionPlan
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks

_SM120 = AcceleratorTarget("cuda", "sm120")
_SM8X_ARCHITECTURES = ["sm80", "sm86", "sm87", "sm89"]
_PREPARATION_FIELDS = (
    "fuse_rotation_quantization",
    "fused_num_warps",
    "rotation_num_warps",
    "quantization_num_warps",
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _assert_valid_nvidia_schedule(plan: NvidiaExecutionPlan) -> None:
    assert _is_power_of_two(plan.matmul_block_m)
    assert _is_power_of_two(plan.matmul_block_n)
    assert _is_power_of_two(plan.matmul_block_k)
    assert plan.matmul_num_warps in (2, 4, 8)
    assert 1 <= plan.matmul_num_stages <= 4
    assert plan.fused_num_warps in (1, 2, 4, 8, 16)
    assert plan.rotation_num_warps in (1, 2, 4, 8)
    assert plan.quantization_num_warps in (1, 2, 4, 8)


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


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 4.0])
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


@pytest.mark.parametrize("architecture", ["sm75", "sm90", "sm100", "sm121"])
@pytest.mark.parametrize("in_features", [512, 5_376, 14_336, 28_672, 49_152, 49_408])
def test_supported_nvidia_targets_keep_the_existing_schedule(architecture, in_features):
    target = AcceleratorTarget("cuda", architecture)

    assert select_execution_plan(
        target, in_features=in_features
    ) == nvidia_policy.baseline_execution_plan(in_features=in_features)


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
    "in_features",
    [16, 128, 512, 4_096, 4_097, 5_376, 8_192, 8_193, 14_336, 16_640, 28_672, 49_152, 49_153],
)
def test_fused_preparation_chunking_and_fusion_extent_contracts(in_features: int) -> None:
    chunks = fused_preparation_chunks(in_features)
    plan = select_execution_plan(_SM120, in_features=in_features)
    _assert_valid_nvidia_schedule(plan)

    if in_features > 49_152:
        assert chunks is None
        assert not plan.fuse_rotation_quantization
        return

    assert chunks is not None
    chunk_count, chunk_size = chunks
    assert chunk_count in (1, 2, 3)
    assert _is_power_of_two(chunk_size)
    assert 128 <= chunk_size <= 16_384
    if chunk_count == 1:
        assert chunk_size <= 8_192
    assert chunk_count * chunk_size >= in_features
    assert (chunk_count - 1) * chunk_size < in_features
    assert plan.fuse_rotation_quantization is True


def test_shape_free_defaults_select_expected_kernel_families() -> None:
    sm120 = select_execution_plan(_SM120, in_features=512)
    baseline = nvidia_policy.baseline_execution_plan(in_features=512)
    _assert_valid_nvidia_schedule(sm120)
    _assert_valid_nvidia_schedule(baseline)

    assert sm120.matmul_kernel == "gluon_tma"
    assert sm120.matmul_specialize_m is False
    assert sm120.matmul_explicit_bias_fma is True
    assert baseline.matmul_kernel == "triton"


def test_sm120_shape_aware_regimes_preserve_preparation() -> None:
    for k in (512, 5_376, 14_336):
        default = select_execution_plan(_SM120, in_features=k)
        small = select_execution_plan(_SM120, in_features=k, rows=16, out_features=1024)
        narrow = select_execution_plan(_SM120, in_features=k, rows=100_000, out_features=64)
        large = select_execution_plan(_SM120, in_features=k, rows=8_192, out_features=4_096)

        for plan in (small, narrow, large):
            _assert_valid_nvidia_schedule(plan)
            for field_name in _PREPARATION_FIELDS:
                assert getattr(plan, field_name) == getattr(default, field_name)

        assert small.matmul_kernel == "triton"
        assert narrow.matmul_kernel == "triton"
        assert large.matmul_kernel == "gluon_tma"
        assert small.matmul_block_m < large.matmul_block_m
        assert narrow.matmul_block_m < large.matmul_block_m


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
    assert all(_is_power_of_two(block) for block in blocks)
    assert len(set(blocks)) <= 3


@pytest.mark.parametrize("architecture", ["sm75", "sm90", "sm100", "sm121"])
def test_shape_aware_schedule_keeps_unmeasured_targets_unchanged(architecture):
    target = AcceleratorTarget("cuda", architecture)
    assert select_execution_plan(
        target, in_features=3072, rows=128, out_features=2048
    ) == select_execution_plan(target, in_features=3072)


@pytest.mark.parametrize("architecture", ["sm120", "sm86"])
@pytest.mark.parametrize(("rows", "out_features"), [(None, None), (128, 2048), (8192, 4096)])
def test_matmul_selection_is_independent_of_preparation_width(architecture, rows, out_features):
    target = AcceleratorTarget("cuda", architecture)
    plans = [
        select_execution_plan(target, in_features=k, rows=rows, out_features=out_features)
        for k in (512, 5376, 14336, 28672, 49408)
    ]
    matmul_choices = [
        {name: value for name, value in plan.as_dict().items() if name.startswith("matmul_")}
        for plan in plans
    ]
    assert all(choices == matmul_choices[0] for choices in matmul_choices)
    assert len({plan.fused_num_warps for plan in plans}) > 1
    assert {plan.fuse_rotation_quantization for plan in plans} == {True, False}


@pytest.mark.parametrize("architecture", _SM8X_ARCHITECTURES)
def test_sm8x_regimes_never_use_tma_and_preserve_preparation(architecture):
    target = AcceleratorTarget("cuda", architecture)
    sm120_prep = select_execution_plan(_SM120, in_features=5376)

    small = select_execution_plan(target, in_features=5376, rows=16, out_features=1024)
    narrow = select_execution_plan(target, in_features=5376, rows=100_000, out_features=128)
    medium_wide = select_execution_plan(target, in_features=5376, rows=4096, out_features=1024)
    large_wide = select_execution_plan(target, in_features=5376, rows=4096, out_features=16384)
    shape_free = select_execution_plan(target, in_features=5376)

    for plan in (small, narrow, medium_wide, large_wide, shape_free):
        assert isinstance(plan, NvidiaExecutionPlan)
        _assert_valid_nvidia_schedule(plan)
        assert plan.matmul_kernel != "gluon_tma"
        assert plan.matmul_specialize_m is False
        assert plan.matmul_explicit_bias_fma is True
        for field_name in _PREPARATION_FIELDS:
            assert getattr(plan, field_name) == getattr(sm120_prep, field_name)

    assert small.matmul_kernel == "triton"
    assert narrow.matmul_kernel == "triton"
    assert medium_wide.matmul_kernel == "gluon_async_copy"
    assert large_wide.matmul_kernel == "gluon_async_copy"
    assert shape_free.matmul_kernel == "gluon_async_copy"
    assert shape_free.matmul_group_m > 0
    assert small.matmul_block_m < medium_wide.matmul_block_m <= large_wide.matmul_block_m


def test_all_sm8x_architectures_share_the_same_schedule():
    shapes = [(None, None), (16, 1024), (256, 2048), (4096, 128), (32768, 16384)]
    for rows, out_features in shapes:
        plans = [
            select_execution_plan(
                AcceleratorTarget("cuda", arch),
                in_features=5376,
                rows=rows,
                out_features=out_features,
            )
            for arch in _SM8X_ARCHITECTURES
        ]
        assert all(plan == plans[0] for plan in plans)


@pytest.mark.parametrize(
    "out_features", [16, 64, 65, 96, 128, 129, 256, 1024, 1025, 3072, 5376, 16384, 65536]
)
def test_sm8x_layer_steps_through_at_most_three_tiles_as_rows_grow(out_features):
    # Each distinct schedule is one compiled GEMM because SM8x launches do not specialize on M.
    target = AcceleratorTarget("cuda", "sm89")
    rows = [*range(1, 9000), *(2**power + d for power in range(14, 21) for d in (-1, 0, 1))]
    schedules = [
        select_execution_plan(target, in_features=2048, rows=row, out_features=out_features)
        for row in rows
    ]
    block_m = [plan.matmul_block_m for plan in schedules]
    assert block_m == sorted(block_m)
    assert len(set(schedules)) <= 3


@pytest.mark.parametrize("architecture", _SM8X_ARCHITECTURES)
@pytest.mark.parametrize("in_features", [16, 512, 1_024, 2_048, 5_376, 14_336, 28_672, 49_408])
def test_sm8x_preparation_depends_only_on_in_features(architecture, in_features):
    target = AcceleratorTarget("cuda", architecture)
    sm120 = select_execution_plan(_SM120, in_features=in_features)
    plans = [
        select_execution_plan(target, in_features=in_features, rows=rows, out_features=n)
        for rows, n in ((None, None), (1, 64), (256, 1024), (8192, 4096))
    ]
    for plan in plans:
        _assert_valid_nvidia_schedule(plan)
        assert plan.fused_num_warps == plans[0].fused_num_warps
        assert plan.fuse_rotation_quantization is sm120.fuse_rotation_quantization
        assert plan.rotation_num_warps == sm120.rotation_num_warps
        assert plan.quantization_num_warps == sm120.quantization_num_warps
    if in_features <= 1_024:
        assert plans[0].fused_num_warps == 1
    else:
        assert plans[0].fused_num_warps == sm120.fused_num_warps


@pytest.mark.parametrize("architecture", ["sm89", "sm120"])
def test_one_warp_preparation_is_an_explicit_nvidia_option(architecture):
    plan = select_execution_plan(AcceleratorTarget("cuda", architecture), in_features=512)
    assert replace(plan, fused_num_warps=1).fused_num_warps == 1
    with pytest.raises(ValueError, match="must be 1, 2, 4, 8, or 16"):
        replace(plan, fused_num_warps=3)


@pytest.mark.parametrize(
    "changes",
    [
        {"matmul_block_k": 128},
        {"matmul_block_n": 64},
        {"matmul_num_warps": 4},
        {"matmul_num_stages": 2},
    ],
)
def test_sm8x_plan_rejects_unsupported_kernel_tiles(changes):
    # The shape-free SM8x plan uses the Gluon async-copy kernel with fixed tile contracts.
    plan = select_execution_plan(AcceleratorTarget("cuda", "sm89"), in_features=512)

    with pytest.raises(ValueError, match="ConvRot async-copy"):
        replace(plan, **changes)


@pytest.mark.parametrize("group_m", [-1, 1, 4, 32, True, False, 8.0, 16.0])
def test_sm8x_plan_rejects_invalid_group_m(group_m):
    plan = select_execution_plan(AcceleratorTarget("cuda", "sm89"), in_features=512)

    with pytest.raises(ValueError, match="group_m"):
        replace(plan, matmul_group_m=group_m)


def test_execution_plan_serializes_flat_tuning_fields() -> None:
    plan = select_execution_plan(_SM120, in_features=512)
    baseline = nvidia_policy.baseline_execution_plan(in_features=512)

    for candidate in (plan, baseline):
        serialized = candidate.as_dict()
        assert "preparation" not in serialized
        assert all(not isinstance(value, dict) for value in serialized.values())
        for field_info in fields(LinearExecutionPlan):
            assert serialized[field_info.name] == getattr(candidate, field_info.name)
        assert serialized["matmul_kernel"] == candidate.matmul_kernel
        assert serialized["matmul_group_m"] == candidate.matmul_group_m
        assert serialized["matmul_specialize_m"] is candidate.matmul_specialize_m
        assert serialized["matmul_explicit_bias_fma"] is candidate.matmul_explicit_bias_fma
    assert plan.as_dict()["matmul_kernel"] == "gluon_tma"
    assert baseline.as_dict()["matmul_kernel"] == "triton"


@pytest.mark.parametrize(
    "changes",
    [
        {"matmul_block_m": 8},
        {"matmul_block_m": 256},
        {"matmul_block_n": 512},
        {"matmul_block_n": 128.0},
        {"matmul_block_k": 16},
        {"matmul_num_warps": 16},
        {"matmul_num_warps": True},
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
        replace(plan, fused_num_warps=3)


def test_kernel_selection_is_independent_of_tile_replacement() -> None:
    baseline = nvidia_policy.baseline_execution_plan(in_features=512)
    triton = replace(
        baseline,
        matmul_block_m=128,
        matmul_block_n=128,
        matmul_block_k=64,
        matmul_num_warps=4,
        matmul_num_stages=3,
    )
    assert triton.matmul_kernel == "triton"
    gluon = replace(triton, matmul_kernel="gluon_async_copy")
    assert replace(gluon, matmul_num_stages=4).matmul_kernel == "gluon_async_copy"
    with pytest.raises(ValueError, match="async-copy GEMM requires"):
        replace(gluon, matmul_block_n=64)

    tma = select_execution_plan(_SM120, in_features=512)
    assert replace(tma, matmul_group_m=8).matmul_kernel == "gluon_tma"
    for invalid in (
        {"matmul_block_m": 256},
        {"matmul_block_n": 128},
        {"matmul_block_k": 128},
        {"matmul_num_warps": 4},
        {"matmul_num_stages": 4},
    ):
        with pytest.raises(ValueError, match="TMA GEMM requires"):
            replace(tma, **invalid)


@pytest.mark.parametrize("kernel", ["gluon_async_copy", "gluon_tma"])
@pytest.mark.parametrize("specialize_m", [False, True])
@pytest.mark.parametrize("explicit_bias_fma", [False, True])
def test_gluon_reports_effective_behavior_and_preserves_triton_options(
    kernel, specialize_m, explicit_bias_fma
) -> None:
    tile = (
        {"matmul_block_n": 128, "matmul_block_k": 64, "matmul_num_warps": 4}
        if kernel == "gluon_async_copy"
        else {"matmul_block_k": 64}
    )
    triton = replace(
        nvidia_policy.baseline_execution_plan(in_features=512),
        **tile,
        triton_specialize_m=specialize_m,
        triton_explicit_bias_fma=explicit_bias_fma,
    )
    gluon = replace(triton, matmul_kernel=kernel)

    assert gluon.matmul_specialize_m is False
    assert gluon.matmul_explicit_bias_fma is True
    assert gluon.as_dict() == triton.as_dict() | {
        "matmul_kernel": kernel,
        "matmul_specialize_m": False,
        "matmul_explicit_bias_fma": True,
    }
    assert replace(gluon, matmul_kernel="triton") == triton
    fallback = nvidia_policy.grouped_triton_plan(gluon)
    assert fallback.matmul_kernel == "triton"
    assert fallback.matmul_specialize_m is False
    assert fallback.matmul_explicit_bias_fma is True


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("architecture", ["sm89", "sm90", "sm120", "sm121"])
@pytest.mark.parametrize("kernel", ["triton", "gluon_async_copy", "gluon_tma"])
def test_supports_residual_epilogue_covers_sm120_triton_and_tma(
    dtype, architecture, kernel
) -> None:
    base = nvidia_policy.baseline_execution_plan(in_features=512)
    if kernel == "gluon_async_copy":
        plan = replace(
            base,
            matmul_kernel=kernel,
            matmul_block_n=128,
            matmul_block_k=64,
            matmul_num_warps=4,
        )
    elif kernel == "gluon_tma":
        plan = select_execution_plan(_SM120, in_features=512)
    else:
        plan = base
    expected = (
        architecture == "sm120"
        and kernel in ("triton", "gluon_tma")
        and dtype in (torch.float16, torch.bfloat16)
    )
    assert (
        nvidia_policy.supports_residual_epilogue(
            AcceleratorTarget("cuda", architecture), plan, dtype
        )
        is expected
    )


@pytest.mark.parametrize(
    "changes",
    [{"matmul_kernel": "unknown"}, {"triton_specialize_m": 1}, {"triton_explicit_bias_fma": 0}],
)
def test_execution_plan_rejects_invalid_implementation_options(changes) -> None:
    plan = select_execution_plan(_SM120, in_features=512)
    with pytest.raises(ValueError, match="ConvRot"):
        replace(plan, **changes)


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
