"""Fused updates share schedules across accelerators and keep rotation groups whole."""

import pytest

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8._plan import AddmmExecutionPlan
from piper_kernels.weights.convrot.int8._policy import select_addmm_plan


@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("cuda", "sm75"),
        AcceleratorTarget("cuda"),
        AcceleratorTarget("cpu"),
    ],
)
def test_unsupported_dot_targets_keep_batched_updates(target):
    assert select_addmm_plan(target, 256) is None


@pytest.mark.parametrize(("group_size", "column_width"), [(16, 128), (64, 128), (256, 256)])
def test_fused_plans_are_shared_across_accelerators(group_size, column_width):
    targets = [
        *[AcceleratorTarget("cuda", arch) for arch in ("sm80", "sm89", "sm120")],
        *[AcceleratorTarget("hip", arch) for arch in ("gfx1030", "gfx1100", "gfx942", "gfx1201")],
    ]
    plans = [select_addmm_plan(target, group_size) for target in targets]
    plan = plans[0]
    assert plan is not None
    assert plan.block_n == column_width
    assert all(candidate == plan for candidate in plans)


def test_fused_plan_rejects_unsupported_column_width():
    with pytest.raises(ValueError, match="feature tiles"):
        AddmmExecutionPlan(block_n=64)
