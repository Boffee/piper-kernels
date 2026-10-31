"""Select shared INT8 updates by instruction support and rotation-group size."""

from piper_kernels._triton.targets import AcceleratorTarget

from ._plan import AddmmExecutionPlan


def select_addmm_plan(
    target: AcceleratorTarget,
    group_size: int,
) -> AddmmExecutionPlan | None:
    """Share one conservative schedule across CUDA and HIP.

    NVIDIA needs SM80+ for BF16/TF32x3 dots. HIP uses IEEE FP32 dots and
    the compiler's native or emulated lowering for the input dtype. Small
    tiles bound live rotation and stochastic-rounding values across targets;
    only the column width changes to keep complete rotation groups together.
    """
    if not (target.cuda_capability_at_least(8) or target.is_amd_hip):
        return None
    return AddmmExecutionPlan(block_n=max(128, group_size))
