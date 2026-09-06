"""NVIDIA policy compatibility surface for existing ConvRot INT8 tooling."""

from ._nvidia.policy import select_execution_plan
from ._plan import LinearExecutionPlan

__all__ = ["LinearExecutionPlan", "select_execution_plan"]
