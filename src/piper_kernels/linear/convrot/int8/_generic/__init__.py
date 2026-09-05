"""Generic preparation and weight updates; tuned linear backends are optional."""

from .dispatch import add_, addmm_, prepare_input

__all__ = ["add_", "addmm_", "prepare_input"]
