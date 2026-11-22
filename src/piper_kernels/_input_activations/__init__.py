"""Portable activations that a linear backend can absorb during input preparation."""

from typing import Literal, cast

import torch

type InputActivation = Literal["gelu_tanh", "swiglu"]

GELU_TANH_CUBIC_COEFFICIENT = 0.044715
GELU_TANH_SCALE_COEFFICIENT = 0.7978845608028654

_INPUT_ACTIVATIONS = ("gelu_tanh", "swiglu")
_SUPPORTED_INPUT_ACTIVATIONS = (None, *_INPUT_ACTIVATIONS)


def validate_input_activation(activation_fn: str | None) -> None:
    """Validate a portable input-activation name."""
    if activation_fn not in _SUPPORTED_INPUT_ACTIVATIONS:
        raise ValueError(
            "unsupported input activation "
            f"{activation_fn!r}; expected 'gelu_tanh', 'swiglu', or None"
        )


def parse_input_activation(activation_fn: str, *, name: str = "FFN") -> InputActivation:
    """Validate and narrow a required input-activation name."""
    if activation_fn not in _INPUT_ACTIVATIONS:
        raise ValueError(
            f"{name} activation_fn must be 'gelu_tanh' or 'swiglu', got {activation_fn!r}"
        )
    return cast(InputActivation, activation_fn)


def input_activation_width(activation_fn: str | None) -> int:
    """Return how many source columns produce one activated column."""
    validate_input_activation(activation_fn)
    return 2 if activation_fn == "swiglu" else 1


def apply_input_activation(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    activation_fn: str | None,
) -> torch.Tensor:
    """Apply a supported activation with FP32 intermediate arithmetic."""
    validate_input_activation(activation_fn)
    if activation_fn == "gelu_tanh":
        return torch.nn.functional.gelu(input.float(), approximate="tanh")
    if activation_fn == "swiglu":
        up, gate = input.float().chunk(2, dim=-1)
        return up * torch.nn.functional.silu(gate)
    return input


__all__ = [
    "GELU_TANH_CUBIC_COEFFICIENT",
    "GELU_TANH_SCALE_COEFFICIENT",
    "InputActivation",
    "apply_input_activation",
    "input_activation_width",
    "parse_input_activation",
    "validate_input_activation",
]
