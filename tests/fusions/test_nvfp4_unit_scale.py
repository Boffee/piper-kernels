"""Unit activation metadata must retain the explicit-one FFN numerics."""

from dataclasses import replace
from importlib import import_module

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
def test_chunked_ffn_implicit_unit_scale_matches_explicit_ones(rotated, activation):
    family = f"{'convrot_' if rotated else ''}nvfp4_{activation}_ffn"
    helpers = import_module(f".{family}._helpers", package=__package__)
    backend = import_module(f"piper_kernels.fusions.{family}.triton")
    operation = getattr(backend, f"_chunked_{activation}_ffn_op")
    modes = (
        {"up_dynamic": False, "down_dynamic": False} if activation == "gelu" else {"dynamic": False}
    )
    operands = helpers.make_operands(rows=385, **modes)
    names = ("up", "down") if activation == "gelu" else ("gate", "value", "down")

    def with_scale(scale):
        return replace(
            operands,
            **{name: replace(getattr(operands, name), activation_scale=scale) for name in names},
        )

    explicit = with_scale(torch.ones((), device="cuda"))
    implicit = with_scale(None)
    expected = operation(*explicit.arguments(128))
    actual = operation(*implicit.arguments(128))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
