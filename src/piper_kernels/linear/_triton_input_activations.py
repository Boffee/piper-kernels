"""Reusable Triton primitives selected by linear backends for fused input activation."""

# Triton JIT values do not have meaningful Python annotations.
# ruff: noqa: ANN001, ANN202

import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels.linear._input_activations import (
    GELU_TANH_CUBIC_COEFFICIENT,
    GELU_TANH_SCALE_COEFFICIENT,
)

_GELU_TANH_CUBIC_COEFFICIENT = tl.constexpr(GELU_TANH_CUBIC_COEFFICIENT)
_GELU_TANH_SCALE_COEFFICIENT = tl.constexpr(GELU_TANH_SCALE_COEFFICIENT)


@triton.jit
def _to_logical_dtype(values, logical_dtype_code: tl.constexpr):
    if logical_dtype_code == 1:
        return values.to(tl.float16).to(tl.float32)
    elif logical_dtype_code == 2:
        return values.to(tl.bfloat16).to(tl.float32)
    return values


@triton.jit
def _tanh_approx(values):
    return tl.inline_asm_elementwise(
        asm="tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[values],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def gelu_tanh(
    values,
    logical_dtype_code: tl.constexpr,
    accelerator_backend: tl.constexpr,
):
    """Apply tanh-approximate GELU using the selected accelerator implementation."""
    inner = _GELU_TANH_SCALE_COEFFICIENT * (
        values + _GELU_TANH_CUBIC_COEFFICIENT * values * values * values
    )
    if accelerator_backend == "cuda":  # noqa: SIM108 - discard target-specific assembly
        tanh_inner = _tanh_approx(inner)
    else:
        tanh_inner = libdevice.tanh(inner)
    return _to_logical_dtype(
        0.5 * values * (1.0 + tanh_inner),  # pyright: ignore[reportOperatorIssue]
        logical_dtype_code,
    )


@triton.jit
def swiglu(up, gate, logical_dtype_code: tl.constexpr):
    """Apply packed ``up * silu(gate)`` with logical-dtype rounding."""
    activated_gate = _to_logical_dtype(gate / (1.0 + tl.exp(-gate)), logical_dtype_code)
    return _to_logical_dtype(up * activated_gate, logical_dtype_code)
