"""Reusable Triton rotation primitives shared by ConvRot storage formats."""

# Triton's JIT launcher accepts compile-time options outside its Python signature.
# pyright: reportCallIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels.linear._triton_input_activations import (
    gelu_tanh,
    swiglu,
)


@triton.jit
def rotate_hadamard_stage(values, block_size: tl.constexpr, stride: tl.constexpr):
    """Apply one H4 factor with eight additions per independent quartet."""
    outer: tl.constexpr = block_size // (4 * stride)
    grouped = tl.reshape(values, (outer, 4, stride))
    quartets = tl.permute(grouped, (0, 2, 1))
    paired = tl.reshape(quartets, (outer, stride, 2, 2))
    ac, bd = tl.split(paired)
    a, c = tl.split(ac)
    b, d = tl.split(bd)
    p = a + b
    q = a - b
    r = c + d
    s = c - d
    y0 = p + s
    y1 = p - s
    y2 = q + r
    y3 = r - q
    y02 = tl.join(y0, y2)
    y13 = tl.join(y1, y3)
    transformed = tl.reshape(tl.join(y02, y13), (outer, stride, 4))
    transformed = tl.permute(transformed, (0, 2, 1))
    return tl.reshape(transformed, (block_size,))


@triton.jit
def rotate_hadamard_groups(
    values,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
):
    """Apply every H4 factor within independent ConvRot groups."""
    values = rotate_hadamard_stage(values, block_size, 1)
    values = rotate_hadamard_stage(values, block_size, 4)
    if group_size >= 64:
        values = rotate_hadamard_stage(values, block_size, 16)
    if group_size >= 256:
        values = rotate_hadamard_stage(values, block_size, 64)
    return values


@triton.jit
def load_activated_rotated_chunk(
    input_ptr,
    input_row_offset,
    row_width,
    chunk_start: tl.constexpr,
    chunk_offsets,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
):
    """Load, optionally activate, and rotate one group-aligned row slice."""
    offsets = chunk_start + chunk_offsets
    mask = offsets < row_width
    if activation_fn == "swiglu":
        up = tl.load(
            input_ptr + input_row_offset + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            input_ptr + input_row_offset + row_width + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = up
    else:
        values = tl.load(
            input_ptr + input_row_offset + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = values

    if activation_fn == "swiglu":
        values = swiglu(values, gate, logical_dtype_code)
    elif activation_fn == "gelu_tanh":
        values = gelu_tanh(values, logical_dtype_code, accelerator_backend)

    values = rotate_hadamard_groups(values, chunk_size, group_size)
    values *= inverse_sqrt_group
    if logical_dtype_code == 1:
        values = values.to(tl.float16)
    elif logical_dtype_code == 2:
        values = values.to(tl.bfloat16)
    return values


@triton.jit
def rotate_groups_kernel(
    input_ptr,
    output_ptr,
    row_width,
    groups_per_row,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
):
    group_id = tl.program_id(0)
    row = group_id // groups_per_row
    group = group_id % groups_per_row
    offsets = tl.arange(0, group_size)
    row_offset = row.to(tl.int64) * row_width
    pointers = input_ptr + row_offset + group * group_size + offsets
    values = tl.load(pointers).to(tl.float32)
    values = rotate_hadamard_groups(values, group_size, group_size)
    tl.store(output_ptr + row_offset + group * group_size + offsets, values * inverse_sqrt_group)


def logical_dtype_code(dtype: torch.dtype) -> int:
    """Encode a logical floating dtype for shared Triton activation helpers."""
    if dtype is torch.float16:
        return 1
    if dtype is torch.bfloat16:
        return 2
    return 0


def rotate_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    rotated: torch.Tensor,
    group_size: int,
    *,
    num_warps: int,
) -> None:
    """Materialize a grouped ConvRot transform into caller-owned storage."""
    rows, features = input.shape
    groups_per_row = features // group_size
    rotate_groups_kernel[(rows * groups_per_row,)](
        input,
        rotated,
        features,
        groups_per_row,
        group_size=group_size,
        inverse_sqrt_group=group_size**-0.5,
        num_warps=num_warps,
    )


__all__ = [
    "load_activated_rotated_chunk",
    "logical_dtype_code",
    "rotate_groups_kernel",
    "rotate_hadamard_groups",
    "rotate_hadamard_stage",
    "rotate_input",
]
