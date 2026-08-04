"""Shared Triton primitives for rotating attention head channels."""

# ruff: noqa: ANN001, ANN202

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl


@triton.jit
def hadamard_stage(values, offsets, stride: tl.constexpr):
    """Apply one H4 Kronecker factor to a flattened row."""
    digit = (offsets // stride) % 4
    base = offsets - digit * stride
    a = tl.gather(values, base, 0)
    b = tl.gather(values, base + stride, 0)
    c = tl.gather(values, base + 2 * stride, 0)
    d = tl.gather(values, base + 3 * stride, 0)
    return tl.where(
        digit == 0,
        a + b + c - d,
        tl.where(
            digit == 1,
            a + b - c + d,
            tl.where(digit == 2, a - b + c + d, -a + b + c + d),
        ),
    )


@triton.jit
def hadamard_stage_rows(values, offsets, stride: tl.constexpr, block_m: tl.constexpr):
    """Apply one H4 Kronecker factor to several rows in registers."""
    digit = (offsets // stride) % 4
    base = offsets - digit * stride
    row_broadcast = tl.zeros((block_m, 1), tl.int32)
    a = tl.gather(values, row_broadcast + base[None, :], 1)
    b = tl.gather(values, row_broadcast + base[None, :] + stride, 1)
    c = tl.gather(values, row_broadcast + base[None, :] + 2 * stride, 1)
    d = tl.gather(values, row_broadcast + base[None, :] + 3 * stride, 1)
    return tl.where(
        digit[None, :] == 0,
        a + b + c - d,
        tl.where(
            digit[None, :] == 1,
            a + b - c + d,
            tl.where(digit[None, :] == 2, a - b + c + d, -a + b + c + d),
        ),
    )


@triton.jit
def rotate_rows_in_registers(
    values,
    offsets,
    block_m: tl.constexpr,
    rotation_group: tl.constexpr,
):
    """Apply a normalized block-Hadamard rotation to rows already in registers."""
    if rotation_group != 0:
        values = hadamard_stage_rows(values, offsets, 1, block_m)
        values = hadamard_stage_rows(values, offsets, 4, block_m)
        if rotation_group >= 64:
            values = hadamard_stage_rows(values, offsets, 16, block_m)
        if rotation_group >= 256:
            values = hadamard_stage_rows(values, offsets, 64, block_m)
        values *= rotation_group**-0.5
    return values


@triton.jit
def _rotate_rows_kernel(
    input_ptr,
    output_ptr,
    head_dim: tl.constexpr,
    rotation_group: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, head_dim)
    values = tl.load(input_ptr + row * head_dim + offsets).to(tl.float32)
    values = hadamard_stage(values, offsets, 1)
    values = hadamard_stage(values, offsets, 4)
    if rotation_group >= 64:
        values = hadamard_stage(values, offsets, 16)
    if rotation_group >= 256:
        values = hadamard_stage(values, offsets, 64)
    tl.store(output_ptr + row * head_dim + offsets, values * rotation_group**-0.5)


def rotate_attention_rows(value: torch.Tensor, rotation_group: int) -> torch.Tensor:
    """Rotate groups along an attention tensor's contiguous head dimension."""
    head_dim = value.shape[-1]
    if rotation_group not in (16, 64, 256):
        raise ValueError(f"rotation group must be 16, 64, or 256, got {rotation_group}")
    if head_dim % rotation_group:
        raise ValueError(
            f"head dimension {head_dim} must be divisible by rotation group {rotation_group}"
        )
    contiguous = value.contiguous()
    rows = contiguous.numel() // head_dim
    output = torch.empty_like(contiguous)
    _rotate_rows_kernel[(rows,)](
        contiguous,
        output,
        head_dim=head_dim,
        rotation_group=rotation_group,
        num_warps=4,
    )
    return output
