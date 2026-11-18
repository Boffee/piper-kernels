"""Use unfilled output rows to pack long RDNA4 SwiGLU projections."""

# Triton launch options are accepted by the JIT launcher.
# pyright: reportCallIssue=false

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._amd.policy import matmul_accumulator_seed
from piper_kernels.linear.convrot.int8._amd.triton import _amd_matmul_compiler_options
from piper_kernels.linear.convrot.int8._kernels.triton import (
    _row_scaled_int8_matmul,
    _weight_offsets,
)

from .policy import PACKED_CHUNK_ROWS

if TYPE_CHECKING:
    from .._core import LinearOperands

# Each weight row contributes 32 adjacent K bytes to a packed panel.
_PACK_K = 32


@triton.jit
def _pack_weight_kernel(source, packed, n: tl.constexpr, k: tl.constexpr, pack_k: tl.constexpr):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    cols = tl.program_id(1) * 512 + tl.arange(0, 512)
    values = tl.load(source + rows[:, None] * k + cols[None, :], cols[None, :] < k, 0)
    offsets = tl.trans(_weight_offsets(rows, cols, n, k, pack_k))
    tl.store(packed + offsets, values, cols[None, :] < k)


@triton.jit
def _paired_projection(
    q,
    value,
    gate,
    rows,
    cols,
    n: tl.constexpr,
    k: tl.constexpr,
    pack_k: tl.constexpr,
    seed: tl.constexpr,
):
    # Two 128-column accumulators share each activation load and its WMMA layout.
    reduction = tl.arange(0, 64)
    activation = q + rows[:, None] * k + reduction[None, :]
    offsets = _weight_offsets(cols, reduction, n, k, pack_k)
    value_ptr = value + offsets
    gate_ptr = gate + offsets
    value_acc = tl.full((128, 128), seed, tl.int32)
    gate_acc = tl.full((128, 128), seed, tl.int32)
    for _ in range(k // 64):
        a = tl.load(activation)
        v = tl.load(value_ptr)
        g = tl.load(gate_ptr)
        value_acc = tl.dot(a, v, value_acc)
        gate_acc = tl.dot(a, g, gate_acc)
        activation += 64
        value_ptr += 64 * n if pack_k else 64
        gate_ptr += 64 * n if pack_k else 64
    return (value_acc - seed).to(tl.float32), (gate_acc - seed).to(tl.float32)


@triton.jit
def _project_kernel(
    q,
    weight,
    output,
    row_scale,
    weight_scale,
    gate,
    gate_scale,
    m,
    n: tl.constexpr,
    k: tl.constexpr,
    paired: tl.constexpr,
    pack_k: tl.constexpr,
    full_rows: tl.constexpr,
    chunk_tiles: tl.constexpr,
    seed: tl.constexpr,
):
    pid = tl.program_id(0)
    # Specialize the paired grid for complete chunks. Tail lengths share one kernel.
    row_tiles = tl.constexpr(chunk_tiles) if chunk_tiles else tl.cdiv(m, 128)
    block_n: tl.constexpr = tl.constexpr(128 if paired else 256)
    col_tiles: tl.constexpr = n // block_n
    group_n: tl.constexpr = tl.constexpr(4 if paired else 2)
    first_col = pid // (row_tiles * group_n) * group_n
    cols_in_group = tl.minimum(group_n, col_tiles - first_col)
    local = pid % (row_tiles * group_n)
    rows = local // cols_in_group * 128 + tl.arange(0, 128)
    cols = (first_col + local % cols_in_group) * block_n + tl.arange(0, block_n)
    # Repeat the last valid input row on the tail. Stores mask those lanes.
    source_rows = rows if full_rows else tl.minimum(rows, m - 1)
    dest = output + rows[:, None] * (n * (2 if paired else 1)) + cols[None, :]
    if paired:
        value, gate_value = _paired_projection(
            q, weight, gate, source_rows, cols, n, k, pack_k, seed
        )
        scale = tl.load(row_scale + source_rows)
        result = value * scale[:, None] * tl.load(weight_scale + cols)[None, :]
        gate_result = gate_value * scale[:, None] * tl.load(gate_scale + cols)[None, :]
        tl.store(dest + n, gate_result, mask=True if full_rows else rows[:, None] < m)
    else:
        scaled, scales = _row_scaled_int8_matmul(
            q,
            weight,
            row_scale,
            weight_scale,
            source_rows,
            cols,
            m,
            n,
            k,
            tl.constexpr(128),
            block_n,
            tl.constexpr(64),
            tl.constexpr(True),
            seed,
            pack_k,
        )
        result = scaled * scales[None, :]
    tl.store(dest, result, mask=True if full_rows else rows[:, None] < m)


@dataclass(frozen=True, slots=True)
class PackedProjections:
    """Hold value, gate and down weights in FFN order.

    Packed views borrow the output suffix until the first chunk that could
    overwrite it. Packing, all GEMMs and suffix writes use the current stream.
    """

    prefix_rows: int
    weights: tuple[torch.Tensor, ...]
    packed: tuple[torch.Tensor, ...]
    scales: tuple[torch.Tensor, ...]
    seeds: tuple[int, int]
    compiler_options: dict[str, object]

    def project(
        self,
        q: torch.Tensor,
        scales: torch.Tensor,
        output: torch.Tensor,
        start: int,
        *,
        paired: bool,
    ) -> None:
        packed = start < self.prefix_rows
        weights = self.packed if packed else self.weights
        index = 0 if paired else 2
        weight = weights[index]
        n, k = weight.shape
        m = q.shape[0]
        grid = (triton.cdiv(m, 128) * (n // (128 if paired else 256)),)
        with device_context(q.device):
            _project_kernel[grid](
                q,
                weight,
                output,
                scales,
                self.scales[index],
                weights[1],
                self.scales[1],
                m,
                n,
                k,
                paired,
                _PACK_K if packed else 0,
                m % 128 == 0,
                m // 128 if paired and m == PACKED_CHUNK_ROWS else 0,
                self.seeds[0 if paired else 1],
                num_warps=8,
                num_stages=2,
                **self.compiler_options,
            )


def pack_projections(
    output: torch.Tensor,
    sources: tuple[LinearOperands, ...],
    down: LinearOperands,
    prefix_rows: int,
    chunk_rows: int,
    target: AcceleratorTarget,
) -> PackedProjections:
    """Write the three panel-packed weights into disjoint slices past the output prefix."""
    n, k = sources[0].weight_qdata.shape
    out_features = output.shape[1]
    operands = (*sources, down)
    weights = tuple(linear.weight_qdata for linear in operands)
    scales = tuple(linear.weight_scale for linear in operands)
    sizes = [weight.numel() for weight in weights]
    seeds = (
        matmul_accumulator_seed(target, k, chunk_rows // 128 * (2 * n // 256)),
        matmul_accumulator_seed(target, n, chunk_rows // 128 * (out_features // 256)),
    )
    scratch = output.view(torch.int8).view(-1)[-sum(sizes) :]
    packed = tuple(view.view_as(w) for view, w in zip(scratch.split(sizes), weights, strict=True))
    with device_context(output.device):
        for src, dst in zip(weights, packed, strict=True):
            height, width = src.shape
            _pack_weight_kernel[(height // 8, triton.cdiv(width, 512))](
                src, dst, height, width, _PACK_K, num_warps=8
            )
    return PackedProjections(
        prefix_rows, weights, packed, scales, seeds, _amd_matmul_compiler_options(target)
    )
