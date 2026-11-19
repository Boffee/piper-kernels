"""SM120 up-to-three-stage TMA ring loads for D128 Q/K/V fused projections."""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from piper_kernels._triton.runtime import device_context

from ._plan import GLUON_BLOCK_K, GLUON_BLOCK_M, GLUON_NUM_STAGES, NvidiaExecutionPlan

_HEAD_DIM = 128
_TILE_SHAPE = [GLUON_BLOCK_M, GLUON_BLOCK_K]

_GL_BLOCK_M = gl.constexpr(GLUON_BLOCK_M)
_GL_BLOCK_K = gl.constexpr(GLUON_BLOCK_K)
_GL_HEAD_DIM = gl.constexpr(_HEAD_DIM)
_GL_MMA_K = gl.constexpr(32)
_GL_MAX_STAGES = gl.constexpr(GLUON_NUM_STAGES)


def build_matrix_descriptor(matrix: torch.Tensor) -> TensorDescriptor:
    """View a contiguous batched activation or head weight as 2D INT8 rows."""
    if not matrix.is_contiguous():
        raise ValueError("TMA projection operands must be contiguous")
    width = matrix.shape[-1]
    tile = list(_TILE_SHAPE)
    with device_context(matrix.device):
        return TensorDescriptor(
            matrix,
            [matrix.numel() // width, width],
            [width, 1],
            tile,
            gl.NVMMASharedLayout.get_default_for(tile, gl.int8),
        )


def resolve_operands(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    execution_plan: NvidiaExecutionPlan,
) -> tuple[torch.Tensor | TensorDescriptor, torch.Tensor | TensorDescriptor]:
    """Return TMA descriptors or raw tensor operands matching ``execution_plan.kernel``."""
    if execution_plan.kernel == "gluon_tma":
        return build_matrix_descriptor(input_qdata), build_matrix_descriptor(weight_qdata)
    if execution_plan.kernel == "gluon_async_copy":
        return input_qdata, weight_qdata
    raise ValueError(f"Unsupported Gluon projection kernel: {execution_plan.kernel}")


@gluon.jit
def _load_operands(
    inputs, weights, barrier, input_tile, weight_tile, first_row, head, feature_offset
):
    mbarrier.expect(barrier, inputs.block_type.nbytes + weights.block_type.nbytes)
    tma.async_load(inputs, [first_row, feature_offset], barrier, input_tile)
    tma.async_load(weights, [head * _GL_HEAD_DIM, feature_offset], barrier, weight_tile)


@gluon.jit
def _accumulate_tile(
    inputs,
    weights,
    input_ring,
    weight_ring,
    barriers,
    first_row,
    head,
    accumulator,
    group,
    slot: gl.constexpr,
    stages: gl.constexpr,
    refill: gl.constexpr,
):
    input_layout: gl.constexpr = gl.DotOperandLayout(0, accumulator.type.layout, k_width=4)
    weight_layout: gl.constexpr = gl.DotOperandLayout(1, accumulator.type.layout, k_width=4)
    barrier = barriers.index(slot)
    input_tile = input_ring.index(slot)
    weight_tile = weight_ring.index(slot)
    mbarrier.wait(barrier, phase=group & 1)
    for offset in gl.static_range(0, _GL_BLOCK_K, _GL_MMA_K):
        lhs = input_tile.slice(offset, _GL_MMA_K, dim=1).load(input_layout)
        rhs = weight_tile.slice(offset, _GL_MMA_K, dim=1).permute([1, 0]).load(weight_layout)
        accumulator = mma_v2(lhs, rhs, accumulator)
    gl.barrier()
    if refill:
        _load_operands(
            inputs,
            weights,
            barrier,
            input_tile,
            weight_tile,
            first_row,
            head,
            ((group + 1) * stages + slot) * _GL_BLOCK_K,
        )
    return accumulator


@gluon.jit
def project_int8(
    inputs,
    weights,
    first_row,
    head,
    input_features: gl.constexpr,
    mma_layout: gl.constexpr,
):
    """Project one 128-row tile onto one D128 head via TMA ring loads.

    Supports any ``input_features`` divisible by 64 using ``min(3, k_tiles)``
    shared-memory ring stages and exact ring-order tail refilling and draining.
    """
    gl.static_assert(input_features >= _GL_BLOCK_K and input_features % _GL_BLOCK_K == 0)
    num_k_tiles: gl.constexpr = input_features // _GL_BLOCK_K
    stages: gl.constexpr = _GL_MAX_STAGES if num_k_tiles >= _GL_MAX_STAGES else num_k_tiles
    refill_tiles: gl.constexpr = num_k_tiles - stages
    full_groups: gl.constexpr = refill_tiles // stages
    tail_refills: gl.constexpr = refill_tiles % stages

    input_ring = gl.allocate_shared_memory(
        gl.int8, [stages, _GL_BLOCK_M, _GL_BLOCK_K], inputs.layout
    )
    weight_ring = gl.allocate_shared_memory(
        gl.int8, [stages, _GL_HEAD_DIM, _GL_BLOCK_K], weights.layout
    )
    barriers = gl.allocate_shared_memory(gl.int64, [stages, 1], mbarrier.MBarrierLayout())
    for slot in gl.static_range(stages):
        mbarrier.init(barriers.index(slot), count=1)
    gl.barrier()
    for slot in gl.static_range(stages):
        _load_operands(
            inputs,
            weights,
            barriers.index(slot),
            input_ring.index(slot),
            weight_ring.index(slot),
            first_row,
            head,
            slot * _GL_BLOCK_K,
        )

    accumulator = gl.full([_GL_BLOCK_M, _GL_HEAD_DIM], 0, gl.int32, mma_layout)
    if full_groups > 0:
        for group in range(full_groups):
            for slot in gl.static_range(stages):
                accumulator = _accumulate_tile(
                    inputs,
                    weights,
                    input_ring,
                    weight_ring,
                    barriers,
                    first_row,
                    head,
                    accumulator,
                    group,
                    slot,
                    stages,
                    True,
                )
    for slot in gl.static_range(tail_refills):
        accumulator = _accumulate_tile(
            inputs,
            weights,
            input_ring,
            weight_ring,
            barriers,
            first_row,
            head,
            accumulator,
            full_groups,
            slot,
            stages,
            True,
        )
    for slot in gl.static_range(tail_refills, stages):
        accumulator = _accumulate_tile(
            inputs,
            weights,
            input_ring,
            weight_ring,
            barriers,
            first_row,
            head,
            accumulator,
            full_groups,
            slot,
            stages,
            False,
        )
        mbarrier.invalidate(barriers.index(slot))
    for slot in gl.static_range(tail_refills):
        accumulator = _accumulate_tile(
            inputs,
            weights,
            input_ring,
            weight_ring,
            barriers,
            first_row,
            head,
            accumulator,
            full_groups + 1,
            slot,
            stages,
            False,
        )
        mbarrier.invalidate(barriers.index(slot))
    return accumulator
