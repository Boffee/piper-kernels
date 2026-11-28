"""NVIDIA SM120 Gluon TMA ring, GEMM, and fused SwiGLU/residual epilogues for ConvRot INT8.

Loads row-major INT8 operands into swizzled shared memory through 2D TMA
descriptors across an up-to-three-stage mbarrier ring and accumulates exact INT32
products with m16n8k32 MMAv2 instructions. The standalone ``128x256`` linear GEMM
(``mma_k=64``, 8 warps), dual-accumulator ``128x128`` SwiGLU GEMM (``mma_k=32``,
8 warps), and fused D128 Q/K/V projections (``block_n=128``, ``mma_k=32``, 4 warps)
share :func:`build_matrix_descriptor` and the three-stage mbarrier ring pattern.

TMA launches require contiguous INT8 operands whose pointers and feature width
``K`` are positive 16-byte multiples (TMA hardware zero-fills out-of-bounds M, N,
and K). All linear kernels keep M unspecialized so one compiled binary serves
every row count. Accumulation is exact INT32, and dequantization preserves the
shared ``(acc * input_scale) * weight_scale`` + explicit FMA bias rounding.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from piper_kernels._residual import ResidualAdd
from piper_kernels._triton.runtime import device_context

from .gluon_async_copy import (
    _build_store_layout,
    _cta_tile_coordinates,
    _dequantize_paired_tile,
    _store_dequantized_tile,
    _store_swiglu_epilogue,
)
from .policy import NvidiaExecutionPlan

_BLOCK_M = 128
_BLOCK_N = 256
_BLOCK_K = 64
_MMA_K = 64
_SWIGLU_BLOCK_N = 128
_SWIGLU_MMA_K = 32
_WARPS_M = 2
_WARPS_N = 4
_NUM_WARPS = _WARPS_M * _WARPS_N
_MAX_STAGES = 3
# Process the rounded projection tile in 32-row SMEM-staged chunks in the residual
# epilogue so two-update ReusableUpdate arithmetic stays free of register spills.
_RESIDUAL_CHUNK_M = 32

_GL_BLOCK_M = gl.constexpr(_BLOCK_M)
_GL_BLOCK_N = gl.constexpr(_BLOCK_N)
_GL_BLOCK_K = gl.constexpr(_BLOCK_K)
_GL_MMA_K = gl.constexpr(_MMA_K)
_GL_SWIGLU_BLOCK_N = gl.constexpr(_SWIGLU_BLOCK_N)
_GL_SWIGLU_MMA_K = gl.constexpr(_SWIGLU_MMA_K)
_GL_MAX_STAGES = gl.constexpr(_MAX_STAGES)
_GL_WARPS_M = gl.constexpr(_WARPS_M)
_GL_WARPS_N = gl.constexpr(_WARPS_N)
_GL_NUM_WARPS = gl.constexpr(_NUM_WARPS)
_GL_RESIDUAL_CHUNK_M = gl.constexpr(_RESIDUAL_CHUNK_M)


@gluon.jit
def _load_operands(
    input_descriptor,
    weight_descriptor,
    second_weight_descriptor,
    barrier,
    input_tile,
    weight_tile,
    second_weight_tile,
    row_m,
    row_n,
    k_offset,
    paired: gl.constexpr,
    second,
    dual: gl.constexpr = False,
):
    expected_bytes: gl.constexpr = (
        input_descriptor.block_type.nbytes
        + weight_descriptor.block_type.nbytes
        + (second_weight_descriptor.block_type.nbytes if dual else 0)
    )
    mbarrier.expect(barrier, expected_bytes)
    tma.async_load(input_descriptor, [row_m, k_offset], barrier, input_tile)
    if dual:
        tma.async_load(weight_descriptor, [row_n, k_offset], barrier, weight_tile)
        tma.async_load(second_weight_descriptor, [row_n, k_offset], barrier, second_weight_tile)
    elif paired:
        if second:
            tma.async_load(second_weight_descriptor, [row_n, k_offset], barrier, weight_tile)
        else:
            tma.async_load(weight_descriptor, [row_n, k_offset], barrier, weight_tile)
    else:
        tma.async_load(weight_descriptor, [row_n, k_offset], barrier, weight_tile)


@gluon.jit
def _accumulate_tile(
    input_descriptor,
    weight_descriptor,
    second_weight_descriptor,
    input_ring,
    weight_ring,
    second_weight_ring,
    barriers,
    row_m,
    row_n,
    accumulator,
    second_accumulator,
    group,
    slot: gl.constexpr,
    stages: gl.constexpr,
    refill: gl.constexpr,
    paired: gl.constexpr,
    second,
    mma_k: gl.constexpr,
    dual: gl.constexpr = False,
):
    input_layout: gl.constexpr = gl.DotOperandLayout(0, accumulator.type.layout, k_width=4)
    weight_layout: gl.constexpr = gl.DotOperandLayout(1, accumulator.type.layout, k_width=4)
    barrier = barriers.index(slot)
    input_tile = input_ring.index(slot)
    weight_tile = weight_ring.index(slot)
    second_weight_tile = second_weight_ring.index(slot) if dual else weight_tile
    mbarrier.wait(barrier, phase=group & 1)
    for k_slice in gl.static_range(0, _GL_BLOCK_K, mma_k):
        lhs = input_tile.slice(k_slice, mma_k, dim=1).load(input_layout)
        rhs = weight_tile.slice(k_slice, mma_k, dim=1).permute([1, 0]).load(weight_layout)
        accumulator = mma_v2(lhs, rhs, accumulator)
        if dual:
            second_rhs = (
                second_weight_tile.slice(k_slice, mma_k, dim=1).permute([1, 0]).load(weight_layout)
            )
            second_accumulator = mma_v2(lhs, second_rhs, second_accumulator)
    gl.barrier()
    if refill:
        _load_operands(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            barrier,
            input_tile,
            weight_tile,
            second_weight_tile,
            row_m,
            row_n,
            ((group + 1) * stages + slot) * _GL_BLOCK_K,
            paired,
            second,
            dual,
        )
    return accumulator, second_accumulator


@gluon.jit
def _accumulate_tma_pair(
    input_descriptor,
    weight_descriptor,
    second_weight_descriptor,
    row_m,
    row_n,
    full_groups,
    mma_layout: gl.constexpr,
    stages: gl.constexpr,
    tail_refills: gl.constexpr,
    paired: gl.constexpr,
    second,
    block_n: gl.constexpr = _GL_BLOCK_N,
    mma_k: gl.constexpr = _GL_MMA_K,
    dual: gl.constexpr = False,
):
    """Accumulate one or two exact INT32 ``[128, block_n]`` tiles through a TMA ring."""
    # Reuse one weight ring for paired mode and add a second 128x64 ring only in dual mode.
    input_ring = gl.allocate_shared_memory(
        gl.int8, [stages, _GL_BLOCK_M, _GL_BLOCK_K], input_descriptor.layout
    )
    weight_ring = gl.allocate_shared_memory(
        gl.int8, [stages, block_n, _GL_BLOCK_K], weight_descriptor.layout
    )
    second_weight_ring = (
        gl.allocate_shared_memory(
            gl.int8, [stages, block_n, _GL_BLOCK_K], second_weight_descriptor.layout
        )
        if dual
        else weight_ring
    )
    barriers = gl.allocate_shared_memory(gl.int64, [stages, 1], mbarrier.MBarrierLayout())
    for slot in gl.static_range(stages):
        mbarrier.init(barriers.index(slot), count=1)
    gl.barrier()
    for slot in gl.static_range(stages):
        _load_operands(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            barriers.index(slot),
            input_ring.index(slot),
            weight_ring.index(slot),
            second_weight_ring.index(slot) if dual else weight_ring.index(slot),
            row_m,
            row_n,
            slot * _GL_BLOCK_K,
            paired,
            second,
            dual,
        )

    accumulator = gl.full([_GL_BLOCK_M, block_n], 0, gl.int32, mma_layout)
    second_accumulator = (
        gl.full([_GL_BLOCK_M, block_n], 0, gl.int32, mma_layout) if dual else accumulator
    )
    for group in range(full_groups):
        for slot in gl.static_range(stages):
            accumulator, second_accumulator = _accumulate_tile(
                input_descriptor,
                weight_descriptor,
                second_weight_descriptor,
                input_ring,
                weight_ring,
                second_weight_ring,
                barriers,
                row_m,
                row_n,
                accumulator,
                second_accumulator,
                group,
                slot,
                stages,
                True,
                paired,
                second,
                mma_k,
                dual,
            )
    for slot in gl.static_range(tail_refills):
        accumulator, second_accumulator = _accumulate_tile(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            input_ring,
            weight_ring,
            second_weight_ring,
            barriers,
            row_m,
            row_n,
            accumulator,
            second_accumulator,
            full_groups,
            slot,
            stages,
            True,
            paired,
            second,
            mma_k,
            dual,
        )
    for slot in gl.static_range(tail_refills, stages):
        accumulator, second_accumulator = _accumulate_tile(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            input_ring,
            weight_ring,
            second_weight_ring,
            barriers,
            row_m,
            row_n,
            accumulator,
            second_accumulator,
            full_groups,
            slot,
            stages,
            False,
            paired,
            second,
            mma_k,
            dual,
        )
        mbarrier.invalidate(barriers.index(slot))
    for slot in gl.static_range(tail_refills):
        accumulator, second_accumulator = _accumulate_tile(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            input_ring,
            weight_ring,
            second_weight_ring,
            barriers,
            row_m,
            row_n,
            accumulator,
            second_accumulator,
            full_groups + 1,
            slot,
            stages,
            False,
            paired,
            second,
            mma_k,
            dual,
        )
        mbarrier.invalidate(barriers.index(slot))
    return accumulator, second_accumulator


@gluon.jit
def _accumulate_tma(
    input_descriptor,
    weight_descriptor,
    second_weight_descriptor,
    row_m,
    row_n,
    full_groups,
    mma_layout: gl.constexpr,
    stages: gl.constexpr,
    tail_refills: gl.constexpr,
    paired: gl.constexpr,
    second,
    block_n: gl.constexpr = _GL_BLOCK_N,
    mma_k: gl.constexpr = _GL_MMA_K,
):
    accumulator, _ = _accumulate_tma_pair(
        input_descriptor,
        weight_descriptor,
        second_weight_descriptor,
        row_m,
        row_n,
        full_groups,
        mma_layout,
        stages,
        tail_refills,
        paired,
        second,
        block_n,
        mma_k,
        False,
    )
    return accumulator


@gluon.jit
def accumulate_tma(
    input_descriptor,
    weight_descriptor,
    row_m,
    row_n,
    input_features: gl.constexpr,
    mma_layout: gl.constexpr,
    block_n: gl.constexpr,
    mma_k: gl.constexpr,
):
    """Accumulate one ``[128, block_n]`` INT32 tile for a compile-time ``input_features``.

    Supports any ``input_features`` divisible by 64 using ``min(3, k_tiles)``
    shared-memory ring stages and exact ring-order tail refilling and draining.
    """
    gl.static_assert(input_features >= _GL_BLOCK_K and input_features % _GL_BLOCK_K == 0)
    num_k_tiles: gl.constexpr = input_features // _GL_BLOCK_K
    stages: gl.constexpr = _GL_MAX_STAGES if num_k_tiles >= _GL_MAX_STAGES else num_k_tiles
    refill_tiles: gl.constexpr = num_k_tiles - stages
    full_groups: gl.constexpr = refill_tiles // stages
    tail_refills: gl.constexpr = refill_tiles % stages
    return _accumulate_tma(
        input_descriptor,
        weight_descriptor,
        weight_descriptor,
        row_m,
        row_n,
        full_groups,
        mma_layout,
        stages,
        tail_refills,
        False,
        False,
        block_n,
        mma_k,
    )


@gluon.jit
def _project_tma_tile(
    input_descriptor,
    weight_descriptor,
    second_weight_descriptor,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_scale_ptr,
    second_bias_ptr,
    m,
    n,
    full_groups,
    stages: gl.constexpr,
    tail_refills: gl.constexpr,
    group_m: gl.constexpr,
    has_bias: gl.constexpr,
    paired: gl.constexpr,
    second_has_bias: gl.constexpr,
    output_dtype: gl.constexpr,
):
    """Project and dequantize one cache-grouped 128x256 output tile."""
    pid_m, pid_n, column_tiles = _cta_tile_coordinates(
        m, n, _GL_BLOCK_M, _GL_BLOCK_N, group_m, paired
    )
    second = pid_n >= column_tiles
    if paired:
        pid_n %= column_tiles
        weight_scale_ptr = gl.where(second, second_scale_ptr, weight_scale_ptr)

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[_GL_WARPS_M, _GL_WARPS_N], instr_shape=[16, 8]
    )
    accumulator = _accumulate_tma(
        input_descriptor,
        weight_descriptor,
        second_weight_descriptor,
        pid_m * _GL_BLOCK_M,
        pid_n * _GL_BLOCK_N,
        full_groups,
        mma_layout,
        stages,
        tail_refills,
        paired,
        second,
        _GL_BLOCK_N,
        _GL_MMA_K,
    )
    result = _dequantize_paired_tile(
        accumulator,
        input_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        second_bias_ptr,
        pid_m,
        pid_n,
        m,
        n,
        second,
        mma_layout,
        _GL_BLOCK_M,
        _GL_BLOCK_N,
        has_bias,
        paired,
        second_has_bias,
        output_dtype,
    )
    return pid_m, pid_n, second, result


@gluon.jit
def _load_gate(
    gate_ptr,
    indices,
    columns,
    valid,
    gate_row_stride: gl.constexpr,
    gate_rows,
    python_indexing: gl.constexpr,
):
    if python_indexing:
        indices = gl.where(indices < 0, indices + gate_rows, indices)
    return gl.load(
        gate_ptr + indices[:, None] * gate_row_stride + columns[None, :],
        mask=valid,
        other=0.0,
    ).to(gl.float32)


@gluon.jit(do_not_specialize=["m"])
def _int8_tma_matmul_kernel(
    input_descriptor,
    weight_descriptor,
    second_weight_descriptor,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_scale_ptr,
    second_bias_ptr,
    m,
    n,
    output_row_stride,
    full_groups,
    stages: gl.constexpr,
    tail_refills: gl.constexpr,
    group_m: gl.constexpr,
    has_bias: gl.constexpr,
    paired: gl.constexpr,
    second_has_bias: gl.constexpr,
):
    """Project one TMA-loaded INT8 tile to a logical-dtype output without residual fusion."""
    pid_m, pid_n, second, result = _project_tma_tile(
        input_descriptor,
        weight_descriptor,
        second_weight_descriptor,
        input_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        second_scale_ptr,
        second_bias_ptr,
        m,
        n,
        full_groups,
        stages,
        tail_refills,
        group_m,
        has_bias,
        paired,
        second_has_bias,
        output_ptr.dtype.element_ty,
    )
    _store_dequantized_tile(
        result,
        output_ptr,
        pid_m,
        pid_n,
        m,
        n,
        output_row_stride,
        second,
        _GL_BLOCK_M,
        _GL_BLOCK_N,
        _GL_NUM_WARPS,
        paired,
    )


@gluon.jit(do_not_specialize=["m", "residual_row_offset"])
def _int8_tma_residual_matmul_kernel(
    input_descriptor,
    weight_descriptor,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    m,
    n,
    full_groups,
    residual_base_ptr,
    residual_gate_ptr,
    residual_indices_ptr,
    residual_reusable_gate_ptr,
    residual_row_offset,
    residual_gate_rows,
    residual_reusable_gate_rows,
    stages: gl.constexpr,
    tail_refills: gl.constexpr,
    group_m: gl.constexpr,
    has_bias: gl.constexpr,
    residual_gate_row_stride: gl.constexpr,
    residual_reusable_gate_row_stride: gl.constexpr,
    residual_has_reusable_update: gl.constexpr,
    residual_python_indexing: gl.constexpr,
):
    """Project one 16-bit TMA-loaded INT8 tile and add a validated residual update.

    Reuses the 73.7 KiB TMA ring shared memory to stage the rounded 64 KiB 16-bit
    projection tile and processes 32-row chunks so single- and two-update
    (``ReusableUpdate``) epilogues stay free of register spills while mirroring
    the shared Triton FP32 residual arithmetic and 16-bit projection rounding
    boundary (see ``policy.supports_residual_epilogue``).
    """
    pid_m, pid_n, _, projection = _project_tma_tile(
        input_descriptor,
        weight_descriptor,
        weight_descriptor,
        input_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        weight_scale_ptr,
        bias_ptr,
        m,
        n,
        full_groups,
        stages,
        tail_refills,
        group_m,
        has_bias,
        False,
        False,
        output_ptr.dtype.element_ty,
    )
    # Reuse the TMA ring SMEM (73.7 KiB) to stage the rounded 64 KiB projection tile and
    # apply the residual update in 32-row chunks without spilling registers.
    projection_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    projection_shared = gl.allocate_shared_memory(
        output_ptr.dtype.element_ty, [_GL_BLOCK_M, _GL_BLOCK_N], projection_shared_layout
    )
    projection_shared.store(projection)
    gl.barrier()

    chunk_layout: gl.constexpr = _build_store_layout(
        output_ptr.dtype.element_ty.primitive_bitwidth, _GL_BLOCK_N, _GL_NUM_WARPS
    )
    columns = pid_n * _GL_BLOCK_N + gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(0, chunk_layout))
    column_valid = columns < n
    for chunk_row in gl.static_range(0, _GL_BLOCK_M, _GL_RESIDUAL_CHUNK_M):
        chunk_projection = projection_shared.slice(chunk_row, _GL_RESIDUAL_CHUNK_M, dim=0).load(
            chunk_layout
        )
        rows = (
            pid_m * _GL_BLOCK_M
            + chunk_row
            + gl.arange(0, _GL_RESIDUAL_CHUNK_M, gl.SliceLayout(1, chunk_layout))
        )
        row_valid = rows < m
        valid = row_valid[:, None] & column_valid[None, :]
        offsets = rows.to(gl.int64)[:, None] * n + columns.to(gl.int64)[None, :]
        indices = gl.load(
            residual_indices_ptr + residual_row_offset + rows,
            mask=row_valid,
            other=0,
        ).to(gl.int64)
        hidden = gl.load(residual_base_ptr + offsets, mask=valid, other=0.0).to(gl.float32)
        if residual_has_reusable_update:
            reusable_update = gl.load(output_ptr + offsets, mask=valid, other=0.0).to(gl.float32)
            reusable_gate = _load_gate(
                residual_reusable_gate_ptr,
                indices,
                columns,
                valid,
                residual_reusable_gate_row_stride,
                residual_reusable_gate_rows,
                residual_python_indexing,
            )
            hidden = hidden + reusable_gate * reusable_update
        gate = _load_gate(
            residual_gate_ptr,
            indices,
            columns,
            valid,
            residual_gate_row_stride,
            residual_gate_rows,
            residual_python_indexing,
        )
        final = (hidden + gate * chunk_projection.to(gl.float32)).to(output_ptr.dtype.element_ty)
        gl.store(output_ptr + offsets, final, mask=valid)


@gluon.jit(do_not_specialize=["m"])
def _int8_tma_swiglu_matmul_kernel(
    input_descriptor,
    weight_descriptor,
    second_weight_descriptor,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_scale_ptr,
    second_bias_ptr,
    m,
    n,
    output_row_stride,
    full_groups,
    stages: gl.constexpr,
    tail_refills: gl.constexpr,
    group_m: gl.constexpr,
    has_bias: gl.constexpr,
    second_has_bias: gl.constexpr,
):
    """Project paired TMA-loaded INT8 weights and fuse 16-bit SwiGLU without spills.

    Preserves the 16-bit projection rounding boundary before FP32 SiLU multiplication
    and stores the activated output in ``output_ptr.dtype.element_ty`` (matching
    eager 16-bit SwiGLU).
    """
    pid_m, pid_n, _ = _cta_tile_coordinates(m, n, _GL_BLOCK_M, _GL_SWIGLU_BLOCK_N, group_m, False)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[_GL_WARPS_M, _GL_WARPS_N], instr_shape=[16, 8]
    )
    up_accumulator, gate_accumulator = _accumulate_tma_pair(
        input_descriptor,
        weight_descriptor,
        second_weight_descriptor,
        pid_m * _GL_BLOCK_M,
        pid_n * _GL_SWIGLU_BLOCK_N,
        full_groups,
        mma_layout,
        stages,
        tail_refills,
        False,
        False,
        _GL_SWIGLU_BLOCK_N,
        _GL_SWIGLU_MMA_K,
        True,
    )
    _store_swiglu_epilogue(
        up_accumulator,
        gate_accumulator,
        output_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        second_scale_ptr,
        second_bias_ptr,
        pid_m,
        pid_n,
        m,
        n,
        output_row_stride,
        mma_layout,
        _GL_BLOCK_M,
        _GL_SWIGLU_BLOCK_N,
        _GL_NUM_WARPS,
        has_bias,
        second_has_bias,
    )


def operands_aligned(*operands: torch.Tensor) -> bool:
    """Return whether 2-D INT8 GEMM operands satisfy 16-byte Gluon contiguity and alignment."""
    return all(
        operand.ndim == 2
        and operand.is_contiguous()
        and operand.shape[1] > 0
        and operand.shape[1] % 16 == 0
        and operand.data_ptr() % 16 == 0
        for operand in operands
    )


def build_matrix_descriptor(matrix: torch.Tensor, block_rows: int = _BLOCK_M) -> TensorDescriptor:
    """Build a 2D TMA descriptor over contiguous INT8 rows (flattening leading dims)."""
    if not matrix.is_contiguous():
        raise ValueError("TMA operands must be contiguous")
    width = matrix.shape[-1]
    tile = [block_rows, _BLOCK_K]
    layout = gl.NVMMASharedLayout.get_default_for(tile, gl.int8)
    with device_context(matrix.device):
        return TensorDescriptor(
            matrix,
            [matrix.numel() // width, width],
            [width, 1],
            tile,
            layout,
        )


def _compute_k_schedule(k: int) -> tuple[int, int, int]:
    num_k_tiles = triton.cdiv(k, _BLOCK_K)
    stages = min(_MAX_STAGES, num_k_tiles)
    refill_tiles = num_k_tiles - stages
    return stages, refill_tiles // stages, refill_tiles % stages


def launch_int8_matmul(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    second_weight: torch.Tensor,
    second_scale: torch.Tensor,
    second_bias: torch.Tensor | None,
    *,
    paired: bool,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Launch one TMA Gluon GEMM over ``[m, k]`` inputs and ``[n, k]`` weights."""
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    stages, full_groups, tail_refills = _compute_k_schedule(k)
    input_descriptor = build_matrix_descriptor(input_qdata, _BLOCK_M)
    weight_descriptor = build_matrix_descriptor(weight_qdata, _BLOCK_N)
    second_weight_descriptor = (
        build_matrix_descriptor(second_weight, _BLOCK_N) if paired else weight_descriptor
    )
    grid = (triton.cdiv(m, _BLOCK_M) * triton.cdiv(n, _BLOCK_N) * (2 if paired else 1),)
    with device_context(input_qdata.device):
        _int8_tma_matmul_kernel[grid](
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            output,
            input_scale,
            weight_scale,
            bias if bias is not None else output,
            second_scale,
            second_bias if second_bias is not None else output,
            m,
            n,
            output.stride(0),
            full_groups,
            stages,
            tail_refills,
            execution_plan.matmul_group_m,
            bias is not None,
            paired,
            second_bias is not None,
            num_warps=_NUM_WARPS,
        )


def launch_int8_residual_matmul(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    residual_add: ResidualAdd,
    *,
    row_offset: int,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Project into validated residual storage under a TMA native-epilogue plan."""
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    stages, full_groups, tail_refills = _compute_k_schedule(k)
    input_descriptor = build_matrix_descriptor(input_qdata, _BLOCK_M)
    weight_descriptor = build_matrix_descriptor(weight_qdata, _BLOCK_N)
    gate = residual_add.gate
    reusable = residual_add.reusable
    reusable_gate = output if reusable is None else reusable.gate
    grid = (triton.cdiv(m, _BLOCK_M) * triton.cdiv(n, _BLOCK_N),)
    with device_context(input_qdata.device):
        _int8_tma_residual_matmul_kernel[grid](
            input_descriptor,
            weight_descriptor,
            output,
            input_scale,
            weight_scale,
            bias if bias is not None else output,
            m,
            n,
            full_groups,
            residual_add.base.view(-1, n)[row_offset:],
            gate,
            residual_add.gate_indices,
            reusable_gate,
            row_offset,
            gate.shape[0],
            0 if reusable is None else reusable.gate.shape[0],
            stages,
            tail_refills,
            execution_plan.matmul_group_m,
            bias is not None,
            gate.stride(0),
            0 if reusable is None else reusable.gate.stride(0),
            reusable is not None,
            residual_add.python_indexing,
            num_warps=_NUM_WARPS,
        )


def launch_int8_swiglu_matmul(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    second_weight: torch.Tensor,
    second_scale: torch.Tensor,
    second_bias: torch.Tensor | None,
    *,
    execution_plan: NvidiaExecutionPlan,
) -> None:
    """Launch one dual-accumulator SM120 TMA GEMM with fused BF16/FP16 SwiGLU."""
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    stages, full_groups, tail_refills = _compute_k_schedule(k)
    input_descriptor = build_matrix_descriptor(input_qdata, _BLOCK_M)
    weight_descriptor = build_matrix_descriptor(weight_qdata, _SWIGLU_BLOCK_N)
    second_weight_descriptor = build_matrix_descriptor(second_weight, _SWIGLU_BLOCK_N)
    grid = (triton.cdiv(m, _BLOCK_M) * triton.cdiv(n, _SWIGLU_BLOCK_N),)
    with device_context(input_qdata.device):
        _int8_tma_swiglu_matmul_kernel[grid](
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            output,
            input_scale,
            weight_scale,
            bias if bias is not None else output,
            second_scale,
            second_bias if second_bias is not None else output,
            m,
            n,
            output.stride(0),
            full_groups,
            stages,
            tail_refills,
            execution_plan.matmul_group_m,
            bias is not None,
            second_bias is not None,
            num_warps=_NUM_WARPS,
        )
