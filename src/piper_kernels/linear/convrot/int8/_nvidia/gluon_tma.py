"""NVIDIA SM120 Gluon TMA GEMM and fused residual epilogue for ConvRot INT8 projections.

The kernel loads row-major INT8 operands into swizzled shared memory through 2D TMA
descriptors across an up-to-three-stage mbarrier ring and accumulates exact INT32
products with m16n8k32 (k=64 sliced) MMAv2 instructions across a 128x256 CTA tile
(two 64-row warp rows by four 64-column warp columns). TMA hardware zero-fills
out-of-bounds M, N, and K elements, so ragged tiles require no masked K-loop
duplication. Both kernels keep M unspecialized so one compiled binary serves every
row count. Accumulation is exact INT32, and dequantization preserves the shared
``(acc * input_scale) * weight_scale`` + explicit FMA bias rounding.
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

from ._plan import NvidiaExecutionPlan

_BLOCK_M = 128
_BLOCK_N = 256
_BLOCK_K = 64
_MMA_K = 64
_WARPS_M = 2
_WARPS_N = 4
_NUM_WARPS = _WARPS_M * _WARPS_N
_MAX_STAGES = 3
_COPY_BYTES = 16
# Process the rounded projection tile in 32-row SMEM-staged chunks in the residual
# epilogue so two-update ReusableUpdate arithmetic stays free of register spills.
_RESIDUAL_CHUNK_M = 32

_GL_BLOCK_M = gl.constexpr(_BLOCK_M)
_GL_BLOCK_N = gl.constexpr(_BLOCK_N)
_GL_BLOCK_K = gl.constexpr(_BLOCK_K)
_GL_MMA_K = gl.constexpr(_MMA_K)
_GL_WARPS_M = gl.constexpr(_WARPS_M)
_GL_WARPS_N = gl.constexpr(_WARPS_N)
_GL_NUM_WARPS = gl.constexpr(_NUM_WARPS)
_GL_COPY_BYTES = gl.constexpr(_COPY_BYTES)
_GL_RESIDUAL_CHUNK_M = gl.constexpr(_RESIDUAL_CHUNK_M)


@gluon.jit
def _load_operands(
    input_descriptor,
    weight_descriptor,
    second_weight_descriptor,
    barrier,
    input_tile,
    weight_tile,
    row_m,
    row_n,
    k_offset,
    paired: gl.constexpr,
    second,
):
    mbarrier.expect(
        barrier, input_descriptor.block_type.nbytes + weight_descriptor.block_type.nbytes
    )
    tma.async_load(input_descriptor, [row_m, k_offset], barrier, input_tile)
    if paired:
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
    barriers,
    row_m,
    row_n,
    accumulator,
    group,
    slot: gl.constexpr,
    stages: gl.constexpr,
    refill: gl.constexpr,
    paired: gl.constexpr,
    second,
):
    input_layout: gl.constexpr = gl.DotOperandLayout(0, accumulator.type.layout, k_width=4)
    weight_layout: gl.constexpr = gl.DotOperandLayout(1, accumulator.type.layout, k_width=4)
    barrier = barriers.index(slot)
    input_tile = input_ring.index(slot)
    weight_tile = weight_ring.index(slot)
    mbarrier.wait(barrier, phase=group & 1)
    for k_slice in gl.static_range(0, _GL_BLOCK_K, _GL_MMA_K):
        lhs = input_tile.slice(k_slice, _GL_MMA_K, dim=1).load(input_layout)
        rhs = weight_tile.slice(k_slice, _GL_MMA_K, dim=1).permute([1, 0]).load(weight_layout)
        accumulator = mma_v2(lhs, rhs, accumulator)
    gl.barrier()
    if refill:
        _load_operands(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            barrier,
            input_tile,
            weight_tile,
            row_m,
            row_n,
            ((group + 1) * stages + slot) * _GL_BLOCK_K,
            paired,
            second,
        )
    return accumulator


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
):
    """Accumulate the exact INT32 tile product through an up-to-three-stage TMA ring."""
    # Allocate one shared-memory ring across paired weights so SMEM stays at 73.7 KiB.
    input_ring = gl.allocate_shared_memory(
        gl.int8, [stages, _GL_BLOCK_M, _GL_BLOCK_K], input_descriptor.layout
    )
    weight_ring = gl.allocate_shared_memory(
        gl.int8, [stages, _GL_BLOCK_N, _GL_BLOCK_K], weight_descriptor.layout
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
            row_m,
            row_n,
            slot * _GL_BLOCK_K,
            paired,
            second,
        )

    accumulator = gl.full([_GL_BLOCK_M, _GL_BLOCK_N], 0, gl.int32, mma_layout)
    for group in range(full_groups):
        for slot in gl.static_range(stages):
            accumulator = _accumulate_tile(
                input_descriptor,
                weight_descriptor,
                second_weight_descriptor,
                input_ring,
                weight_ring,
                barriers,
                row_m,
                row_n,
                accumulator,
                group,
                slot,
                stages,
                True,
                paired,
                second,
            )
    for slot in gl.static_range(tail_refills):
        accumulator = _accumulate_tile(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            input_ring,
            weight_ring,
            barriers,
            row_m,
            row_n,
            accumulator,
            full_groups,
            slot,
            stages,
            True,
            paired,
            second,
        )
    for slot in gl.static_range(tail_refills, stages):
        accumulator = _accumulate_tile(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            input_ring,
            weight_ring,
            barriers,
            row_m,
            row_n,
            accumulator,
            full_groups,
            slot,
            stages,
            False,
            paired,
            second,
        )
        mbarrier.invalidate(barriers.index(slot))
    for slot in gl.static_range(tail_refills):
        accumulator = _accumulate_tile(
            input_descriptor,
            weight_descriptor,
            second_weight_descriptor,
            input_ring,
            weight_ring,
            barriers,
            row_m,
            row_n,
            accumulator,
            full_groups + 1,
            slot,
            stages,
            False,
            paired,
            second,
        )
        mbarrier.invalidate(barriers.index(slot))
    return accumulator


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
    column_tiles = gl.cdiv(n, _GL_BLOCK_N)
    num_pid_n = column_tiles * (2 if paired else 1)
    group_rows: gl.constexpr = group_m if group_m else 1
    group_size = group_rows * num_pid_n
    pid = gl.program_id(0)
    first_m = (pid // group_size) * group_rows
    rows_in_group = gl.minimum(gl.cdiv(m, _GL_BLOCK_M) - first_m, group_rows)
    pid_m = first_m + (pid % group_size) % rows_in_group
    pid_n = (pid % group_size) // rows_in_group
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
    )

    offsets_m = pid_m * _GL_BLOCK_M + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, mma_layout))
    offsets_n = pid_n * _GL_BLOCK_N + gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(0, mma_layout))
    input_scale = gl.load(input_scale_ptr + offsets_m, mask=offsets_m < m, other=0.0)
    weight_scale = gl.load(weight_scale_ptr + offsets_n, mask=offsets_n < n, other=0.0)
    row_scaled = accumulator.to(gl.float32) * input_scale[:, None]
    if paired and (has_bias or second_has_bias):
        bias = gl.full([_GL_BLOCK_N], 0, gl.float32, gl.SliceLayout(0, mma_layout))
        second_bias = gl.full([_GL_BLOCK_N], 0, gl.float32, gl.SliceLayout(0, mma_layout))
        if has_bias:
            bias = gl.load(bias_ptr + offsets_n, mask=(offsets_n < n) & ~second, other=0.0).to(
                gl.float32
            )
        if second_has_bias:
            second_bias = gl.load(
                second_bias_ptr + offsets_n, mask=(offsets_n < n) & second, other=0.0
            ).to(gl.float32)
        result = gl.fma(
            row_scaled, weight_scale[None, :], gl.where(second, second_bias, bias)[None, :]
        )
    elif has_bias:
        bias = gl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0).to(gl.float32)
        result = gl.fma(row_scaled, weight_scale[None, :], bias[None, :])
    else:
        result = row_scaled * weight_scale[None, :]
    return pid_m, pid_n, second, result.to(output_dtype)


@gluon.jit
def _build_store_layout(element_bitwidth: gl.constexpr):
    """Return a row-major 16-byte vectorized store layout for 256-column tiles."""
    store_width: gl.constexpr = 8 * _GL_COPY_BYTES // element_bitwidth
    threads_n: gl.constexpr = _GL_BLOCK_N // store_width
    # Clamp warp width along N to 32 threads (needed for 32-bit outputs across 256 columns).
    warp_threads_n: gl.constexpr = 32 if threads_n > 32 else threads_n
    warp_threads_m: gl.constexpr = 32 // warp_threads_n
    return gl.BlockedLayout(
        [1, store_width],
        [warp_threads_m, warp_threads_n],
        [_GL_NUM_WARPS, 1],
        [1, 0],
    )


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
    store_layout: gl.constexpr = _build_store_layout(output_ptr.dtype.element_ty.primitive_bitwidth)
    result = gl.convert_layout(result, store_layout)
    offsets_m = pid_m * _GL_BLOCK_M + gl.arange(0, _GL_BLOCK_M, gl.SliceLayout(1, store_layout))
    offsets_n = pid_n * _GL_BLOCK_N + gl.arange(0, _GL_BLOCK_N, gl.SliceLayout(0, store_layout))
    output_pointers = (
        output_ptr + offsets_m.to(gl.int64)[:, None] * output_row_stride + offsets_n[None, :]
    )
    if paired:
        output_pointers += second.to(gl.int64) * n
    gl.store(output_pointers, result, mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n))


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
    """Project one 16-bit TMA-loaded INT8 tile and add a validated residual update."""
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

    chunk_layout: gl.constexpr = _build_store_layout(output_ptr.dtype.element_ty.primitive_bitwidth)
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


def operands_aligned(*operands: torch.Tensor) -> bool:
    """Return whether 2-D INT8 GEMM operands satisfy TMA contiguity and 16-byte alignment."""
    return all(
        operand.ndim == 2
        and operand.is_contiguous()
        and operand.shape[1] > 0
        and operand.shape[1] % 16 == 0
        and operand.data_ptr() % 16 == 0
        for operand in operands
    )


def _build_matrix_descriptor(matrix: torch.Tensor, block_rows: int) -> TensorDescriptor:
    layout = gl.NVMMASharedLayout.get_default_for([block_rows, _BLOCK_K], gl.int8)
    with device_context(matrix.device):
        return TensorDescriptor(
            matrix,
            [matrix.shape[0], matrix.shape[1]],
            [matrix.stride(0), matrix.stride(1)],
            [block_rows, _BLOCK_K],
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
    input_descriptor = _build_matrix_descriptor(input_qdata, _BLOCK_M)
    weight_descriptor = _build_matrix_descriptor(weight_qdata, _BLOCK_N)
    second_weight_descriptor = (
        _build_matrix_descriptor(second_weight, _BLOCK_N) if paired else weight_descriptor
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
    input_descriptor = _build_matrix_descriptor(input_qdata, _BLOCK_M)
    weight_descriptor = _build_matrix_descriptor(weight_qdata, _BLOCK_N)
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
