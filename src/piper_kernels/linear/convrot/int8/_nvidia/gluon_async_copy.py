"""NVIDIA async-copy Gluon GEMM for ConvRot INT8 projections.

The kernel follows the CUTLASS SM80 INT8 schedule: 16-byte ``cp.async`` copies into
swizzled shared memory, ``ldmatrix`` operands, and m16n8k32 INT8 MMAs with 64x64 warp tiles.
Each pipeline stage has its own shared-memory allocation and stage indices stay static, so
the barrier analysis inserts one barrier per K tile instead of two. Interior tiles whose K
tiles are whole skip per-element copy masks; edge tiles zero-fill rows and K columns outside the
problem. The kernel does not specialize on M, so one compiled kernel serves every row count.
Accumulation is exact INT32, and the epilogue computes
``(acc * input_scale) * weight_scale`` with bias added through explicit FMAs, which matches
the shared Triton arithmetic bitwise. The implementation runs on SM8x and SM120;
architecture policies choose when to use it.
"""

# Gluon exposes low-level signatures that are not fully modeled by type checkers.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

from piper_kernels._triton.runtime import device_context

from ._plan import NvidiaExecutionPlan

# Warp tiles are 64 columns wide, so every supported tile uses two warp columns.
_WARPS_N = 2
# ``cp.async`` moves at most 16 bytes per thread and instruction.
_COPY_BYTES = 16

_GL_WARPS_N = gl.constexpr(_WARPS_N)
_GL_COPY_BYTES = gl.constexpr(_COPY_BYTES)


@gluon.jit
def _stage_buffers(
    block_rows: gl.constexpr,
    block_k: gl.constexpr,
    stages: gl.constexpr,
):
    """Allocate one shared-memory buffer per pipeline stage."""
    layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([block_rows, block_k], gl.int8)
    buffers = (
        gl.allocate_shared_memory(gl.int8, [block_rows, block_k], layout),
        gl.allocate_shared_memory(gl.int8, [block_rows, block_k], layout),
        gl.allocate_shared_memory(gl.int8, [block_rows, block_k], layout),
    )
    if stages == 4:
        # Gluon's frontend has no starred unpacking; tuple concatenation is supported.
        fourth = gl.allocate_shared_memory(gl.int8, [block_rows, block_k], layout)
        buffers = buffers + (fourth,)  # noqa: RUF005
    return buffers


@gluon.jit
def _copy_tile(
    buffer, pointers, rows_valid, columns, k_offset, k, num_k, tile, masked: gl.constexpr
):
    """Start one K tile's copy; missing rows, columns, and tiles are zero-filled."""
    mask = rows_valid[:, None] & (columns[None, :] < k - k_offset) if masked else tile < num_k
    async_copy.async_load(buffer, pointers + k_offset, mask=mask)


@gluon.jit
def _accumulate(
    input_pointers,
    weight_pointers,
    input_valid,
    weight_valid,
    columns,
    k,
    num_k,
    accumulator,
    block_k: gl.constexpr,
    stages: gl.constexpr,
    input_layout: gl.constexpr,
    weight_layout: gl.constexpr,
    masked: gl.constexpr,
):
    """Accumulate the exact INT32 tile product through a ``stages``-deep copy pipeline."""
    # Buffers allocated here end before the epilogue, which reuses their shared memory.
    input_buffers = _stage_buffers(accumulator.shape[0], block_k, stages)
    weight_buffers = _stage_buffers(accumulator.shape[1], block_k, stages)
    for stage in gl.static_range(stages - 1):
        offset = stage * block_k
        _copy_tile(
            input_buffers[stage],
            input_pointers,
            input_valid,
            columns,
            offset,
            k,
            num_k,
            stage,
            masked,
        )
        _copy_tile(
            weight_buffers[stage],
            weight_pointers,
            weight_valid,
            columns,
            offset,
            k,
            num_k,
            stage,
            masked,
        )
        async_copy.commit_group()
    # Stage indices stay static within each group of K tiles.
    for first_tile in range(0, num_k, stages):
        for stage in gl.static_range(stages):
            async_copy.wait_group(stages - 2)
            gl.barrier()
            tile = first_tile + stage + stages - 1
            offset = tile * block_k
            _copy_tile(
                input_buffers[(stage + stages - 1) % stages],
                input_pointers,
                input_valid,
                columns,
                offset,
                k,
                num_k,
                tile,
                masked,
            )
            _copy_tile(
                weight_buffers[(stage + stages - 1) % stages],
                weight_pointers,
                weight_valid,
                columns,
                offset,
                k,
                num_k,
                tile,
                masked,
            )
            async_copy.commit_group()
            values = input_buffers[stage].load(input_layout)
            weight = weight_buffers[stage].permute([1, 0]).load(weight_layout)
            accumulator = mma_v2(values, weight, accumulator)
    async_copy.wait_group(0)
    return accumulator


@gluon.jit(do_not_specialize=["m"])
def _int8_matmul_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_weight_ptr,
    second_scale_ptr,
    second_bias_ptr,
    m,
    n,
    k,
    output_row_stride,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    block_k: gl.constexpr,
    stages: gl.constexpr,
    group_m: gl.constexpr,
    warps_m: gl.constexpr,
    has_bias: gl.constexpr,
    paired: gl.constexpr,
    second_has_bias: gl.constexpr,
    whole_k_tiles: gl.constexpr,
):
    # Gluon has no ``in`` operator for constexpr tuples.
    three_or_four: gl.constexpr = stages == 3 or stages == 4  # noqa: PLR1714
    gl.static_assert(three_or_four, "Async-copy GEMM uses three or four stages")
    num_warps: gl.constexpr = warps_m * _GL_WARPS_N
    column_tiles = gl.cdiv(n, block_n)
    num_pid_n = column_tiles * (2 if paired else 1)
    # Without grouping (group_m == 0), each group holds a single row of tiles.
    group_rows: gl.constexpr = group_m if group_m else 1
    group_size = group_rows * num_pid_n
    pid = gl.program_id(0)
    first_m = (pid // group_size) * group_rows
    rows_in_group = gl.minimum(gl.cdiv(m, block_m) - first_m, group_rows)
    pid_m = first_m + (pid % group_size) % rows_in_group
    pid_n = (pid % group_size) // rows_in_group
    second = pid_n >= column_tiles
    if paired:
        pid_n %= column_tiles
        weight_ptr = gl.where(second, second_weight_ptr, weight_ptr)
        weight_scale_ptr = gl.where(second, second_scale_ptr, weight_scale_ptr)

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[warps_m, _GL_WARPS_N], instr_shape=[16, 8]
    )
    input_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    weight_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, _GL_COPY_BYTES],
        [32 // (block_k // _GL_COPY_BYTES), block_k // _GL_COPY_BYTES],
        [num_warps, 1],
        [1, 0],
    )
    input_rows = pid_m * block_m + gl.arange(0, block_m, gl.SliceLayout(1, copy_layout))
    weight_rows = pid_n * block_n + gl.arange(0, block_n, gl.SliceLayout(1, copy_layout))
    columns = gl.arange(0, block_k, gl.SliceLayout(0, copy_layout))
    input_pointers = input_ptr + input_rows.to(gl.int64)[:, None] * k + columns[None, :]
    weight_pointers = weight_ptr + weight_rows.to(gl.int64)[:, None] * k + columns[None, :]
    num_k = gl.cdiv(k, block_k)
    accumulator = gl.zeros([block_m, block_n], gl.int32, mma_layout)
    # Unmasked interior tiles keep the unrolled loop within the register budget.
    interior = ((pid_m + 1) * block_m <= m) & ((pid_n + 1) * block_n <= n)
    if whole_k_tiles and interior:
        accumulator = _accumulate(
            input_pointers,
            weight_pointers,
            input_rows < m,
            weight_rows < n,
            columns,
            k,
            num_k,
            accumulator,
            block_k,
            stages,
            input_layout,
            weight_layout,
            False,
        )
    else:
        accumulator = _accumulate(
            input_pointers,
            weight_pointers,
            input_rows < m,
            weight_rows < n,
            columns,
            k,
            num_k,
            accumulator,
            block_k,
            stages,
            input_layout,
            weight_layout,
            True,
        )

    offsets_m = pid_m * block_m + gl.arange(0, block_m, gl.SliceLayout(1, mma_layout))
    offsets_n = pid_n * block_n + gl.arange(0, block_n, gl.SliceLayout(0, mma_layout))
    input_scale = gl.load(input_scale_ptr + offsets_m, mask=offsets_m < m, other=0.0)
    weight_scale = gl.load(weight_scale_ptr + offsets_n, mask=offsets_n < n, other=0.0)
    row_scaled = accumulator.to(gl.float32) * input_scale[:, None]
    if paired and (has_bias or second_has_bias):
        # Select FP32 values rather than pointers: biases may have different dtypes.
        bias = gl.full([block_n], 0, gl.float32, gl.SliceLayout(0, mma_layout))
        second_bias = gl.full([block_n], 0, gl.float32, gl.SliceLayout(0, mma_layout))
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
    result = result.to(output_ptr.dtype.element_ty)
    # Store through a row-major layout so each thread writes 16 contiguous bytes.
    store_width: gl.constexpr = 8 * _GL_COPY_BYTES // output_ptr.dtype.element_ty.primitive_bitwidth
    store_layout: gl.constexpr = gl.BlockedLayout(
        [1, store_width],
        [32 // (block_n // store_width), block_n // store_width],
        [num_warps, 1],
        [1, 0],
    )
    result = gl.convert_layout(result, store_layout)
    offsets_m = pid_m * block_m + gl.arange(0, block_m, gl.SliceLayout(1, store_layout))
    offsets_n = pid_n * block_n + gl.arange(0, block_n, gl.SliceLayout(0, store_layout))
    output_pointers = (
        output_ptr + offsets_m.to(gl.int64)[:, None] * output_row_stride + offsets_n[None, :]
    )
    if paired:
        output_pointers += second.to(gl.int64) * n
    gl.store(output_pointers, result, mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n))


def operands_aligned(*operands: torch.Tensor) -> bool:
    """Return whether INT8 GEMM operands start and step in whole 16-byte copies."""
    return all(operand.data_ptr() % 16 == 0 and operand.stride(0) % 16 == 0 for operand in operands)


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
    """Launch one Gluon GEMM over ``[m, k]`` inputs and ``[n, k]`` weights."""
    plan = execution_plan
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    block_m, block_n, block_k = plan.matmul_block_m, plan.matmul_block_n, plan.matmul_block_k
    num_warps = plan.matmul_num_warps
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n) * (2 if paired else 1),)
    with device_context(input_qdata.device):
        _int8_matmul_kernel[grid](
            input_qdata,
            weight_qdata,
            output,
            input_scale,
            weight_scale,
            bias if bias is not None else output,
            second_weight,
            second_scale,
            second_bias if second_bias is not None else output,
            m,
            n,
            k,
            output.stride(0),
            block_m,
            block_n,
            block_k,
            plan.matmul_num_stages,
            plan.matmul_group_m,
            num_warps // _WARPS_N,
            bias is not None,
            paired,
            second_bias is not None,
            k % block_k == 0,
            num_warps=num_warps,
        )
