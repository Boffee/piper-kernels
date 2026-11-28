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
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2

from piper_kernels._triton.runtime import device_context

from .policy import NvidiaExecutionPlan

# Warp tiles are 64 columns wide, so every supported tile uses two warp columns.
_WARPS_N = 2
# ``cp.async`` moves at most 16 bytes per thread and instruction.
_COPY_BYTES = 16
# Dual-accumulator async-copy SwiGLU splits 128 columns into 64 up + 64 gate columns.
_SWIGLU_BLOCK_N = 64
_SWIGLU_WARPS_M = 4
_SWIGLU_NUM_WARPS = _SWIGLU_WARPS_M * _WARPS_N
_SWIGLU_MMA_K = 32
_SWIGLU_CHUNK_M = 32

_GL_WARPS_N = gl.constexpr(_WARPS_N)
_GL_COPY_BYTES = gl.constexpr(_COPY_BYTES)
_GL_SWIGLU_BLOCK_N = gl.constexpr(_SWIGLU_BLOCK_N)
_GL_SWIGLU_WARPS_M = gl.constexpr(_SWIGLU_WARPS_M)
_GL_SWIGLU_MMA_K = gl.constexpr(_SWIGLU_MMA_K)
_GL_SWIGLU_CHUNK_M = gl.constexpr(_SWIGLU_CHUNK_M)


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
def _accumulate_pair(
    input_pointers,
    weight_pointers,
    second_weight_pointers,
    input_valid,
    weight_valid,
    columns,
    k,
    num_k,
    accumulator,
    second_accumulator,
    block_k: gl.constexpr,
    stages: gl.constexpr,
    input_layout: gl.constexpr,
    weight_layout: gl.constexpr,
    masked: gl.constexpr,
    mma_k: gl.constexpr,
    dual: gl.constexpr = False,
):
    """Accumulate one or two exact INT32 tile products through a copy pipeline.

    Each K tile's operands load and multiply in ``mma_k``-wide slices. Slices narrower
    than ``block_k`` hold fewer operand registers at once, for callers whose warp tiles
    or dual accumulators leave little room beside the accumulators.
    """
    # Buffers allocated here end before the epilogue, which reuses their shared memory.
    input_buffers = _stage_buffers(accumulator.shape[0], block_k, stages)
    weight_buffers = _stage_buffers(accumulator.shape[1], block_k, stages)
    second_weight_buffers = (
        _stage_buffers(accumulator.shape[1], block_k, stages) if dual else weight_buffers
    )
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
        if dual:
            _copy_tile(
                second_weight_buffers[stage],
                second_weight_pointers,
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
            if dual:
                _copy_tile(
                    second_weight_buffers[(stage + stages - 1) % stages],
                    second_weight_pointers,
                    weight_valid,
                    columns,
                    offset,
                    k,
                    num_k,
                    tile,
                    masked,
                )
            async_copy.commit_group()
            for first_column in gl.static_range(0, block_k, mma_k):
                values = input_buffers[stage].slice(first_column, mma_k, dim=1).load(input_layout)
                weight = weight_buffers[stage].slice(first_column, mma_k, dim=1).permute([1, 0])
                accumulator = mma_v2(values, weight.load(weight_layout), accumulator)
                if dual:
                    second_weight = (
                        second_weight_buffers[stage]
                        .slice(first_column, mma_k, dim=1)
                        .permute([1, 0])
                    )
                    second_accumulator = mma_v2(
                        values, second_weight.load(weight_layout), second_accumulator
                    )
    async_copy.wait_group(0)
    return accumulator, second_accumulator


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
    mma_k: gl.constexpr,
):
    """Accumulate the exact INT32 tile product through a ``stages``-deep copy pipeline."""
    accumulator, _ = _accumulate_pair(
        input_pointers,
        weight_pointers,
        weight_pointers,
        input_valid,
        weight_valid,
        columns,
        k,
        num_k,
        accumulator,
        0,
        block_k,
        stages,
        input_layout,
        weight_layout,
        masked,
        mma_k,
        False,
    )
    return accumulator


@gluon.jit
def _cta_tile_coordinates(
    m,
    n,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    group_m: gl.constexpr,
    paired: gl.constexpr,
):
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
    return pid_m, pid_n, column_tiles


@gluon.jit
def _dequantize_tile(
    accumulator,
    input_scale,
    weight_scale_ptr,
    bias_ptr,
    offsets_n,
    n,
    has_bias: gl.constexpr,
    output_dtype: gl.constexpr,
):
    weight_scale = gl.load(weight_scale_ptr + offsets_n, mask=offsets_n < n, other=0.0)
    row_scaled = accumulator.to(gl.float32) * input_scale[:, None]
    if has_bias:
        bias = gl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0).to(gl.float32)
        result = gl.fma(row_scaled, weight_scale[None, :], bias[None, :])
    else:
        result = row_scaled * weight_scale[None, :]
    return result.to(output_dtype)


@gluon.jit
def _dequantize_paired_tile(
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
    mma_layout: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    has_bias: gl.constexpr,
    paired: gl.constexpr,
    second_has_bias: gl.constexpr,
    output_dtype: gl.constexpr,
):
    """Dequantize one output tile, selecting unpaired or paired biases safely."""
    offsets_m = pid_m * block_m + gl.arange(0, block_m, gl.SliceLayout(1, mma_layout))
    offsets_n = pid_n * block_n + gl.arange(0, block_n, gl.SliceLayout(0, mma_layout))
    input_scale = gl.load(input_scale_ptr + offsets_m, mask=offsets_m < m, other=0.0)
    if paired and (has_bias or second_has_bias):
        weight_scale = gl.load(weight_scale_ptr + offsets_n, mask=offsets_n < n, other=0.0)
        row_scaled = accumulator.to(gl.float32) * input_scale[:, None]
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
        return gl.fma(
            row_scaled, weight_scale[None, :], gl.where(second, second_bias, bias)[None, :]
        ).to(output_dtype)
    return _dequantize_tile(
        accumulator,
        input_scale,
        weight_scale_ptr,
        bias_ptr,
        offsets_n,
        n,
        has_bias,
        output_dtype,
    )


@gluon.jit
def _build_copy_layout(block_k: gl.constexpr, num_warps: gl.constexpr):
    """Return a row-major 16-byte vectorized async-copy layout for ``block_k`` columns."""
    return gl.BlockedLayout(
        [1, _GL_COPY_BYTES],
        [32 // (block_k // _GL_COPY_BYTES), block_k // _GL_COPY_BYTES],
        [num_warps, 1],
        [1, 0],
    )


@gluon.jit
def _build_store_layout(
    element_bitwidth: gl.constexpr,
    block_n: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Return a row-major 16-byte vectorized store layout for ``block_n``-column tiles."""
    store_width: gl.constexpr = 8 * _GL_COPY_BYTES // element_bitwidth
    threads_n: gl.constexpr = block_n // store_width
    # Clamp warp width along N to 32 threads (needed for 32-bit outputs across wide tiles).
    warp_threads_n: gl.constexpr = 32 if threads_n > 32 else threads_n
    warp_threads_m: gl.constexpr = 32 // warp_threads_n
    return gl.BlockedLayout(
        [1, store_width],
        [warp_threads_m, warp_threads_n],
        [num_warps, 1],
        [1, 0],
    )


@gluon.jit
def _store_dequantized_tile(
    result,
    output_ptr,
    pid_m,
    pid_n,
    m,
    n,
    output_row_stride,
    second,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_warps: gl.constexpr,
    paired: gl.constexpr,
):
    """Store a dequantized output tile through a 16-byte vectorized row-major layout."""
    store_layout: gl.constexpr = _build_store_layout(
        output_ptr.dtype.element_ty.primitive_bitwidth, block_n, num_warps
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


@gluon.jit
def _store_swiglu_epilogue(
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
    mma_layout: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_warps: gl.constexpr,
    has_bias: gl.constexpr,
    second_has_bias: gl.constexpr,
    chunk_m: gl.constexpr = _GL_SWIGLU_CHUNK_M,
):
    """Stage 16-bit up/gate projections in SMEM and store fused SwiGLU in row chunks."""
    offsets_m = pid_m * block_m + gl.arange(0, block_m, gl.SliceLayout(1, mma_layout))
    offsets_n = pid_n * block_n + gl.arange(0, block_n, gl.SliceLayout(0, mma_layout))
    input_scale = gl.load(input_scale_ptr + offsets_m, mask=offsets_m < m, other=0.0)
    # Store up before dequantizing gate so only one FP32 tile is live at a time.
    projection_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    up_shared = gl.allocate_shared_memory(
        output_ptr.dtype.element_ty, [block_m, block_n], projection_shared_layout
    )
    gate_shared = gl.allocate_shared_memory(
        output_ptr.dtype.element_ty, [block_m, block_n], projection_shared_layout
    )
    up_shared.store(
        _dequantize_tile(
            up_accumulator,
            input_scale,
            weight_scale_ptr,
            bias_ptr,
            offsets_n,
            n,
            has_bias,
            output_ptr.dtype.element_ty,
        )
    )
    gate_shared.store(
        _dequantize_tile(
            gate_accumulator,
            input_scale,
            second_scale_ptr,
            second_bias_ptr,
            offsets_n,
            n,
            second_has_bias,
            output_ptr.dtype.element_ty,
        )
    )
    gl.barrier()

    chunk_layout: gl.constexpr = _build_store_layout(
        output_ptr.dtype.element_ty.primitive_bitwidth, block_n, num_warps
    )
    columns = pid_n * block_n + gl.arange(0, block_n, gl.SliceLayout(0, chunk_layout))
    column_valid = columns < n
    for chunk_row in gl.static_range(0, block_m, chunk_m):
        up_chunk = up_shared.slice(chunk_row, chunk_m, dim=0).load(chunk_layout).to(gl.float32)
        gate_chunk = gate_shared.slice(chunk_row, chunk_m, dim=0).load(chunk_layout).to(gl.float32)
        activated = (up_chunk * (gate_chunk / (1.0 + gl.exp(-gate_chunk)))).to(
            output_ptr.dtype.element_ty
        )
        rows = pid_m * block_m + chunk_row + gl.arange(0, chunk_m, gl.SliceLayout(1, chunk_layout))
        output_pointers = (
            output_ptr + rows.to(gl.int64)[:, None] * output_row_stride + columns[None, :]
        )
        gl.store(output_pointers, activated, mask=(rows[:, None] < m) & column_valid[None, :])


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
    pid_m, pid_n, column_tiles = _cta_tile_coordinates(m, n, block_m, block_n, group_m, paired)
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
    copy_layout: gl.constexpr = _build_copy_layout(block_k, num_warps)
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
            block_k,
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
            block_k,
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
        block_m,
        block_n,
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
        block_m,
        block_n,
        num_warps,
        paired,
    )


@gluon.jit(do_not_specialize=["m"])
def _int8_swiglu_matmul_kernel(
    input_ptr,
    weight_ptr,
    second_weight_ptr,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    second_scale_ptr,
    second_bias_ptr,
    m,
    n,
    k,
    output_row_stride,
    block_m: gl.constexpr,
    block_k: gl.constexpr,
    stages: gl.constexpr,
    group_m: gl.constexpr,
    has_bias: gl.constexpr,
    second_has_bias: gl.constexpr,
    whole_k_tiles: gl.constexpr,
):
    """Project paired async-copy INT8 weights and fuse 16-bit SwiGLU without spills.

    Uses 8 warps and K32 MMA slices so both 128x64 and 256x64 schedules stay
    spill-free across SM80/SM86/SM89/SM120 while reusing pipeline shared memory
    across 32-row SwiGLU chunks and preserving the 16-bit rounding boundary.
    """
    three_or_four: gl.constexpr = stages == 3 or stages == 4  # noqa: PLR1714
    gl.static_assert(three_or_four, "Async-copy GEMM uses three or four stages")
    gl.static_assert(
        output_ptr.dtype.element_ty.primitive_bitwidth == 16,
        "Async-copy SwiGLU epilogue supports 16-bit outputs",
    )
    num_warps: gl.constexpr = _GL_SWIGLU_WARPS_M * _GL_WARPS_N
    pid_m, pid_n, _ = _cta_tile_coordinates(m, n, block_m, _GL_SWIGLU_BLOCK_N, group_m, False)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[_GL_SWIGLU_WARPS_M, _GL_WARPS_N], instr_shape=[16, 8]
    )
    input_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, k_width=4)
    weight_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, k_width=4)
    copy_layout: gl.constexpr = _build_copy_layout(block_k, num_warps)
    input_rows = pid_m * block_m + gl.arange(0, block_m, gl.SliceLayout(1, copy_layout))
    weight_rows = pid_n * _GL_SWIGLU_BLOCK_N + gl.arange(
        0, _GL_SWIGLU_BLOCK_N, gl.SliceLayout(1, copy_layout)
    )
    columns = gl.arange(0, block_k, gl.SliceLayout(0, copy_layout))
    input_pointers = input_ptr + input_rows.to(gl.int64)[:, None] * k + columns[None, :]
    weight_pointers = weight_ptr + weight_rows.to(gl.int64)[:, None] * k + columns[None, :]
    second_weight_pointers = (
        second_weight_ptr + weight_rows.to(gl.int64)[:, None] * k + columns[None, :]
    )
    num_k = gl.cdiv(k, block_k)
    up_accumulator = gl.zeros([block_m, _GL_SWIGLU_BLOCK_N], gl.int32, mma_layout)
    gate_accumulator = gl.zeros([block_m, _GL_SWIGLU_BLOCK_N], gl.int32, mma_layout)
    interior = ((pid_m + 1) * block_m <= m) & ((pid_n + 1) * _GL_SWIGLU_BLOCK_N <= n)
    if whole_k_tiles and interior:
        up_accumulator, gate_accumulator = _accumulate_pair(
            input_pointers,
            weight_pointers,
            second_weight_pointers,
            input_rows < m,
            weight_rows < n,
            columns,
            k,
            num_k,
            up_accumulator,
            gate_accumulator,
            block_k,
            stages,
            input_layout,
            weight_layout,
            False,
            _GL_SWIGLU_MMA_K,
            True,
        )
    else:
        up_accumulator, gate_accumulator = _accumulate_pair(
            input_pointers,
            weight_pointers,
            second_weight_pointers,
            input_rows < m,
            weight_rows < n,
            columns,
            k,
            num_k,
            up_accumulator,
            gate_accumulator,
            block_k,
            stages,
            input_layout,
            weight_layout,
            True,
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
        block_m,
        _GL_SWIGLU_BLOCK_N,
        num_warps,
        has_bias,
        second_has_bias,
    )


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
    """Launch one dual-accumulator async-copy Gluon GEMM with fused 16-bit SwiGLU."""
    plan = execution_plan
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    block_m, block_k = plan.matmul_block_m, plan.matmul_block_k
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, _SWIGLU_BLOCK_N),)
    with device_context(input_qdata.device):
        _int8_swiglu_matmul_kernel[grid](
            input_qdata,
            weight_qdata,
            second_weight,
            output,
            input_scale,
            weight_scale,
            bias if bias is not None else output,
            second_scale,
            second_bias if second_bias is not None else output,
            m,
            n,
            k,
            output.stride(0),
            block_m,
            block_k,
            plan.matmul_num_stages,
            plan.matmul_group_m,
            bias is not None,
            second_bias is not None,
            k % block_k == 0,
            num_warps=_SWIGLU_NUM_WARPS,
        )
