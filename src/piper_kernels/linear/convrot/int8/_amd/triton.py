"""ROCm ConvRot INT8 preparation, RDNA4 paired/residual SRD kernels, and launchers.

Shared host orchestration (buffer allocation, reshaping, target/plan resolution,
and standalone residual fallback) lives in ``.._backend``. Matrix launches pass
LLVM attributes from ``_triton.amd_compiler_workarounds`` so RDNA4 LDS barriers
omit redundant ``global_inv`` fences, and long ``gfx1201`` reductions seed INT32
accumulators with ``0x155555`` (see ``policy.matmul_accumulator_seed``) to avoid
signed zero-crossing clock drops.
"""

# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

from piper_kernels._input_activations import input_activation_width
from piper_kernels._residual import ResidualAdd
from piper_kernels._residual.triton import add_residual
from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget
from piper_kernels.weights.convrot.int8.triton import (
    quantize_input,  # noqa: F401 - shared LinearBackend kernel contract
    rotate_quantize_rows_kernel,
)

from .._plan import LinearExecutionPlan
from ..triton import launch_int8_matmul
from . import policy


def fused_rotate_quantize_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    input_qdata: torch.Tensor,
    row_scales: torch.Tensor,
    group_size: int,
    *,
    activation_fn: str | None = None,
    input_scale: torch.Tensor | None = None,
    num_warps: int,
    target: AcceleratorTarget | None = None,
) -> None:
    """Rotate and quantize to ``input_qdata`` without a rotated intermediate.

    ``input_qdata`` defines the activated row width. SwiGLU requires a raw
    ``[up | gate]`` input with twice that width.
    """
    if input.ndim != 2 or input_qdata.ndim != 2:
        raise ValueError(
            "fused preparation tensors must be 2-D, "
            f"got shapes {tuple(input.shape)} and {tuple(input_qdata.shape)}"
        )
    m, k = input_qdata.shape
    expected_input_shape = (m, k * input_activation_width(activation_fn))
    if tuple(input.shape) != expected_input_shape:
        raise ValueError(
            f"fused preparation input must have shape {expected_input_shape}, "
            f"got {tuple(input.shape)}"
        )
    target = AcceleratorTarget.from_device(input.device) if target is None else target
    if not policy.supports_target(target):
        raise ValueError(f"ConvRot INT8 execution has no optimized policy for {target}")
    if not policy.supports_fused_preparation(k):
        raise ValueError(f"fused preparation does not support row width {k}")
    blocks = policy.preparation_blocks(k)
    if input.dtype is torch.bfloat16 and blocks[1] != 0:
        block0, block1, block2 = blocks
        fold_norm = True
    else:
        block0, block1, block2 = max(128, triton.next_power_of_2(k)), 0, 0
        fold_norm = input.dtype is not torch.float16 and block0 > 8_192
    with device_context(input.device):
        rotate_quantize_rows_kernel[(m,)](
            input,
            input_qdata,
            row_scales,
            k,
            block0=block0,
            block1=block1,
            block2=block2,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=activation_fn,
            accelerator_backend="hip",
            gguf_quant_type=-1,
            static_scale_ptr=input_scale,
            fold_norm_into_scale=fold_norm,
            num_warps=num_warps,
        )


def _amd_matmul_compiler_options(
    target: AcceleratorTarget,
) -> dict[str, object]:
    """Return RDNA4 scheduling and LDS-only barrier attributes."""
    if not target.is_hip_architecture(*GFX120X_ARCHITECTURES):
        return {}
    fence_attrs = amd_compiler_workarounds.lds_only_fence_options(True).get("llvm_fn_attrs", ())
    assert isinstance(fence_attrs, tuple)
    return {
        "llvm_fn_attrs": (
            ("amdgpu-sched-strategy", "iterative-ilp"),
            # Barriers here only guard LDS tile reuse, so prefetches need not drain.
            *fence_attrs,
        )
    }


_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _fence_register_value(value):
    """Sequence AMDGPU register scheduling to bound VGPR pressure in wide epilogues.

    In 256-column residual tiles, splitting the epilogue into two 128-column
    halves separated by this zero-cost inline-asm fence and ``tl.debug_barrier()``
    prevents AMDGPU from hoisting residual global loads across both live
    ``128x256`` INT32 accumulators. In paired 16-bit SwiGLU epilogues, fencing
    each rounded ``128x128`` projection before gate dequantization keeps FP16
    bias kernels at 232 VGPRs with zero spills.
    """
    return tl.inline_asm_elementwise(
        ";;", "=v,0", [value], dtype=value.dtype, is_pure=False, pack=1
    )


@triton.jit
def _amd_rcp_f32(value):
    """Compute single-instruction hardware FP32 reciprocal on AMDGPU."""
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [value], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _amd_fast_swiglu(up, gate):
    """Apply ``up * silu(gate)`` via ``exp2`` + ``v_rcp_f32`` in GEMM epilogues.

    Avoids Triton/HIP's multi-instruction IEEE division and ``tl.exp``
    range-reduction wrapper across 64 elements/lane to keep zero VGPR spills.
    Matches eager 16-bit SwiGLU within ``2 * eps`` across all FP16 values and
    normal BF16 activations; for extreme negative BF16 inputs (``gate <= -88.5``),
    ``1.0 + exp2(-gate * log2(e))`` exceeds ``2**126`` and ``v_rcp_f32`` flushes
    the subnormal FP32 reciprocal to ``0.0``.
    """
    return up * (gate * _amd_rcp_f32(1.0 + tl.exp2(-gate * _LOG2E)))


@triton.jit(do_not_specialize=["m"])
def _paired_int8_matmul_kernel(
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
    output_row_stride,
    n: tl.constexpr,
    k: tl.constexpr,
    has_bias: tl.constexpr,
    second_has_bias: tl.constexpr,
    full_rows: tl.constexpr,
    accumulator_seed: tl.constexpr,
    fuse_swiglu: tl.constexpr,
):
    """Project one prepared input into two adjacent 128-column-tiled weights.

    Selected by ``policy.supports_paired_matmul`` and
    ``policy.supports_swiglu_epilogue`` on aligned ``gfx1201`` ``(128, 256, 64)``
    shapes with ``rows > 64`` and signed 32-bit element/byte offsets for SRD
    buffer addressing. Reuses each ``128x64`` activation tile across both
    128-column INT32 accumulators; when ``fuse_swiglu`` is enabled, rounds each
    projection to ``output_ptr``'s 16-bit dtype and stores ``up * silu(gate)``.
    """
    # Caller validates that 32-bit element indexing fits inputs, weights, and outputs.
    pid = tl.program_id(0)
    row_tiles = tl.cdiv(m, 128)
    col_tiles: tl.constexpr = n // 128
    group_n: tl.constexpr = tl.constexpr(4)
    first_col = pid // (row_tiles * group_n) * group_n
    cols_in_group = tl.minimum(group_n, col_tiles - first_col)
    local = pid % (row_tiles * group_n)
    rows = local // cols_in_group * 128 + tl.arange(0, 128)
    cols = (first_col + local % cols_in_group) * 128 + tl.arange(0, 128)
    # Repeat the final valid input row on partial tiles; stores mask out-of-range rows.
    source_rows = rows if full_rows else tl.minimum(rows, m - 1)
    offsets_k = tl.arange(0, 64)
    activation_pointers = input_ptr + source_rows[:, None] * k + offsets_k[None, :]
    weight_offsets = cols[None, :] * k + offsets_k[:, None]
    first_weight_pointers = weight_ptr + weight_offsets
    second_weight_pointers = second_weight_ptr + weight_offsets
    first_accumulator = tl.full((128, 128), accumulator_seed, tl.int32)
    second_accumulator = tl.full((128, 128), accumulator_seed, tl.int32)
    for _ in range(k // 64):
        activation = tl.load(activation_pointers)
        first_weight = tl.load(first_weight_pointers)
        second_weight = tl.load(second_weight_pointers)
        first_accumulator = tl.dot(activation, first_weight, first_accumulator)
        second_accumulator = tl.dot(activation, second_weight, second_accumulator)
        activation_pointers += 64
        first_weight_pointers += 64
        second_weight_pointers += 64
    row_scale = tl.load(input_scale_ptr + source_rows)
    first_result = (
        (first_accumulator - accumulator_seed).to(tl.float32)
        * row_scale[:, None]
        * tl.load(weight_scale_ptr + cols)[None, :]
    )
    if has_bias:
        first_result += tl.load(bias_ptr + cols).to(tl.float32)[None, :]
    if fuse_swiglu:
        # Round to 16-bit and fence before dequantizing the gate tile so FP16
        # bias kernels stay at 232 VGPRs with zero spills.
        first_rounded = _fence_register_value(first_result.to(output_ptr.dtype.element_ty))
    second_result = (
        (second_accumulator - accumulator_seed).to(tl.float32)
        * row_scale[:, None]
        * tl.load(second_scale_ptr + cols)[None, :]
    )
    if second_has_bias:
        second_result += tl.load(second_bias_ptr + cols).to(tl.float32)[None, :]
    dest = output_ptr + rows[:, None] * output_row_stride + cols[None, :]
    valid_rows = True if full_rows else rows[:, None] < m
    if fuse_swiglu:
        second_rounded = _fence_register_value(second_result.to(output_ptr.dtype.element_ty))
        activated = _amd_fast_swiglu(
            first_rounded.to(tl.float32),  # pyright: ignore[reportPossiblyUnboundVariable]
            second_rounded.to(tl.float32),
        )
        tl.store(dest, activated, mask=valid_rows)
    else:
        tl.store(dest + n, second_result, mask=valid_rows)
        tl.store(dest, first_result, mask=valid_rows)


def _launch_paired_int8_matmul(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    second_weight: torch.Tensor,
    second_scale: torch.Tensor,
    second_bias: torch.Tensor | None,
    *,
    target: AcceleratorTarget,
    fuse_swiglu: bool,
) -> None:
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    output_tiles = triton.cdiv(m, 128) * (n // 128)
    with device_context(input_qdata.device):
        _paired_int8_matmul_kernel[(output_tiles,)](
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
            output.stride(0),
            n=n,
            k=k,
            has_bias=bias is not None,
            second_has_bias=second_bias is not None,
            full_rows=m % 128 == 0,
            accumulator_seed=policy.matmul_accumulator_seed(target, k, output_tiles),
            fuse_swiglu=fuse_swiglu,
            **_amd_matmul_compiler_options(target),
            num_stages=2,
            num_warps=8,
        )


def launch_prepared_linear(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    second_weight: torch.Tensor,
    second_scale: torch.Tensor,
    second_bias: torch.Tensor | None,
    *,
    paired: bool,
    logical_dtype: torch.dtype,
    execution_plan: LinearExecutionPlan,
    target: AcceleratorTarget | None = None,
) -> None:
    """Launch RDNA4 paired SRD GEMM or shared INT8 GEMM for validated 2-D operands."""
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    plan = execution_plan
    target = AcceleratorTarget.from_device(input_qdata.device) if target is None else target
    if paired and policy.supports_paired_matmul(
        target,
        plan,
        logical_dtype,
        rows=m,
        in_features=k,
        out_features=n,
        output_row_stride=output.stride(0),
    ):
        _launch_paired_int8_matmul(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            output,
            second_weight,
            second_scale,
            second_bias,
            target=target,
            fuse_swiglu=False,
        )
        return
    launch_int8_matmul(
        input_qdata,
        weight_qdata,
        output,
        input_scale,
        weight_scale,
        bias,
        second_weight,
        second_scale,
        second_bias,
        paired=paired,
        execution_plan=plan,
        group_m=policy.matmul_group_m(plan),
        target=target,
        compiler_options=_amd_matmul_compiler_options(target),
    )


@triton.jit
def _store_amd_residual_half(
    centered_accumulator,
    cols,
    barrier_before_residual: tl.constexpr,
    row_scale,
    weight_scale_ptr,
    bias_ptr,
    base_ptr,
    output_ptr,
    gate_ptr,
    reusable_gate_ptr,
    indices_ptr,
    rows,
    valid_rows,
    n: tl.constexpr,
    row_offset,
    gate_row_stride: tl.constexpr,
    gate_rows,
    reusable_gate_row_stride: tl.constexpr,
    reusable_gate_rows,
    has_bias: tl.constexpr,
    has_reusable_update: tl.constexpr,
    python_indexing: tl.constexpr,
):
    """Dequantize, round, add gated residual, and store one 128-column half."""
    result = (
        centered_accumulator.to(tl.float32)
        * row_scale[:, None]
        * tl.load(weight_scale_ptr + cols)[None, :]
    )
    if has_bias:
        result += tl.load(bias_ptr + cols).to(tl.float32)[None, :]
    # Round to the logical dtype in registers before FP32 residual math.
    rounded = _fence_register_value(result.to(output_ptr.dtype.element_ty))
    if barrier_before_residual:
        tl.debug_barrier()
    output = add_residual(
        rounded.to(tl.float32),
        base_ptr,
        output_ptr,
        gate_ptr,
        reusable_gate_ptr,
        indices_ptr,
        rows[:, None],
        cols[None, :],
        valid_rows,
        valid_rows,
        n,
        row_offset,
        gate_row_stride,
        gate_rows,
        reusable_gate_row_stride,
        reusable_gate_rows,
        has_reusable_update,
        python_indexing,
    )
    tl.store(
        output_ptr + rows.to(tl.int64)[:, None] * n + cols[None, :],
        output,
        mask=valid_rows,
    )


@triton.jit(do_not_specialize=["m", "row_offset", "gate_rows", "reusable_gate_rows"])
def _amd_int8_residual_matmul_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    base_ptr,
    gate_ptr,
    reusable_gate_ptr,
    indices_ptr,
    row_offset,
    gate_rows,
    reusable_gate_rows,
    m,
    n: tl.constexpr,
    k: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    full_rows: tl.constexpr,
    gate_row_stride: tl.constexpr,
    reusable_gate_row_stride: tl.constexpr,
    has_reusable_update: tl.constexpr,
    python_indexing: tl.constexpr,
    accumulator_seed: tl.constexpr,
):
    """Project and fuse a gated residual update across one or two 128-column halves.

    Selected on aligned ``gfx1201`` 16-bit projections with ``rows > 64`` and
    signed 32-bit SRD offsets. Rounds each 128-column half to its logical dtype in
    registers before calling shared Triton residual arithmetic; 256-column tiles
    sequence the two halves with ``_fence_register_value`` and ``tl.debug_barrier()``
    so residual global loads do not hoist across both live INT32 accumulators.
    """
    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = n // block_n
    row_block_count = tl.num_programs(0) // num_pid_n
    group_id = pid // (8 * num_pid_n)
    pid_in_group = pid % (8 * num_pid_n)
    first_pid_m = group_id * 8
    actual_group_m = tl.minimum(row_block_count - first_pid_m, 8)
    pid_m = first_pid_m + pid_in_group % actual_group_m
    pid_n = pid_in_group // actual_group_m

    rows = pid_m * 128 + tl.arange(0, 128)
    source_rows = rows if full_rows else tl.minimum(rows, m - 1)
    first_cols = pid_n * block_n + tl.arange(0, 128)
    offsets_k = tl.arange(0, block_k)
    activation_pointers = input_ptr + source_rows[:, None] * k + offsets_k[None, :]
    first_weight_pointers = weight_ptr + first_cols[None, :] * k + offsets_k[:, None]
    first_accumulator = tl.full((128, 128), accumulator_seed, tl.int32)
    split_n: tl.constexpr = tl.constexpr(block_n == 256)
    second_cols = first_cols + 128
    second_weight_pointers = weight_ptr + second_cols[None, :] * k + offsets_k[:, None]
    second_accumulator = tl.full((128, 128), accumulator_seed, tl.int32)
    for _ in range(k // block_k):
        activation = tl.load(activation_pointers)
        first_accumulator = tl.dot(activation, tl.load(first_weight_pointers), first_accumulator)
        if split_n:
            second_accumulator = tl.dot(
                activation, tl.load(second_weight_pointers), second_accumulator
            )
            second_weight_pointers += block_k
        activation_pointers += block_k
        first_weight_pointers += block_k

    row_scale = tl.load(input_scale_ptr + source_rows)
    valid_rows = True if full_rows else rows[:, None] < m
    epilogue_args = (
        row_scale,
        weight_scale_ptr,
        bias_ptr,
        base_ptr,
        output_ptr,
        gate_ptr,
        reusable_gate_ptr,
        indices_ptr,
        rows,
        valid_rows,
        n,
        row_offset,
        gate_row_stride,
        gate_rows,
        reusable_gate_row_stride,
        reusable_gate_rows,
        has_bias,
        has_reusable_update,
        python_indexing,
    )
    _store_amd_residual_half(
        first_accumulator - accumulator_seed,
        first_cols,
        split_n,
        *epilogue_args,
    )
    if split_n:
        tl.debug_barrier()
        _store_amd_residual_half(
            _fence_register_value(second_accumulator - accumulator_seed),
            second_cols,
            tl.constexpr(False),
            *epilogue_args,
        )


def supports_prepared_swiglu(
    weight_qdata: torch.Tensor,
    second_weight: torch.Tensor,
    logical_dtype: torch.dtype,
    *,
    rows: int | None = None,
    execution_plan: LinearExecutionPlan,
    target: AcceleratorTarget,
) -> bool:
    """Return whether paired INT8 projections can fuse SwiGLU on this target."""
    if (
        weight_qdata.ndim != 2
        or second_weight.shape != weight_qdata.shape
        or not weight_qdata.is_contiguous()
        or not second_weight.is_contiguous()
    ):
        return False
    out_features, in_features = weight_qdata.shape
    return policy.supports_swiglu_epilogue(
        target,
        execution_plan,
        logical_dtype,
        in_features=in_features,
        out_features=out_features,
        rows=rows,
    )


def try_launch_prepared_swiglu(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    second_weight: torch.Tensor,
    second_scale: torch.Tensor,
    second_bias: torch.Tensor | None,
    *,
    execution_plan: LinearExecutionPlan,
    target: AcceleratorTarget,
) -> bool:
    """Launch fused RDNA4 SwiGLU GEMM when supported; return False for fallback."""
    if (
        not input_qdata.is_contiguous()
        or not weight_qdata.is_contiguous()
        or not second_weight.is_contiguous()
        or second_weight.shape != weight_qdata.shape
    ):
        return False
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    if not policy.supports_swiglu_epilogue(
        target,
        execution_plan,
        output.dtype,
        in_features=k,
        out_features=n,
        rows=m,
        output_row_stride=output.stride(0),
    ):
        return False
    _launch_paired_int8_matmul(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        output,
        second_weight,
        second_scale,
        second_bias,
        target=target,
        fuse_swiglu=True,
    )
    return True


def try_launch_prepared_residual(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    residual_add: ResidualAdd,
    *,
    row_offset: int,
    execution_plan: LinearExecutionPlan,
    target: AcceleratorTarget,
) -> bool:
    """Launch the RDNA4 fused residual GEMM when supported; return False for fallback."""
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    plan = execution_plan
    if not policy.supports_residual_epilogue(
        target,
        plan,
        residual_add.base.dtype,
        rows=m,
        in_features=k,
        out_features=n,
    ):
        return False
    if m and n:
        base = residual_add.base.view(-1, n)[row_offset : row_offset + m]
        output_tiles = triton.cdiv(m, 128) * (n // plan.matmul_block_n)
        gate = residual_add.gate
        reusable = residual_add.reusable
        reusable_gate = gate if reusable is None else reusable.gate
        with device_context(input_qdata.device):
            _amd_int8_residual_matmul_kernel[(output_tiles,)](
                input_qdata,
                weight_qdata,
                output,
                input_scale,
                weight_scale,
                bias if bias is not None else output,
                base,
                gate,
                reusable_gate,
                residual_add.gate_indices,
                row_offset,
                gate.shape[0],
                0 if reusable is None else reusable.gate.shape[0],
                m,
                n=n,
                k=k,
                block_n=plan.matmul_block_n,
                block_k=plan.matmul_block_k,
                has_bias=bias is not None,
                full_rows=m % 128 == 0,
                gate_row_stride=gate.stride(0),
                reusable_gate_row_stride=0 if reusable is None else reusable.gate.stride(0),
                has_reusable_update=reusable is not None,
                python_indexing=residual_add.python_indexing,
                accumulator_seed=policy.matmul_accumulator_seed(target, k, output_tiles),
                **_amd_matmul_compiler_options(target),
                num_stages=2,
                num_warps=8,
            )
    return True
