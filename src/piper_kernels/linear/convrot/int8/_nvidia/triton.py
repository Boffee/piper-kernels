"""NVIDIA ConvRot INT8 preparation and kernel-dialect launchers.

Routes validated 2-D GEMM and fused-residual calls between SM120 Gluon TMA
(``gluon_tma``), SM8x Gluon async-copy (``gluon_async_copy``), and shared Triton
kernels, falling back to the rounding-compatible grouped 128x64 Triton schedule
when operands do not meet the selected Gluon kernel's alignment contract.
"""

# Triton launch options are not represented in Python call signatures.
# pyright: reportCallIssue=false

import torch
import triton

from piper_kernels._input_activations import input_activation_width
from piper_kernels._residual import ResidualAdd
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks
from piper_kernels.weights.convrot.int8.triton import (
    quantize_rows_kernel,
    rotate_quantize_rows_kernel,
)

from .._plan import LinearExecutionPlan
from ..triton import (
    int8_matmul_kernel,
    int8_residual_matmul_kernel,
    try_launch_int8_matmul_splitk,
)
from . import gluon_async_copy, gluon_tma, policy
from .policy import NvidiaExecutionPlan

# A runtime M and per-tile tail branch keep one compiled kernel per schedule.
dynamic_m_int8_matmul_kernel = triton.jit(do_not_specialize=["m"])(int8_matmul_kernel.fn)
dynamic_m_int8_residual_matmul_kernel = triton.jit(do_not_specialize=["m"])(
    int8_residual_matmul_kernel.fn
)


def quantize_input(
    rotated: torch.Tensor,
    input_qdata: torch.Tensor,
    row_scales: torch.Tensor,
    *,
    input_scale: torch.Tensor | None = None,
    num_warps: int,
) -> None:
    """Apply the portable split-path rowwise quantization."""
    m, k = rotated.shape
    with device_context(rotated.device):
        quantize_rows_kernel[(m,)](
            rotated,
            input_qdata,
            row_scales,
            k,
            block_size=max(128, triton.next_power_of_2(k)),
            accelerator_backend="cuda",
            static_scale_ptr=input_scale,
            num_warps=num_warps,
        )


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
    if not policy.supports_preparation_target(target):
        raise ValueError(f"ConvRot INT8 preparation has no optimized policy for {target}")
    fused_chunks = fused_preparation_chunks(k)
    if fused_chunks is None:
        raise ValueError(f"fused preparation does not support row width {k}")
    chunk_count, chunk_size = fused_chunks
    with device_context(input.device):
        rotate_quantize_rows_kernel[(m,)](
            input,
            input_qdata,
            row_scales,
            k,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=activation_fn,
            static_scale_ptr=input_scale,
            accelerator_backend=target.backend,
            gguf_quant_type=-1,
            num_warps=num_warps,
        )


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
    """Launch the shared INT8 arithmetic with the plan's explicit Triton schedule."""
    plan = execution_plan
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    group_m = plan.matmul_group_m
    if try_launch_int8_matmul_splitk(
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
        block_m=plan.matmul_block_m,
        block_n=plan.matmul_block_n,
        block_k=plan.matmul_block_k,
        group_m=group_m,
        explicit_bias_fma=plan.matmul_explicit_bias_fma,
        num_warps=plan.matmul_num_warps,
        num_stages=plan.matmul_num_stages,
    ):
        return
    num_n_tiles = triton.cdiv(n, plan.matmul_block_n) * (2 if paired else 1)
    row_block_count = triton.cdiv(m, plan.matmul_block_m)
    grid = (row_block_count * num_n_tiles,) if group_m else (row_block_count, num_n_tiles)
    kernel = int8_matmul_kernel if plan.matmul_specialize_m else dynamic_m_int8_matmul_kernel
    with device_context(input_qdata.device):
        kernel[grid](
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
            block_m=plan.matmul_block_m,
            block_n=plan.matmul_block_n,
            block_k=plan.matmul_block_k,
            has_bias=bias is not None,
            paired=paired,
            second_has_bias=second_bias is not None,
            aligned_m=plan.matmul_specialize_m and m % plan.matmul_block_m == 0,
            aligned_nk=n % plan.matmul_block_n == 0 and k % plan.matmul_block_k == 0,
            group_m=group_m,
            explicit_bias_fma=plan.matmul_explicit_bias_fma,
            per_tile_tail=bool(group_m) or not plan.matmul_specialize_m,
            num_stages=plan.matmul_num_stages,
            num_warps=plan.matmul_num_warps,
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
    """Project into validated residual storage under a native-epilogue plan."""
    plan = execution_plan
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    num_n_tiles = triton.cdiv(n, plan.matmul_block_n)
    row_block_count = triton.cdiv(m, plan.matmul_block_m)
    group_m = plan.matmul_group_m
    grid = (row_block_count * num_n_tiles,) if group_m else (row_block_count, num_n_tiles)
    kernel = (
        int8_residual_matmul_kernel
        if plan.matmul_specialize_m
        else dynamic_m_int8_residual_matmul_kernel
    )
    gate = residual_add.gate
    reusable = residual_add.reusable
    reusable_gate = None if reusable is None else reusable.gate
    with device_context(input_qdata.device):
        kernel[grid](
            input_qdata,
            weight_qdata,
            output,
            input_scale,
            weight_scale,
            bias if bias is not None else output,
            m,
            n,
            k,
            block_m=plan.matmul_block_m,
            block_n=plan.matmul_block_n,
            block_k=plan.matmul_block_k,
            has_bias=bias is not None,
            aligned_m=plan.matmul_specialize_m and m % plan.matmul_block_m == 0,
            aligned_nk=n % plan.matmul_block_n == 0 and k % plan.matmul_block_k == 0,
            group_m=group_m,
            explicit_bias_fma=plan.matmul_explicit_bias_fma,
            per_tile_tail=bool(group_m) or not plan.matmul_specialize_m,
            residual_base_ptr=residual_add.base.view(-1, n)[row_offset:],
            residual_gate_ptr=gate,
            residual_indices_ptr=residual_add.gate_indices,
            residual_reusable_gate_ptr=reusable_gate,
            residual_row_offset=row_offset,
            residual_gate_row_stride=gate.stride(0),
            residual_gate_rows=gate.shape[0],
            residual_reusable_gate_row_stride=0
            if reusable_gate is None
            else reusable_gate.stride(0),
            residual_reusable_gate_rows=0 if reusable_gate is None else reusable_gate.shape[0],
            residual_has_reusable_update=reusable is not None,
            residual_python_indexing=residual_add.python_indexing,
            num_stages=plan.matmul_num_stages,
            num_warps=plan.matmul_num_warps,
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
    """Launch the plan's NVIDIA GEMM kernel, falling back to grouped Triton if unaligned."""
    del logical_dtype, target
    if not isinstance(execution_plan, NvidiaExecutionPlan):
        raise TypeError("NVIDIA ConvRot execution requires a NvidiaExecutionPlan")
    plan = execution_plan
    launcher = launch_int8_matmul
    if plan.matmul_kernel in ("gluon_async_copy", "gluon_tma"):
        aligned = _gluon_operands_supported(plan, input_qdata, weight_qdata)
        if paired:
            aligned = aligned and _gluon_operands_supported(plan, second_weight)
        if aligned:
            launcher = _gluon_module(plan).launch_int8_matmul
        else:
            plan = policy.grouped_triton_plan(plan)
    launcher(
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
    )


def _gluon_module(execution_plan: NvidiaExecutionPlan):
    if execution_plan.matmul_kernel == "gluon_tma":
        return gluon_tma
    if execution_plan.matmul_kernel == "gluon_async_copy":
        return gluon_async_copy
    raise ValueError(f"Unsupported Gluon matmul kernel: {execution_plan.matmul_kernel!r}")


def _gluon_operands_supported(execution_plan: NvidiaExecutionPlan, *operands: torch.Tensor) -> bool:
    if not _gluon_module(execution_plan).operands_aligned(*operands):
        return False
    if execution_plan.matmul_kernel == "gluon_tma":
        return True
    return all(
        operand.ndim == 2
        and operand.is_contiguous()
        and operand.shape[1] > 0
        and operand.shape[1] % 16 == 0
        for operand in operands
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
    """Return whether paired INT8 projections can fuse SwiGLU via Gluon TMA or async copy."""
    del rows
    if not isinstance(execution_plan, NvidiaExecutionPlan):
        raise TypeError("NVIDIA ConvRot execution requires a NvidiaExecutionPlan")
    if not policy.supports_swiglu_epilogue(target, execution_plan, logical_dtype):
        return False
    return _gluon_operands_supported(execution_plan, weight_qdata, second_weight)


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
    """Launch the dual-accumulator Gluon SwiGLU GEMM when supported; else False."""
    if not supports_prepared_swiglu(
        weight_qdata,
        second_weight,
        output.dtype,
        execution_plan=execution_plan,
        target=target,
    ):
        return False
    assert isinstance(execution_plan, NvidiaExecutionPlan)
    if not _gluon_operands_supported(execution_plan, input_qdata):
        return False
    m, _k = input_qdata.shape
    n = weight_qdata.shape[0]
    if m and n:
        _gluon_module(execution_plan).launch_int8_swiglu_matmul(
            input_qdata,
            weight_qdata,
            output,
            input_scale,
            weight_scale,
            bias,
            second_weight,
            second_scale,
            second_bias,
            execution_plan=execution_plan,
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
    """Launch an SM120 fused residual GEMM when supported; return False for fallback."""
    if not isinstance(execution_plan, NvidiaExecutionPlan):
        raise TypeError("NVIDIA ConvRot execution requires a NvidiaExecutionPlan")
    plan = execution_plan
    if not policy.supports_residual_epilogue(target, plan, residual_add.base.dtype):
        return False
    m, _k = input_qdata.shape
    n = weight_qdata.shape[0]
    if m and n:
        residual_launcher = launch_int8_residual_matmul
        if plan.matmul_kernel == "gluon_tma":
            if gluon_tma.operands_aligned(input_qdata, weight_qdata):
                residual_launcher = gluon_tma.launch_int8_residual_matmul
            else:
                plan = policy.grouped_triton_plan(plan)
        residual_launcher(
            input_qdata,
            weight_qdata,
            output,
            input_scale,
            weight_scale,
            bias,
            residual_add,
            row_offset=row_offset,
            execution_plan=plan,
        )
    return True
