"""Triton preparation and GEMM launchers for NVIDIA ConvRot INT8."""

# Triton launch options are not represented in Python call signatures.
# pyright: reportCallIssue=false

import torch
import triton

from piper_kernels._input_activations import input_activation_width
from piper_kernels._triton.convrot_int8 import quantize_rows_kernel, rotate_quantize_rows_kernel
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8._packing import fused_preparation_chunks

from .._kernels.triton import int8_matmul_kernel
from . import policy
from ._plan import NvidiaExecutionPlan

# A runtime M and per-tile tail branch keep one compiled kernel per schedule.
dynamic_m_int8_matmul_kernel = triton.jit(do_not_specialize=["m"])(int8_matmul_kernel.fn)


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
    plan: NvidiaExecutionPlan,
) -> None:
    """Launch the shared INT8 arithmetic with the plan's explicit Triton schedule."""
    m, k = input_qdata.shape
    n = weight_qdata.shape[0]
    num_n_tiles = triton.cdiv(n, plan.matmul_block_n) * (2 if paired else 1)
    row_block_count = triton.cdiv(m, plan.matmul_block_m)
    group_m = plan.matmul_group_m
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
