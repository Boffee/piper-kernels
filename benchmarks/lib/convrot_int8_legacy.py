"""Original ConvRot INT8 GEMM retained only as a benchmark/correctness control."""

# pyright: reportCallIssue=false
# ruff: noqa: ANN001, ANN201, PLR0913, PLR0917

import math

import torch
import triton
import triton.language as tl

from piper_kernels.linear.convrot.int8._kernels.triton import scaled_int8_matmul
from piper_kernels.linear.convrot.int8._plan import LinearExecutionPlan


@triton.jit
def legacy_int8_matmul_kernel(
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
    row_block_offset,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
    paired: tl.constexpr,
    second_has_bias: tl.constexpr,
    aligned_tiles: tl.constexpr,
    group_m: tl.constexpr,
):
    if group_m:
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(n, block_n) * (2 if paired else 1)
        row_block_count = tl.num_programs(0) // num_pid_n
        num_pid_in_group = group_m * num_pid_n
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * group_m
        actual_group_m = tl.minimum(row_block_count - first_pid_m, group_m)
        pid_m = first_pid_m + pid_in_group % actual_group_m + row_block_offset
        pid_n = pid_in_group // actual_group_m
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    second = pid_n >= tl.cdiv(n, block_n)
    if paired:
        pid_n %= tl.cdiv(n, block_n)
        weight_ptr = tl.where(second, second_weight_ptr, weight_ptr)
        weight_scale_ptr = tl.where(second, second_scale_ptr, weight_scale_ptr)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_n_i64 = offsets_n.to(tl.int64)
    result = scaled_int8_matmul(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        offsets_m,
        offsets_n,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        aligned_tiles,
    )
    if paired and (has_bias or second_has_bias):
        # Select FP32 values rather than pointers: biases may have different dtypes.
        bias = tl.full((block_n,), 0, tl.float32)
        second_bias = tl.full((block_n,), 0, tl.float32)
        if has_bias:
            bias = tl.load(bias_ptr + offsets_n, (offsets_n < n) & ~second, other=0.0).to(
                tl.float32
            )
        if second_has_bias:
            second_bias = tl.load(
                second_bias_ptr + offsets_n, (offsets_n < n) & second, other=0.0
            ).to(tl.float32)
        result += tl.where(second, second_bias, bias)[None, :]
    elif has_bias:
        if aligned_tiles:
            bias = tl.load(bias_ptr + offsets_n)
        else:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < n, other=0.0)
        result += bias[None, :]

    output_pointers = (
        output_ptr + offsets_m_i64[:, None] * output_row_stride + offsets_n_i64[None, :]
    )
    if paired:
        output_pointers += second * n
    if aligned_tiles:
        tl.store(output_pointers, result)
    else:
        tl.store(
            output_pointers,
            result,
            mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
        )


def legacy_matmul(
    prepared: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    plan: LinearExecutionPlan,
    *,
    bias: torch.Tensor | None = None,
    second_projection: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
    split_tail: bool = True,
) -> torch.Tensor:
    """Reproduce the old launch policy, or its fully masked single-launch control."""
    qdata, row_scales = prepared
    m, k, n = math.prod(qdata.shape[:-1]), qdata.shape[-1], weight.shape[0]
    paired = second_projection is not None
    second_weight, second_scale, second_bias = (
        (weight, scale, None) if second_projection is None else second_projection
    )
    group_m = 16 if (plan.matmul_block_m, plan.matmul_block_n) == (128, 256) else 0
    num_n_tiles = triton.cdiv(n, plan.matmul_block_n) * (2 if paired else 1)
    if not m or not n:
        return out

    def launch(row_blocks: int, offset: int, *, aligned_m: bool) -> None:
        grid = (row_blocks * num_n_tiles,) if group_m else (row_blocks, num_n_tiles)
        legacy_int8_matmul_kernel[grid](
            qdata,
            weight,
            out,
            row_scales,
            scale,
            bias if bias is not None else out,
            second_weight,
            second_scale,
            second_bias if second_bias is not None else out,
            m,
            n,
            k,
            out.stride(-2),
            offset,
            block_m=plan.matmul_block_m,
            block_n=plan.matmul_block_n,
            block_k=plan.matmul_block_k,
            has_bias=bias is not None,
            paired=paired,
            second_has_bias=second_bias is not None,
            aligned_tiles=aligned_m
            and n % plan.matmul_block_n == 0
            and k % plan.matmul_block_k == 0,
            group_m=group_m,
            num_warps=plan.matmul_num_warps,
            num_stages=plan.matmul_num_stages,
        )

    if split_tail and group_m:
        if m // plan.matmul_block_m:
            launch(m // plan.matmul_block_m, 0, aligned_m=True)
        if m % plan.matmul_block_m:
            launch(1, m // plan.matmul_block_m, aligned_m=False)
    else:
        launch(triton.cdiv(m, plan.matmul_block_m), 0, aligned_m=m % plan.matmul_block_m == 0)
    return out
