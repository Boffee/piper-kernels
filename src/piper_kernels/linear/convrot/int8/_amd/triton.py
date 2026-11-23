"""ROCm ConvRot INT8 preparation, RDNA4 paired/residual SRD kernels, and launchers.

Shared host orchestration (buffer allocation, reshaping, target/plan resolution,
and standalone residual fallback) lives in ``.._backend``. Matrix launches pass
LLVM attributes from ``_triton.amd_compiler_workarounds`` so RDNA4 LDS barriers
omit redundant ``global_inv`` fences, and long ``gfx1201`` reductions seed INT32
accumulators with ``0x155555`` (see ``policy.matmul_accumulator_seed``) to avoid
signed zero-crossing clock drops.
"""

# pyright: reportCallIssue=false

import importlib

import torch
import triton
import triton.language as tl

from piper_kernels._input_activations import input_activation_width
from piper_kernels._input_activations.triton import gelu_tanh, swiglu
from piper_kernels._residual import ResidualAdd
from piper_kernels._residual.triton import add_residual
from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot import triton as convrot_backend
from piper_kernels.weights.convrot.int8.triton import (
    _quantize_int8,
    int8_scale_from_max,
    normalize_for_int8,
    quantize_rows_kernel,
    round_to_int8,
)

from .._plan import LinearExecutionPlan
from ..triton import int8_matmul_kernel
from . import policy


@triton.jit
def _normalize_rotated_values(
    values,
    inverse_sqrt_group: tl.constexpr,
):
    """Apply the portable group normalization and construct an INT8 row scale."""
    values *= inverse_sqrt_group
    scale = tl.maximum(
        int8_scale_from_max(tl.max(tl.abs(values).to(tl.float32), axis=0), tl.constexpr(True)),
        1e-30,
    )
    return normalize_for_int8(values, scale), scale


@triton.jit
def _amd_normalization_scale(
    absolute_max,
    inverse_sqrt_group: tl.constexpr,
):
    """Construct the value scale used by AMD's folded normalization."""
    return tl.maximum(
        int8_scale_from_max(absolute_max, tl.constexpr(True)),
        1e-30 / inverse_sqrt_group,
    )


@triton.jit
def _normalize_rotated_values_amd_wide(
    values,
    inverse_sqrt_group: tl.constexpr,
):
    """Fold wide AMD group normalization into the stored row scale."""
    # Every supported group has an exact power-of-two normalization factor.
    # Wide HIP BF16/FP32 quantization is invariant to that common factor, so
    # apply it once to the stored row scale instead of every rotated value.
    normalization_scale = _amd_normalization_scale(
        tl.max(tl.abs(values).to(tl.float32), axis=0),
        inverse_sqrt_group,
    )
    scale = normalization_scale * inverse_sqrt_group
    return normalize_for_int8(values, normalization_scale), scale


@triton.jit
def rotate_quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    static_scale_ptr=None,
):
    """Rotate and quantize one complete row without a global-memory intermediate."""
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    offsets = tl.arange(0, block_size)
    mask = offsets < row_width
    input_row_width = row_width * (2 if activation_fn == "swiglu" else 1)
    input_row_offset = row_i64 * input_row_width
    output_row_offset = row_i64 * row_width

    if activation_fn == "swiglu":
        up = tl.load(
            x_ptr + input_row_offset + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            x_ptr + input_row_offset + row_width + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = swiglu(up, gate)
    else:
        values = tl.load(
            x_ptr + input_row_offset + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if activation_fn == "gelu_tanh":
            values = gelu_tanh(values, accelerator_backend)

    values = convrot_backend.rotate_hadamard_groups(values, block_size, group_size)
    if static_scale_ptr is not None:
        scale = tl.load(static_scale_ptr)
        scaled = normalize_for_int8(values * inverse_sqrt_group, scale)
    elif (
        accelerator_backend != "hip" or x_ptr.dtype.element_ty == tl.float16 or block_size <= 8_192
    ):
        scaled, scale = _normalize_rotated_values(
            values,
            inverse_sqrt_group,
        )
    else:
        scaled, scale = _normalize_rotated_values_amd_wide(
            values,
            inverse_sqrt_group,
        )
    quantized = round_to_int8(scaled, accelerator_backend)
    tl.store(q_ptr + output_row_offset + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row_i64, scale)


@triton.jit
def _load_activated_input_chunk(
    x_ptr,
    input_row_offset,
    row_width,
    offsets,
    mask,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
):
    """Load one group-aligned chunk and apply the optional input activation."""
    if activation_fn == "swiglu":
        up = tl.load(x_ptr + input_row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(
            x_ptr + input_row_offset + row_width + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        return swiglu(up, gate)

    values = tl.load(x_ptr + input_row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    if activation_fn == "gelu_tanh":
        values = gelu_tanh(values, accelerator_backend)
    return values


@triton.jit
def rotate_quantize_rows_chunked_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block0: tl.constexpr,
    block1: tl.constexpr,
    block2: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
    static_scale_ptr=None,
):
    """Rotate a BF16 input row in FP32 chunks before terminal INT8 quantization.

    ConvRot groups never cross a chunk boundary. Avoiding a single next-power-
    of-two pad substantially reduces work and live storage for wide ragged rows.
    """
    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    input_row_width = row_width * (2 if activation_fn == "swiglu" else 1)
    input_row_offset = row_i64 * input_row_width
    output_row_offset = row_i64 * row_width

    offsets0 = tl.arange(0, block0)
    mask0 = offsets0 < row_width
    values0 = _load_activated_input_chunk(
        x_ptr,
        input_row_offset,
        row_width,
        offsets0,
        mask0,
        activation_fn,
        accelerator_backend,
    )
    values0 = convrot_backend.rotate_hadamard_groups(values0, block0, group_size)
    max0 = tl.max(tl.abs(values0).to(tl.float32), axis=0)

    offsets1 = block0 + tl.arange(0, block1)
    mask1 = offsets1 < row_width
    values1 = _load_activated_input_chunk(
        x_ptr,
        input_row_offset,
        row_width,
        offsets1,
        mask1,
        activation_fn,
        accelerator_backend,
    )
    values1 = convrot_backend.rotate_hadamard_groups(values1, block1, group_size)
    max1 = tl.max(tl.abs(values1).to(tl.float32), axis=0)
    absolute_max = tl.maximum(max0, max1)

    if block2:
        offsets2 = block0 + block1 + tl.arange(0, block2)
        mask2 = offsets2 < row_width
        values2 = _load_activated_input_chunk(
            x_ptr,
            input_row_offset,
            row_width,
            offsets2,
            mask2,
            activation_fn,
            accelerator_backend,
        )
        values2 = convrot_backend.rotate_hadamard_groups(values2, block2, group_size)
        max2 = tl.max(tl.abs(values2).to(tl.float32), axis=0)
        absolute_max = tl.maximum(absolute_max, max2)

    # All supported group normalizations are exact powers of two. Quantizing
    # the unnormalized FP32 values and applying that factor once to the scale
    # preserves the logical result while avoiding a vector multiply per chunk.
    if static_scale_ptr is None:
        normalization_scale = _amd_normalization_scale(absolute_max, inverse_sqrt_group)
        scale = normalization_scale * inverse_sqrt_group
    else:
        scale = tl.load(static_scale_ptr)
        normalization_scale = scale
        values0 = values0 * inverse_sqrt_group
        values1 = values1 * inverse_sqrt_group
        if block2:
            values2 = values2 * inverse_sqrt_group  # pyright: ignore[reportPossiblyUnboundVariable]
    quantized0 = _quantize_int8(values0, normalization_scale, accelerator_backend)
    quantized1 = _quantize_int8(values1, normalization_scale, accelerator_backend)
    tl.store(q_ptr + output_row_offset + offsets0, quantized0, mask=mask0)
    tl.store(q_ptr + output_row_offset + offsets1, quantized1, mask=mask1)
    if block2:
        quantized2 = _quantize_int8(values2, normalization_scale, accelerator_backend)  # pyright: ignore[reportPossiblyUnboundVariable]
        tl.store(q_ptr + output_row_offset + offsets2, quantized2, mask=mask2)  # pyright: ignore[reportPossiblyUnboundVariable]
    tl.store(scale_ptr + row_i64, scale)


def _uses_amd_chunked_preparation(
    target: AcceleratorTarget,
    input_dtype: torch.dtype,
    blocks: tuple[int, int, int],
) -> bool:
    """Return whether AMD prepares BF16 input in separate row segments."""
    return target.is_amd_hip and input_dtype is torch.bfloat16 and blocks[1] != 0


def _launch_amd_chunked_preparation(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    input_qdata: torch.Tensor,
    row_scales: torch.Tensor,
    group_size: int,
    activation_fn: str | None,
    num_warps: int,
    blocks: tuple[int, int, int],
    input_scale: torch.Tensor | None = None,
) -> None:
    """Launch AMD's chunked fused rotation and quantization path."""
    m, k = input_qdata.shape
    with device_context(input.device):
        rotate_quantize_rows_chunked_kernel[(m,)](
            input,
            input_qdata,
            row_scales,
            k,
            block0=blocks[0],
            block1=blocks[1],
            block2=blocks[2],
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=activation_fn,
            static_scale_ptr=input_scale,
            accelerator_backend="hip",
            num_warps=num_warps,
        )


def _launch_full_row_preparation(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    input_qdata: torch.Tensor,
    row_scales: torch.Tensor,
    group_size: int,
    activation_fn: str | None,
    num_warps: int,
    target: AcceleratorTarget,
    input_scale: torch.Tensor | None = None,
) -> None:
    """Launch the shared full-row fused rotation and quantization path."""
    m, k = input_qdata.shape
    with device_context(input.device):
        rotate_quantize_rows_kernel[(m,)](
            input,
            input_qdata,
            row_scales,
            k,
            block_size=max(128, triton.next_power_of_2(k)),
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            activation_fn=activation_fn,
            static_scale_ptr=input_scale,
            accelerator_backend=target.backend,
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
    if not policy.supports_target(target):
        raise ValueError(f"ConvRot INT8 execution has no optimized policy for {target}")
    if not policy.supports_fused_preparation(k):
        raise ValueError(f"fused preparation does not support row width {k}")
    blocks = policy.preparation_blocks(k)
    if _uses_amd_chunked_preparation(target, input.dtype, blocks):
        _launch_amd_chunked_preparation(
            input,
            input_qdata,
            row_scales,
            group_size,
            activation_fn,
            num_warps,
            blocks,
            input_scale,
        )
        return
    _launch_full_row_preparation(
        input,
        input_qdata,
        row_scales,
        group_size,
        activation_fn,
        num_warps,
        target,
        input_scale,
    )


def _amd_matmul_compiler_options(
    target: AcceleratorTarget,
) -> dict[str, object]:
    """Return optional AMD compiler controls supported by the installed Triton.

    On RDNA4 (``gfx1200``/``gfx1201``), adds ``LDS_ONLY_FENCE_ATTRS`` from
    ``_triton.amd_compiler_workarounds`` so LDS barriers omit redundant
    ``global_inv`` fences while keeping ``iterative-ilp`` scheduling.
    """
    if not target.is_architecture("gfx1200", "gfx1201"):
        return {}
    # llvm_fn_attrs was added to Triton's HIP options after the initial gfx12
    # backend. Older compatible releases retain the portable schedule.
    try:
        hip_compiler = importlib.import_module("triton.backends.amd.compiler")
    except ImportError:
        return {}
    hip_options = getattr(hip_compiler, "HIPOptions", None)
    hip_option_fields = getattr(hip_options, "__dataclass_fields__", {})
    if "llvm_fn_attrs" not in hip_option_fields:
        return {}
    return {
        "llvm_fn_attrs": (
            ("amdgpu-sched-strategy", "iterative-ilp"),
            # Barriers here only guard LDS tile reuse, so prefetches need not drain.
            *amd_compiler_workarounds.LDS_ONLY_FENCE_ATTRS,
        )
    }


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
            accelerator_backend="hip",
            reciprocal_scale=True,
            static_scale_ptr=input_scale,
            num_warps=num_warps,
        )


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
):
    """Project one prepared input into two adjacent 128-column-tiled weights.

    Selected by ``policy.supports_paired_matmul`` on aligned ``gfx1201``
    ``(128, 256, 64)`` shapes with ``rows > 64`` and signed 32-bit element/byte
    offsets for SRD buffer addressing. Reuses each ``128x64`` activation tile
    across both 128-column INT32 accumulators.
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
    second_result = (
        (second_accumulator - accumulator_seed).to(tl.float32)
        * row_scale[:, None]
        * tl.load(second_scale_ptr + cols)[None, :]
    )
    if second_has_bias:
        second_result += tl.load(second_bias_ptr + cols).to(tl.float32)[None, :]
    dest = output_ptr + rows[:, None] * output_row_stride + cols[None, :]
    valid_rows = True if full_rows else rows[:, None] < m
    tl.store(dest + n, second_result, mask=valid_rows)
    tl.store(dest, first_result, mask=valid_rows)


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
    compiler_options = _amd_matmul_compiler_options(target)
    bias_pointer = bias if bias is not None else output
    if paired and policy.supports_paired_matmul(
        target,
        plan,
        logical_dtype,
        rows=m,
        in_features=k,
        out_features=n,
        output_row_stride=output.stride(0),
    ):
        output_tiles = triton.cdiv(m, 128) * (n // 128)
        with device_context(input_qdata.device):
            _paired_int8_matmul_kernel[(output_tiles,)](
                input_qdata,
                weight_qdata,
                output,
                input_scale,
                weight_scale,
                bias_pointer,
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
                **compiler_options,
                num_stages=2,
                num_warps=8,
            )
        return
    num_n_tiles = triton.cdiv(n, plan.matmul_block_n) * (2 if paired else 1)
    group_m = policy.matmul_group_m(plan)
    row_block_count = triton.cdiv(m, plan.matmul_block_m)
    output_tiles = row_block_count * num_n_tiles
    grid = (output_tiles,) if group_m else (row_block_count, num_n_tiles)
    with device_context(input_qdata.device):
        int8_matmul_kernel[grid](
            input_qdata,
            weight_qdata,
            output,
            input_scale,
            weight_scale,
            bias_pointer,
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
            aligned_m=m % plan.matmul_block_m == 0,
            aligned_nk=n % plan.matmul_block_n == 0 and k % plan.matmul_block_k == 0,
            group_m=group_m,
            explicit_bias_fma=False,
            per_tile_tail=bool(group_m),
            accumulator_seed=policy.matmul_accumulator_seed(target, k, output_tiles),
            **compiler_options,
            num_stages=plan.matmul_num_stages,
            num_warps=plan.matmul_num_warps,
        )


@triton.jit
def _fence_register_value(value):
    """Sequence AMDGPU register scheduling to bound VGPR spills in 256-column epilogues.

    In 256-column tiles, splitting the residual epilogue into two 128-column halves
    separated by this zero-cost inline-asm fence and ``tl.debug_barrier()`` prevents
    AMDGPU from hoisting residual global loads across both live ``128x256`` INT32
    accumulators, keeping ``128x128`` and non-reusable ``256x64`` kernels spill-free
    while keeping reusable-update ``256x64`` spills bounded.
    """
    return tl.inline_asm_elementwise(
        ";;", "=v,0", [value], dtype=value.dtype, is_pure=False, pack=1
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
    execution_plan: LinearExecutionPlan,
    target: AcceleratorTarget,
) -> bool:
    """Return whether paired INT8 projections can fuse SwiGLU on this target."""
    del weight_qdata, second_weight, logical_dtype, execution_plan, target
    return False


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
    """Launch fused SwiGLU GEMM when supported; return False for fallback."""
    del (
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        output,
        second_weight,
        second_scale,
        second_bias,
        execution_plan,
        target,
    )
    return False


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
