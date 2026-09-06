"""ROCm ConvRot INT8 preparation and projection; weight updates are shared."""

# pyright: reportCallIssue=false

import importlib
import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear._input_activations import apply_input_activation, input_activation_width
from piper_kernels.linear._triton_input_activations import gelu_tanh, swiglu
from piper_kernels.linear.convrot import triton as convrot_backend

from .._kernels.triton import (
    int8_matmul_kernel,
    int8_scale_from_max,
    normalize_for_int8,
    quantize_rows_kernel,
)
from .._plan import LinearExecutionPlan
from . import policy

# Smaller row groups improve cache reuse at the validated large/wide RDNA4 shapes.
_LARGE_MATMUL_GROUP_M_TILES = 8


@triton.jit
def _normalize_rotated_values(
    values,
    inverse_sqrt_group: tl.constexpr,
    logical_dtype_code: tl.constexpr,
):
    """Apply the portable group normalization and construct an INT8 row scale."""
    values *= inverse_sqrt_group
    if logical_dtype_code == 1:
        values = values.to(tl.float16)
    elif logical_dtype_code == 2:
        values = values.to(tl.bfloat16)
    scale = tl.maximum(
        int8_scale_from_max(tl.max(tl.abs(values).to(tl.float32), axis=0), True), 1e-30
    )
    return normalize_for_int8(values, scale, logical_dtype_code), scale


@triton.jit
def _amd_normalization_scale(
    absolute_max,
    inverse_sqrt_group: tl.constexpr,
):
    """Construct the value scale used by AMD's folded normalization."""
    return tl.maximum(
        int8_scale_from_max(absolute_max, True),
        1e-30 / inverse_sqrt_group,
    )


@triton.jit
def _normalize_rotated_values_amd_wide(
    values,
    inverse_sqrt_group: tl.constexpr,
    logical_dtype_code: tl.constexpr,
):
    """Fold wide AMD group normalization into the stored row scale."""
    if logical_dtype_code == 2:
        values = values.to(tl.bfloat16)
    # Every supported group has an exact power-of-two normalization factor.
    # Wide HIP BF16/FP32 quantization is invariant to that common factor, so
    # apply it once to the stored row scale instead of every rotated value.
    normalization_scale = _amd_normalization_scale(
        tl.max(tl.abs(values).to(tl.float32), axis=0),
        inverse_sqrt_group,
    )
    scale = normalization_scale * inverse_sqrt_group
    return normalize_for_int8(values, normalization_scale, logical_dtype_code), scale


@triton.jit
def rotate_quantize_rows_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    row_width,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    inverse_sqrt_group: tl.constexpr,
    logical_dtype_code: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
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
        values = swiglu(up, gate, logical_dtype_code)
    else:
        values = tl.load(
            x_ptr + input_row_offset + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if activation_fn == "gelu_tanh":
            values = gelu_tanh(values, logical_dtype_code, accelerator_backend)

    values = convrot_backend.rotate_hadamard_groups(values, block_size, group_size)
    if accelerator_backend != "hip" or logical_dtype_code == 1 or block_size <= 8_192:
        scaled, scale = _normalize_rotated_values(
            values,
            inverse_sqrt_group,
            logical_dtype_code,
        )
    else:
        scaled, scale = _normalize_rotated_values_amd_wide(
            values,
            inverse_sqrt_group,
            logical_dtype_code,
        )
    quantized = tl.clamp(libdevice.rint(scaled.to(tl.float32)), -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + output_row_offset + offsets, quantized, mask=mask)
    tl.store(scale_ptr + row_i64, scale)


@triton.jit
def _load_activated_input_chunk(
    x_ptr,
    input_row_offset,
    row_width,
    offsets,
    mask,
    logical_dtype_code: tl.constexpr,
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
        return swiglu(up, gate, logical_dtype_code)

    values = tl.load(x_ptr + input_row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    if activation_fn == "gelu_tanh":
        values = gelu_tanh(values, logical_dtype_code, accelerator_backend)
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
    logical_dtype_code: tl.constexpr,
    activation_fn: tl.constexpr,
    accelerator_backend: tl.constexpr,
):
    """Rotate a BF16 row as up to three power-of-two chunks before quantizing.

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
        logical_dtype_code,
        activation_fn,
        accelerator_backend,
    )
    values0 = convrot_backend.rotate_hadamard_groups(values0, block0, group_size).to(tl.bfloat16)
    max0 = tl.max(tl.abs(values0).to(tl.float32), axis=0)

    offsets1 = block0 + tl.arange(0, block1)
    mask1 = offsets1 < row_width
    values1 = _load_activated_input_chunk(
        x_ptr,
        input_row_offset,
        row_width,
        offsets1,
        mask1,
        logical_dtype_code,
        activation_fn,
        accelerator_backend,
    )
    values1 = convrot_backend.rotate_hadamard_groups(values1, block1, group_size).to(tl.bfloat16)
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
            logical_dtype_code,
            activation_fn,
            accelerator_backend,
        )
        values2 = convrot_backend.rotate_hadamard_groups(values2, block2, group_size).to(
            tl.bfloat16
        )
        max2 = tl.max(tl.abs(values2).to(tl.float32), axis=0)
        absolute_max = tl.maximum(absolute_max, max2)

    # All supported group normalizations are exact powers of two. Quantizing
    # the unnormalized BF16 values and applying that factor once to the scale
    # preserves the logical result while avoiding a vector multiply per chunk.
    normalization_scale = _amd_normalization_scale(
        absolute_max,
        inverse_sqrt_group,
    )
    quantized0 = tl.clamp(
        libdevice.rint(
            normalize_for_int8(values0, normalization_scale, logical_dtype_code).to(tl.float32)
        ),
        -128.0,
        127.0,
    ).to(tl.int8)
    quantized1 = tl.clamp(
        libdevice.rint(
            normalize_for_int8(values1, normalization_scale, logical_dtype_code).to(tl.float32)
        ),
        -128.0,
        127.0,
    ).to(tl.int8)
    tl.store(q_ptr + output_row_offset + offsets0, quantized0, mask=mask0)
    tl.store(q_ptr + output_row_offset + offsets1, quantized1, mask=mask1)
    if block2:
        quantized2 = tl.clamp(
            libdevice.rint(
                normalize_for_int8(values2, normalization_scale, logical_dtype_code).to(tl.float32)
            ),
            -128.0,
            127.0,
        ).to(tl.int8)
        tl.store(q_ptr + output_row_offset + offsets2, quantized2, mask=mask2)
    tl.store(scale_ptr + row_i64, normalization_scale * inverse_sqrt_group)


def _uses_amd_chunked_preparation(
    target: AcceleratorTarget,
    logical_dtype_code: int,
    blocks: tuple[int, int, int],
) -> bool:
    """Return whether AMD uses the lower-live-range BF16 preparation kernel."""
    return target.is_amd_hip and logical_dtype_code == 2 and blocks[1] != 0


def _launch_amd_chunked_preparation(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    group_size: int,
    logical_dtype_code: int,
    activation_fn: str | None,
    num_warps: int,
    blocks: tuple[int, int, int],
) -> None:
    """Launch AMD's chunked fused rotation and quantization path."""
    m, k = input_qdata.shape
    with device_context(input.device):
        rotate_quantize_rows_chunked_kernel[(m,)](
            input,
            input_qdata,
            input_scale,
            k,
            block0=blocks[0],
            block1=blocks[1],
            block2=blocks[2],
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            logical_dtype_code=logical_dtype_code,
            activation_fn=activation_fn,
            accelerator_backend="hip",
            num_warps=num_warps,
        )


def _launch_full_row_preparation(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    group_size: int,
    logical_dtype_code: int,
    activation_fn: str | None,
    num_warps: int,
    target: AcceleratorTarget,
) -> None:
    """Launch the shared full-row fused rotation and quantization path."""
    m, k = input_qdata.shape
    with device_context(input.device):
        rotate_quantize_rows_kernel[(m,)](
            input,
            input_qdata,
            input_scale,
            k,
            block_size=max(128, triton.next_power_of_2(k)),
            group_size=group_size,
            inverse_sqrt_group=group_size**-0.5,
            logical_dtype_code=logical_dtype_code,
            activation_fn=activation_fn,
            accelerator_backend=target.backend,
            num_warps=num_warps,
        )


def fused_rotate_quantize_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    group_size: int,
    logical_dtype_code: int,
    *,
    activation_fn: str | None = None,
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
    if not policy.select_execution_plan(target, in_features=k).fuse_rotation_quantization:
        raise ValueError(f"fused preparation does not support row width {k}")
    blocks = policy.preparation_blocks(k)
    if _uses_amd_chunked_preparation(target, logical_dtype_code, blocks):
        _launch_amd_chunked_preparation(
            input,
            input_qdata,
            input_scale,
            group_size,
            logical_dtype_code,
            activation_fn,
            num_warps,
            blocks,
        )
        return
    _launch_full_row_preparation(
        input,
        input_qdata,
        input_scale,
        group_size,
        logical_dtype_code,
        activation_fn,
        num_warps,
        target,
    )


def _amd_matmul_compiler_options(
    target: AcceleratorTarget,
) -> dict[str, object]:
    """Return optional AMD compiler controls supported by the installed Triton."""
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
    return {"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"}


def quantize_input(
    rotated: torch.Tensor,
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    logical_dtype_code: int,
    *,
    num_warps: int,
) -> None:
    """Apply the portable split-path rowwise quantization."""
    m, k = rotated.shape
    with device_context(rotated.device):
        quantize_rows_kernel[(m,)](
            rotated,
            input_qdata,
            input_scale,
            k,
            block_size=max(128, triton.next_power_of_2(k)),
            logical_dtype_code=logical_dtype_code,
            reciprocal_scale=True,
            num_warps=num_warps,
        )


def default_execution_plan(
    weight_qdata: torch.Tensor,
    *,
    target: AcceleratorTarget | None = None,
) -> LinearExecutionPlan:
    """Resolve production policy, accepting an explicit target for offline tuning."""
    target = AcceleratorTarget.from_device(weight_qdata.device) if target is None else target
    return policy.select_execution_plan(
        target,
        in_features=weight_qdata.shape[1],
    )


def prepare_input_with_plan(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    in_features: int,
    group_size: int,
    *,
    activation_fn: str | None,
    execution_plan: LinearExecutionPlan,
    target: AcceleratorTarget,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate and quantize an input, optionally into caller-owned storage."""
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    m = input_2d.shape[0]
    if out is None:
        input_qdata = torch.empty(
            (m, in_features),
            device=input.device,
            dtype=torch.int8,
        )
        input_scale = torch.empty(m, device=input.device, dtype=torch.float32)
        result = (
            input_qdata.reshape(*input.shape[:-1], in_features),
            input_scale.reshape(input.shape[:-1]),
        )
    else:
        result = out
    input_qdata = result[0].reshape(m, in_features)
    input_scale = result[1].reshape(m)
    logical_dtype_code = convrot_backend.logical_dtype_code(input.dtype)
    if execution_plan.fuse_rotation_quantization:
        fused_rotate_quantize_input(
            input_2d,
            input_qdata,
            input_scale,
            group_size,
            logical_dtype_code,
            activation_fn=activation_fn,
            num_warps=execution_plan.fused_num_warps,
            target=target,
        )
    else:
        transformed_input = apply_input_activation(input_2d, activation_fn)
        rotated = torch.empty_like(transformed_input)
        convrot_backend.rotate_input(
            transformed_input,
            rotated,
            group_size,
            num_warps=execution_plan.rotation_num_warps,
        )
        quantize_input(
            rotated,
            input_qdata,
            input_scale,
            logical_dtype_code,
            num_warps=execution_plan.quantization_num_warps,
        )
    return result


def _prepare_input_with_production_plan(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    *,
    activation_fn: str | None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare an ordinary or activated input under production policy."""
    in_features = input.shape[-1] // input_activation_width(activation_fn)
    target = AcceleratorTarget.from_device(input.device)
    plan = policy.select_execution_plan(
        target,
        in_features=in_features,
    )
    return prepare_input_with_plan(
        input,
        in_features,
        group_size,
        activation_fn=activation_fn,
        execution_plan=plan,
        target=target,
        out=out,
    )


def execute_prepared_linear(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
    execution_plan: LinearExecutionPlan,
    *,
    out: torch.Tensor | None = None,
    second_projection: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Project prepared INT8 input, optionally to two equal-width independent weights.

    Paired projections share a launch and write adjacent output columns without
    packing weights. ``out`` may provide caller-owned, row-strided storage.
    """
    leading_shape = input_qdata.shape[:-1]
    m = math.prod(leading_shape)
    k = input_qdata.shape[-1]
    n = weight_qdata.shape[0]
    paired = second_projection is not None
    second_weight, second_scale, second_bias = (
        (weight_qdata, weight_scale, None) if second_projection is None else second_projection
    )
    if second_weight.shape != weight_qdata.shape:
        raise ValueError("paired INT8 projections must have matching weight shapes")
    output_features = n * (2 if paired else 1)
    if out is None:
        output = torch.empty(
            (m, output_features),
            device=input_qdata.device,
            dtype=logical_dtype,
        )
        result = output.reshape(*leading_shape, output_features)
    else:
        result = out
    input_qdata_2d = input_qdata.reshape(m, k)
    input_scale_1d = input_scale.reshape(m)
    output = result.reshape(m, output_features)
    if output.stride(1) != 1:
        raise ValueError("prepared INT8 GEMM output must be column-contiguous")
    plan = execution_plan
    num_n_tiles = triton.cdiv(n, plan.matmul_block_n) * (2 if paired else 1)
    # Cache grouping and the M-tail split are intrinsic to the selected large-tile family,
    # rather than additional execution-plan axes.
    group_m = (
        _LARGE_MATMUL_GROUP_M_TILES
        if plan.matmul_block_m == 128 and plan.matmul_block_n == 256
        else 0
    )
    compiler_options = _amd_matmul_compiler_options(
        AcceleratorTarget.from_device(input_qdata.device)
    )
    bias_pointer = bias if bias is not None else output

    with device_context(input_qdata.device):

        def launch_tiles(row_block_count: int, row_block_offset: int, *, aligned_m: bool) -> None:
            grid = (row_block_count * num_n_tiles,) if group_m else (row_block_count, num_n_tiles)
            int8_matmul_kernel[grid](
                input_qdata_2d,
                weight_qdata,
                output,
                input_scale_1d,
                weight_scale,
                bias_pointer,
                second_weight,
                second_scale,
                second_bias if second_bias is not None else output,
                m,
                n,
                k,
                output.stride(0),
                row_block_offset,
                block_m=plan.matmul_block_m,
                block_n=plan.matmul_block_n,
                block_k=plan.matmul_block_k,
                has_bias=bias is not None,
                paired=paired,
                second_has_bias=second_bias is not None,
                aligned_tiles=aligned_m
                and (n % plan.matmul_block_n == 0)
                and (k % plan.matmul_block_k == 0),
                group_m=group_m,
                **compiler_options,
                num_stages=plan.matmul_num_stages,
                num_warps=plan.matmul_num_warps,
            )

        full_row_blocks = m // plan.matmul_block_m
        if group_m:
            if full_row_blocks:
                launch_tiles(full_row_blocks, 0, aligned_m=True)
            if m % plan.matmul_block_m:
                launch_tiles(1, full_row_blocks, aligned_m=False)
        else:
            launch_tiles(
                (m + plan.matmul_block_m - 1) // plan.matmul_block_m,
                0,
                aligned_m=False,
            )
        return result


def run_linear(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    *,
    activation_fn: str | None = None,
    execution_plan: LinearExecutionPlan | None = None,
) -> torch.Tensor:
    """Run ConvRot input preparation and INT8 GEMM under one plan."""
    original_shape = input.shape
    k = weight_qdata.shape[1]
    expected_width = k * input_activation_width(activation_fn)
    if original_shape[-1] != expected_width:
        operation = "activated linear input" if activation_fn is not None else "linear input"
        raise ValueError(
            f"{operation} has {original_shape[-1]} features, expected {expected_width}"
        )
    target = AcceleratorTarget.from_device(input.device)
    plan = (
        execution_plan
        if execution_plan is not None
        else default_execution_plan(weight_qdata, target=target)
    )
    input_qdata, input_scale = prepare_input_with_plan(
        input,
        k,
        group_size,
        activation_fn=activation_fn,
        execution_plan=plan,
        target=target,
    )
    return execute_prepared_linear(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        input.dtype,
        plan,
    )


def linear(
    input: torch.Tensor,  # noqa: A002
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    activation_fn: str | None = None,
) -> torch.Tensor:
    """Run ConvRot input rotation, dynamic quantization, and INT8 GEMM."""
    return run_linear(
        input,
        weight_qdata,
        weight_scale,
        bias,
        group_size,
        activation_fn=activation_fn,
    )


def prepare_input(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    group_size: int,
    activation_fn: str | None = None,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an optional activation, then rotate and quantize a linear input."""
    return _prepare_input_with_production_plan(
        input,
        group_size,
        activation_fn=activation_fn,
        out=out,
    )


def linear_prepared(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    logical_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
    second_projection: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Apply one weight to an input prepared by the matching operator."""
    plan = default_execution_plan(weight_qdata)
    return execute_prepared_linear(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        logical_dtype,
        plan,
        out=out,
        second_projection=second_projection,
    )
