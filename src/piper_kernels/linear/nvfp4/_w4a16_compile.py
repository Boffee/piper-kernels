"""Offline Triton compilation helpers for NVFP4 W4A16 kernels."""

from __future__ import annotations

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton.primitives import (
    TRITON_FLOAT_POINTER_TYPES,
    TritonCompileFn,
    default_triton_compile,
)
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot._rotation import validate_group_size

from ._w4a16_schedule import W4A16_REDUCE_BLOCK_N
from ._w4a16_triton import (
    dequantize_nvfp4_swizzled_kernel,
    select_w4a16_schedule,
    w4a16_gemv_splitk_kernel,
    w4a16_splitk_reduce_kernel,
)


def _optional_ptr_binding(
    name: str, ptr_type: str | None
) -> tuple[dict[str, str], dict[str, object]]:
    return ({name: ptr_type}, {}) if ptr_type is not None else ({}, {name: None})


def _compile_w4a16_reduce_kernel(
    target: GPUTarget,
    *,
    bias_ty: str | None,
    out_ty: str,
    has_weight_per_tensor_scale: bool,
    has_bias: bool,
    post_scale_only: bool = False,
    compile_fn: TritonCompileFn = default_triton_compile,
) -> triton.compiler.CompiledKernel:
    scale_sig, scale_const = _optional_ptr_binding(
        "weight_per_tensor_scale_ptr", "*fp32" if has_weight_per_tensor_scale else None
    )
    bias_sig, bias_const = _optional_ptr_binding("bias_ptr", bias_ty if has_bias else None)
    return compile_fn(
        ASTSource(
            w4a16_splitk_reduce_kernel,
            {
                "partial_ptr": "*fp32",
                **scale_sig,
                **bias_sig,
                "output_ptr": out_ty,
                "rows": "i32",
                "out_features": "i32",
                "split_k": "i32",
            },
            constexprs={
                **scale_const,
                **bias_const,
                "block_n": W4A16_REDUCE_BLOCK_N,
                "has_weight_per_tensor_scale": has_weight_per_tensor_scale,
                "has_bias": has_bias,
                "post_scale_only": post_scale_only,
            },
        ),
        target,
        {"num_warps": 4},
    )


def compile_w4a16_kernels(  # noqa: PLR0913
    target: GPUTarget,
    *,
    rows: int,
    out_features: int,
    input_features: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype | None = None,
    group_size: int = 0,
    high_first: bool = False,
    has_weight_per_tensor_scale: bool = True,
    num_sms: int = 64,
    compile_fn: TritonCompileFn = default_triton_compile,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile the production W4A16 Triton kernels selected for one workload."""
    if rows <= 0 or out_features <= 0 or input_features <= 0 or num_sms <= 0:
        raise ValueError("Workload dimensions and SM count must be positive integers")
    if input_dtype not in TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported W4A16 input dtype: {input_dtype}")
    if bias_dtype is not None and bias_dtype not in TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported W4A16 bias dtype: {bias_dtype}")
    if group_size != 0:
        validate_group_size(group_size)

    active_dtype = (
        torch.float32 if (group_size > 0 and input_dtype is torch.float16) else input_dtype
    )
    active_ty = TRITON_FLOAT_POINTER_TYPES[active_dtype]
    out_ty = TRITON_FLOAT_POINTER_TYPES[input_dtype]
    has_bias = bias_dtype is not None
    bias_ty = None if bias_dtype is None else TRITON_FLOAT_POINTER_TYPES[bias_dtype]
    target_info = AcceleratorTarget.from_compiler_target(target)
    if not target_info.supports_nvfp4_w4a16:
        raise ValueError(
            "NVFP4 W4A16 requires NVIDIA SM12x (use ConvRot INT8 or ConvRot Trellis on other GPUs)"
        )
    schedule = select_w4a16_schedule(
        num_sms,
        rows,
        active_dtype,
        out_features,
        input_features,
        target=target_info,
    )
    if schedule.mode == "dequant":
        compiled_large: dict[str, triton.compiler.CompiledKernel] = {
            "dequant": compile_fn(
                ASTSource(
                    dequantize_nvfp4_swizzled_kernel,
                    {
                        "weight_ptr": "*u8",
                        "weight_scale_ptr": "*u8",
                        "output_ptr": active_ty,
                        "out_features": "i32",
                        "packed_k": "i32",
                        "scale_column_blocks": "i32",
                    },
                    constexprs={
                        "block_n": schedule.block_n,
                        "block_k": schedule.block_k,
                        "high_first": high_first,
                    },
                ),
                target,
                {"num_warps": schedule.num_warps},
            )
        }
        if has_weight_per_tensor_scale or has_bias or active_dtype is not input_dtype:
            compiled_large["reduce"] = _compile_w4a16_reduce_kernel(
                target,
                bias_ty=bias_ty,
                out_ty=out_ty,
                has_weight_per_tensor_scale=has_weight_per_tensor_scale,
                has_bias=has_bias,
                post_scale_only=has_weight_per_tensor_scale and active_dtype is not torch.float16,
                compile_fn=compile_fn,
            )
        return compiled_large

    is_splitk = schedule.split_k > 1
    scale_sig, scale_const = _optional_ptr_binding(
        "weight_per_tensor_scale_ptr", "*fp32" if has_weight_per_tensor_scale else None
    )
    bias_sig, bias_const = _optional_ptr_binding("bias_ptr", bias_ty if has_bias else None)
    compile_opts: dict[str, object] = {"num_warps": schedule.num_warps}
    if schedule.num_stages is not None:
        compile_opts["num_stages"] = schedule.num_stages
    compiled: dict[str, triton.compiler.CompiledKernel] = {
        "gemv": compile_fn(
            ASTSource(
                w4a16_gemv_splitk_kernel,
                {
                    "input_ptr": active_ty,
                    "weight_ptr": "*u8",
                    "weight_scale_ptr": "*u8",
                    **scale_sig,
                    **bias_sig,
                    "output_ptr": "*fp32" if is_splitk else out_ty,
                    "rows": "i32",
                    "out_features": "i32",
                    "packed_k": "i32",
                    "scale_column_blocks": "i32",
                    "split_k": "i32",
                },
                constexprs={
                    **scale_const,
                    **bias_const,
                    "block_m": schedule.block_m,
                    "block_n": schedule.block_n,
                    "block_k": schedule.block_k,
                    "high_first": high_first,
                    "has_weight_per_tensor_scale": has_weight_per_tensor_scale,
                    "has_bias": has_bias,
                    "is_splitk": is_splitk,
                    "use_dot": schedule.use_dot,
                    "single_dot": schedule.single_dot,
                    "group_m": schedule.group_m,
                },
            ),
            target,
            compile_opts,
        )
    }
    if is_splitk:
        compiled["reduce"] = _compile_w4a16_reduce_kernel(
            target,
            bias_ty=bias_ty,
            out_ty=out_ty,
            has_weight_per_tensor_scale=has_weight_per_tensor_scale,
            has_bias=has_bias,
            post_scale_only=(
                has_weight_per_tensor_scale
                and not (schedule.use_dot and active_dtype is torch.float16)
            ),
            compile_fn=compile_fn,
        )
    return compiled
