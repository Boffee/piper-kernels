"""Offline Triton kernel compilation helpers for quantized MoE operators."""

# Wide compile helpers mirror the runtime kernel parameter lists cleanly.
# ruff: noqa: PLR0913

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.primitives import (
    TRITON_FLOAT_POINTER_TYPES,
    TritonCompileFn,
    default_triton_compile,
)
from piper_kernels._triton.targets import AcceleratorTarget, rdna4_int8_accumulator_seed
from piper_kernels.linear.convrot.int8.triton import splitk_accumulator_seed
from piper_kernels.trellis._layout import (
    TrellisLayout,
    default_layout_for_block_size,
    validate_block_size,
)
from piper_kernels.trellis.triton import _decode_stream_flags, decode_slots_int8_kernel
from piper_kernels.weights.convrot._rotation import validate_group_size
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

from . import _policy, _schedule
from ._int8_triton import (
    int8_moe_activate_quant_kernel,
    int8_moe_batched_matmul_kernel,
    int8_moe_grouped_gemm_kernel,
    int8_moe_m1_matmul_kernel,
)
from ._nvfp4_triton import (
    nvfp4_moe_activate_quant_kernel,
    nvfp4_moe_batched_gemv_kernel,
    nvfp4_moe_gather_kernel,
    nvfp4_moe_grouped_gemm_kernel,
    nvfp4_moe_m1_gemv_kernel,
    nvfp4_moe_row_quant_kernel,
    nvfp4_w4a4_moe_grouped_gemm_kernel,
)
from ._ops import _INT8_MAX_REDUCTION_WIDTH
from ._routing_triton import (
    moe_bucket_tokens_kernel,
    moe_m1_activate_rotate_kernel,
    moe_m1_weighted_reduce_kernel,
    moe_prefill_bucket_count_kernel,
    moe_prefill_bucket_scatter_kernel,
    moe_prefill_gather_kernel,
    moe_splitk_reduce_kernel,
)
from ._trellis_triton import trellis_moe_gemv_kernel
from .reference import is_gated_activation, validate_moe_activation_bounds


def _bind_opt_ptrs(
    sig: dict[str, str],
    constexprs: dict[str, object],
    **opt_ptrs: str | None,
) -> None:
    for name, ptr_type in opt_ptrs.items():
        if ptr_type is None:
            constexprs[name] = None
        else:
            sig[name] = ptr_type


def _div16_attrs(
    arg_names: Sequence[str],
    signature: dict[str, str],
    *extra_int_args: str,
) -> dict[tuple[int, ...], list[list[object]]]:
    """Attach 16-alignment hints for pointer args and selected integer strides."""
    aligned = [name for name, ty in signature.items() if ty.startswith("*")]
    aligned.extend(extra_int_args)
    return {(arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned}


def _nvfp4_aligned_attrs(
    arg_names: Sequence[str],
    signature: dict[str, str],
    *,
    out_features: int,
    in_features: int,
) -> dict[tuple[int, ...], list[list[object]]]:
    """Attach runtime pointer and packed-stride divisibility hints for offline compile."""
    extra = ["expert_scale_stride"]
    if out_features % 16 == 0:
        extra.append("out_features")
    if (in_features // 2) % 16 == 0:
        extra.append("packed_k")
    return _div16_attrs(arg_names, signature, *extra)


def _compile_bucket_kernel(
    target: GPUTarget,
    *,
    num_pairs: int,
    has_slot_map: bool,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    block_p = triton.next_power_of_2(max(16, num_pairs))
    sig: dict[str, str] = {
        "topk_ids_ptr": "*i64",
        "tile_slots_ptr": "*i32",
        "tile_pair_ids_ptr": "*i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "max_tiles": "i32",
    }
    constexprs: dict[str, object] = {
        "block_p": block_p,
        "block_m": _schedule.BATCH_BLOCK_M,
        "has_slot_map": has_slot_map,
    }
    _bind_opt_ptrs(sig, constexprs, expert_to_slot_ptr="*i64" if has_slot_map else None)
    return compile_fn(
        ASTSource(moe_bucket_tokens_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": 4},
    )


def _compile_weighted_reduce(
    target: GPUTarget,
    *,
    out_ty: str,
    is_int8: bool,
    has_slot_map: bool,
    accumulate: bool,
    has_global_scale: bool,
    post_scale_only: bool,
    use_compact_map: bool = False,
    has_row_scale: bool = False,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    eff_int8 = False if use_compact_map else is_int8
    eff_slot_map = False if use_compact_map else has_slot_map
    eff_global_scale = False if use_compact_map else has_global_scale
    eff_post_scale_only = True if use_compact_map else post_scale_only
    eff_row_scale = False if use_compact_map else has_row_scale
    idx_ty = "*i32" if use_compact_map else "*i64"
    sig: dict[str, str] = {
        "partial_ptr": "*i32" if eff_int8 else "*fp32",
        "topk_ids_ptr": idx_ty,
        "topk_weights_ptr": out_ty,
        "output_ptr": out_ty,
        "out_features": "i32",
        "out_row_stride": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "top_k": "i32",
        "split_k": "i32",
    }
    constexprs: dict[str, object] = {
        "block_n": _schedule.WEIGHTED_REDUCE_BLOCK_N,
        "is_int8": eff_int8,
        "has_slot_map": eff_slot_map,
        "accumulate": accumulate,
        "has_global_scale": eff_global_scale,
        "post_scale_only": eff_post_scale_only,
        "use_compact_map": use_compact_map,
        "has_row_scale": eff_row_scale,
    }
    _bind_opt_ptrs(
        sig,
        constexprs,
        input_scale_ptr="*fp32" if eff_int8 else None,
        weight_scale_ptr="*fp32" if (eff_int8 or eff_row_scale) else None,
        weight_global_scale_ptr="*fp32" if (not eff_int8 and eff_global_scale) else None,
        expert_to_slot_ptr="*i64" if eff_slot_map else None,
    )
    return compile_fn(
        ASTSource(moe_m1_weighted_reduce_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": 4},
    )


def _compile_splitk_reduce(
    target: GPUTarget,
    *,
    out_ty: str,
    is_int8: bool,
    has_slot_map: bool,
    has_global_scale: bool = False,
    post_scale_only: bool = True,
    has_row_scale: bool = False,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    eff_global_scale = False if is_int8 else has_global_scale
    eff_row_scale = False if is_int8 else has_row_scale
    sig: dict[str, str] = {
        "partial_ptr": "*i32" if is_int8 else "*fp32",
        "topk_ids_ptr": "*i64",
        "output_ptr": out_ty if is_int8 else "*fp32",
        "out_features": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "top_k": "i32",
        "split_k": "i32",
    }
    constexprs: dict[str, object] = {
        "block_n": _schedule.INT8_REDUCE_BLOCK_N if is_int8 else _schedule.NVFP4_REDUCE_BLOCK_N,
        "is_int8": is_int8,
        "has_slot_map": has_slot_map,
        "has_global_scale": eff_global_scale,
        "post_scale_only": True if is_int8 else post_scale_only,
        "has_row_scale": eff_row_scale,
    }
    _bind_opt_ptrs(
        sig,
        constexprs,
        input_scale_ptr="*fp32" if is_int8 else None,
        weight_scale_ptr="*fp32" if (is_int8 or eff_row_scale) else None,
        weight_global_scale_ptr="*fp32" if eff_global_scale else None,
        expert_to_slot_ptr="*i64" if has_slot_map else None,
    )
    return compile_fn(
        ASTSource(moe_splitk_reduce_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": 4},
    )


def _compile_activate_rotate(
    target: GPUTarget,
    *,
    active_ty: str,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    accelerator_backend: str,
    has_slot_map: bool,
    use_compact_map: bool = False,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    eff_slot_map = False if use_compact_map else has_slot_map
    idx_ty = "*i32" if use_compact_map else "*i64"
    sig: dict[str, str] = {
        "mid_ptr": "*fp32",
        "act_ptr": active_ty,
        "logical_ref_ptr": out_ty,
        "topk_ids_ptr": idx_ty,
        "intermediate_size": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "gate_bound": "fp32",
        "up_bound": "fp32",
    }
    constexprs: dict[str, object] = {
        "block_size": _schedule.ACTIVATE_BLOCK_SIZE,
        "group_size": group_size,
        "is_gated": is_gated_activation(activation_fn),
        "gate_first": gate_first,
        "has_gate_bound": gate_bound is not None,
        "has_up_bound": up_bound is not None,
        "accelerator_backend": accelerator_backend,
        "has_slot_map": eff_slot_map,
    }
    _bind_opt_ptrs(sig, constexprs, expert_to_slot_ptr="*i64" if eff_slot_map else None)
    return compile_fn(
        ASTSource(moe_m1_activate_rotate_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": 4},
    )


def _compile_int8_activate_quant(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    out_ty: str,
    intermediate_size: int,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    use_compact_map: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    eff_slot_map = False if use_compact_map else has_slot_map
    idx_ty = "*i32" if use_compact_map else "*i64"
    single_tile, block_k, num_warps = _schedule.select_int8_activate_schedule(intermediate_size)
    sig: dict[str, str] = {
        "mid_ptr": out_ty,
        "routing_ids_ptr": idx_ty,
        "q_out_ptr": "*i8",
        "scale_out_ptr": "*fp32",
        "intermediate_size": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "gate_bound": "fp32",
        "up_bound": "fp32",
    }
    constexprs: dict[str, object] = {
        "block_k": block_k,
        "group_size": group_size,
        "single_tile": single_tile,
        "is_gated": is_gated_activation(activation_fn),
        "gate_first": gate_first,
        "has_gate_bound": gate_bound is not None,
        "has_up_bound": up_bound is not None,
        "has_slot_map": eff_slot_map,
        "use_compact_map": use_compact_map,
        "accelerator_backend": accel.backend,
    }
    _bind_opt_ptrs(sig, constexprs, expert_to_slot_ptr="*i64" if eff_slot_map else None)
    return compile_fn(
        ASTSource(int8_moe_activate_quant_kernel, sig, constexprs=constexprs),
        target,
        {"num_warps": num_warps},
    )


def _validate_nvfp4_compile_format(
    accel: AcceleratorTarget,
    *,
    weight_format: str,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    if not accel.supports_nvfp4_w4a16:
        raise ValueError(
            "NVFP4 MoE requires NVIDIA SM12x "
            f"(use ConvRot INT8 or ConvRot Trellis on other GPUs), got {accel}"
        )
    if hidden_size % nvfp4_layout.BLOCK_SIZE or intermediate_size % nvfp4_layout.BLOCK_SIZE:
        raise ValueError("NVFP4 MoE hidden_size and intermediate_size must be multiples of 16")
    if weight_format in ("nvfp4", "nvfp4_w4a16"):
        if group_size != 0:
            suggested = "convrot_nvfp4" if weight_format == "nvfp4" else "convrot_nvfp4_w4a16"
            raise ValueError(f"{weight_format} requires group_size=0; use {suggested}")
        return
    if group_size == 0:
        raise ValueError(f"{weight_format} requires group_size in (16, 64, 256)")
    validate_group_size(group_size)
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError(
            f"ConvRot NVFP4 MoE dimensions must be divisible by group_size {group_size}"
        )


def _validate_convrot_compile_format(
    accel: AcceleratorTarget,
    *,
    weight_format: str,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
    high_first: bool,
    block_size: int,
    variable_rows: bool,
    has_slot_offsets: bool,
    bytes_per_block: int | None,
) -> None:
    label = "ConvRot INT8 MoE" if weight_format == "convrot_int8" else "ConvRot Trellis MoE"
    if high_first:
        raise ValueError(f"{label} weights do not support high_first")
    if (
        weight_format == "convrot_int8"
        and accel.is_nvidia_cuda
        and not accel.cuda_capability_at_least(8)
    ):
        raise ValueError(
            f"ConvRot INT8 MoE requires NVIDIA SM80+ or supported AMD GPU, got {accel}"
        )
    validate_group_size(group_size)
    if hidden_size % group_size or intermediate_size % group_size:
        raise ValueError(f"{label} dimensions must be divisible by group_size {group_size}")
    if hidden_size > _INT8_MAX_REDUCTION_WIDTH or intermediate_size > _INT8_MAX_REDUCTION_WIDTH:
        raise ValueError(
            f"{label} widths must not exceed {_INT8_MAX_REDUCTION_WIDTH} "
            f"for INT32 accumulation, got hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}"
        )
    if weight_format != "convrot_trellis":
        return
    if has_slot_offsets and not variable_rows:
        raise ValueError("has_slot_offsets requires variable_rows=True")
    validate_block_size(block_size)
    if hidden_size % block_size or intermediate_size % block_size:
        raise ValueError(
            f"ConvRot Trellis MoE dimensions must be divisible by block_size {block_size}"
        )
    if bytes_per_block is not None:
        if variable_rows:
            raise ValueError(
                "bytes_per_block applies only to uniform (variable_rows=False) layouts"
            )
        TrellisLayout.from_storage(block_size, bytes_per_block)


def _validate_compile_format(
    accel: AcceleratorTarget,
    *,
    weight_format: str,
    group_size: int,
    hidden_size: int,
    intermediate_size: int,
    high_first: bool,
    block_size: int = 256,
    variable_rows: bool = False,
    has_slot_offsets: bool = False,
    bytes_per_block: int | None = None,
) -> None:
    if weight_format in ("nvfp4", "convrot_nvfp4", "nvfp4_w4a16", "convrot_nvfp4_w4a16"):
        _validate_nvfp4_compile_format(
            accel,
            weight_format=weight_format,
            group_size=group_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        return
    if weight_format not in ("convrot_int8", "convrot_trellis"):
        raise ValueError(f"Unsupported MoE weight_format {weight_format!r}")
    _validate_convrot_compile_format(
        accel,
        weight_format=weight_format,
        group_size=group_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        high_first=high_first,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
        bytes_per_block=bytes_per_block,
    )


def _compile_prefill_utility_kernels(
    target: GPUTarget,
    *,
    gather_ty: str,
    out_ty: str,
    is_int8: bool,
    block_m: int,
    has_slot_map: bool,
    accumulate: bool,
    compile_fn: TritonCompileFn,
    include_gather: bool = True,
) -> dict[str, triton.compiler.CompiledKernel]:
    count_sig: dict[str, str] = {
        "topk_ids_ptr": "*i64",
        "slot_counts_ptr": "*i32",
        "pair_slots_ptr": "*i32",
        "pair_pos_ptr": "*i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
    }
    count_constexprs: dict[str, object] = {
        "block_p": _schedule.PREFILL_BUCKET_BLOCK_P,
        "has_slot_map": has_slot_map,
    }
    _bind_opt_ptrs(count_sig, count_constexprs, expert_to_slot_ptr="*i64" if has_slot_map else None)
    compiled: dict[str, triton.compiler.CompiledKernel] = {
        "bucket_count": compile_fn(
            ASTSource(moe_prefill_bucket_count_kernel, count_sig, constexprs=count_constexprs),
            target,
            {"num_warps": 4},
        ),
        "bucket_scatter": compile_fn(
            ASTSource(
                moe_prefill_bucket_scatter_kernel,
                {
                    "pair_slots_ptr": "*i32",
                    "pair_pos_ptr": "*i32",
                    "slot_counts_ptr": "*i32",
                    "slot_offsets_ptr": "*i32",
                    "slot_tile_base_ptr": "*i32",
                    "compact_to_pair_ptr": "*i32",
                    "pair_to_compact_ptr": "*i32",
                    "tile_slots_ptr": "*i32",
                    "tile_m_start_ptr": "*i32",
                    "tile_m_len_ptr": "*i32",
                    "num_pairs": "i32",
                },
                constexprs={"block_p": _schedule.PREFILL_BUCKET_BLOCK_P, "block_m": block_m},
            ),
            target,
            {"num_warps": 4},
        ),
        "weighted_reduce": _compile_weighted_reduce(
            target,
            out_ty=out_ty,
            is_int8=False,
            has_slot_map=False,
            accumulate=accumulate,
            has_global_scale=False,
            post_scale_only=True,
            use_compact_map=True,
            compile_fn=compile_fn,
        ),
    }
    if include_gather:
        gather_sig: dict[str, str] = {
            "input_ptr": gather_ty,
            "compact_to_pair_ptr": "*i32",
            "out_ptr": gather_ty,
            "hidden_size": "i32",
            "top_k": "i32",
        }
        gather_constexprs: dict[str, object] = {
            "block_k": _schedule.PREFILL_GATHER_BLOCK_K,
            "has_scale": is_int8,
        }
        _bind_opt_ptrs(
            gather_sig,
            gather_constexprs,
            input_scale_ptr="*fp32" if is_int8 else None,
            out_scale_ptr="*fp32" if is_int8 else None,
        )
        compiled["gather"] = compile_fn(
            ASTSource(moe_prefill_gather_kernel, gather_sig, constexprs=gather_constexprs),
            target,
            {"num_warps": 4},
        )
    return compiled


def _compile_int8_prefill_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    accumulate: bool,
    slot_window: bool = False,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    up_gate_out = intermediate_size * (2 if is_gated_activation(activation_fn) else 1)
    schedule = _policy.select_int8_prefill_schedule(
        accel,
        chunk_rows=rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
    )
    compiler_options = amd_compiler_workarounds.lds_only_fence_options(schedule.lds_only_fences)
    expected_tiles = _schedule.expected_bucket_tiles(
        rows * top_k, num_slots, active_slots_hint, block_m=schedule.block_m, batched=True
    )
    up_seed = rdna4_int8_accumulator_seed(
        accel, hidden_size, expected_tiles * triton.cdiv(up_gate_out, schedule.block_n)
    )
    down_seed = rdna4_int8_accumulator_seed(
        accel, intermediate_size, expected_tiles * triton.cdiv(hidden_size, schedule.block_n)
    )
    sig = {
        "input_ptr": "*i8",
        "weight_ptr": "*i8",
        "tile_slots_ptr": "*i32",
        "tile_m_start_ptr": "*i32",
        "tile_m_len_ptr": "*i32",
        "input_scale_ptr": "*fp32",
        "weight_scale_ptr": "*fp32",
        "output_ptr": out_ty,
        "max_tiles": "i32",
        "out_features": "i32",
        "in_features": "i32",
        "slot_start": "i32",
        "slot_end": "i32",
        "n_start": "i32",
        "n_rows": "i32",
    }
    opts = {
        "num_stages": schedule.num_stages,
        "num_warps": schedule.num_warps,
        **compiler_options,
    }
    compiled = _compile_prefill_utility_kernels(
        target,
        gather_ty="*i8",
        out_ty=out_ty,
        is_int8=True,
        block_m=schedule.block_m,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        compile_fn=compile_fn,
    )
    compiled["int8_grouped_up"] = compile_fn(
        ASTSource(
            int8_moe_grouped_gemm_kernel,
            sig,
            constexprs={
                "block_m": schedule.block_m,
                "block_n": schedule.block_n,
                "block_k": schedule.block_k,
                "group_m": _schedule.PREFILL_GROUP_M,
                "aligned_nk": (
                    up_gate_out % schedule.block_n == 0 and hidden_size % schedule.block_k == 0
                ),
                "accumulator_seed": up_seed,
                "slot_window": slot_window,
            },
        ),
        target,
        opts,
    )
    compiled["int8_activate_quant"] = _compile_int8_activate_quant(
        target,
        accel,
        out_ty=out_ty,
        intermediate_size=intermediate_size,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        has_slot_map=has_slot_map,
        use_compact_map=True,
        gate_bound=gate_bound,
        up_bound=up_bound,
        compile_fn=compile_fn,
    )
    compiled["int8_grouped_down"] = compile_fn(
        ASTSource(
            int8_moe_grouped_gemm_kernel,
            {**sig, "output_ptr": "*fp32"},
            constexprs={
                "block_m": schedule.block_m,
                "block_n": schedule.block_n,
                "block_k": schedule.block_k,
                "group_m": _schedule.PREFILL_GROUP_M,
                "aligned_nk": (
                    hidden_size % schedule.block_n == 0
                    and intermediate_size % schedule.block_k == 0
                ),
                "accumulator_seed": down_seed,
                "slot_window": slot_window,
            },
        ),
        target,
        opts,
    )
    return compiled


def _compile_nvfp4_prefill_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    hidden_size: int,
    intermediate_size: int,
    logical_dtype: torch.dtype,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    accumulate: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    active_dtype = torch.float32 if group_size > 0 else logical_dtype
    active_ty = TRITON_FLOAT_POINTER_TYPES[active_dtype]
    schedule = _policy.select_nvfp4_prefill_schedule(
        accel,
        chunk_rows=rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        active_dtype=active_dtype,
        logical_dtype=logical_dtype,
    )
    sig: dict[str, str] = {
        "input_ptr": active_ty,
        "weight_ptr": "*u8",
        "weight_scale_ptr": "*u8",
        "tile_slots_ptr": "*i32",
        "tile_m_start_ptr": "*i32",
        "tile_m_len_ptr": "*i32",
        "output_ptr": "*fp32",
        "max_tiles": "i32",
        "out_features": "i32",
        "packed_k": "i32",
        "scale_column_blocks": "i32",
        "expert_scale_stride": "i32",
    }
    constexprs: dict[str, object] = {
        "block_m": schedule.block_m,
        "block_n": schedule.block_n,
        "block_k": schedule.block_k,
        "group_m": _schedule.PREFILL_GROUP_M,
        "high_first": high_first,
        "has_global_scale": has_global_scale,
        "cast_fp32_to_bf16": schedule.cast_fp32_to_bf16,
        "force_fp32_dot": schedule.force_fp32_dot,
        "single_dot": schedule.single_dot,
    }
    _bind_opt_ptrs(
        sig,
        constexprs,
        weight_global_scale_ptr="*fp32" if has_global_scale else None,
    )
    opts: dict[str, object] = {"num_warps": schedule.num_warps, "num_stages": 1}
    compiled = _compile_prefill_utility_kernels(
        target,
        gather_ty=active_ty,
        out_ty=out_ty,
        is_int8=False,
        block_m=schedule.block_m,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        compile_fn=compile_fn,
    )
    up_gate_out = intermediate_size * (2 if is_gated_activation(activation_fn) else 1)
    grouped_attrs = _nvfp4_aligned_attrs(
        nvfp4_moe_grouped_gemm_kernel.arg_names,
        sig,
        out_features=math.gcd(up_gate_out, hidden_size),
        in_features=math.gcd(hidden_size, intermediate_size),
    )
    compiled["nvfp4_grouped"] = compile_fn(
        ASTSource(nvfp4_moe_grouped_gemm_kernel, sig, constexprs, attrs=grouped_attrs),
        target,
        opts,
    )
    compiled["activate_rotate"] = _compile_activate_rotate(
        target,
        active_ty=active_ty,
        out_ty=out_ty,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        accelerator_backend=accel.backend,
        has_slot_map=has_slot_map,
        use_compact_map=True,
        gate_bound=gate_bound,
        up_bound=up_bound,
        compile_fn=compile_fn,
    )
    return compiled


def _compile_nvfp4_w4a4_grouped_gemm(
    target: GPUTarget,
    *,
    schedule: _schedule.Nvfp4W4A4PrefillSchedule,
    output_ty: str,
    out_features: int,
    in_features: int,
    has_global_scale: bool,
    compile_fn: TritonCompileFn,
) -> triton.compiler.CompiledKernel:
    sig: dict[str, str] = {
        "a_q_ptr": "*u8",
        "a_sc_ptr": "*fp8e4nv",
        "a_rs_ptr": "*fp32",
        "b_q_ptr": "*u8",
        "b_sc_ptr": "*fp8e4nv",
        "tile_slots_ptr": "*i32",
        "tile_m_start_ptr": "*i32",
        "tile_m_len_ptr": "*i32",
        "output_ptr": output_ty,
        "max_tiles": "i32",
        "out_features": "i32",
        "packed_k": "i32",
        "scale_column_blocks": "i32",
        "expert_scale_stride": "i32",
    }
    constexprs: dict[str, object] = {
        "block_m": schedule.block_m,
        "block_n": schedule.block_n,
        "block_k": schedule.block_k,
        "group_m": _schedule.PREFILL_GROUP_M,
        "has_global_scale": has_global_scale,
    }
    _bind_opt_ptrs(sig, constexprs, b_gs_ptr="*fp32" if has_global_scale else None)
    attrs = _nvfp4_aligned_attrs(
        nvfp4_w4a4_moe_grouped_gemm_kernel.arg_names,
        sig,
        out_features=out_features,
        in_features=in_features,
    )
    return compile_fn(
        ASTSource(nvfp4_w4a4_moe_grouped_gemm_kernel, sig, constexprs, attrs=attrs),
        target,
        {"num_warps": schedule.num_warps, "num_stages": schedule.num_stages},
    )


def _compile_nvfp4_w4a4_prefill_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    hidden_size: int,
    intermediate_size: int,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    accumulate: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    schedule = _policy.select_nvfp4_w4a4_prefill_schedule(
        accel,
        chunk_rows=rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
    )
    compiled = _compile_prefill_utility_kernels(
        target,
        gather_ty=out_ty,
        out_ty=out_ty,
        is_int8=False,
        block_m=schedule.block_m,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        compile_fn=compile_fn,
        include_gather=False,
    )
    rq_sig = {
        "x_ptr": out_ty,
        "q_out_ptr": "*u8",
        "sc_out_ptr": "*fp8e4nv",
        "row_scale_ptr": "*fp32",
        "hidden_size": "i32",
        "x_row_stride": "i32",
    }
    rq_attrs = _div16_attrs(
        nvfp4_moe_row_quant_kernel.arg_names,
        rq_sig,
        *(("hidden_size", "x_row_stride") if hidden_size % 16 == 0 else ()),
    )
    compiled["nvfp4_row_quant"] = compile_fn(
        ASTSource(
            nvfp4_moe_row_quant_kernel,
            rq_sig,
            constexprs={
                "block_k": _schedule.NVFP4_W4A4_QUANT_BLOCK_K,
                "group_size": group_size,
                "high_first": high_first,
            },
            attrs=rq_attrs,
        ),
        target,
        {"num_warps": 4},
    )
    g_sig = {
        "in_q_ptr": "*u8",
        "in_sc_ptr": "*fp8e4nv",
        "in_rs_ptr": "*fp32",
        "compact_to_pair_ptr": "*i32",
        "out_q_ptr": "*u8",
        "out_sc_ptr": "*fp8e4nv",
        "out_rs_ptr": "*fp32",
        "packed_k": "i32",
        "top_k": "i32",
    }
    g_attrs = _div16_attrs(
        nvfp4_moe_gather_kernel.arg_names,
        g_sig,
        *(("packed_k",) if (hidden_size // 2) % 16 == 0 else ()),
    )
    compiled["nvfp4_gather"] = compile_fn(
        ASTSource(
            nvfp4_moe_gather_kernel,
            g_sig,
            constexprs={"block_bytes": _schedule.NVFP4_W4A4_GATHER_BLOCK_BYTES},
            attrs=g_attrs,
        ),
        target,
        {"num_warps": 4},
    )
    aq_sig = {
        "mid_ptr": out_ty,
        "compact_to_pair_ptr": "*i32",
        "q_out_ptr": "*u8",
        "sc_out_ptr": "*fp8e4nv",
        "row_scale_ptr": "*fp32",
        "intermediate_size": "i32",
        "gate_bound": "fp32",
        "up_bound": "fp32",
    }
    aq_attrs = _div16_attrs(
        nvfp4_moe_activate_quant_kernel.arg_names,
        aq_sig,
        *(("intermediate_size",) if intermediate_size % 16 == 0 else ()),
    )
    compiled["nvfp4_activate_quant"] = compile_fn(
        ASTSource(
            nvfp4_moe_activate_quant_kernel,
            aq_sig,
            constexprs={
                "block_k": _schedule.NVFP4_W4A4_QUANT_BLOCK_K,
                "group_size": group_size,
                "is_gated": is_gated_activation(activation_fn),
                "gate_first": gate_first,
                "high_first": high_first,
                "has_gate_bound": gate_bound is not None,
                "has_up_bound": up_bound is not None,
                "accelerator_backend": accel.backend,
            },
            attrs=aq_attrs,
        ),
        target,
        {"num_warps": 4},
    )
    up_gate_out = intermediate_size * (2 if is_gated_activation(activation_fn) else 1)
    compiled["nvfp4_w4a4_grouped_up"] = _compile_nvfp4_w4a4_grouped_gemm(
        target,
        schedule=schedule,
        output_ty=out_ty,
        out_features=up_gate_out,
        in_features=hidden_size,
        has_global_scale=has_global_scale,
        compile_fn=compile_fn,
    )
    compiled["nvfp4_w4a4_grouped_down"] = _compile_nvfp4_w4a4_grouped_gemm(
        target,
        schedule=schedule,
        output_ty="*fp32",
        out_features=hidden_size,
        in_features=intermediate_size,
        has_global_scale=has_global_scale,
        compile_fn=compile_fn,
    )
    return compiled


def _compile_int8_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    batched: bool,
    num_pairs: int = 1,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    split_k: int,
    accumulate: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    up_gate_out = intermediate_size * (2 if is_gated_activation(activation_fn) else 1)
    schedule = _policy.select_int8_launch_schedule(accel)
    block_k = schedule.block_k
    num_stages = schedule.num_stages
    compiler_options = amd_compiler_workarounds.lds_only_fence_options(schedule.lds_only_fences)
    is_splitk = split_k > 1
    seed_rows = (num_pairs if batched else 1) * _schedule.INT8_GEMM_BLOCK_M
    up_seed = splitk_accumulator_seed(
        accel,
        rows=seed_rows,
        out_features=up_gate_out,
        in_features=hidden_size,
        block_m=_schedule.INT8_GEMM_BLOCK_M,
        block_n=_schedule.INT8_GEMM_BLOCK_N,
        block_k=block_k,
        paired=False,
        split_k=split_k,
    )
    down_seed = splitk_accumulator_seed(
        accel,
        rows=seed_rows,
        out_features=hidden_size,
        in_features=intermediate_size,
        block_m=_schedule.INT8_GEMM_BLOCK_M,
        block_n=_schedule.INT8_GEMM_BLOCK_N,
        block_k=block_k,
        paired=False,
        split_k=split_k,
    )
    kernel_fn = int8_moe_batched_matmul_kernel if batched else int8_moe_m1_matmul_kernel
    int8_signature: dict[str, str] = (
        {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "tile_slots_ptr": "*i32",
            "tile_pair_ids_ptr": "*i32",
            "input_scale_ptr": "*fp32",
            "weight_scale_ptr": "*fp32",
            "output_ptr": "*i32" if is_splitk else out_ty,
            "out_features": "i32",
            "in_features": "i32",
            "num_pairs": "i32",
            "top_k": "i32",
            "split_k": "i32",
        }
        if batched
        else {
            "input_ptr": "*i8",
            "weight_ptr": "*i8",
            "topk_ids_ptr": "*i64",
            "input_scale_ptr": "*fp32",
            "weight_scale_ptr": "*fp32",
            "output_ptr": "*i32" if is_splitk else out_ty,
            "out_features": "i32",
            "in_features": "i32",
            "num_experts": "i32",
            "num_slots": "i32",
            "top_k": "i32",
            "split_k": "i32",
        }
    )
    slot_map_cfg: dict[str, object] = {}
    if not batched:
        slot_map_cfg["has_slot_map"] = has_slot_map
        _bind_opt_ptrs(
            int8_signature,
            slot_map_cfg,
            expert_to_slot_ptr="*i64" if has_slot_map else None,
        )
    up_block_m = _schedule.BATCH_BLOCK_M if batched else _schedule.INT8_GEMM_BLOCK_M
    compiled_int8: dict[str, triton.compiler.CompiledKernel] = {
        "int8_matmul_up": compile_fn(
            ASTSource(
                kernel_fn,
                int8_signature,
                constexprs={
                    "block_m": up_block_m,
                    "block_n": _schedule.INT8_GEMM_BLOCK_N,
                    "block_k": block_k,
                    **slot_map_cfg,
                    "per_expert_input": False,
                    "is_splitk": is_splitk,
                    "dequant_output": not is_splitk,
                    "aligned_nk": (
                        up_gate_out % _schedule.INT8_GEMM_BLOCK_N == 0
                        and hidden_size % block_k == 0
                    ),
                    "accumulator_seed": up_seed,
                },
            ),
            target,
            {"num_stages": num_stages, "num_warps": 4, **compiler_options},
        ),
        "int8_matmul_down": compile_fn(
            ASTSource(
                kernel_fn,
                {**int8_signature, "output_ptr": "*i32"},
                constexprs={
                    "block_m": up_block_m,
                    "block_n": _schedule.INT8_GEMM_BLOCK_N,
                    "block_k": block_k,
                    **slot_map_cfg,
                    "per_expert_input": True,
                    "is_splitk": True,
                    "dequant_output": False,
                    "aligned_nk": (
                        hidden_size % _schedule.INT8_GEMM_BLOCK_N == 0
                        and intermediate_size % block_k == 0
                    ),
                    "accumulator_seed": down_seed,
                },
            ),
            target,
            {"num_stages": num_stages, "num_warps": 4, **compiler_options},
        ),
        "int8_activate_quant": _compile_int8_activate_quant(
            target,
            accel,
            out_ty=out_ty,
            intermediate_size=intermediate_size,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            has_slot_map=has_slot_map,
            use_compact_map=False,
            gate_bound=gate_bound,
            up_bound=up_bound,
            compile_fn=compile_fn,
        ),
    }
    if is_splitk:
        compiled_int8["int8_reduce"] = _compile_splitk_reduce(
            target,
            out_ty=out_ty,
            is_int8=True,
            has_slot_map=has_slot_map,
            compile_fn=compile_fn,
        )
    compiled_int8["weighted_reduce"] = _compile_weighted_reduce(
        target,
        out_ty=out_ty,
        is_int8=True,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        has_global_scale=False,
        post_scale_only=True,
        compile_fn=compile_fn,
    )
    return compiled_int8


def _compile_a16_epilogue(
    compiled: dict[str, triton.compiler.CompiledKernel],
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    reduce_key: str,
    active_ty: str,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    post_scale_only: bool,
    has_row_scale: bool,
    is_splitk: bool,
    accumulate: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> None:
    if is_splitk:
        compiled[reduce_key] = _compile_splitk_reduce(
            target,
            out_ty=out_ty,
            is_int8=False,
            has_slot_map=has_slot_map,
            has_global_scale=has_global_scale,
            post_scale_only=post_scale_only,
            has_row_scale=has_row_scale,
            compile_fn=compile_fn,
        )
    compiled["activate_rotate"] = _compile_activate_rotate(
        target,
        active_ty=active_ty,
        out_ty=out_ty,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        accelerator_backend=accel.backend,
        has_slot_map=has_slot_map,
        use_compact_map=False,
        gate_bound=gate_bound,
        up_bound=up_bound,
        compile_fn=compile_fn,
    )
    compiled["weighted_reduce"] = _compile_weighted_reduce(
        target,
        out_ty=out_ty,
        is_int8=False,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        has_global_scale=has_global_scale,
        post_scale_only=post_scale_only,
        has_row_scale=has_row_scale,
        compile_fn=compile_fn,
    )


def _compile_nvfp4_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    batched: bool,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    logical_dtype: torch.dtype,
    out_ty: str,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    split_k: int,
    accumulate: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    active_dtype = torch.float32 if group_size > 0 else logical_dtype
    active_ty = TRITON_FLOAT_POINTER_TYPES[active_dtype]
    is_splitk = split_k > 1
    if batched:
        schedule = _policy.select_nvfp4_batched_schedule(
            accel,
            rows=rows,
            active_dtype=active_dtype,
            logical_dtype=logical_dtype,
        )
        post_scale_only = schedule.post_scale_only
        nvfp4_signature: dict[str, str] = {
            "input_ptr": active_ty,
            "weight_ptr": "*u8",
            "weight_scale_ptr": "*u8",
            "tile_slots_ptr": "*i32",
            "tile_pair_ids_ptr": "*i32",
            "output_ptr": "*fp32",
            "out_features": "i32",
            "packed_k": "i32",
            "scale_column_blocks": "i32",
            "expert_scale_stride": "i32",
            "num_pairs": "i32",
            "top_k": "i32",
            "split_k": "i32",
        }
        kernel_fn = nvfp4_moe_batched_gemv_kernel
        mode_constexprs: dict[str, object] = {
            "block_m": _schedule.BATCH_BLOCK_M,
            "block_n": schedule.block_n,
            "block_k": schedule.block_k,
            "use_dot": schedule.use_dot,
            "cast_fp32_to_bf16": schedule.cast_fp32_to_bf16,
            "single_dot": schedule.single_dot,
        }
        gemv_opts: dict[str, object] = {"num_warps": schedule.num_warps, "num_stages": 1}
    else:
        m1_schedule = _policy.select_nvfp4_m1_schedule(accel)
        post_scale_only = True
        nvfp4_signature = {
            "input_ptr": active_ty,
            "weight_ptr": "*u8",
            "weight_scale_ptr": "*u8",
            "topk_ids_ptr": "*i64",
            "output_ptr": "*fp32",
            "out_features": "i32",
            "packed_k": "i32",
            "scale_column_blocks": "i32",
            "expert_scale_stride": "i32",
            "num_experts": "i32",
            "num_slots": "i32",
            "top_k": "i32",
            "split_k": "i32",
        }
        kernel_fn = nvfp4_moe_m1_gemv_kernel
        mode_constexprs = {
            "block_n": m1_schedule.block_n,
            "block_k": m1_schedule.block_k,
            "has_slot_map": has_slot_map,
        }
        _bind_opt_ptrs(
            nvfp4_signature,
            mode_constexprs,
            expert_to_slot_ptr="*i64" if has_slot_map else None,
        )
        gemv_opts = {"num_warps": m1_schedule.num_warps}
    _bind_opt_ptrs(
        nvfp4_signature,
        mode_constexprs,
        weight_global_scale_ptr="*fp32" if has_global_scale else None,
    )

    up_gate_out = intermediate_size * (2 if is_gated_activation(activation_fn) else 1)
    up_attrs = _nvfp4_aligned_attrs(
        kernel_fn.arg_names,
        nvfp4_signature,
        out_features=up_gate_out,
        in_features=hidden_size,
    )
    down_attrs = _nvfp4_aligned_attrs(
        kernel_fn.arg_names,
        nvfp4_signature,
        out_features=hidden_size,
        in_features=intermediate_size,
    )
    compiled_nvfp4: dict[str, triton.compiler.CompiledKernel] = {
        "nvfp4_gemv_up": compile_fn(
            ASTSource(
                kernel_fn,
                nvfp4_signature,
                constexprs={
                    **mode_constexprs,
                    "high_first": high_first,
                    "has_global_scale": has_global_scale,
                    "per_expert_input": False,
                    "is_splitk": is_splitk,
                },
                attrs=up_attrs,
            ),
            target,
            gemv_opts,
        ),
        "nvfp4_gemv_down": compile_fn(
            ASTSource(
                kernel_fn,
                nvfp4_signature,
                constexprs={
                    **mode_constexprs,
                    "high_first": high_first,
                    "has_global_scale": has_global_scale,
                    "per_expert_input": True,
                    "is_splitk": True,
                },
                attrs=down_attrs,
            ),
            target,
            gemv_opts,
        ),
    }
    _compile_a16_epilogue(
        compiled_nvfp4,
        target,
        accel,
        reduce_key="nvfp4_reduce",
        active_ty=active_ty,
        out_ty=out_ty,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        has_slot_map=has_slot_map,
        has_global_scale=has_global_scale,
        post_scale_only=post_scale_only,
        has_row_scale=False,
        is_splitk=is_splitk,
        accumulate=accumulate,
        gate_bound=gate_bound,
        up_bound=up_bound,
        compile_fn=compile_fn,
    )
    return compiled_nvfp4


def _compile_trellis_prefill_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    block_size: int,
    variable_rows: bool,
    has_slot_offsets: bool,
    bytes_per_block: int | None,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    accumulate: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel]:
    compiled = _compile_int8_prefill_kernels(
        target,
        accel,
        rows=rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        out_ty=out_ty,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        has_slot_map=has_slot_map,
        accumulate=accumulate,
        slot_window=True,
        gate_bound=gate_bound,
        up_bound=up_bound,
        compile_fn=compile_fn,
    )

    tile_steps = triton.next_power_of_2(block_size)
    per_slot_2d = variable_rows and not has_slot_offsets
    up_gate_out = intermediate_size * (2 if is_gated_activation(activation_fn) else 1)
    if variable_rows:
        needs_w2 = True
        stream_bytes = 1
    else:
        layout = default_layout_for_block_size(block_size, bytes_per_block)
        needs_w2 = layout.block_bits > 5 * layout.block_size
        up_slot_bytes = up_gate_out * (hidden_size // block_size) * layout.bytes_per_block
        down_slot_bytes = hidden_size * (intermediate_size // block_size) * layout.bytes_per_block
        stream_bytes = up_slot_bytes if (down_slot_bytes % 4 == 0) else down_slot_bytes
    use_dp4a, use_sad_u8, has_tail_bytes, clamped_load, num_warps = _decode_stream_flags(
        accel,
        stream_bytes,
        variable_rows=variable_rows,
        ptr_aligned=True,
        per_slot_2d=per_slot_2d,
    )
    dec_sig: dict[str, str] = {
        "bitstreams": "*u8",
        "block_scales": "*u8",
        "qdata": "*i8",
        "active_counts_ptr": "*i32",
        "num_blocks": "i32",
        "blocks_per_row": "i32",
        "bitstreams_bytes": "i64",
        "block_size": "i32",
        "step_bits": "i32",
        "num_wide_steps": "i32",
        "bytes_per_block": "i32",
        "rows_per_slot": "i32",
        "slot_bytes": "i32",
    }
    dec_constexprs: dict[str, object] = {
        "tile_steps": tile_steps,
        "tile_blocks": 4,
        "variable_rows": variable_rows,
        "exact_pow2": block_size == tile_steps,
        "filter_active": True,
        "has_slot_offsets": has_slot_offsets,
        "per_slot_2d": per_slot_2d,
        "needs_w2": needs_w2,
        "use_dp4a": use_dp4a,
        "use_sad_u8": use_sad_u8,
        "has_tail_bytes": has_tail_bytes,
        "ptr_aligned": True,
        "clamped_load": clamped_load,
        "q_u32_aligned": (block_size % 4 == 0),
    }
    _bind_opt_ptrs(
        dec_sig,
        dec_constexprs,
        row_widths="*u8" if variable_rows else None,
        row_offsets="*i32" if variable_rows else None,
        slot_offsets_ptr="*i64" if has_slot_offsets else None,
    )

    compiled["trellis_decode"] = compile_fn(
        ASTSource(decode_slots_int8_kernel, dec_sig, constexprs=dec_constexprs),
        target,
        {"num_warps": num_warps},
    )
    return compiled


def _compile_trellis_kernels(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    batched: bool,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    block_size: int,
    variable_rows: bool,
    has_slot_offsets: bool,
    bytes_per_block: int | None,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    has_slot_map: bool,
    split_k: int,
    accumulate: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
    rows: int = 1,
    top_k: int = 1,
    num_slots: int = 64,
    active_slots_hint: int = 0,
) -> dict[str, triton.compiler.CompiledKernel]:
    tile_steps = triton.next_power_of_2(block_size)
    exact_pow2 = block_size == tile_steps
    expected_slots = min(num_slots, active_slots_hint) if active_slots_hint > 0 else num_slots
    schedule = _policy.select_trellis_schedule(
        accel,
        batched=batched,
        num_pairs=rows * top_k,
        expected_slots=expected_slots,
    )
    is_splitk = split_k > 1
    up_gate_out = intermediate_size * (2 if is_gated_activation(activation_fn) else 1)
    per_slot_2d = variable_rows and not has_slot_offsets
    if variable_rows:
        needs_w2 = True
        up_has_tail = True
        down_has_tail = True
    else:
        layout = default_layout_for_block_size(block_size, bytes_per_block)
        needs_w2 = layout.block_bits > 5 * layout.block_size
        up_bytes = num_slots * up_gate_out * (hidden_size // block_size) * layout.bytes_per_block
        down_bytes = (
            num_slots * hidden_size * (intermediate_size // block_size) * layout.bytes_per_block
        )
        up_has_tail = _decode_stream_flags(
            accel, up_bytes, variable_rows=False, per_slot_2d=per_slot_2d
        )[2]
        down_has_tail = _decode_stream_flags(
            accel, down_bytes, variable_rows=False, per_slot_2d=per_slot_2d
        )[2]

    gemv_sig: dict[str, str] = {
        "x_rot_ptr": "*fp32",
        "bitstreams_ptr": "*u8",
        "block_scales_ptr": "*u8",
        "row_scale_ptr": "*fp32",
        "output_ptr": "*fp32",
        "out_features": "i32",
        "in_features": "i32",
        "blocks_per_row": "i32",
        "bitstreams_bytes": "i64",
        "block_size": "i32",
        "step_bits": "i32",
        "num_wide_steps": "i32",
        "bytes_per_block": "i32",
        "slot_bytes": "i32",
        "num_experts": "i32",
        "num_slots": "i32",
        "num_pairs": "i32",
        "top_k": "i32",
        "grid_tiles": "i32",
        "split_k": "i32",
    }
    gemv_constexprs: dict[str, object] = {
        "tile_steps": tile_steps,
        "block_m": schedule.block_m,
        "block_n": schedule.block_n,
        "batched": batched,
        "has_slot_map": False if batched else has_slot_map,
        "is_splitk": is_splitk,
        "variable_rows": variable_rows,
        "has_slot_offsets": has_slot_offsets,
        "exact_pow2": exact_pow2,
        "needs_w2": needs_w2,
        "use_dp4a": schedule.use_dp4a,
        "use_sad_u8": schedule.use_sad_u8,
        "split_x_loads": schedule.split_x_loads,
        "ptr_aligned": True,
        "x_mul4": (block_size % 4 == 0),
        "unroll_k": schedule.unroll_k,
    }
    _bind_opt_ptrs(
        gemv_sig,
        gemv_constexprs,
        row_widths_ptr="*u8" if variable_rows else None,
        row_offsets_ptr="*i32" if variable_rows else None,
        slot_offsets_ptr="*i64" if has_slot_offsets else None,
        tile_slots_ptr="*i32" if batched else None,
        tile_pair_ids_ptr="*i32" if batched else None,
        topk_ids_ptr=None if batched else "*i64",
        expert_to_slot_ptr="*i64" if (not batched and has_slot_map) else None,
    )

    compiled: dict[str, triton.compiler.CompiledKernel] = {
        "trellis_gemv_up": compile_fn(
            ASTSource(
                trellis_moe_gemv_kernel,
                gemv_sig,
                constexprs={
                    **gemv_constexprs,
                    "has_tail_bytes": up_has_tail,
                    "per_expert_input": False,
                },
            ),
            target,
            {"num_warps": schedule.num_warps},
        ),
        "trellis_gemv_down": compile_fn(
            ASTSource(
                trellis_moe_gemv_kernel,
                gemv_sig,
                constexprs={
                    **gemv_constexprs,
                    "has_tail_bytes": down_has_tail,
                    "per_expert_input": True,
                    "is_splitk": True,
                },
            ),
            target,
            {"num_warps": schedule.num_warps},
        ),
    }
    _compile_a16_epilogue(
        compiled,
        target,
        accel,
        reduce_key="trellis_reduce",
        active_ty="*fp32",
        out_ty=out_ty,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        has_slot_map=has_slot_map,
        has_global_scale=False,
        post_scale_only=True,
        has_row_scale=True,
        is_splitk=is_splitk,
        accumulate=accumulate,
        gate_bound=gate_bound,
        up_bound=up_bound,
        compile_fn=compile_fn,
    )
    return compiled


def _compile_prefill_kernels_dispatch(
    target: GPUTarget,
    accel: AcceleratorTarget,
    *,
    is_trellis: bool,
    has_int8_backend: bool,
    is_nvfp4_w4a4: bool,
    chunk_rows: int,
    top_k: int,
    num_slots: int,
    active_slots_hint: int,
    logical_dtype: torch.dtype,
    out_ty: str,
    hidden_size: int,
    intermediate_size: int,
    block_size: int,
    variable_rows: bool,
    has_slot_offsets: bool,
    bytes_per_block: int | None,
    group_size: int,
    activation_fn: str,
    gate_first: bool,
    high_first: bool,
    has_slot_map: bool,
    has_global_scale: bool,
    accumulate: bool,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compile_fn: TritonCompileFn,
) -> dict[str, triton.compiler.CompiledKernel] | None:
    if is_trellis and has_int8_backend:
        return _compile_trellis_prefill_kernels(
            target,
            accel,
            rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            group_size=group_size,
            block_size=block_size,
            variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets,
            bytes_per_block=bytes_per_block,
            activation_fn=activation_fn,
            gate_first=gate_first,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            gate_bound=gate_bound,
            up_bound=up_bound,
            compile_fn=compile_fn,
        )
    if has_int8_backend:
        return _compile_int8_prefill_kernels(
            target,
            accel,
            rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            has_slot_map=has_slot_map,
            accumulate=accumulate,
            gate_bound=gate_bound,
            up_bound=up_bound,
            compile_fn=compile_fn,
        )
    if is_nvfp4_w4a4:
        return _compile_nvfp4_w4a4_prefill_kernels(
            target,
            accel,
            rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            out_ty=out_ty,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            has_slot_map=has_slot_map,
            has_global_scale=has_global_scale,
            accumulate=accumulate,
            gate_bound=gate_bound,
            up_bound=up_bound,
            compile_fn=compile_fn,
        )
    if not is_trellis:
        return _compile_nvfp4_prefill_kernels(
            target,
            accel,
            rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            logical_dtype=logical_dtype,
            out_ty=out_ty,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            has_slot_map=has_slot_map,
            has_global_scale=has_global_scale,
            accumulate=accumulate,
            gate_bound=gate_bound,
            up_bound=up_bound,
            compile_fn=compile_fn,
        )
    return None


def compile_moe_kernels(
    target: GPUTarget,
    *,
    weight_format: str = "nvfp4_w4a16",
    logical_dtype: torch.dtype = torch.bfloat16,
    rows: int = 1,
    top_k: int = 1,
    num_slots: int = 64,
    active_slots_hint: int = 0,
    hidden_size: int = 256,
    intermediate_size: int = 256,
    group_size: int = 0,
    activation_fn: str = "swiglu",
    gate_first: bool = True,
    high_first: bool = False,
    has_slot_map: bool = True,
    has_global_scale: bool = True,
    split_k: int = 1,
    accumulate: bool = False,
    block_size: int = 256,
    variable_rows: bool = False,
    has_slot_offsets: bool = False,
    bytes_per_block: int | None = None,
    gate_bound: float | None = None,
    up_bound: float | None = None,
    compiler: TritonCompileFn | None = None,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile quantized MoE Triton kernels matching runtime regime and chunking selection."""
    if logical_dtype not in TRITON_FLOAT_POINTER_TYPES:
        raise ValueError(f"Unsupported MoE logical dtype: {logical_dtype}")
    if (
        hidden_size <= 0
        or intermediate_size <= 0
        or split_k <= 0
        or rows <= 0
        or top_k <= 0
        or num_slots <= 0
        or active_slots_hint < 0
    ):
        raise ValueError("MoE dimensions, rows, top_k, num_slots, and split_k must be positive")
    resolved_gate_bound, resolved_up_bound = validate_moe_activation_bounds(
        activation_fn, gate_bound, up_bound
    )
    accel = AcceleratorTarget.from_compiler_target(target)
    _validate_compile_format(
        accel,
        weight_format=weight_format,
        group_size=group_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        high_first=high_first,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
        bytes_per_block=bytes_per_block,
    )

    up_gate_out = intermediate_size * (2 if is_gated_activation(activation_fn) else 1)
    is_int8 = weight_format == "convrot_int8"
    is_trellis = weight_format == "convrot_trellis"
    is_nvfp4_w4a4 = weight_format in ("nvfp4", "convrot_nvfp4")
    compile_fn = compiler or default_triton_compile
    out_ty = TRITON_FLOAT_POINTER_TYPES[logical_dtype]

    has_int8_backend = is_int8
    if is_trellis:
        from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

        has_int8_backend = (
            _policy.supports_trellis_int8_prefill(accel)
            and int8_backend_mod.select_linear_backend_for_target(accel) is not None
        )

    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=has_int8_backend,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        up_gate_out=up_gate_out,
        logical_elem_bytes=logical_dtype.itemsize,
        group_size=group_size,
        is_nvfp4_w4a4=is_nvfp4_w4a4,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)

    if regime == "prefill":
        prefill_compiled = _compile_prefill_kernels_dispatch(
            target,
            accel,
            is_trellis=is_trellis,
            has_int8_backend=has_int8_backend,
            is_nvfp4_w4a4=is_nvfp4_w4a4,
            chunk_rows=chunk_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            logical_dtype=logical_dtype,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_size=block_size,
            variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets,
            bytes_per_block=bytes_per_block,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            has_slot_map=has_slot_map,
            has_global_scale=has_global_scale,
            accumulate=accumulate,
            gate_bound=resolved_gate_bound,
            up_bound=resolved_up_bound,
            compile_fn=compile_fn,
        )
        if prefill_compiled is not None:
            return prefill_compiled

    if regime == "prefill":
        a16_rows = min(
            rows, _schedule.BATCH_MAX_ROWS, max(1, _schedule.BUCKET_MAX_PAIRS // max(1, top_k))
        )
        batched = a16_rows > 1
    else:
        a16_rows = chunk_rows
        batched = regime == "batched"
    num_pairs = a16_rows * top_k

    if is_trellis:
        compiled = _compile_trellis_kernels(
            target,
            accel,
            batched=batched,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_size=block_size,
            variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets,
            bytes_per_block=bytes_per_block,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            has_slot_map=has_slot_map,
            split_k=split_k,
            accumulate=accumulate,
            gate_bound=resolved_gate_bound,
            up_bound=resolved_up_bound,
            compile_fn=compile_fn,
            rows=a16_rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
        )
    elif is_int8:
        compiled = _compile_int8_kernels(
            target,
            accel,
            batched=batched,
            num_pairs=num_pairs,
            out_ty=out_ty,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            has_slot_map=has_slot_map,
            split_k=split_k,
            accumulate=accumulate,
            gate_bound=resolved_gate_bound,
            up_bound=resolved_up_bound,
            compile_fn=compile_fn,
        )
    else:
        compiled = _compile_nvfp4_kernels(
            target,
            accel,
            batched=batched,
            rows=chunk_rows,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            logical_dtype=logical_dtype,
            out_ty=out_ty,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            has_slot_map=has_slot_map,
            has_global_scale=has_global_scale,
            split_k=split_k,
            accumulate=accumulate,
            gate_bound=resolved_gate_bound,
            up_bound=resolved_up_bound,
            compile_fn=compile_fn,
        )
    if batched:
        compiled["bucket"] = _compile_bucket_kernel(
            target,
            num_pairs=num_pairs,
            has_slot_map=has_slot_map,
            compile_fn=compile_fn,
        )
    return compiled
