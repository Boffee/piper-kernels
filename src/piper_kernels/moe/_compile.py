"""Offline Triton kernel compilation helpers for quantized MoE operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

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
from piper_kernels.trellis._layout import default_layout_for_block_size
from piper_kernels.trellis.triton import _decode_stream_flags, decode_slots_int8_kernel

from . import (
    _int8_triton,
    _nvfp4_triton,
    _ops,
    _policy,
    _routing_triton,
    _schedule,
    _trellis_triton,
)
from ._nvidia import policy as _nvidia_policy
from .reference import _ActivationConfig, validate_moe_activation_bounds


class _JitKernel(Protocol):
    arg_names: list[str]


@dataclass(frozen=True, slots=True)
class _CompileConfig:
    target: GPUTarget
    accel: AcceleratorTarget
    weight_format: str
    logical_dtype: torch.dtype
    out_ty: str
    rows: int
    top_k: int
    num_slots: int
    active_slots_hint: int
    hidden_size: int
    intermediate_size: int
    up_gate_out: int
    group_size: int
    activation: _ActivationConfig
    high_first: bool
    has_slot_map: bool
    has_global_scale: bool
    split_k: int
    accumulate: bool
    block_size: int
    variable_rows: bool
    has_slot_offsets: bool
    bytes_per_block: int | None
    compile_fn: TritonCompileFn

    @property
    def active_dtype(self) -> torch.dtype:
        return torch.float32 if self.group_size > 0 else self.logical_dtype

    @property
    def active_ty(self) -> str:
        return TRITON_FLOAT_POINTER_TYPES[self.active_dtype]

    @property
    def is_splitk(self) -> bool:
        return self.split_k > 1


_EXACT_ARG_TYPES: dict[str, str] = {
    "gate_bound": "fp32",
    "up_bound": "fp32",
    "bitstreams_bytes": "i64",
    "topk_ids_ptr": "*i64",
    "sc_out_ptr": "*fp8e4nv",
}
_SUFFIX_ARG_TYPES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("_sc_ptr",), "*fp8e4nv"),
    (("_q_ptr", "bitstreams", "bitstreams_ptr", "block_scales", "block_scales_ptr"), "*u8"),
    (("_scale_ptr", "_rs_ptr", "_gs_ptr", "scale_out_ptr", "x_rot_ptr"), "*fp32"),
    (("_ptr",), "*i32"),
)


def _default_arg_type(name: str) -> str:
    if name in _EXACT_ARG_TYPES:
        return _EXACT_ARG_TYPES[name]
    for suffixes, ty in _SUFFIX_ARG_TYPES:
        if name.endswith(suffixes):
            return ty
    return "i32"


def build_kernel_source(
    kernel: _JitKernel,
    *,
    constexprs: Mapping[str, object],
    overrides: Mapping[str, str | None] | None = None,
    div16_ints: Sequence[str] = (),
    align_ptrs: bool = False,
) -> ASTSource:
    """Build a Triton ``ASTSource`` with canonical pointer defaults and ``i32`` scalar defaults."""
    resolved_constexprs = dict(constexprs)
    ov = overrides or {}
    sig: dict[str, str] = {}
    for name in kernel.arg_names:
        if name in resolved_constexprs:
            continue
        if name in ov:
            ty = ov[name]
            if ty is None:
                resolved_constexprs[name] = None
            else:
                sig[name] = ty
        else:
            sig[name] = _default_arg_type(name)
    attrs: dict[tuple[int, ...], list[list[object]]] | None = None
    if align_ptrs or div16_ints:
        aligned = [name for name, ty in sig.items() if ty.startswith("*")] if align_ptrs else []
        aligned.extend(div16_ints)
        attrs = {(kernel.arg_names.index(name),): [["tt.divisibility", 16]] for name in aligned}
    return ASTSource(kernel, sig, constexprs=resolved_constexprs, attrs=attrs)


def _compile_kernel(
    cfg: _CompileConfig,
    kernel: _JitKernel,
    *,
    constexprs: Mapping[str, object],
    overrides: Mapping[str, str | None] | None = None,
    options: Mapping[str, object] | None = None,
    div16_ints: Sequence[str] = (),
    align_ptrs: bool = False,
) -> triton.compiler.CompiledKernel:
    src = build_kernel_source(
        kernel,
        constexprs=constexprs,
        overrides=overrides,
        div16_ints=div16_ints,
        align_ptrs=align_ptrs,
    )
    return cfg.compile_fn(src, cfg.target, dict(options or {"num_warps": 4}))


def _nvfp4_div16_ints(out_features: int, in_features: int) -> tuple[str, ...]:
    extra = ["expert_scale_stride"]
    if out_features % 16 == 0:
        extra.append("out_features")
    if (in_features // 2) % 16 == 0:
        extra.append("packed_k")
    return tuple(extra)


def _opt_ptr(flag: bool, ty: str) -> str | None:
    return ty if flag else None


def _activation_constexprs(cfg: _CompileConfig, *, active: bool = True) -> dict[str, object]:
    act = cfg.activation
    return {
        "group_size": cfg.group_size,
        "is_gated": act.is_gated if active else False,
        "gate_first": act.gate_first if active else True,
        "has_gate_bound": active and act.gate_bound is not None,
        "has_up_bound": active and act.up_bound is not None,
        "accelerator_backend": cfg.accel.backend if active else "cuda",
    }


def _compile_reduce(
    cfg: _CompileConfig,
    *,
    weighted: bool,
    is_int8: bool,
    has_global_scale: bool = False,
    post_scale_only: bool = True,
    use_compact_map: bool = False,
    has_row_scale: bool = False,
) -> triton.compiler.CompiledKernel:
    eff_int8 = False if use_compact_map else is_int8
    eff_slot_map = False if use_compact_map else cfg.has_slot_map
    eff_global = False if (use_compact_map or eff_int8) else has_global_scale
    eff_row = False if (use_compact_map or eff_int8) else has_row_scale
    block_n = (
        _schedule.WEIGHTED_REDUCE_BLOCK_N
        if weighted
        else (_schedule.INT8_REDUCE_BLOCK_N if eff_int8 else _schedule.NVFP4_REDUCE_BLOCK_N)
    )
    constexprs: dict[str, object] = {
        "block_n": block_n,
        "is_int8": eff_int8,
        "has_slot_map": eff_slot_map,
        "has_global_scale": eff_global,
        "post_scale_only": True if (use_compact_map or eff_int8) else post_scale_only,
        "has_row_scale": eff_row,
    }
    overrides: dict[str, str | None] = {
        "partial_ptr": "*i32" if eff_int8 else "*fp32",
        "input_scale_ptr": _opt_ptr(eff_int8, "*fp32"),
        "weight_scale_ptr": _opt_ptr(eff_int8 or eff_row, "*fp32"),
        "weight_global_scale_ptr": _opt_ptr(eff_global, "*fp32"),
        "topk_ids_ptr": "*i32" if use_compact_map else "*i64",
        "expert_to_slot_ptr": _opt_ptr(eff_slot_map, "*i64"),
        "output_ptr": cfg.out_ty if (weighted or eff_int8) else "*fp32",
    }
    if weighted:
        constexprs.update({"accumulate": cfg.accumulate, "use_compact_map": use_compact_map})
        overrides["topk_weights_ptr"] = cfg.out_ty
    kernel: _JitKernel = (
        _routing_triton.moe_m1_weighted_reduce_kernel
        if weighted
        else _routing_triton.moe_splitk_reduce_kernel
    )
    return _compile_kernel(cfg, kernel, constexprs=constexprs, overrides=overrides)


def _compile_activate_rotate(
    cfg: _CompileConfig, *, active_ty: str, use_compact_map: bool = False
) -> triton.compiler.CompiledKernel:
    eff_slot_map = False if use_compact_map else cfg.has_slot_map
    return _compile_kernel(
        cfg,
        _routing_triton.moe_m1_activate_rotate_kernel,
        constexprs={
            "block_size": _schedule.ACTIVATE_BLOCK_SIZE,
            **_activation_constexprs(cfg),
            "has_slot_map": eff_slot_map,
        },
        overrides={
            "mid_ptr": "*fp32",
            "act_ptr": active_ty,
            "logical_ref_ptr": cfg.out_ty,
            "topk_ids_ptr": "*i32" if use_compact_map else "*i64",
            "expert_to_slot_ptr": _opt_ptr(eff_slot_map, "*i64"),
        },
    )


def _compile_int8_activate_quant(
    cfg: _CompileConfig, *, use_compact_map: bool
) -> triton.compiler.CompiledKernel:
    eff_slot_map = False if use_compact_map else cfg.has_slot_map
    single_tile, block_k, num_warps = _schedule.select_int8_activate_schedule(cfg.intermediate_size)
    return _compile_kernel(
        cfg,
        _int8_triton.int8_moe_activate_quant_kernel,
        constexprs={
            "block_k": block_k,
            "single_tile": single_tile,
            **_activation_constexprs(cfg),
            "has_slot_map": eff_slot_map,
            "use_compact_map": use_compact_map,
        },
        overrides={
            "mid_ptr": cfg.out_ty,
            "routing_ids_ptr": "*i32" if use_compact_map else "*i64",
            "expert_to_slot_ptr": _opt_ptr(eff_slot_map, "*i64"),
            "q_out_ptr": "*i8",
        },
        options={"num_warps": num_warps},
    )


def _compile_prefill_utilities(
    cfg: _CompileConfig,
    *,
    gather_ty: str,
    is_int8: bool,
    block_m: int,
    include_gather: bool = True,
) -> dict[str, triton.compiler.CompiledKernel]:
    compiled: dict[str, triton.compiler.CompiledKernel] = {
        "bucket_count": _compile_kernel(
            cfg,
            _routing_triton.moe_prefill_bucket_count_kernel,
            constexprs={
                "block_p": _schedule.PREFILL_BUCKET_BLOCK_P,
                "has_slot_map": cfg.has_slot_map,
            },
            overrides={"expert_to_slot_ptr": _opt_ptr(cfg.has_slot_map, "*i64")},
        ),
        "bucket_scatter": _compile_kernel(
            cfg,
            _routing_triton.moe_prefill_bucket_scatter_kernel,
            constexprs={"block_p": _schedule.PREFILL_BUCKET_BLOCK_P, "block_m": block_m},
        ),
        "weighted_reduce": _compile_reduce(cfg, weighted=True, is_int8=False, use_compact_map=True),
    }
    if include_gather:
        compiled["gather"] = _compile_kernel(
            cfg,
            _routing_triton.moe_prefill_gather_kernel,
            constexprs={"block_k": _schedule.PREFILL_GATHER_BLOCK_K, "has_scale": is_int8},
            overrides={
                "input_ptr": gather_ty,
                "input_scale_ptr": _opt_ptr(is_int8, "*fp32"),
                "out_ptr": gather_ty,
                "out_scale_ptr": _opt_ptr(is_int8, "*fp32"),
            },
        )
    return compiled


def _compile_int8_prefill(
    cfg: _CompileConfig, chunk_rows: int, *, slot_window: bool = False
) -> dict[str, triton.compiler.CompiledKernel]:
    sched = _policy.select_int8_prefill_schedule(
        cfg.accel,
        chunk_rows=chunk_rows,
        top_k=cfg.top_k,
        num_slots=cfg.num_slots,
        active_slots_hint=cfg.active_slots_hint,
    )
    opts: dict[str, object] = {
        "num_stages": sched.num_stages,
        "num_warps": sched.num_warps,
        **amd_compiler_workarounds.lds_only_fence_options(sched.lds_only_fences),
    }
    expected_tiles = _schedule.expected_bucket_tiles(
        chunk_rows * cfg.top_k,
        cfg.num_slots,
        cfg.active_slots_hint,
        block_m=sched.block_m,
        batched=True,
    )
    base_const: dict[str, object] = {
        "block_m": sched.block_m,
        "block_n": sched.block_n,
        "block_k": sched.block_k,
        "group_m": _schedule.PREFILL_GROUP_M,
        "slot_window": slot_window,
    }
    compiled = _compile_prefill_utilities(cfg, gather_ty="*i8", is_int8=True, block_m=sched.block_m)
    compiled["int8_activate_quant"] = _compile_int8_activate_quant(cfg, use_compact_map=True)
    for key, out_ty, out_f, in_f in (
        ("int8_grouped_up", cfg.out_ty, cfg.up_gate_out, cfg.hidden_size),
        ("int8_grouped_down", "*fp32", cfg.hidden_size, cfg.intermediate_size),
    ):
        seed = rdna4_int8_accumulator_seed(
            cfg.accel, in_f, expected_tiles * triton.cdiv(out_f, sched.block_n)
        )
        compiled[key] = _compile_kernel(
            cfg,
            _int8_triton.int8_moe_grouped_gemm_kernel,
            constexprs={
                **base_const,
                "aligned_nk": out_f % sched.block_n == 0 and in_f % sched.block_k == 0,
                "accumulator_seed": seed,
            },
            overrides={"input_ptr": "*i8", "weight_ptr": "*i8", "output_ptr": out_ty},
            options=opts,
        )
    return compiled


def _compile_nvfp4_prefill(
    cfg: _CompileConfig, chunk_rows: int
) -> dict[str, triton.compiler.CompiledKernel]:
    sched = _nvidia_policy.select_nvfp4_prefill_schedule(
        cfg.accel,
        chunk_rows=chunk_rows,
        top_k=cfg.top_k,
        num_slots=cfg.num_slots,
        active_slots_hint=cfg.active_slots_hint,
        active_dtype=cfg.active_dtype,
        logical_dtype=cfg.logical_dtype,
    )
    constexprs: dict[str, object] = {
        "block_m": sched.block_m,
        "block_n": sched.block_n,
        "block_k": sched.block_k,
        "group_m": _schedule.PREFILL_GROUP_M,
        "high_first": cfg.high_first,
        "has_global_scale": cfg.has_global_scale,
        "cast_fp32_to_bf16": sched.cast_fp32_to_bf16,
        "force_fp32_dot": sched.force_fp32_dot,
        "single_dot": sched.single_dot,
    }
    overrides: dict[str, str | None] = {
        "input_ptr": cfg.active_ty,
        "weight_ptr": "*u8",
        "weight_scale_ptr": "*u8",
        "weight_global_scale_ptr": _opt_ptr(cfg.has_global_scale, "*fp32"),
        "output_ptr": "*fp32",
    }
    opts: dict[str, object] = {"num_warps": sched.num_warps, "num_stages": 1}

    def _compile_grouped(div16: tuple[str, ...]) -> triton.compiler.CompiledKernel:
        return _compile_kernel(
            cfg,
            _nvfp4_triton.nvfp4_moe_grouped_gemm_kernel,
            constexprs=constexprs,
            overrides=overrides,
            options=opts,
            div16_ints=div16,
            align_ptrs=True,
        )

    compiled = _compile_prefill_utilities(
        cfg, gather_ty=cfg.active_ty, is_int8=False, block_m=sched.block_m
    )
    up_div16 = _nvfp4_div16_ints(cfg.up_gate_out, cfg.hidden_size)
    down_div16 = _nvfp4_div16_ints(cfg.hidden_size, cfg.intermediate_size)
    up_kernel = _compile_grouped(up_div16)
    if down_div16 == up_div16:
        compiled["nvfp4_grouped"] = up_kernel
    else:
        compiled["nvfp4_grouped_up"] = up_kernel
        compiled["nvfp4_grouped_down"] = _compile_grouped(down_div16)
    compiled["activate_rotate"] = _compile_activate_rotate(
        cfg, active_ty=cfg.active_ty, use_compact_map=True
    )
    return compiled


def _compile_nvfp4_w4a4_prefill(
    cfg: _CompileConfig, chunk_rows: int
) -> dict[str, triton.compiler.CompiledKernel]:
    sched = _nvidia_policy.select_nvfp4_w4a4_prefill_schedule(
        cfg.accel,
        chunk_rows=chunk_rows,
        top_k=cfg.top_k,
        num_slots=cfg.num_slots,
        active_slots_hint=cfg.active_slots_hint,
    )
    compiled = _compile_prefill_utilities(
        cfg, gather_ty=cfg.out_ty, is_int8=False, block_m=sched.block_m, include_gather=False
    )
    for key, activate, div16 in (
        (
            "nvfp4_row_quant",
            False,
            ("width", "src_row_stride") if cfg.hidden_size % 16 == 0 else (),
        ),
        ("nvfp4_activate_quant", True, ("width",) if cfg.intermediate_size % 16 == 0 else ()),
    ):
        compiled[key] = _compile_kernel(
            cfg,
            _nvfp4_triton.nvfp4_moe_row_quant_kernel,
            constexprs={
                "block_k": _schedule.NVFP4_W4A4_QUANT_BLOCK_K,
                "high_first": cfg.high_first,
                "activate": activate,
                **_activation_constexprs(cfg, active=activate),
            },
            overrides={
                "src_ptr": cfg.out_ty,
                "q_out_ptr": "*u8",
                "compact_to_pair_ptr": _opt_ptr(activate, "*i32"),
            },
            div16_ints=div16,
            align_ptrs=True,
        )
    compiled["nvfp4_gather"] = _compile_kernel(
        cfg,
        _nvfp4_triton.nvfp4_moe_gather_kernel,
        constexprs={"block_bytes": _schedule.NVFP4_W4A4_GATHER_BLOCK_BYTES},
        div16_ints=("packed_k",) if (cfg.hidden_size // 2) % 16 == 0 else (),
        align_ptrs=True,
    )
    gemm_const: dict[str, object] = {
        "block_m": sched.block_m,
        "block_n": sched.block_n,
        "block_k": sched.block_k,
        "group_m": _schedule.PREFILL_GROUP_M,
        "has_global_scale": cfg.has_global_scale,
    }
    gemm_opts: dict[str, object] = {"num_warps": sched.num_warps, "num_stages": sched.num_stages}
    for key, out_ty, out_f, in_f in (
        ("nvfp4_w4a4_grouped_up", cfg.out_ty, cfg.up_gate_out, cfg.hidden_size),
        ("nvfp4_w4a4_grouped_down", "*fp32", cfg.hidden_size, cfg.intermediate_size),
    ):
        compiled[key] = _compile_kernel(
            cfg,
            _nvfp4_triton.nvfp4_w4a4_moe_grouped_gemm_kernel,
            constexprs=gemm_const,
            overrides={"output_ptr": out_ty, "b_gs_ptr": _opt_ptr(cfg.has_global_scale, "*fp32")},
            options=gemm_opts,
            div16_ints=_nvfp4_div16_ints(out_f, in_f),
            align_ptrs=True,
        )
    return compiled


def _compile_int8_decode(
    cfg: _CompileConfig, *, batched: bool, num_pairs: int
) -> dict[str, triton.compiler.CompiledKernel]:
    sched = _policy.select_int8_launch_schedule(cfg.accel)
    block_k = sched.block_k
    opts: dict[str, object] = {
        "num_stages": sched.num_stages,
        "num_warps": 4,
        **amd_compiler_workarounds.lds_only_fence_options(sched.lds_only_fences),
    }
    expected_tiles = _schedule.expected_bucket_tiles(
        num_pairs, cfg.num_slots, cfg.active_slots_hint, batched=batched
    )
    seed_rows = expected_tiles * _schedule.INT8_GEMM_BLOCK_M
    kernel_fn: _JitKernel = (
        _int8_triton.int8_moe_batched_matmul_kernel
        if batched
        else _int8_triton.int8_moe_m1_matmul_kernel
    )
    base_ptrs: dict[str, str | None] = {
        "input_ptr": "*i8",
        "weight_ptr": "*i8",
        **({} if batched else {"expert_to_slot_ptr": _opt_ptr(cfg.has_slot_map, "*i64")}),
    }
    base_const: dict[str, object] = {
        "block_m": _schedule.BATCH_BLOCK_M if batched else _schedule.INT8_GEMM_BLOCK_M,
        "block_n": _schedule.INT8_GEMM_BLOCK_N,
        "block_k": block_k,
        **({} if batched else {"has_slot_map": cfg.has_slot_map}),
    }
    compiled: dict[str, triton.compiler.CompiledKernel] = {}
    for key, per_expert, is_splitk, out_ty, out_f, in_f in (
        (
            "int8_matmul_up",
            False,
            cfg.is_splitk,
            "*i32" if cfg.is_splitk else cfg.out_ty,
            cfg.up_gate_out,
            cfg.hidden_size,
        ),
        ("int8_matmul_down", True, True, "*i32", cfg.hidden_size, cfg.intermediate_size),
    ):
        seed = splitk_accumulator_seed(
            cfg.accel,
            rows=seed_rows,
            out_features=out_f,
            in_features=in_f,
            block_m=_schedule.INT8_GEMM_BLOCK_M,
            block_n=_schedule.INT8_GEMM_BLOCK_N,
            block_k=block_k,
            paired=False,
            split_k=cfg.split_k,
        )
        compiled[key] = _compile_kernel(
            cfg,
            kernel_fn,
            constexprs={
                **base_const,
                "per_expert_input": per_expert,
                "is_splitk": is_splitk,
                "dequant_output": not is_splitk,
                "aligned_nk": out_f % _schedule.INT8_GEMM_BLOCK_N == 0 and in_f % block_k == 0,
                "accumulator_seed": seed,
            },
            overrides={**base_ptrs, "output_ptr": out_ty},
            options=opts,
        )
    compiled["int8_activate_quant"] = _compile_int8_activate_quant(cfg, use_compact_map=False)
    if cfg.is_splitk:
        compiled["int8_reduce"] = _compile_reduce(cfg, weighted=False, is_int8=True)
    compiled["weighted_reduce"] = _compile_reduce(cfg, weighted=True, is_int8=True)
    return compiled


def _compile_a16_epilogue(
    compiled: dict[str, triton.compiler.CompiledKernel],
    cfg: _CompileConfig,
    *,
    reduce_key: str,
    active_ty: str,
    has_global_scale: bool = False,
    post_scale_only: bool = True,
    has_row_scale: bool = False,
) -> None:
    if cfg.is_splitk:
        compiled[reduce_key] = _compile_reduce(
            cfg,
            weighted=False,
            is_int8=False,
            has_global_scale=has_global_scale,
            post_scale_only=post_scale_only,
            has_row_scale=has_row_scale,
        )
    compiled["activate_rotate"] = _compile_activate_rotate(
        cfg, active_ty=active_ty, use_compact_map=False
    )
    compiled["weighted_reduce"] = _compile_reduce(
        cfg,
        weighted=True,
        is_int8=False,
        has_global_scale=has_global_scale,
        post_scale_only=post_scale_only,
        has_row_scale=has_row_scale,
    )


def _compile_nvfp4_decode(
    cfg: _CompileConfig, *, batched: bool, rows: int
) -> dict[str, triton.compiler.CompiledKernel]:
    if batched:
        sched = _nvidia_policy.select_nvfp4_batched_schedule(
            cfg.accel, rows=rows, active_dtype=cfg.active_dtype, logical_dtype=cfg.logical_dtype
        )
        post_scale_only = sched.post_scale_only
        kernel_fn: _JitKernel = _nvfp4_triton.nvfp4_moe_batched_gemv_kernel
        mode_ptrs: dict[str, str | None] = {}
        mode_const: dict[str, object] = {
            "block_m": _schedule.BATCH_BLOCK_M,
            "block_n": sched.block_n,
            "block_k": sched.block_k,
            "use_dot": sched.use_dot,
            "cast_fp32_to_bf16": sched.cast_fp32_to_bf16,
            "single_dot": sched.single_dot,
        }
        gemv_opts: dict[str, object] = {"num_warps": sched.num_warps, "num_stages": 1}
    else:
        m1 = _nvidia_policy.select_nvfp4_m1_schedule(cfg.accel)
        post_scale_only = True
        kernel_fn = _nvfp4_triton.nvfp4_moe_m1_gemv_kernel
        mode_ptrs = {"expert_to_slot_ptr": _opt_ptr(cfg.has_slot_map, "*i64")}
        mode_const = {
            "block_n": m1.block_n,
            "block_k": m1.block_k,
            "has_slot_map": cfg.has_slot_map,
        }
        gemv_opts = {"num_warps": m1.num_warps}

    base_ptrs: dict[str, str | None] = {
        "input_ptr": cfg.active_ty,
        "weight_ptr": "*u8",
        "weight_scale_ptr": "*u8",
        "weight_global_scale_ptr": _opt_ptr(cfg.has_global_scale, "*fp32"),
        "output_ptr": "*fp32",
        **mode_ptrs,
    }
    base_const: dict[str, object] = {
        **mode_const,
        "high_first": cfg.high_first,
        "has_global_scale": cfg.has_global_scale,
    }
    compiled: dict[str, triton.compiler.CompiledKernel] = {}
    for key, per_expert, is_splitk, out_f, in_f in (
        ("nvfp4_gemv_up", False, cfg.is_splitk, cfg.up_gate_out, cfg.hidden_size),
        ("nvfp4_gemv_down", True, True, cfg.hidden_size, cfg.intermediate_size),
    ):
        compiled[key] = _compile_kernel(
            cfg,
            kernel_fn,
            constexprs={**base_const, "per_expert_input": per_expert, "is_splitk": is_splitk},
            overrides=base_ptrs,
            options=gemv_opts,
            div16_ints=_nvfp4_div16_ints(out_f, in_f),
            align_ptrs=True,
        )
    _compile_a16_epilogue(
        compiled,
        cfg,
        reduce_key="nvfp4_reduce",
        active_ty=cfg.active_ty,
        has_global_scale=cfg.has_global_scale,
        post_scale_only=post_scale_only,
    )
    return compiled


def _trellis_stream_meta(cfg: _CompileConfig) -> tuple[int, bool, bool, int, int]:
    tile_steps = triton.next_power_of_2(cfg.block_size)
    per_slot_2d = cfg.variable_rows and not cfg.has_slot_offsets
    if cfg.variable_rows:
        return tile_steps, per_slot_2d, True, 1, 1
    layout = default_layout_for_block_size(cfg.block_size, cfg.bytes_per_block)
    needs_w2 = layout.block_bits > 5 * layout.block_size
    up_slot = cfg.up_gate_out * (cfg.hidden_size // cfg.block_size) * layout.bytes_per_block
    dn_slot = cfg.hidden_size * (cfg.intermediate_size // cfg.block_size) * layout.bytes_per_block
    return tile_steps, per_slot_2d, needs_w2, up_slot, dn_slot


def _compile_trellis_prefill(
    cfg: _CompileConfig, chunk_rows: int
) -> dict[str, triton.compiler.CompiledKernel]:
    compiled = _compile_int8_prefill(cfg, chunk_rows, slot_window=True)
    tile_steps, per_slot_2d, needs_w2, up_slot, dn_slot = _trellis_stream_meta(cfg)
    stream_bytes = up_slot if (cfg.variable_rows or dn_slot % 4 == 0) else dn_slot
    use_dp4a, use_sad_u8, has_tail_bytes, clamped_load, num_warps = _decode_stream_flags(
        cfg.accel,
        stream_bytes,
        variable_rows=cfg.variable_rows,
        ptr_aligned=True,
        per_slot_2d=per_slot_2d,
    )
    compiled["trellis_decode"] = _compile_kernel(
        cfg,
        decode_slots_int8_kernel,
        constexprs={
            "tile_steps": tile_steps,
            "tile_blocks": 4,
            "variable_rows": cfg.variable_rows,
            "exact_pow2": cfg.block_size == tile_steps,
            "filter_active": True,
            "has_slot_offsets": cfg.has_slot_offsets,
            "per_slot_2d": per_slot_2d,
            "needs_w2": needs_w2,
            "use_dp4a": use_dp4a,
            "use_sad_u8": use_sad_u8,
            "has_tail_bytes": has_tail_bytes,
            "ptr_aligned": True,
            "clamped_load": clamped_load,
            "q_u32_aligned": (cfg.block_size % 4 == 0),
        },
        overrides={
            "qdata": "*i8",
            "row_widths": _opt_ptr(cfg.variable_rows, "*u8"),
            "row_offsets": _opt_ptr(cfg.variable_rows, "*i32"),
            "slot_offsets_ptr": _opt_ptr(cfg.has_slot_offsets, "*i64"),
        },
        options={"num_warps": num_warps},
    )
    return compiled


def _compile_trellis_decode(
    cfg: _CompileConfig, *, batched: bool, rows: int
) -> dict[str, triton.compiler.CompiledKernel]:
    tile_steps, per_slot_2d, needs_w2, up_slot, dn_slot = _trellis_stream_meta(cfg)
    expected_slots = (
        min(cfg.num_slots, cfg.active_slots_hint) if cfg.active_slots_hint > 0 else cfg.num_slots
    )
    sched = _policy.select_trellis_schedule(
        cfg.accel, batched=batched, num_pairs=rows * cfg.top_k, expected_slots=expected_slots
    )
    up_tail, dn_tail = (
        (True, True)
        if cfg.variable_rows
        else (
            _decode_stream_flags(
                cfg.accel, cfg.num_slots * sb, variable_rows=False, per_slot_2d=per_slot_2d
            )[2]
            for sb in (up_slot, dn_slot)
        )
    )
    gemv_ptrs: dict[str, str | None] = {
        "row_widths_ptr": _opt_ptr(cfg.variable_rows, "*u8"),
        "row_offsets_ptr": _opt_ptr(cfg.variable_rows, "*i32"),
        "slot_offsets_ptr": _opt_ptr(cfg.has_slot_offsets, "*i64"),
        "topk_ids_ptr": _opt_ptr(not batched, "*i64"),
        "expert_to_slot_ptr": _opt_ptr(not batched and cfg.has_slot_map, "*i64"),
        "tile_slots_ptr": _opt_ptr(batched, "*i32"),
        "tile_pair_ids_ptr": _opt_ptr(batched, "*i32"),
        "output_ptr": "*fp32",
    }
    gemv_const: dict[str, object] = {
        "tile_steps": tile_steps,
        "block_m": sched.block_m,
        "block_n": sched.block_n,
        "batched": batched,
        "has_slot_map": False if batched else cfg.has_slot_map,
        "variable_rows": cfg.variable_rows,
        "has_slot_offsets": cfg.has_slot_offsets,
        "exact_pow2": cfg.block_size == tile_steps,
        "needs_w2": needs_w2,
        "use_dp4a": sched.use_dp4a,
        "use_sad_u8": sched.use_sad_u8,
        "split_x_loads": sched.split_x_loads,
        "ptr_aligned": True,
        "x_mul4": (cfg.block_size % 4 == 0),
        "unroll_k": sched.unroll_k,
    }
    compiled: dict[str, triton.compiler.CompiledKernel] = {}
    for key, per_expert, is_splitk, has_tail in (
        ("trellis_gemv_up", False, cfg.is_splitk, up_tail),
        ("trellis_gemv_down", True, True, dn_tail),
    ):
        compiled[key] = _compile_kernel(
            cfg,
            _trellis_triton.trellis_moe_gemv_kernel,
            constexprs={
                **gemv_const,
                "is_splitk": is_splitk,
                "has_tail_bytes": has_tail,
                "per_expert_input": per_expert,
            },
            overrides=gemv_ptrs,
            options={"num_warps": sched.num_warps},
        )
    _compile_a16_epilogue(
        compiled, cfg, reduce_key="trellis_reduce", active_ty="*fp32", has_row_scale=True
    )
    return compiled


def _dispatch_compile(cfg: _CompileConfig) -> dict[str, triton.compiler.CompiledKernel]:
    is_int8 = cfg.weight_format == "convrot_int8"
    is_trellis = cfg.weight_format == "convrot_trellis"
    is_nvfp4_w4a4 = cfg.weight_format in ("nvfp4", "convrot_nvfp4")

    has_int8_backend = is_int8
    if is_trellis:
        from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

        has_int8_backend = (
            _policy.supports_trellis_int8_prefill(cfg.accel)
            and int8_backend_mod.select_linear_backend_for_target(cfg.accel) is not None
        )

    bytes_per_pair = _schedule.prefill_bytes_per_pair(
        is_int8=has_int8_backend,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        up_gate_out=cfg.up_gate_out,
        logical_elem_bytes=cfg.logical_dtype.itemsize,
        group_size=cfg.group_size,
        is_nvfp4_w4a4=is_nvfp4_w4a4,
    )
    regime, chunk_rows = _schedule.select_moe_execution_regime(cfg.rows, cfg.top_k, bytes_per_pair)

    if regime == "prefill":
        if is_trellis and has_int8_backend:
            return _compile_trellis_prefill(cfg, chunk_rows)
        if has_int8_backend:
            return _compile_int8_prefill(cfg, chunk_rows)
        if is_nvfp4_w4a4:
            return _compile_nvfp4_w4a4_prefill(cfg, chunk_rows)
        if not is_trellis:
            return _compile_nvfp4_prefill(cfg, chunk_rows)
        a16_rows = min(
            cfg.rows,
            _schedule.BATCH_MAX_ROWS,
            max(1, _schedule.BUCKET_MAX_PAIRS // max(1, cfg.top_k)),
        )
        batched = a16_rows > 1
    else:
        a16_rows = chunk_rows
        batched = regime == "batched"

    num_pairs = a16_rows * cfg.top_k
    if is_trellis:
        compiled = _compile_trellis_decode(cfg, batched=batched, rows=a16_rows)
    elif is_int8:
        compiled = _compile_int8_decode(cfg, batched=batched, num_pairs=num_pairs)
    else:
        compiled = _compile_nvfp4_decode(cfg, batched=batched, rows=chunk_rows)
    if batched:
        compiled["bucket"] = _compile_kernel(
            cfg,
            _routing_triton.moe_bucket_tokens_kernel,
            constexprs={
                "block_p": triton.next_power_of_2(max(16, num_pairs)),
                "block_m": _schedule.BATCH_BLOCK_M,
                "has_slot_map": cfg.has_slot_map,
            },
            overrides={"expert_to_slot_ptr": _opt_ptr(cfg.has_slot_map, "*i64")},
        )
    return compiled


def compile_moe_kernels(  # noqa: PLR0913 - public offline compile contract
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
        min(hidden_size, intermediate_size, split_k, rows, top_k, num_slots) <= 0
        or active_slots_hint < 0
    ):
        raise ValueError("MoE dimensions, rows, top_k, num_slots, and split_k must be positive")
    resolved_gate_bound, resolved_up_bound = validate_moe_activation_bounds(
        activation_fn, gate_bound, up_bound
    )
    accel = AcceleratorTarget.from_compiler_target(target)
    _ops.validate_named_weight_format(
        weight_format,
        group_size=group_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        high_first=high_first,
        accel=accel,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
        bytes_per_block=bytes_per_block,
    )
    activation = _ActivationConfig(
        activation_fn=activation_fn,
        gate_first=gate_first,
        gate_bound=resolved_gate_bound,
        up_bound=resolved_up_bound,
    )
    return _dispatch_compile(
        _CompileConfig(
            target=target,
            accel=accel,
            weight_format=weight_format,
            logical_dtype=logical_dtype,
            out_ty=TRITON_FLOAT_POINTER_TYPES[logical_dtype],
            rows=rows,
            top_k=top_k,
            num_slots=num_slots,
            active_slots_hint=active_slots_hint,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            up_gate_out=intermediate_size * (2 if activation.is_gated else 1),
            group_size=group_size,
            activation=activation,
            high_first=high_first,
            has_slot_map=has_slot_map,
            has_global_scale=has_global_scale,
            split_k=split_k,
            accumulate=accumulate,
            block_size=block_size,
            variable_rows=variable_rows,
            has_slot_offsets=has_slot_offsets,
            bytes_per_block=bytes_per_block,
            compile_fn=compiler or default_triton_compile,
        )
    )
