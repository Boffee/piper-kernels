"""Offline Triton/Gluon compile harnesses for Piper KV freeze, decode, prefill, and LSE merge."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch
import triton

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from . import _backend, _quantization
from ._decode_schedule import BLOCK_QH, REDUCE_NUM_WARPS
from ._lse_merge import (
    MAX_MERGE_WAVES,
    MERGE_STRIDE_FIELDS,
    _merge_two_attention_waves_kernel,
)
from ._nvidia import policy as _nvidia_policy
from ._validation import WindowMode, resolve_window_mode
from .triton import (
    _piper_decode_frozen_split_kernel,
    _piper_decode_splitkv_reduce_kernel,
    _piper_decode_tail_split_kernel,
    select_decode_schedule,
)

POOL_SLOT_STRIDE_NAMES: tuple[str, ...] = (
    "stride_ki_s",
    "stride_ks_s",
    "stride_kc_s",
    "stride_vi_s",
    "stride_vs_s",
    "stride_vm_s",
)

_KEY_DIV16_BASE = frozenset(
    {
        "output_ptr",
        "scale_ptr",
        "center_ptr",
        "stride_ob",
        "stride_oh",
        "stride_on",
        "stride_ch",
    }
)
_VALUE_DIV16_BASE = frozenset(
    {
        "output_ptr",
        "scale_multiplier_ptr",
        "log_scale_ptr",
        "value_mean_ptr",
        "stride_ob",
        "stride_oh",
        "stride_mh",
        "stride_ch",
    }
)


def _div16_attrs(
    arg_names: list[str], div16_names: frozenset[str]
) -> dict[tuple[int, ...], list[list[object]]]:
    return {
        (idx,): [["tt.divisibility", 16]]
        for idx, name in enumerate(arg_names)
        if name in div16_names
    }


def compile_pointer_attrs(
    arg_names: list[str],
    signature: dict[str, str],
    *,
    div16_names: frozenset[str] = frozenset(),
    all_strides_div16: bool = False,
    pointer_range_32: bool = True,
) -> dict[tuple[int, ...], list[list[object]]]:
    """Attach live pointer divisibility/range attrs and selected stride attrs."""
    ptr_attrs: list[list[object]] = [["tt.divisibility", 16]]
    if pointer_range_32:
        ptr_attrs.append(["tt.pointer_range", 32])
    attrs: dict[tuple[int, ...], list[list[object]]] = {}
    for idx, name in enumerate(arg_names):
        ty = signature.get(name)
        if isinstance(ty, str) and ty.startswith("*"):
            attrs[(idx,)] = list(ptr_attrs)
        elif ty is not None and (
            name in div16_names or (all_strides_div16 and name.startswith("stride_"))
        ):
            attrs[(idx,)] = [["tt.divisibility", 16]]
    return attrs


def validate_kv_compile_args(dtype: torch.dtype, head_dim: int) -> str:
    """Validate offline KV compile dtype/head_dim and return the Triton pointer dtype stem."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"dtype must be float16 or bfloat16, got {dtype}")
    if head_dim not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")
    return "bf16" if dtype is torch.bfloat16 else "fp16"


class _JitKernel(Protocol):
    arg_names: list[str]


def build_kernel_source(
    kernel: _JitKernel,
    *,
    constants: dict[str, object],
    overrides: dict[str, str],
    attrs: Mapping[tuple[int, ...], object] | None = None,
    gluon: bool = False,
    div16_names: frozenset[str] = frozenset(),
    all_strides_div16: bool = False,
    pointer_range_32: bool = True,
) -> object:
    """Construct a Triton ASTSource or GluonASTSource with i32 defaults and pointer attrs."""
    arg_names: list[str] = list(kernel.arg_names)
    sig: dict[str, str] = {name: "i32" for name in arg_names if name not in constants}
    sig.update(overrides)
    resolved_attrs: Any = (
        compile_pointer_attrs(
            arg_names,
            sig,
            div16_names=div16_names,
            all_strides_div16=all_strides_div16,
            pointer_range_32=pointer_range_32,
        )
        if attrs is None
        else attrs
    )
    jit_fn: Any = kernel
    if gluon:
        from triton.experimental.gluon._runtime import GluonASTSource  # noqa: PLC0415

        return GluonASTSource(jit_fn, sig, constexprs=constants, attrs=resolved_attrs)
    from triton.compiler import ASTSource  # noqa: PLC0415

    return ASTSource(jit_fn, sig, constexprs=constants, attrs=resolved_attrs)


def _frozen_pool_prefill_overrides(
    triton_dtype: str,
    constants: dict[str, object],
    *,
    wide_pool: bool,
    has_chunk_splits: bool,
    has_tail: bool,
) -> dict[str, str]:
    """Build shared pointer and scalar overrides for NVIDIA and AMD frozen-pool prefill kernels."""
    overrides: dict[str, str] = {
        "query_ptr": "*i8",
        "query_scale_ptr": "*fp32",
        "key_int8_ptr": "*i8",
        "key_scale_ptr": "*fp32",
        "key_center_ptr": "*fp32",
        "value_int8_ptr": "*i8",
        "value_scale_ptr": "*fp32",
        "value_mean_ptr": "*fp32",
        "chunk_to_slot_ptr": "*i32",
        "num_chunks_ptr": "*i32",
        "output_ptr": "*fp32" if has_chunk_splits else f"*{triton_dtype}",
        "lse_ptr": "*fp32",
        "partial_bias_ptr": "*fp32",
        "partial_log_denom_ptr": "*fp32",
        "window_left": "i64",
        "window_right": "i64",
        "logical_seq_q": "i32",
    }
    if wide_pool:
        overrides.update(dict.fromkeys(POOL_SLOT_STRIDE_NAMES, "i64"))
    if has_tail:
        overrides["tail_len_ptr"] = "*i32"
    else:
        constants["tail_len_ptr"] = None
        overrides["tail_len_ptr"] = "constexpr"
    return overrides


def _compile_quantize_pair(
    target: object,
    *,
    dtype: torch.dtype,
    head_dim: int,
    scatter_slots: bool,
    has_key_lengths: bool,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object]:
    triton_dtype = validate_kv_compile_args(dtype, head_dim)
    accel = AcceleratorTarget.from_compiler_target(target)
    _, grouped_qk, pack_wmma = _backend.kv_chunk_layout(accel)

    key_kernel = _quantization._quantize_key_k1024_kernel
    value_kernel = _quantization._quantize_value_k1024_kernel

    key_source = build_kernel_source(
        key_kernel,
        constants={
            "grouped_qk": grouped_qk,
            "scatter_slots": scatter_slots,
            "head_dim": head_dim,
            "has_key_lengths": has_key_lengths,
        },
        overrides={
            "key_ptr": f"*{triton_dtype}",
            "output_ptr": "*i8",
            "scale_ptr": "*fp32",
            "center_ptr": "*fp32",
            "slot_ids_ptr": "*i32",
        },
        attrs=(
            _div16_attrs(key_kernel.arg_names, _KEY_DIV16_BASE | {"stride_sh"})
            if scatter_slots
            else None
        ),
        div16_names=_KEY_DIV16_BASE | {"stride_cb", "stride_kb", "stride_kh", "stride_kn"},
    )

    store_log_scale = pack_wmma and not scatter_slots
    scale_by_p_range = pack_wmma and not scatter_slots
    use_tile_layout = True
    value_source = build_kernel_source(
        value_kernel,
        constants={
            "stride_ok": 1,
            "store_log_scale": store_log_scale,
            "scale_by_p_range": scale_by_p_range,
            "pack_wmma": pack_wmma,
            "tile_layout": use_tile_layout,
            "scatter_slots": scatter_slots,
            "head_dim": head_dim,
            "has_key_lengths": has_key_lengths,
        },
        overrides={
            "value_ptr": f"*{triton_dtype}",
            "output_ptr": "*i8",
            "scale_multiplier_ptr": "*fp32",
            "log_scale_ptr": "*fp32",
            "value_mean_ptr": "*fp32",
            "slot_ids_ptr": "*i32",
        },
        attrs=(_div16_attrs(value_kernel.arg_names, _VALUE_DIV16_BASE) if scatter_slots else None),
        div16_names=_VALUE_DIV16_BASE
        | {"stride_mb", "stride_cb", "stride_vb", "stride_vh", "stride_vn"},
    )

    value_options: dict[str, object] = {
        "num_warps": _quantization._value_quantize_num_warps(
            head_dim,
            tile_layout=use_tile_layout,
            pack_wmma=pack_wmma,
            has_key_lengths=has_key_lengths,
        )
    }
    compiler = triton.compile if compile_fn is None else compile_fn
    return (
        compiler(key_source, target, {"num_warps": 4}),
        compiler(value_source, target, value_options),
    )


def compile_kv_freeze_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object]:
    """Offline-compile K1024 and V1024 slot-scatter freeze kernels for ``target``."""
    return _compile_quantize_pair(
        target,
        dtype=dtype,
        head_dim=head_dim,
        scatter_slots=True,
        has_key_lengths=False,
        compile_fn=compile_fn,
    )


def compile_kv_tail_quantize_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object]:
    """Offline-compile K1024 and V1024 ragged tail quantization kernels for ``target``."""
    return _compile_quantize_pair(
        target,
        dtype=dtype,
        head_dim=head_dim,
        scatter_slots=False,
        has_key_lengths=True,
        compile_fn=compile_fn,
    )


def compile_nvidia_prefill_kernels(  # noqa: PLR0913
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    wide_pool: bool = False,
    has_chunk_splits: bool = False,
    per_row_query_scale: bool = False,
    window_mode: WindowMode | None = None,
    has_tail: bool = False,
    include_tail: bool = True,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, ...]:
    """Offline-compile unified NVIDIA frozen-pool and offset-window tail prefill kernels."""
    from ._nvidia import prefill as nvidia_prefill  # noqa: PLC0415

    accel = AcceleratorTarget.from_compiler_target(target)
    triton_dtype = validate_kv_compile_args(dtype, head_dim)
    grouped_qk = _nvidia_policy.groups_qk_scales(accel)
    compiler = triton.compile if compile_fn is None else compile_fn
    frozen_mode: WindowMode = "none" if window_mode is None else window_mode
    tail_mode = resolve_window_mode(is_causal, window_mode)

    def _compile_variant(
        *,
        tail_mode_flag: bool,
        variant_window_mode: WindowMode,
        chunk_map: bool,
        chunk_splits: bool,
        per_row: bool,
        with_tail: bool,
    ) -> object:
        block_m, num_warps, split_pv, num_stages, loop_num_stages = (
            _nvidia_policy.select_prefill_config(
                accel,
                head_dim=head_dim,
                is_tail=tail_mode_flag,
                has_chunk_map=chunk_map,
                has_chunk_splits=chunk_splits,
                window_mode=variant_window_mode,
            )
        )
        constants: dict[str, object] = {
            "query_heads_per_kv_head": 3,
            "head_dim": head_dim,
            "grouped_qk": grouped_qk,
            "split_pv_head_dim": split_pv,
            "has_chunk_map": chunk_map,
            "is_tail": tail_mode_flag,
            "window_mode": variant_window_mode,
            "block_m": block_m,
            "loop_num_stages": loop_num_stages,
            "has_chunk_splits": chunk_splits,
            "per_row_query_scale": per_row,
            "has_tail": with_tail,
        }
        wide = wide_pool and not tail_mode_flag
        overrides = _frozen_pool_prefill_overrides(
            triton_dtype,
            constants,
            wide_pool=wide,
            has_chunk_splits=chunk_splits,
            has_tail=with_tail,
        )
        source = build_kernel_source(
            nvidia_prefill._piper_chunked_prefill_kernel,
            constants=constants,
            overrides=overrides,
            pointer_range_32=not wide,
        )
        return compiler(source, target, {"num_warps": num_warps, "num_stages": num_stages})

    frozen_compiled = _compile_variant(
        tail_mode_flag=False,
        variant_window_mode=frozen_mode,
        chunk_map=has_chunk_map,
        chunk_splits=has_chunk_splits,
        per_row=per_row_query_scale,
        with_tail=has_tail,
    )
    if not include_tail:
        return (frozen_compiled,)
    tail_compiled = _compile_variant(
        tail_mode_flag=True,
        variant_window_mode=tail_mode,
        chunk_map=False,
        chunk_splits=False,
        per_row=False,
        with_tail=False,
    )
    return (frozen_compiled, tail_compiled)


def compile_amd_prefill_kernels(  # noqa: PLR0913
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    wide_pool: bool = False,
    has_chunk_splits: bool = False,
    per_row_query_scale: bool = False,
    window_mode: WindowMode | None = None,
    has_tail: bool = False,
    include_tail: bool = True,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, ...]:
    """Offline-compile RDNA4 frozen-pool and optional offset-window tail prefill kernels."""
    from ._amd import gluon as amd_gluon  # noqa: PLC0415

    triton_dtype = validate_kv_compile_args(dtype, head_dim)
    options: dict[str, object] = {
        "num_warps": 4,
        "num_stages": 1,
        "llvm_fn_attrs": amd_gluon._LLVM_FN_ATTRS,
    }
    compiler = triton.compile if compile_fn is None else compile_fn
    frozen_mode: WindowMode = "none" if window_mode is None else window_mode
    tail_mode = resolve_window_mode(is_causal, window_mode)

    frozen_constants: dict[str, object] = {
        "query_heads_per_kv_head": 3,
        "head_dim": head_dim,
        "has_chunk_map": has_chunk_map,
        "wide_query_offsets": False,
        "has_chunk_splits": has_chunk_splits,
        "per_row_query_scale": per_row_query_scale,
        "has_tail": has_tail,
        "window_mode": frozen_mode,
    }
    frozen_overrides = _frozen_pool_prefill_overrides(
        triton_dtype,
        frozen_constants,
        wide_pool=wide_pool,
        has_chunk_splits=has_chunk_splits,
        has_tail=has_tail,
    )
    frozen_source = build_kernel_source(
        amd_gluon._frozen_pool_piper_kernel,
        constants=frozen_constants,
        overrides=frozen_overrides,
        gluon=True,
        pointer_range_32=not wide_pool,
    )

    frozen_compiled = compiler(frozen_source, target, options)
    if not include_tail:
        return (frozen_compiled,)

    tail_overrides: dict[str, str] = dict.fromkeys(
        (
            "query_scale_ptr",
            "key_scale_ptr",
            "key_center_ptr",
            "multiplier_ptr",
            "log_scale_ptr",
            "value_mean_ptr",
            "lse_ptr",
        ),
        "*fp32",
    )
    tail_overrides.update(
        query_ptr="*i8",
        key_ptr="*i8",
        value_ptr="*i8",
        key_lengths_ptr="*i32",
        output_ptr=f"*{triton_dtype}",
        window_left="i64",
        window_right="i64",
    )
    tail_source = build_kernel_source(
        amd_gluon._dense_piper_kernel,
        constants={
            "query_heads_per_kv_head": 3,
            "head_dim": head_dim,
            "window_mode": tail_mode,
            "wide_query_offsets": False,
            "wide_context_offsets": False,
            "store_lse": True,
            "has_key_lengths": True,
            "offset_causal_from_key_length": tail_mode != "none",
        },
        overrides=tail_overrides,
        gluon=True,
    )
    return (frozen_compiled, compiler(tail_source, target, options))


def compile_kv_decode_kernels(  # noqa: PLR0913
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    contiguous_inputs: bool = True,
    wide_pool: bool = False,
    reduce_vectorize_splits: bool = False,
    window_mode: WindowMode | None = None,
    has_tail: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object, object]:
    """Compile frozen-split, tail-split, and reduce decode kernels offline."""
    triton_dtype = validate_kv_compile_args(dtype, head_dim)
    accel = AcceleratorTarget.from_compiler_target(target)
    backend = _backend.prefill_backend(accel)
    schedule = select_decode_schedule(accel, head_dim)
    frozen_mode: WindowMode = "none" if window_mode is None else window_mode
    tail_mode = resolve_window_mode(is_causal, window_mode)

    compiler = triton.compile if compile_fn is None else compile_fn
    split_options: dict[str, object] = {
        "num_warps": schedule.num_warps,
        "num_stages": schedule.num_stages,
        **dict(schedule.compiler_options),
    }
    if backend == "amd":
        frozen_compiled = compile_amd_prefill_kernels(
            target,
            dtype=dtype,
            head_dim=head_dim,
            has_chunk_map=has_chunk_map,
            is_causal=False,
            wide_pool=wide_pool,
            has_chunk_splits=True,
            per_row_query_scale=True,
            window_mode=frozen_mode,
            has_tail=has_tail,
            include_tail=False,
            compile_fn=compile_fn,
        )[0]
    elif backend == "nvidia":
        frozen_compiled = compile_nvidia_prefill_kernels(
            target,
            dtype=dtype,
            head_dim=head_dim,
            has_chunk_map=has_chunk_map,
            is_causal=False,
            wide_pool=wide_pool,
            has_chunk_splits=True,
            per_row_query_scale=True,
            window_mode=frozen_mode,
            has_tail=has_tail,
            include_tail=False,
            compile_fn=compile_fn,
        )[0]
    else:
        _, grouped_qk, pack_wmma = _backend.kv_chunk_layout(accel)
        frozen_overrides: dict[str, str] = {
            "query_ptr": f"*{triton_dtype}",
            "key_int8_ptr": "*i8",
            "key_scale_ptr": "*fp32",
            "key_center_ptr": "*fp32",
            "value_int8_ptr": "*i8",
            "value_scale_ptr": "*fp32",
            "value_mean_ptr": "*fp32",
            "chunk_to_slot_ptr": "*i32",
            "num_chunks_ptr": "*i32",
            "partial_out_ptr": "*fp32",
            "partial_lse_ptr": "*fp32",
            "partial_bias_ptr": "*fp32",
            "partial_log_denom_ptr": "*fp32",
            "scale": "fp32",
            "tail_cap": "i64",
            "window_left": "i64",
            "window_right": "i64",
        }
        frozen_constants: dict[str, object] = {
            "has_chunk_map": has_chunk_map,
            "grouped_qk": grouped_qk,
            "pack_wmma": pack_wmma,
            "head_dim": head_dim,
            "block_qh": BLOCK_QH,
            "block_n": schedule.block_n,
            "pv_precision": schedule.pv_precision,
            "has_tail": has_tail,
            "window_mode": frozen_mode,
        }
        if has_tail:
            frozen_overrides["tail_len_ptr"] = "*i32"
        else:
            frozen_constants["tail_len_ptr"] = None
            frozen_overrides["tail_len_ptr"] = "constexpr"
        if wide_pool:
            frozen_overrides.update(dict.fromkeys(POOL_SLOT_STRIDE_NAMES, "i64"))
        frozen_noncontig = frozenset(
            {
                "key_int8_ptr",
                "key_scale_ptr",
                "key_center_ptr",
                "value_int8_ptr",
                "value_scale_ptr",
                "value_mean_ptr",
                "partial_out_ptr",
                "stride_ki_s",
                "stride_vi_s",
                "stride_pob",
                "stride_poh",
                "stride_pos",
                "stride_pop",
            }
        )
        frozen_source = build_kernel_source(
            _piper_decode_frozen_split_kernel,
            constants=frozen_constants,
            overrides=frozen_overrides,
            attrs=(
                None
                if contiguous_inputs
                else _div16_attrs(_piper_decode_frozen_split_kernel.arg_names, frozen_noncontig)
            ),
            all_strides_div16=True,
            pointer_range_32=not wide_pool,
        )
        frozen_compiled = compiler(frozen_source, target, split_options)

    tail_noncontig = frozenset(
        {"partial_out_ptr", "stride_pob", "stride_poh", "stride_pos", "stride_pop"}
    )
    tail_source = build_kernel_source(
        _piper_decode_tail_split_kernel,
        constants={
            "window_mode": tail_mode,
            "head_dim": head_dim,
            "block_qh": BLOCK_QH,
            "block_n": schedule.block_n,
            "qk_precision": schedule.tail_qk_precision,
            "pv_precision": schedule.pv_precision,
        },
        overrides={
            "query_ptr": f"*{triton_dtype}",
            "tail_key_ptr": f"*{triton_dtype}",
            "tail_value_ptr": f"*{triton_dtype}",
            "tail_len_ptr": "*i32",
            "partial_out_ptr": "*fp32",
            "partial_lse_ptr": "*fp32",
            "partial_bias_ptr": "*fp32",
            "partial_log_denom_ptr": "*fp32",
            "scale": "fp32",
            "window_left": "i64",
            "window_right": "i64",
        },
        attrs=(
            None
            if contiguous_inputs
            else _div16_attrs(_piper_decode_tail_split_kernel.arg_names, tail_noncontig)
        ),
        all_strides_div16=True,
    )

    reduce_noncontig = frozenset(
        {
            "partial_out_ptr",
            "output_ptr",
            "stride_pob",
            "stride_poh",
            "stride_pos",
            "stride_pop",
            "stride_ob",
            "stride_oh",
            "stride_os",
        }
    )
    reduce_source = build_kernel_source(
        _piper_decode_splitkv_reduce_kernel,
        constants={
            "head_dim": head_dim,
            "block_r": 1 if reduce_vectorize_splits else 8,
            "block_s": 16 if reduce_vectorize_splits else 1,
        },
        overrides={
            "partial_out_ptr": "*fp32",
            "partial_lse_ptr": "*fp32",
            "partial_bias_ptr": "*fp32",
            "partial_log_denom_ptr": "*fp32",
            "output_ptr": f"*{triton_dtype}",
            "lse_ptr": "*fp32",
        },
        attrs=(
            None
            if contiguous_inputs
            else _div16_attrs(_piper_decode_splitkv_reduce_kernel.arg_names, reduce_noncontig)
        ),
        all_strides_div16=True,
    )
    reduce_options: dict[str, object] = {
        "num_warps": REDUCE_NUM_WARPS,
        **amd_compiler_workarounds.lds_only_fence_options(
            reduce_vectorize_splits and accel.is_hip_architecture(*GFX120X_ARCHITECTURES)
        ),
    }

    return (
        frozen_compiled,
        compiler(tail_source, target, split_options),
        compiler(reduce_source, target, reduce_options),
    )


def compile_merge_attention_lse_kernel(  # noqa: PLR0912
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    num_waves: int = 2,
    is_contiguous: bool = True,
    in_place_0: bool = True,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> object:
    """Offline-compile the unified 1..8-wave LSE merge kernel for ``target``."""
    if not 1 <= num_waves <= MAX_MERGE_WAVES:
        raise ValueError(f"num_waves must be in 1..{MAX_MERGE_WAVES}, got {num_waves}")
    if head_dim < 1:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    triton_dtype = (
        "bf16" if dtype is torch.bfloat16 else ("fp16" if dtype is torch.float16 else "fp32")
    )
    compiler = triton.compile if compile_fn is None else compile_fn
    block_d = min(256, max(16, triton.next_power_of_2(head_dim)))
    exact_d = head_dim == block_d
    constants: dict[str, object] = {
        "block_d": block_d,
        "exact_d": exact_d,
        "num_waves": num_waves,
        "is_contiguous": is_contiguous,
        "in_place_0": in_place_0,
    }
    overrides: dict[str, str] = {
        "out_dst_ptr": f"*{triton_dtype}",
        "lse_dst_ptr": "*fp32",
        "dim1": "i64",
        "dim2": "i64",
        "dim3": "i64",
        "head_dim": "i64",
    }
    for wave_idx in range(MAX_MERGE_WAVES):
        if wave_idx < num_waves:
            overrides[f"out_{wave_idx}_ptr"] = f"*{triton_dtype}"
            overrides[f"lse_{wave_idx}_ptr"] = "*fp32"
        else:
            constants[f"out_{wave_idx}_ptr"] = None
            constants[f"lse_{wave_idx}_ptr"] = None
            overrides[f"out_{wave_idx}_ptr"] = "constexpr"
            overrides[f"lse_{wave_idx}_ptr"] = "constexpr"
    if is_contiguous:
        for field in MERGE_STRIDE_FIELDS:
            constants[field] = None
            overrides[field] = "constexpr"
    else:
        for wave_idx in range(MAX_MERGE_WAVES):
            wave_fields = MERGE_STRIDE_FIELDS[wave_idx * 9 : (wave_idx + 1) * 9]
            if wave_idx < num_waves:
                for field in wave_fields:
                    overrides[field] = "i64"
            else:
                for field in wave_fields:
                    constants[field] = None
                    overrides[field] = "constexpr"
    merge_source = build_kernel_source(
        _merge_two_attention_waves_kernel,
        constants=constants,
        overrides=overrides,
    )
    return compiler(merge_source, target, {"num_warps": REDUCE_NUM_WARPS})


def compile_kv_prefill_kernels(  # noqa: PLR0913
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    wide_pool: bool = False,
    has_chunk_splits: bool = False,
    per_row_query_scale: bool = False,
    window_mode: WindowMode | None = None,
    has_tail: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, ...]:
    """Offline-compile two-wave merge and native chunked prefill kernels for ``target``."""
    validate_kv_compile_args(dtype, head_dim)
    merge_compiled = compile_merge_attention_lse_kernel(
        target,
        dtype=dtype,
        head_dim=head_dim,
        num_waves=2,
        is_contiguous=True,
        in_place_0=True,
        compile_fn=compile_fn,
    )

    accel = AcceleratorTarget.from_compiler_target(target)
    backend = _backend.prefill_backend(accel)
    if backend == "amd":
        return (
            merge_compiled,
            *compile_amd_prefill_kernels(
                target,
                dtype=dtype,
                head_dim=head_dim,
                has_chunk_map=has_chunk_map,
                is_causal=is_causal,
                wide_pool=wide_pool,
                has_chunk_splits=has_chunk_splits,
                per_row_query_scale=per_row_query_scale,
                window_mode=window_mode,
                has_tail=has_tail,
                compile_fn=compile_fn,
            ),
        )
    if backend == "nvidia":
        return (
            merge_compiled,
            *compile_nvidia_prefill_kernels(
                target,
                dtype=dtype,
                head_dim=head_dim,
                has_chunk_map=has_chunk_map,
                is_causal=is_causal,
                wide_pool=wide_pool,
                has_chunk_splits=has_chunk_splits,
                per_row_query_scale=per_row_query_scale,
                window_mode=window_mode,
                has_tail=has_tail,
                compile_fn=compile_fn,
            ),
        )
    return (merge_compiled,)
