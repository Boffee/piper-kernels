"""Offline Triton compilation harnesses for Piper KV decode and chunked prefill."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import triton

from piper_kernels._triton import amd_compiler_workarounds
from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from . import _backend, _quantization
from ._decode_schedule import BLOCK_QH, REDUCE_NUM_WARPS
from ._kv_prefill import (
    _MAX_MERGE_WAVES,
    _MERGE_STRIDE_FIELDS,
    _merge_two_attention_waves_kernel,
    _piper_decode_splitkv_reduce_kernel,
    _piper_decode_tail_split_kernel,
    select_decode_schedule,
)
from ._quantization import _div16_attrs
from .triton import _piper_decode_frozen_split_kernel


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
    window_mode: Literal["none", "causal", "windowed"] | None = None,
    has_tail: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object, object]:
    """Compile frozen-split, tail-split, and reduce decode kernels offline."""
    triton_dtype = _quantization.validate_kv_compile_args(dtype, head_dim)
    accel = AcceleratorTarget.from_compiler_target(target)
    backend = _backend.prefill_backend(accel)
    schedule = select_decode_schedule(accel, head_dim)
    frozen_mode: Literal["none", "causal", "windowed"] = (
        "none" if window_mode is None else window_mode
    )
    tail_mode: Literal["none", "causal", "windowed"] = (
        ("causal" if is_causal else "none") if window_mode is None else window_mode
    )

    compiler = triton.compile if compile_fn is None else compile_fn
    split_options: dict[str, object] = {
        "num_warps": schedule.num_warps,
        "num_stages": schedule.num_stages,
        **dict(schedule.compiler_options),
    }
    if backend == "amd":
        from ._amd import gluon as amd_gluon  # noqa: PLC0415

        frozen_compiled = amd_gluon.compile_amd_prefill_kernels(
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
        from ._nvidia import prefill as nvidia_prefill  # noqa: PLC0415

        frozen_compiled = nvidia_prefill.compile_nvidia_prefill_kernels(
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
            "qk_mode": schedule.frozen_qk_mode,
            "pv_precision": schedule.pv_precision,
            # Free a pipeline buffer stage when windowed D=128 spills at loop=2.
            "loop_num_stages": (
                1 if (frozen_mode == "windowed" and head_dim == 128) else schedule.loop_num_stages
            ),
            "has_tail": has_tail,
            "window_mode": frozen_mode,
        }
        if has_tail:
            frozen_overrides["tail_len_ptr"] = "*i32"
        else:
            frozen_constants["tail_len_ptr"] = None
            frozen_overrides["tail_len_ptr"] = "constexpr"
        if wide_pool:
            frozen_overrides.update(dict.fromkeys(_quantization.POOL_SLOT_STRIDE_NAMES, "i64"))
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
        frozen_source = _quantization.build_kernel_source(
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
    tail_source = _quantization.build_kernel_source(
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
    reduce_source = _quantization.build_kernel_source(
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
    if not 1 <= num_waves <= _MAX_MERGE_WAVES:
        raise ValueError(f"num_waves must be in 1..{_MAX_MERGE_WAVES}, got {num_waves}")
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
    for wave_idx in range(_MAX_MERGE_WAVES):
        if wave_idx < num_waves:
            overrides[f"out_{wave_idx}_ptr"] = f"*{triton_dtype}"
            overrides[f"lse_{wave_idx}_ptr"] = "*fp32"
        else:
            constants[f"out_{wave_idx}_ptr"] = None
            constants[f"lse_{wave_idx}_ptr"] = None
            overrides[f"out_{wave_idx}_ptr"] = "constexpr"
            overrides[f"lse_{wave_idx}_ptr"] = "constexpr"
    if is_contiguous:
        for field in _MERGE_STRIDE_FIELDS:
            constants[field] = None
            overrides[field] = "constexpr"
    else:
        for wave_idx in range(_MAX_MERGE_WAVES):
            wave_fields = _MERGE_STRIDE_FIELDS[wave_idx * 9 : (wave_idx + 1) * 9]
            if wave_idx < num_waves:
                for field in wave_fields:
                    overrides[field] = "i64"
            else:
                for field in wave_fields:
                    constants[field] = None
                    overrides[field] = "constexpr"
    merge_source = _quantization.build_kernel_source(
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
    window_mode: Literal["none", "causal", "windowed"] | None = None,
    has_tail: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, ...]:
    """Offline-compile two-wave merge and native chunked prefill kernels for ``target``."""
    _quantization.validate_kv_compile_args(dtype, head_dim)
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
        from ._amd import gluon as amd_gluon  # noqa: PLC0415

        return (
            merge_compiled,
            *amd_gluon.compile_amd_prefill_kernels(
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
        from ._nvidia import prefill as nvidia_prefill  # noqa: PLC0415

        return (
            merge_compiled,
            *nvidia_prefill.compile_nvidia_prefill_kernels(
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
