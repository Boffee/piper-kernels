"""Offline Triton compilation harnesses for Piper KV decode and chunked prefill."""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton

from piper_kernels._triton.targets import AcceleratorTarget

from . import _backend, _quantization
from ._decode_schedule import BLOCK_QH, REDUCE_NUM_WARPS
from ._kv_prefill import (
    _merge_two_attention_waves_kernel,
    _piper_decode_splitkv_reduce_kernel,
)
from ._quantization import _div16_attrs
from .triton import (
    _piper_decode_frozen_split_kernel,
    _piper_decode_tail_split_kernel,
    select_decode_schedule,
)


def compile_kv_decode_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    contiguous_inputs: bool = True,
    wide_pool: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, object, object]:
    """Compile frozen-split, tail-split, and reduce decode kernels offline."""
    triton_dtype = _quantization.validate_kv_compile_args(dtype, head_dim)
    accel = AcceleratorTarget.from_compiler_target(target)
    backend = _backend.prefill_backend(accel)
    schedule = select_decode_schedule(accel, head_dim)

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
        }
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
            constants={
                "has_chunk_map": has_chunk_map,
                "grouped_qk": grouped_qk,
                "pack_wmma": pack_wmma,
                "head_dim": head_dim,
                "block_qh": BLOCK_QH,
                "block_n": schedule.block_n,
                "qk_mode": schedule.frozen_qk_mode,
                "pv_precision": schedule.pv_precision,
                "loop_num_stages": schedule.loop_num_stages,
            },
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
            "is_causal": is_causal,
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
        constants={"head_dim": head_dim},
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

    return (
        frozen_compiled,
        compiler(tail_source, target, split_options),
        compiler(reduce_source, target, {"num_warps": REDUCE_NUM_WARPS}),
    )


def compile_kv_prefill_kernels(
    target: object,
    *,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = 128,
    has_chunk_map: bool = True,
    is_causal: bool = True,
    wide_pool: bool = False,
    has_chunk_splits: bool = False,
    compile_fn: Callable[[object, object, dict[str, object]], object] | None = None,
) -> tuple[object, ...]:
    """Offline-compile two-wave merge and native chunked prefill kernels for ``target``."""
    triton_dtype = _quantization.validate_kv_compile_args(dtype, head_dim)
    compiler = triton.compile if compile_fn is None else compile_fn
    merge_source = _quantization.build_kernel_source(
        _merge_two_attention_waves_kernel,
        constants={"head_dim": head_dim},
        overrides={
            "out_a_ptr": f"*{triton_dtype}",
            "lse_a_ptr": "*fp32",
            "out_b_ptr": f"*{triton_dtype}",
            "lse_b_ptr": "*fp32",
        },
    )
    merge_compiled = compiler(merge_source, target, {"num_warps": REDUCE_NUM_WARPS})

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
                compile_fn=compile_fn,
            ),
        )
    return (merge_compiled,)
