"""Triton kernels and GPU dispatcher for quantized MoE decode and prefill."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import triton

from piper_kernels._triton.primitives import TritonCompileFn
from piper_kernels._triton.targets import AcceleratorTarget

from . import _int8_triton, _nvfp4_triton, _routing_triton, _schedule, _trellis_triton
from ._int8_triton import (
    int8_moe_batched_matmul_kernel,
    int8_moe_grouped_gemm_kernel,
    int8_moe_m1_matmul_kernel,
)
from ._nvfp4_triton import (
    nvfp4_moe_batched_gemv_kernel,
    nvfp4_moe_grouped_gemm_kernel,
    nvfp4_moe_m1_gemv_kernel,
)
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

if TYPE_CHECKING:
    from triton.backends.compiler import GPUTarget


def moe_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    up_gate_qdata: torch.Tensor,
    up_gate_scale: torch.Tensor,
    down_qdata: torch.Tensor,
    down_scale: torch.Tensor,
    up_gate_global_scale: torch.Tensor | None = None,
    down_global_scale: torch.Tensor | None = None,
    expert_to_slot: torch.Tensor | None = None,
    prepared_data: torch.Tensor | None = None,
    prepared_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    target: AcceleratorTarget | None = None,
    accumulate: bool = False,
    group_size: int = 0,
    activation_fn: str = "swiglu",
    gate_first: bool = True,
    high_first: bool = False,
    active_slots_hint: int = 0,
    up_gate_block_scales: torch.Tensor | None = None,
    down_block_scales: torch.Tensor | None = None,
    up_gate_row_widths: torch.Tensor | None = None,
    down_row_widths: torch.Tensor | None = None,
    up_gate_slot_offsets: torch.Tensor | None = None,
    down_slot_offsets: torch.Tensor | None = None,
    up_gate_block_size: int = 0,
    down_block_size: int = 0,
) -> torch.Tensor:
    """Execute single-token, small-batch, or prefill quantized MoE kernels on GPU."""
    device = input.device
    rows = math.prod(input.shape[:-1])
    hidden_size = input.shape[-1]
    top_k = topk_ids.shape[-1]
    result = torch.empty_like(input, memory_format=torch.contiguous_format) if out is None else out
    out_2d = result.view(rows, hidden_size)
    input_2d = input.reshape(rows, hidden_size)
    ids_2d = topk_ids.reshape(rows, top_k).contiguous()
    weights_2d = topk_weights.reshape(rows, top_k).contiguous()
    resolved_target = AcceleratorTarget.from_device(device) if target is None else target

    if up_gate_block_scales is not None:
        assert down_block_scales is not None
        from piper_kernels.linear.convrot.int8 import _backend as int8_backend_mod  # noqa: PLC0415

        int8_backend = (
            int8_backend_mod.select_linear_backend_for_target(resolved_target)
            if _routing_triton.supports_trellis_int8_prefill(resolved_target)
            else None
        )
        intermediate_size = down_block_scales.shape[2] * down_block_size
        up_gate_out = up_gate_block_scales.shape[1]
        bytes_per_pair = _schedule.prefill_bytes_per_pair(
            is_int8=int8_backend is not None,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            up_gate_out=up_gate_out,
            logical_elem_bytes=out_2d.element_size(),
            group_size=group_size,
        )
        regime, chunk_rows = _schedule.select_moe_execution_regime(rows, top_k, bytes_per_pair)
        if regime == "prefill" and int8_backend is not None:
            _int8_triton.dispatch_int8_prefill_chunks(
                input_2d,
                ids_2d,
                weights_2d,
                up_gate_qdata,
                up_gate_scale,
                down_qdata,
                down_scale,
                expert_to_slot,
                prepared_data,
                None,
                out_2d,
                resolved_target=resolved_target,
                int8_backend=int8_backend,
                chunk_rows=chunk_rows,
                accumulate=accumulate,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                active_slots_hint=active_slots_hint,
                up_gate_block_scales=up_gate_block_scales,
                up_gate_row_widths=up_gate_row_widths,
                up_gate_slot_offsets=up_gate_slot_offsets,
                up_gate_block_size=up_gate_block_size,
                down_block_scales=down_block_scales,
                down_row_widths=down_row_widths,
                down_slot_offsets=down_slot_offsets,
                down_block_size=down_block_size,
            )
        else:
            _trellis_triton.dispatch_trellis_a16_moe(
                input_2d,
                ids_2d,
                weights_2d,
                up_gate_qdata,
                up_gate_block_scales,
                up_gate_scale,
                up_gate_row_widths,
                up_gate_slot_offsets,
                up_gate_block_size,
                down_qdata,
                down_block_scales,
                down_scale,
                down_row_widths,
                down_slot_offsets,
                down_block_size,
                expert_to_slot,
                prepared_data,
                out_2d,
                resolved_target=resolved_target,
                regime=regime,
                chunk_rows=chunk_rows,
                accumulate=accumulate,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                active_slots_hint=active_slots_hint,
            )
    elif up_gate_qdata.dtype is torch.int8:
        _int8_triton.dispatch_int8_moe(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            expert_to_slot,
            prepared_data,
            prepared_scale,
            out_2d,
            resolved_target=resolved_target,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            active_slots_hint=active_slots_hint,
        )
    else:
        _nvfp4_triton.dispatch_nvfp4_moe(
            input_2d,
            ids_2d,
            weights_2d,
            up_gate_qdata,
            up_gate_scale,
            down_qdata,
            down_scale,
            up_gate_global_scale,
            down_global_scale,
            expert_to_slot,
            prepared_data,
            out_2d,
            resolved_target=resolved_target,
            accumulate=accumulate,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            high_first=high_first,
            active_slots_hint=active_slots_hint,
        )
    return result


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
    compiler: TritonCompileFn | None = None,
) -> dict[str, triton.compiler.CompiledKernel]:
    """Compile quantized MoE Triton kernels matching runtime regime and chunking selection."""
    from . import _compile  # noqa: PLC0415

    return _compile.compile_moe_kernels(
        target,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        rows=rows,
        top_k=top_k,
        num_slots=num_slots,
        active_slots_hint=active_slots_hint,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        high_first=high_first,
        has_slot_map=has_slot_map,
        has_global_scale=has_global_scale,
        split_k=split_k,
        accumulate=accumulate,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
        bytes_per_block=bytes_per_block,
        compiler=compiler,
    )


__all__ = [
    "compile_moe_kernels",
    "int8_moe_batched_matmul_kernel",
    "int8_moe_grouped_gemm_kernel",
    "int8_moe_m1_matmul_kernel",
    "moe_bucket_tokens_kernel",
    "moe_ffn",
    "moe_m1_activate_rotate_kernel",
    "moe_m1_weighted_reduce_kernel",
    "moe_prefill_bucket_count_kernel",
    "moe_prefill_bucket_scatter_kernel",
    "moe_prefill_gather_kernel",
    "moe_splitk_reduce_kernel",
    "nvfp4_moe_batched_gemv_kernel",
    "nvfp4_moe_grouped_gemm_kernel",
    "nvfp4_moe_m1_gemv_kernel",
    "trellis_moe_gemv_kernel",
]
