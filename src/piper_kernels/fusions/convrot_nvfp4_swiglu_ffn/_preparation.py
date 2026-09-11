"""FP32 SwiGLU preparation inside the chunked ConvRot NVFP4 FFN."""

# pyright: reportCallIssue=false, reportIndexIssue=false
# ruff: noqa: ANN001, ANN202
# Triton infers device pointer types from the launch arguments.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from piper_kernels._triton import convrot as rotation
from piper_kernels._triton.convrot_nvfp4 import (
    _preparation_num_warps,
    _rotated_row_amax_kernel,
)
from piper_kernels._triton.input_activations import swiglu
from piper_kernels._triton.nvfp4 import dynamic_scale as nvfp4_dynamic_scale
from piper_kernels._triton.nvfp4 import encode_nvfp4_blocks, swizzled_scale_offsets
from piper_kernels._triton.runtime import device_context
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_backend
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend
from piper_kernels.linear.nvfp4._storage import prepare_activation_storage

# RTX 5090 measurements favored reuse up to 32 MiB; larger buffers added enough
# memory traffic to favor recomputation. Reproduce with benchmark_nvfp4_ffn.py
# and its --rotated-workspace-mib override (benchmark-only, not a runtime option).
_ROTATED_WORKSPACE_BYTES = 32 * 1024 * 1024


@triton.jit
def _swiglu_rotated_amax_kernel(
    projections_ptr,
    rotated_ptr,
    row_amax_ptr,
    features: tl.constexpr,
    group_size: tl.constexpr,
    chunk_size: tl.constexpr,
):
    row_offset = tl.program_id(0).to(tl.int64) * features
    input_offset = row_offset * 2
    columns = tl.arange(0, chunk_size)
    row_amax = tl.full((), 0.0, tl.float32)
    for chunk in range(tl.cdiv(features, chunk_size)):
        column = chunk * chunk_size + columns
        gate = tl.load(
            projections_ptr + input_offset + features + column, column < features, 0.0
        ).to(tl.float32)
        value = tl.load(projections_ptr + input_offset + column, column < features, 0.0).to(
            tl.float32
        )
        rotated = rotation.rotate_hadamard_groups(swiglu(value, gate), chunk_size, group_size) * (
            group_size**-0.5
        )
        tl.store(rotated_ptr + row_offset + column, rotated, column < features)
        row_amax = tl.maximum(row_amax, tl.max(tl.abs(rotated), 0))
    tl.store(row_amax_ptr + tl.program_id(0), row_amax)


@triton.jit
def _swiglu_quantize_kernel(
    projections_ptr,
    global_scale_ptr,
    qdata_ptr,
    scale_ptr,
    elements,
    features: tl.constexpr,
    group_size: tl.constexpr,
    block_size: tl.constexpr,
    high_first: tl.constexpr,
):
    """Recompute FP32 activation/rotation in group-aligned tiles before encoding."""
    start = tl.program_id(0).to(tl.int64) * block_size
    offsets = start + tl.arange(0, block_size)
    input_offsets = (offsets // features) * (2 * features) + offsets % features
    gate = tl.load(projections_ptr + input_offsets + features, offsets < elements, 0.0).to(
        tl.float32
    )
    value = tl.load(projections_ptr + input_offsets, offsets < elements, 0.0).to(tl.float32)
    rotated = rotation.rotate_hadamard_groups(swiglu(value, gate), block_size, group_size) * (
        group_size**-0.5
    )
    packed, scales = encode_nvfp4_blocks(
        tl.reshape(rotated, (block_size // 16, 16)),
        tl.load(global_scale_ptr),
        block_size // 16,
        high_first,
    )
    qdata_offsets = start // 2 + tl.arange(0, block_size // 2)
    tl.store(
        qdata_ptr + qdata_offsets,
        tl.reshape(packed, (block_size // 2,)),
        qdata_offsets < elements // 2,
    )
    blocks = start // 16 + tl.arange(0, block_size // 16)
    scale_offsets = swizzled_scale_offsets(
        blocks // (features // 16), blocks % (features // 16), tl.cdiv(features, 64)
    )
    tl.store(scale_ptr + scale_offsets, scales, blocks < elements // 16)


def prepare(
    projections: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    group_size: int,
    high_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read the runner's adjacent [value | gate] workspace without copying either half.

    Dynamic scaling remains local to the current FFN chunk. SwiGLU and rotation
    retain FP32 arithmetic through NVFP4 encoding for both scale modes.
    """
    contiguous_input, rows, features, chunks, target = convrot_backend._validate_input(
        projections, group_size, "swiglu"
    )
    if not dynamic_activation_scale and (
        per_tensor_scale is None
        or per_tensor_scale.shape != ()
        or per_tensor_scale.dtype is not torch.float32
        or per_tensor_scale.device != projections.device
        or not per_tensor_scale.is_contiguous()
    ):
        raise ValueError("ConvRot NVFP4 scale must be a contiguous FP32 scalar on the input device")
    with device_context(projections.device):
        if dynamic_activation_scale:
            row_amax = torch.empty(rows, device=projections.device, dtype=torch.float32)
            if rows * features * torch.float32.itemsize <= _ROTATED_WORKSPACE_BYTES:
                rotated = torch.empty(
                    (rows, features), device=projections.device, dtype=torch.float32
                )
                chunk_size = min(triton.next_power_of_2(features), 8_192)
                warps = 2 if chunk_size <= 2_048 else (4 if chunk_size <= 4_096 else 8)
                _swiglu_rotated_amax_kernel[(rows,)](
                    contiguous_input,
                    rotated,
                    row_amax,
                    features,
                    group_size,
                    chunk_size,
                    num_warps=warps,
                )
                global_scale = nvfp4_dynamic_scale(row_amax)
                qdata, scale = nvfp4_backend._prepare_static_storage(
                    rotated, global_scale, swiglu=False, high_first=high_first
                )
                return qdata, scale, global_scale
            chunk_count = sum(chunk > 0 for chunk in chunks)
            amax_warps, _ = _preparation_num_warps(chunks, group_size)
            if chunk_count == 1 and chunks[0] >= 8_192:
                amax_warps = 8
            _rotated_row_amax_kernel[(rows,)](
                contiguous_input,
                row_amax,
                features,
                chunk_count,
                *chunks,
                group_size,
                group_size**-0.5,
                "swiglu",
                target.backend,
                -1,
                num_warps=amax_warps,
            )
            per_tensor_scale = nvfp4_dynamic_scale(row_amax)
        assert per_tensor_scale is not None
        qdata, scale = prepare_activation_storage(contiguous_input, rows, features)
        block_size = 1_024
        _swiglu_quantize_kernel[(triton.cdiv(rows * features, block_size),)](
            contiguous_input,
            per_tensor_scale,
            qdata,
            scale,
            rows * features,
            features,
            group_size,
            block_size,
            high_first,
            num_warps=2,
        )
    return qdata, scale, per_tensor_scale
