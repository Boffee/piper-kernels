"""Prepare dense per-token V directly in RDNA4's K64 WMMA layout."""

# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

from piper_kernels._triton.runtime import device_context

from .._quantization import quantize_value_rows


@triton.jit(do_not_specialize=["length", "storage_length", "heads"])
def _prepare_value_kernel(
    value_ptr,
    mean_ptr,
    packed_ptr,
    multiplier_ptr,
    log_scale_ptr,
    length,
    storage_length,
    stride_b,
    stride_h,
    stride_n,
    heads,
    head_dim: tl.constexpr,
    is_causal: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    batch_head = batch * heads + head
    token = tl.arange(0, 64)
    feature = tl.arange(0, head_dim)
    rows = tile * 64 + token
    valid = rows < length
    values = tl.load(
        value_ptr
        + batch * stride_b
        + head * stride_h
        + rows[:, None] * stride_n
        + feature[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)
    if not is_causal:
        mean = tl.load(mean_ptr + batch_head * head_dim + feature)
        values -= mean[None, :]
    values = tl.where(valid[:, None], values, 0.0)
    codes, scales = quantize_value_rows(values)
    # Match the shared PV fragments without an intermediate V transpose/repack.
    packed_token = (token & ~24) | ((token & 8) << 1) | ((token & 16) >> 1)
    offsets = (
        (batch_head * (storage_length // 64) + tile) * head_dim * 64
        + feature[None, :] * 64
        + packed_token[:, None]
    )
    tl.store(packed_ptr + offsets, codes)
    metadata_offsets = batch_head * storage_length + rows
    tl.store(multiplier_ptr + metadata_offsets, scales * 255.0)
    tl.store(log_scale_ptr + metadata_offsets, tl.log2(scales))


def prepare_value(
    value: torch.Tensor,
    mean: torch.Tensor,
    *,
    is_causal: bool,
    storage_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, length, head_dim = value.shape
    with device_context(value.device):
        packed = torch.empty(
            (batch, heads, storage_length // 64, head_dim, 64),
            dtype=torch.int8,
            device=value.device,
        )
        multiplier = torch.empty(
            (batch, heads, storage_length),
            dtype=torch.float32,
            device=value.device,
        )
        log_scale = torch.empty_like(multiplier)
        _prepare_value_kernel[(storage_length // 64, heads, batch)](
            value,
            mean,
            packed,
            multiplier,
            log_scale,
            length,
            storage_length,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            heads,
            head_dim,
            is_causal,
            num_warps=4,
        )
    return packed, multiplier, log_scale
