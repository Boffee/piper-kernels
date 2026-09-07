"""Backend-owned K64 value layout and query-independent FP32 parameters."""

# Gluon device function parameters are not Python runtime values.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

from dataclasses import dataclass

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.runtime import device_context

from .._prepared import _PreparedSparsePiperContext

# One FP32 record per K64 tile, shared by the table writer and attention reader.
KEY_SCALE = gl.constexpr(0)
LOG_MULTIPLIER = gl.constexpr(1)
LOG_MULTIPLIER_OVER_255 = gl.constexpr(2)
INVERSE_MULTIPLIER = gl.constexpr(3)
PARAMETER_COUNT = gl.constexpr(4)


@gluon.jit
def _pack_values(values_ptr, packed_ptr, storage_length: gl.constexpr):
    tile = gl.program_id(0).to(gl.int64)
    head = gl.program_id(1).to(gl.int64)
    element = gl.arange(0, 128 * 64, gl.BlockedLayout([16], [32], [4], [0]))
    column = element // 64
    token = element % 64
    # Swap token bits 3/4 so adjacent WMMA reductions use contiguous words.
    token = (token & ~24) | ((token & 8) << 1) | ((token & 16) >> 1)
    source = (head * 128 + column) * storage_length + tile * 64 + token
    target = (head * (storage_length // 64) + tile) * (128 * 64) + element
    gl.store(packed_ptr + target, gl.load(values_ptr + source))


@gluon.jit
def _prepare_parameters(key_scale_ptr, multiplier_ptr, table_ptr, count: gl.constexpr):
    index = gl.program_id(0).to(gl.int64) * 256 + gl.arange(
        0, 256, gl.BlockedLayout([1], [32], [4], [0])
    )
    scale = gl.load(key_scale_ptr + index, index < count, 0)
    multiplier = gl.load(multiplier_ptr + index, index < count, 1)
    record = table_ptr + index * PARAMETER_COUNT
    gl.store(record + KEY_SCALE, scale, index < count)
    gl.store(record + LOG_MULTIPLIER, gl.log2(multiplier), index < count)
    gl.store(record + LOG_MULTIPLIER_OVER_255, gl.log2(multiplier * (1.0 / 255.0)), index < count)
    gl.store(record + INVERSE_MULTIPLIER, 1.0 / multiplier, index < count)


@dataclass(frozen=True, slots=True)
class PackedContext:
    """Explicit snapshot owned by a bound launcher, never a global tensor cache.

    Source K/V and scales must remain unchanged while this context is reused.
    Rebinding after writes rebuilds the packed values and parameter table.
    """

    source: _PreparedSparsePiperContext
    value: torch.Tensor
    parameters: torch.Tensor


def pack_context(context: _PreparedSparsePiperContext) -> PackedContext:
    batch, heads, storage, _ = context.key.shape
    with device_context(context.key.device):
        value = torch.empty(
            (batch, heads, storage // 64, 128, 64), device=context.key.device, dtype=torch.int8
        )
        parameters = torch.empty(
            (batch, heads, storage // 64, PARAMETER_COUNT.value),
            device=context.key.device,
            dtype=torch.float32,
        )
        _pack_values[(storage // 64, batch * heads)](
            context.value, value, storage, num_warps=4, num_stages=1
        )
        count = batch * heads * (storage // 64)
        _prepare_parameters[((count + 255) // 256,)](
            context.key_scale, context.value_scale_multiplier, parameters, count, num_warps=4
        )
    return PackedContext(context, value, parameters)
