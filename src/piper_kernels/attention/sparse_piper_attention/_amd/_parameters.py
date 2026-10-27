"""Query-independent FP32 parameters owned by an AMD attention context."""

# Gluon device function parameters are not Python runtime values.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportIndexIssue=false

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


def prepare_parameters(context: _PreparedSparsePiperContext) -> torch.Tensor:
    """Fold immutable K/V scales once; V already has its final physical format."""
    with device_context(context.key.device):
        parameters = torch.empty(
            (*context.key_scale.shape, PARAMETER_COUNT.value),
            device=context.key.device,
            dtype=torch.float32,
        )
        count = context.key_scale.numel()
        _prepare_parameters[((count + 255) // 256,)](
            context.key_scale, context.value_scale_multiplier, parameters, count, num_warps=4
        )
    return parameters
