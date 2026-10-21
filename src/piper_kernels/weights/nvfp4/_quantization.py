"""Portable NVFP4 encoding with E4M3 subnormal block scales.

Global scales are decoding multipliers, the reciprocal of FlashInfer's encoding
scale. Each block scale is round-to-nearest E4M3 of ``amax / (6 * global_scale)``;
there is no smallest-normal floor. A nonzero block whose scale rounds to zero
uses saturated codes, matching FlashInfer, and reconstructs to zero. An all-zero
block uses zero codes. Both cases avoid NaNs during packing.
"""

from __future__ import annotations

import math

import torch
from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, pack_uint4
from torchao.prototype.mx_formats.utils import to_blocked

from piper_kernels.weights.nvfp4 import _layout


def quantize_nvfp4(
    source: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    *,
    swizzled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed low-first E2M1 data and FP8 scales for a matrix or matrix batch."""
    if source.ndim not in (2, 3) or source.shape[-1] % _layout.BLOCK_SIZE:
        raise ValueError("NVFP4 requires matrices with a feature width divisible by 16")
    values = source.float().reshape(*source.shape[:-1], -1, _layout.BLOCK_SIZE)
    block_amax = values.abs().amax(dim=-1)
    global_scale = per_tensor_scale
    if global_scale is None:
        global_scale = source.new_ones((), dtype=torch.float32)
    effective_global_scale = torch.where(global_scale == 0, 1.0, global_scale)
    scales = (
        (block_amax * (1.0 / 6.0) / effective_global_scale).clamp(max=448).to(torch.float8_e4m3fn)
    )
    reciprocal = (1.0 / effective_global_scale) / scales.float()
    normalized = values * reciprocal.unsqueeze(-1)
    # A zero block scale produces signed infinities for nonzero values and NaN
    # for zeros. Saturate these in place; the FP4 converter saturates finite values.
    normalized.nan_to_num_(nan=6.0, posinf=6.0, neginf=-6.0)
    qdata = pack_uint4(f32_to_f4_unpacked(normalized.reshape(source.shape)))
    # All-zero blocks use zero codes, including signed-zero inputs. Correct the
    # packed bytes instead of selecting another source-sized FP32 tensor.
    qdata.reshape(*block_amax.shape, _layout.QDATA_BLOCK_SIZE).masked_fill_(
        (block_amax == 0).unsqueeze(-1), 0
    )
    if swizzled:
        rows, features = source.shape[-2:]
        scales = to_blocked(
            scales.reshape(math.prod(source.shape[:-1]), features // _layout.BLOCK_SIZE)
        )
        scales = scales.reshape(*source.shape[:-2], *_layout.scale_shape(rows, features))
    return qdata, scales
