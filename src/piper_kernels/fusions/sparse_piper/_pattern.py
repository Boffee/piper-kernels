"""Projection-independent RMSNorm/RoPE sparse-Piper graph grammar."""

from __future__ import annotations

import operator
from collections.abc import Callable

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg

type ProjectionPattern = Callable[[str], CallFunction]

_SLICE_END = torch.iinfo(torch.int64).max


def _normalized_rope_pattern(
    projection: CallFunction,
    prefix: str,
) -> CallFunction:
    reshaped = CallFunction(
        torch.ops.aten.reshape.default,
        projection,
        KeywordArg("sparse_attention_shape"),
        _users=1,
    )
    promoted = CallFunction(
        torch.ops.prims.convert_element_type.default,
        reshaped,
        torch.float32,
        _users=2,
    )
    squared = CallFunction(torch.ops.aten.pow.Tensor_Scalar, promoted, 2, _users=1)
    mean = CallFunction(torch.ops.aten.mean.dim, squared, [3], True, _users=1)
    variance = CallFunction(
        torch.ops.aten.add.Scalar,
        mean,
        KeywordArg(f"{prefix}_norm_epsilon"),
        _users=1,
    )
    inverse_rms = CallFunction(torch.ops.aten.rsqrt.default, variance, _users=1)
    normalized = CallFunction(torch.ops.aten.mul.Tensor, promoted, inverse_rms, _users=1)
    scaled = CallFunction(
        torch.ops.aten.mul.Tensor,
        normalized,
        KeywordArg(f"{prefix}_norm_weight"),
        _users=1,
    )
    rounded = CallFunction(
        torch.ops.prims.convert_element_type.default,
        scaled,
        torch.bfloat16,
        _users=2,
    )
    rotary = CallFunction(
        torch.ops.aten.slice.Tensor,
        rounded,
        3,
        0,
        KeywordArg("sparse_rotary_dim"),
        _users=2,
    )
    split = CallFunction(
        torch.ops.aten.split.Tensor,
        rotary,
        KeywordArg("sparse_half_rotary_dim"),
        -1,
        _users=2,
    )
    first = CallFunction(operator.getitem, split, 0, _users=1)
    second = CallFunction(operator.getitem, split, 1, _users=1)
    cos = CallFunction(
        torch.ops.prims.convert_element_type.default,
        KeywordArg("sparse_cos"),
        torch.bfloat16,
        _users=1,
    )
    cos = CallFunction(torch.ops.aten.unsqueeze.default, cos, 0, _users=1)
    cos = CallFunction(torch.ops.aten.unsqueeze.default, cos, 2, _users=1)
    direct = CallFunction(torch.ops.aten.mul.Tensor, rotary, cos, _users=1)
    rotated = CallFunction(
        torch.ops.aten.cat.default,
        [CallFunction(torch.ops.aten.neg.default, second, _users=1), first],
        -1,
        _users=1,
    )
    sin = CallFunction(
        torch.ops.prims.convert_element_type.default,
        KeywordArg("sparse_sin"),
        torch.bfloat16,
        _users=1,
    )
    sin = CallFunction(torch.ops.aten.unsqueeze.default, sin, 0, _users=1)
    sin = CallFunction(torch.ops.aten.unsqueeze.default, sin, 2, _users=1)
    rotated = CallFunction(torch.ops.aten.mul.Tensor, rotated, sin, _users=1)
    rotary_output = CallFunction(torch.ops.aten.add.Tensor, direct, rotated, _users=1)
    passthrough = CallFunction(
        torch.ops.aten.slice.Tensor,
        rounded,
        3,
        KeywordArg("sparse_rotary_dim"),
        _SLICE_END,
        _users=1,
    )
    return CallFunction(
        torch.ops.aten.cat.default,
        [rotary_output, passthrough],
        -1,
        _users=1,
    )


def sparse_piper_projection_pattern(
    projection: ProjectionPattern,
) -> CallFunction:
    """Match the common projected Q/K/V sparse-Piper attention region."""
    projected_value = CallFunction(
        torch.ops.aten.reshape.default,
        projection("sparse_v"),
        KeywordArg("sparse_attention_shape"),
        _users=1,
    )
    return CallFunction(
        torch.ops.piper_kernels.sparse_piper_attention.default,
        _normalized_rope_pattern(projection("sparse_q"), "sparse_q"),
        _normalized_rope_pattern(projection("sparse_k"), "sparse_k"),
        projected_value,
        KeywordArg("sparse_head_keep_ratio_units"),
        KeywordArg("sparse_key_blocks"),
        KeywordArg("sparse_softmax_scale"),
    )


__all__ = ["sparse_piper_projection_pattern"]
