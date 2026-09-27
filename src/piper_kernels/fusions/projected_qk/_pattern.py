"""Projection-independent graph grammar for RMSNorm and split-half RoPE."""

from __future__ import annotations

import operator

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg

_SLICE_END = torch.iinfo(torch.int64).max


def _rope_table_pattern(name: str, activation_dtype: torch.dtype) -> CallFunction:
    table: KeywordArg | CallFunction = KeywordArg(name)
    if activation_dtype is not torch.float32:
        table = CallFunction(
            torch.ops.prims.convert_element_type.default, table, activation_dtype, _users=1
        )
    table = CallFunction(torch.ops.aten.unsqueeze.default, table, 0, _users=1)
    return CallFunction(torch.ops.aten.unsqueeze.default, table, 2, _users=1)


def normalized_rope_pattern(  # noqa: PLR0913
    projection: CallFunction,
    *,
    shape_name: str,
    norm_weight_name: str,
    norm_epsilon_name: str,
    cos_name: str,
    sin_name: str,
    rotary_dim_name: str,
    half_rotary_dim_name: str,
    output_users: int = 1,
    activation_dtype: torch.dtype = torch.bfloat16,
    affine: bool = True,
    full_rotary: bool = False,
) -> CallFunction:
    """Match projected RMSNorm/RoPE using caller-owned capture names.

    Full-width RoPE drops the identity slice and empty passthrough in canonical
    graphs. That form captures only the half width; callers validate that its
    tables cover the complete head dimension.
    """
    # FP32 graphs omit redundant casts around RMSNorm and RoPE.
    low_precision = activation_dtype is not torch.float32
    reshaped = CallFunction(
        torch.ops.aten.reshape.default,
        projection,
        KeywordArg(shape_name),
        _users=1 if low_precision else 2,
    )
    promoted = (
        CallFunction(
            torch.ops.prims.convert_element_type.default, reshaped, torch.float32, _users=2
        )
        if low_precision
        else reshaped
    )
    squared = CallFunction(torch.ops.aten.pow.Tensor_Scalar, promoted, 2, _users=1)
    mean = CallFunction(torch.ops.aten.mean.dim, squared, [3], True, _users=1)
    variance = CallFunction(
        torch.ops.aten.add.Scalar,
        mean,
        KeywordArg(norm_epsilon_name),
        _users=1,
    )
    inverse_rms = CallFunction(torch.ops.aten.rsqrt.default, variance, _users=1)
    normalized = CallFunction(
        torch.ops.aten.mul.Tensor, promoted, inverse_rms, _users=1 if affine or low_precision else 2
    )
    scaled = (
        CallFunction(
            torch.ops.aten.mul.Tensor,
            normalized,
            KeywordArg(norm_weight_name),
            _users=1 if low_precision else 2,
        )
        if affine
        else normalized
    )
    rounded = (
        CallFunction(
            torch.ops.prims.convert_element_type.default, scaled, activation_dtype, _users=2
        )
        if low_precision
        else scaled
    )
    rotary = (
        rounded
        if full_rotary
        else CallFunction(
            torch.ops.aten.slice.Tensor,
            rounded,
            3,
            0,
            KeywordArg(rotary_dim_name),
            _users=2,
        )
    )
    split = CallFunction(
        torch.ops.aten.split.Tensor,
        rotary,
        KeywordArg(half_rotary_dim_name),
        -1,
        _users=2,
    )
    first = CallFunction(operator.getitem, split, 0, _users=1)
    second = CallFunction(operator.getitem, split, 1, _users=1)
    cos = _rope_table_pattern(cos_name, activation_dtype)
    direct = CallFunction(torch.ops.aten.mul.Tensor, rotary, cos, _users=1)
    rotated = CallFunction(
        torch.ops.aten.cat.default,
        [CallFunction(torch.ops.aten.neg.default, second, _users=1), first],
        -1,
        _users=1,
    )
    sin = _rope_table_pattern(sin_name, activation_dtype)
    rotated = CallFunction(torch.ops.aten.mul.Tensor, rotated, sin, _users=1)
    rotary_output = CallFunction(
        torch.ops.aten.add.Tensor, direct, rotated, _users=output_users if full_rotary else 1
    )
    if full_rotary:
        return rotary_output
    passthrough = CallFunction(
        torch.ops.aten.slice.Tensor,
        rounded,
        3,
        KeywordArg(rotary_dim_name),
        _SLICE_END,
        _users=1,
    )
    return CallFunction(
        torch.ops.aten.cat.default,
        [rotary_output, passthrough],
        -1,
        _users=output_users,
    )


__all__ = ["normalized_rope_pattern"]
