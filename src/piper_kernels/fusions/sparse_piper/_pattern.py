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
    *,
    output_users: int = 1,
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
        _users=output_users,
    )


def _sparse_piper_projection_components(
    projection: ProjectionPattern,
    *,
    operand_users: int,
    attention_users: int | None = None,
    with_block_lengths: bool = False,
) -> tuple[
    CallFunction,
    CallFunction,
    CallFunction,
    CallFunction,
    KeywordArg,
    KeywordArg | None,
]:
    query = _normalized_rope_pattern(
        projection("sparse_q"),
        "sparse_q",
        output_users=operand_users,
    )
    key = _normalized_rope_pattern(
        projection("sparse_k"),
        "sparse_k",
        output_users=operand_users,
    )
    value = CallFunction(
        torch.ops.aten.reshape.default,
        projection("sparse_v"),
        KeywordArg("sparse_attention_shape"),
        _users=operand_users,
    )
    sparse_key_blocks = KeywordArg("sparse_key_blocks")
    block_lengths = KeywordArg("sparse_block_lengths") if with_block_lengths else None
    arguments = (
        torch.ops.piper_kernels.sparse_piper_attention.default,
        query,
        key,
        value,
        KeywordArg("sparse_head_keep_ratio_units"),
        sparse_key_blocks,
        KeywordArg("sparse_softmax_scale"),
        KeywordArg("sparse_routing_mode"),
        *((block_lengths,) if block_lengths is not None else ()),
    )
    attention = (
        CallFunction(*arguments)
        if attention_users is None
        else CallFunction(*arguments, _users=attention_users)
    )
    return attention, query, key, value, sparse_key_blocks, block_lengths


def _sparse_piper_attention_projection_pattern(
    projection: ProjectionPattern,
    *,
    with_block_lengths: bool,
) -> CallFunction:
    attention, _query, _key, _value, _sparse_key_blocks, _block_lengths = (
        _sparse_piper_projection_components(
            projection,
            operand_users=1,
            with_block_lengths=with_block_lengths,
        )
    )
    return attention


def sparse_piper_projection_pattern(
    projection: ProjectionPattern,
) -> CallFunction:
    """Match the common projected Q/K/V sparse-Piper attention region."""
    return _sparse_piper_attention_projection_pattern(
        projection,
        with_block_lengths=False,
    )


def sparse_piper_block_lengths_projection_pattern(
    projection: ProjectionPattern,
) -> CallFunction:
    """Match projected sparse attention over valid-front padded K64 storage."""
    return _sparse_piper_attention_projection_pattern(
        projection,
        with_block_lengths=True,
    )


def _sparse_piper_coarse_residual_projection_pattern(
    projection: ProjectionPattern,
    *,
    with_block_lengths: bool,
) -> CallFunction:
    fine_output, query, key, value, sparse_key_blocks, block_lengths = (
        _sparse_piper_projection_components(
            projection,
            operand_users=2,
            attention_users=1,
            with_block_lengths=with_block_lengths,
        )
    )
    arguments = (
        torch.ops.piper_kernels.sparse_piper_coarse_residual.default,
        fine_output,
        query,
        key,
        value,
        KeywordArg("coarse_compression_gate"),
        sparse_key_blocks,
        KeywordArg("coarse_key_blocks"),
        KeywordArg("coarse_scale"),
        KeywordArg("coarse_routing_mode"),
        *((block_lengths,) if block_lengths is not None else ()),
    )
    return CallFunction(*arguments)


def sparse_piper_coarse_residual_projection_pattern(
    projection: ProjectionPattern,
) -> CallFunction:
    """Match projected sparse attention plus a Q/K/V-derived coarse residual."""
    return _sparse_piper_coarse_residual_projection_pattern(
        projection,
        with_block_lengths=False,
    )


def sparse_piper_coarse_residual_block_lengths_projection_pattern(
    projection: ProjectionPattern,
) -> CallFunction:
    """Match sparse plus coarse attention over valid-front padded K64 storage."""
    return _sparse_piper_coarse_residual_projection_pattern(
        projection,
        with_block_lengths=True,
    )


__all__ = [
    "sparse_piper_block_lengths_projection_pattern",
    "sparse_piper_coarse_residual_block_lengths_projection_pattern",
    "sparse_piper_coarse_residual_projection_pattern",
    "sparse_piper_projection_pattern",
]
