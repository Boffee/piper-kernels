"""Shared graph grammar for sparse-Piper projection fusion passes."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import cast

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, Match
from torch.fx.node import Argument

type ProjectionPattern = Callable[[str], CallFunction]

_SLICE_END = torch.iinfo(torch.int64).max

_QUANTIZED_ATTENTION_OPERAND_NAMES = (
    "output_query",
    "output_query_scale",
    "output_query_summary",
    "output_key",
    "output_key_scale",
    "output_key_summary",
    "output_key_aux",
    "output_value",
    "output_value_scale_multiplier",
    "output_value_mean",
)
_QUANTIZED_ATTENTION_POLICY_NAMES = (
    "output_head_keep_ratio_units",
    "output_sparse_key_blocks",
    "output_logical_sequence_length",
    "output_routing_mode",
)
_QUANTIZED_ATTENTION_ARGUMENT_NAMES = (
    _QUANTIZED_ATTENTION_OPERAND_NAMES + _QUANTIZED_ATTENTION_POLICY_NAMES
)


def optional_attention_layout_arguments[ArgumentT](
    block_lengths: ArgumentT | None,
    sparse_query_blocks: ArgumentT | None,
) -> tuple[ArgumentT | None, ...]:
    """Preserve positional defaults only when a later layout argument is present."""
    if sparse_query_blocks is not None:
        return block_lengths, sparse_query_blocks
    return (block_lengths,) if block_lengths is not None else ()


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


def sparse_piper_projection_pattern(
    projection: ProjectionPattern,
    *,
    with_block_lengths: bool = False,
    with_coarse: bool = False,
    with_sparse_query_blocks: bool = False,
) -> CallFunction:
    """Match projected sparse attention with optional padding, coarse, and Q scopes."""
    operand_users = 2 if with_coarse else 1
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
    sparse_query_blocks = KeywordArg("sparse_query_blocks") if with_sparse_query_blocks else None
    optional_arguments = optional_attention_layout_arguments(
        block_lengths,
        sparse_query_blocks,
    )
    arguments = (
        torch.ops.piper_kernels.sparse_piper_attention.default,
        query,
        key,
        value,
        KeywordArg("sparse_head_keep_ratio_units"),
        sparse_key_blocks,
        KeywordArg("sparse_softmax_scale"),
        KeywordArg("sparse_routing_mode"),
        *optional_arguments,
    )
    fine_output = CallFunction(*arguments, _users=1) if with_coarse else CallFunction(*arguments)
    if not with_coarse:
        return fine_output
    coarse_arguments = (
        torch.ops.piper_kernels.sparse_piper_coarse_residual.default,
        query,
        key,
        value,
        KeywordArg("coarse_gate"),
        KeywordArg("coarse_key_blocks"),
        KeywordArg("coarse_scale"),
        KeywordArg("coarse_routing_mode"),
        *((block_lengths,) if block_lengths is not None else ()),
    )
    coarse_output = CallFunction(*coarse_arguments, _users=1)
    return CallFunction(torch.ops.aten.add.Tensor, fine_output, coarse_output)


def reshaped_quantized_attention_pattern(
    *,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
) -> CallFunction:
    """Match one materialized quantized attention result before projection."""
    operands = tuple(KeywordArg(name) for name in _QUANTIZED_ATTENTION_OPERAND_NAMES)
    policy = tuple(KeywordArg(name) for name in _QUANTIZED_ATTENTION_POLICY_NAMES)
    if with_coarse:
        attention = CallFunction(
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default,
            *operands,
            KeywordArg("output_block_mean"),
            KeywordArg("output_coarse_gate"),
            *policy,
            KeywordArg("output_coarse_scale"),
            KeywordArg("output_block_lengths") if with_block_lengths else None,
            KeywordArg("output_coarse_key_blocks"),
            *((KeywordArg("output_sparse_query_blocks"),) if with_sparse_query_blocks else ()),
            _users=1,
        )
    else:
        block_lengths = KeywordArg("output_block_lengths") if with_block_lengths else None
        sparse_query_blocks = (
            KeywordArg("output_sparse_query_blocks") if with_sparse_query_blocks else None
        )
        optional_arguments = optional_attention_layout_arguments(
            block_lengths,
            sparse_query_blocks,
        )
        attention = CallFunction(
            torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
            *operands,
            *policy,
            *optional_arguments,
            _users=1,
        )
    return CallFunction(
        torch.ops.aten.reshape.default,
        attention,
        KeywordArg("output_attention_shape"),
        _users=1,
    )


def quantized_attention_arguments(match: Match) -> tuple[Argument, ...]:
    """Return the quantized attention operands and routing policy."""
    return cast(
        tuple[Argument, ...],
        tuple(match.kwargs[name] for name in _QUANTIZED_ATTENTION_ARGUMENT_NAMES),
    )


def bounded_attention_arguments(match: Match) -> tuple[Argument, ...]:
    """Return optional padding, coarse-residual, and query-scope arguments."""
    return cast(
        tuple[Argument, ...],
        (
            match.kwargs.get("output_block_lengths"),
            match.kwargs.get("output_block_mean"),
            match.kwargs.get("output_coarse_gate"),
            match.kwargs.get("output_coarse_scale"),
            match.kwargs.get("output_coarse_key_blocks"),
            match.kwargs.get("output_sparse_query_blocks"),
        ),
    )


__all__ = [
    "bounded_attention_arguments",
    "optional_attention_layout_arguments",
    "quantized_attention_arguments",
    "reshaped_quantized_attention_pattern",
    "sparse_piper_projection_pattern",
]
