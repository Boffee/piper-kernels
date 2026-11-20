"""Imperative FX matchers and metadata guards for projected RMSNorm + RoPE."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch.fx.node import Argument

from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _preparation_sharing as preparation_sharing

_aten = torch.ops.aten
_SLICE_END = torch.iinfo(torch.int64).max


@dataclass(frozen=True, slots=True)
class NormalizedRopeMatch:
    """Matched ``projection -> reshape -> RMSNorm -> split-half RoPE`` chain."""

    projection_node: torch.fx.Node
    shape: Argument
    norm_weight: torch.fx.Node | None
    norm_epsilon: Argument
    cos: torch.fx.Node
    sin: torch.fx.Node
    rotary_dim: Argument | None
    half_rotary_dim: Argument
    activation_dtype: torch.dtype | None


def _match_rope_table_view(node: Argument) -> tuple[torch.fx.Node, torch.dtype | None] | None:  # noqa: PLR0911
    """Match ``unsqueeze(unsqueeze(convert?(table), 0), 2)``."""
    if not isinstance(node, torch.fx.Node) or len(node.users) != 1:
        return None
    u2 = linear_fx.bind_call(node, _aten.unsqueeze.default)
    if u2 is None or u2["dim"] != 2:
        return None
    mid = u2["self"]
    if not isinstance(mid, torch.fx.Node) or len(mid.users) != 1:
        return None
    u0 = linear_fx.bind_call(mid, _aten.unsqueeze.default)
    if u0 is None or u0["dim"] != 0:
        return None
    inner = u0["self"]
    if not isinstance(inner, torch.fx.Node):
        return None
    converted = linear_fx.convert_dtype(inner)
    if converted is not None:
        if len(inner.users) != 1:
            return None
        return converted
    return inner, None


def _match_rotated_half(  # noqa: PLR0911
    rot_cat: Argument,
    rotary: torch.fx.Node,
) -> tuple[torch.fx.Node, Argument] | None:
    """Match ``cat([-split[1], split[0]], -1)`` over ``rotary`` and return ``(split, half_dim)``."""
    if not isinstance(rot_cat, torch.fx.Node) or len(rot_cat.users) != 1:
        return None
    cat_bound = linear_fx.bind_call(rot_cat, _aten.cat.default)
    if cat_bound is None or cat_bound["dim"] not in (-1, 3):
        return None
    tensors = cat_bound["tensors"]
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 2:
        return None
    neg_second, first = tensors
    if (
        not isinstance(neg_second, torch.fx.Node)
        or not isinstance(first, torch.fx.Node)
        or len(neg_second.users) != 1
        or len(first.users) != 1
    ):
        return None
    second = linear_fx.unary_arg(neg_second, _aten.neg.default)
    if second is None or len(second.users) != 1:
        return None
    split = linear_fx.ordered_tuple_output_producer((first, second), _aten.split.Tensor)
    if split is None or set(split.users) != {first, second}:
        return None
    split_bound = linear_fx.bind_call(split, _aten.split.Tensor)
    if (
        split_bound is None
        or split_bound["self"] is not rotary
        or split_bound["dim"] not in (-1, 3)
    ):
        return None
    return split, split_bound["split_size"]


def _match_rope_addition(
    rotary_output: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, Argument, torch.dtype | None] | None:
    """Match ``rotary * cos_view + rotated * sin_view`` in any commutative operand order."""
    add_args = linear_fx.binary_args(rotary_output, _aten.add.Tensor)
    if add_args is None:
        return None
    for direct_arg, rotated_arg in (add_args, add_args[::-1]):
        if (
            not isinstance(direct_arg, torch.fx.Node)
            or not isinstance(rotated_arg, torch.fx.Node)
            or len(direct_arg.users) != 1
            or len(rotated_arg.users) != 1
        ):
            continue
        direct_mul = linear_fx.binary_args(direct_arg, _aten.mul.Tensor)
        rotated_mul = linear_fx.binary_args(rotated_arg, _aten.mul.Tensor)
        if direct_mul is None or rotated_mul is None:
            continue
        for rotary_cand, cos_view in (direct_mul, direct_mul[::-1]):
            if not isinstance(rotary_cand, torch.fx.Node):
                continue
            cos_match = _match_rope_table_view(cos_view)
            if cos_match is None:
                continue
            cos_node, cos_dtype = cos_match
            for rot_cat, sin_view in (rotated_mul, rotated_mul[::-1]):
                sin_match = _match_rope_table_view(sin_view)
                if sin_match is None or sin_match[1] is not cos_dtype:
                    continue
                sin_node = sin_match[0]
                half_match = _match_rotated_half(rot_cat, rotary_cand)
                if half_match is None:
                    continue
                split_node, half_rotary_dim = half_match
                if set(rotary_cand.users) != {direct_arg, split_node}:
                    continue
                return rotary_cand, cos_node, sin_node, half_rotary_dim, cos_dtype
    return None


def _match_rms_core(node: torch.fx.Node) -> tuple[torch.fx.Node, Argument] | None:
    """Match ``promoted * rsqrt(mean(promoted**2, [3], True) + eps)``."""
    mul_args = linear_fx.binary_args(node, _aten.mul.Tensor)
    if mul_args is None:
        return None
    for promoted, inv_rms in (mul_args, mul_args[::-1]):
        if (
            not isinstance(promoted, torch.fx.Node)
            or not isinstance(inv_rms, torch.fx.Node)
            or len(inv_rms.users) != 1
        ):
            continue
        variance = linear_fx.unary_arg(inv_rms, _aten.rsqrt.default)
        if variance is None or len(variance.users) != 1:
            continue
        var_args = linear_fx.binary_args(variance, _aten.add.Scalar)
        if var_args is None:
            continue
        mean_arg, norm_epsilon = var_args
        if not isinstance(mean_arg, torch.fx.Node) or len(mean_arg.users) != 1:
            continue
        mean_bound = linear_fx.bind_call(mean_arg, _aten.mean.dim)
        if (
            mean_bound is None
            or mean_bound["dim"] not in ([3], (3,), [-1], (-1,))
            or mean_bound["keepdim"] is not True
            or mean_bound["dtype"] is not None
        ):
            continue
        squared = mean_bound["self"]
        if not isinstance(squared, torch.fx.Node) or len(squared.users) != 1:
            continue
        pow_bound = linear_fx.bind_call(squared, _aten.pow.Tensor_Scalar)
        if (
            pow_bound is None
            or pow_bound["self"] is not promoted
            or not linear_fx.is_scalar(pow_bound["exponent"], 2.0)
        ):
            continue
        if set(promoted.users) != {node, squared}:
            continue
        return promoted, norm_epsilon
    return None


def _match_rmsnorm(
    scaled: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node | None, Argument] | None:
    """Match affine or weightless RMSNorm and return ``(promoted, norm_weight, epsilon)``."""
    core = _match_rms_core(scaled)
    if core is not None:
        return core[0], None, core[1]
    mul_args = linear_fx.binary_args(scaled, _aten.mul.Tensor)
    if mul_args is None:
        return None
    for norm_cand, weight_cand in (mul_args, mul_args[::-1]):
        if (
            not isinstance(norm_cand, torch.fx.Node)
            or not isinstance(weight_cand, torch.fx.Node)
            or len(norm_cand.users) != 1
        ):
            continue
        core = _match_rms_core(norm_cand)
        if core is not None:
            return core[0], weight_cand, core[1]
    return None


def match_normalized_rope(  # noqa: PLR0911, PLR0912
    node: object,
    *,
    expected_users: int = 1,
) -> NormalizedRopeMatch | None:
    """Imperatively match ``projection -> reshape -> RMSNorm -> split-half RoPE``.

    Handles FP16/BF16 (promoted FP32 RMSNorm) and unpromoted FP32 graphs,
    affine and non-affine RMSNorm, and partial or canonical full-width RoPE.
    Every intermediate node's exact user set is verified so escaping values are
    never silently discarded.
    """
    if not isinstance(node, torch.fx.Node) or len(node.users) != expected_users:
        return None

    cat_bound = linear_fx.bind_call(node, _aten.cat.default)
    if cat_bound is not None:
        if cat_bound["dim"] not in (-1, 3):
            return None
        tensors = cat_bound["tensors"]
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 2:
            return None
        rotary_output, passthrough = tensors
        if (
            not isinstance(rotary_output, torch.fx.Node)
            or not isinstance(passthrough, torch.fx.Node)
            or len(rotary_output.users) != 1
            or len(passthrough.users) != 1
        ):
            return None
        tail_slice = linear_fx.bind_call(passthrough, _aten.slice.Tensor)
        if (
            tail_slice is None
            or tail_slice["dim"] not in (3, -1)
            or tail_slice["step"] != 1
            or tail_slice["end"] not in (None, _SLICE_END)
            or not isinstance(tail_slice["self"], torch.fx.Node)
        ):
            return None
        rounded_from_tail = tail_slice["self"]
        rotary_dim: Argument | None = tail_slice["start"]
    else:
        rotary_output = node
        passthrough = None
        rounded_from_tail = None
        rotary_dim = None

    rope = _match_rope_addition(rotary_output)
    if rope is None:
        return None
    rotary_node, cos_node, sin_node, half_rotary_dim, activation_dtype = rope

    if rounded_from_tail is not None:
        head_slice = linear_fx.bind_call(rotary_node, _aten.slice.Tensor)
        if (
            head_slice is None
            or head_slice["self"] is not rounded_from_tail
            or head_slice["dim"] not in (3, -1)
            or head_slice["start"] not in (0, None)
            or head_slice["step"] != 1
            or not linear_fx.same_argument(head_slice["end"], rotary_dim)
            or set(rounded_from_tail.users) != {rotary_node, passthrough}
        ):
            return None
        rounded = rounded_from_tail
    else:
        rounded = rotary_node

    if activation_dtype is not None:
        downcast = linear_fx.convert_dtype(rounded)
        if downcast is None or downcast[1] is not activation_dtype or len(downcast[0].users) != 1:
            return None
        scaled = downcast[0]
    else:
        scaled = rounded

    norm = _match_rmsnorm(scaled)
    if norm is None:
        return None
    promoted, norm_weight, norm_epsilon = norm

    if activation_dtype is not None:
        upcast = linear_fx.convert_dtype(promoted)
        if upcast is None or upcast[1] is not torch.float32 or len(upcast[0].users) != 1:
            return None
        reshaped = upcast[0]
    else:
        reshaped = promoted

    reshape_bound = linear_fx.bind_call(reshaped, _aten.reshape.default)
    if reshape_bound is None:
        return None
    projection_node = reshape_bound["self"]
    if not isinstance(projection_node, torch.fx.Node) or len(projection_node.users) != 1:
        return None
    return NormalizedRopeMatch(
        projection_node=projection_node,
        shape=reshape_bound["shape"],
        norm_weight=norm_weight,
        norm_epsilon=norm_epsilon,
        cos=cos_node,
        sin=sin_node,
        rotary_dim=rotary_dim,
        half_rotary_dim=half_rotary_dim,
        activation_dtype=activation_dtype,
    )


def static_int(value: object) -> int | None:
    """Return a non-boolean static integer."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def integer_scalar_metadata(value: object) -> int | torch.SymInt | None:
    """Resolve static or symbolic integer metadata from an FX argument."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, torch.SymInt)):
        return value
    if isinstance(value, torch.fx.Node):
        metadata = cast(torch.fx.Node, value).meta.get("val")
        if isinstance(metadata, (int, torch.SymInt)) and not isinstance(metadata, bool):
            return metadata
    return None


def integer_scalar_argument(value: object) -> Argument | None:
    """Return an FX-compatible integer argument when its metadata is valid."""
    if integer_scalar_metadata(value) is None:
        return None
    return value if isinstance(value, (int, torch.SymInt, torch.fx.Node)) else None


def positive_float(value: object) -> float | None:
    """Return a finite positive Python configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted > 0 else None


def valid_rmsnorm(
    norm_weight: object,
    epsilon: object,
    *,
    head_dim: int,
    device: torch.device,
    supported_dtypes: Sequence[torch.dtype],
) -> bool:
    """Check an optional affine weight and the Python RMSNorm epsilon."""
    if norm_weight is not None:
        if not isinstance(norm_weight, torch.fx.Node):
            return False
        norm = preparation_sharing.tensor_metadata(norm_weight)
        if (
            norm is None
            or norm.layout is not torch.strided
            or not norm.is_contiguous()
            or norm.dtype not in supported_dtypes
            or tuple(norm.shape) != (head_dim,)
            or norm.device != device
        ):
            return False
    return positive_float(epsilon) is not None


def valid_rope_tables(
    cos_node: object,
    sin_node: object,
    rotary_dim_value: object,
    half_rotary_dim_value: object,
    *,
    sequence_length: int | torch.SymInt,
    head_dim: int,
    device: torch.device,
) -> bool:
    """Check contiguous FP32 split-half tables without reading tensor contents.

    Keep symbolic dimension comparisons structural: symbolic widths are checked
    against their captured expressions, while static widths enforce the range
    and parity requirements.
    """
    if not isinstance(cos_node, torch.fx.Node) or not isinstance(sin_node, torch.fx.Node):
        return False
    rotary_dim = integer_scalar_metadata(rotary_dim_value)
    half_rotary_dim = integer_scalar_metadata(half_rotary_dim_value)
    if rotary_dim is None or half_rotary_dim is None:
        return False
    for node in (cos_node, sin_node):
        table = preparation_sharing.tensor_metadata(node)
        if (
            table is None
            or table.layout is not torch.strided
            or not table.is_contiguous()
            or table.dtype is not torch.float32
            or table.ndim != 2
            or table.device != device
            or preparation_sharing.dimension_key(table.shape[0])
            != preparation_sharing.dimension_key(sequence_length)
            or preparation_sharing.dimension_key(table.shape[1])
            != preparation_sharing.dimension_key(rotary_dim)
        ):
            return False
    if preparation_sharing.dimension_key(half_rotary_dim) != preparation_sharing.dimension_key(
        (rotary_dim + 1) // 2
    ):
        return False
    if isinstance(rotary_dim, int):
        return (
            2 <= rotary_dim <= head_dim
            and rotary_dim % 2 == 0
            and isinstance(half_rotary_dim, int)
            and half_rotary_dim == rotary_dim // 2
        )
    return True


__all__ = [
    "NormalizedRopeMatch",
    "integer_scalar_argument",
    "integer_scalar_metadata",
    "match_normalized_rope",
    "positive_float",
    "static_int",
    "valid_rmsnorm",
    "valid_rope_tables",
]
