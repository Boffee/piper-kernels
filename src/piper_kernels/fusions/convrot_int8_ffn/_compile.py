"""Format-level compiler matching for bounded ConvRot INT8 FFNs."""

from __future__ import annotations

from typing import cast

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, Match

from piper_kernels.linear import _bias
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _backend, _compile_fx


def source_files() -> tuple[str, ...]:
    """Return source files whose changes invalidate format-level matching."""
    return tuple(
        file_name
        for file_name in (
            __file__,
            _bias.__file__,
            preparation_sharing.__file__,
            _backend.__file__,
            _compile_fx.__file__,
        )
        if file_name is not None
    )


def semantic_linear_pattern(
    input_pattern: object,
    prefix: str,
    users: int | None,
) -> CallFunction:
    """Match one ordinary ConvRot INT8 projection with named operands."""
    arguments = (
        input_pattern,
        KeywordArg(f"{prefix}_weight_qdata"),
        KeywordArg(f"{prefix}_weight_scale"),
        KeywordArg(f"{prefix}_bias"),
        KeywordArg(f"{prefix}_group_size"),
        None,
        KeywordArg(f"{prefix}_input_scale"),
    )
    if users is None:
        return CallFunction(torch.ops.piper_kernels.convrot_int8_linear.default, *arguments)
    return CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        *arguments,
        _users=users,
    )


def argument_metadata(match: Match, name: str) -> torch.Tensor | None:
    """Return fake tensor metadata for one named pattern argument."""
    argument = match.kwargs[name]
    return (
        preparation_sharing.tensor_metadata(argument)
        if isinstance(argument, torch.fx.Node)
        else None
    )


def _dimension_matches(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


def _valid_bias(
    match: Match,
    name: str,
    *,
    features: int | torch.SymInt,
    input_value: torch.Tensor,
) -> bool:
    argument = match.kwargs[name]
    if argument is None:
        return True
    value = argument_metadata(match, name)
    return bool(
        value is not None
        and value.ndim == 1
        and _dimension_matches(value.shape[0], features)
        and _bias.is_supported_dtype(value.dtype)
        and value.device == input_value.device
        and value.layout is torch.strided
        and value.is_contiguous()
    )


def valid_semantic_ffn(  # noqa: PLR0911
    match: Match,
    source_prefixes: tuple[str, ...],
) -> bool:
    """Validate format metadata shared by one- and two-source ConvRot INT8 FFNs."""
    if len(source_prefixes) not in (1, 2):
        return False
    input_value = argument_metadata(match, "ffn_input")
    projection_prefixes = (*source_prefixes, "down")
    weights = tuple(
        argument_metadata(match, f"{prefix}_weight_qdata") for prefix in projection_prefixes
    )
    scales = tuple(
        argument_metadata(match, f"{prefix}_weight_scale") for prefix in projection_prefixes
    )
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    if (
        input_value is None
        or output_value is None
        or any(value is None for value in (*weights, *scales))
    ):
        return False
    concrete_weights = cast(tuple[torch.Tensor, ...], weights)
    concrete_scales = cast(tuple[torch.Tensor, ...], scales)
    if any(
        not _compile_fx.valid_input_scale(match.kwargs[f"{prefix}_input_scale"], input_value.device)
        for prefix in projection_prefixes
    ):
        return False
    if (
        input_value.ndim == 0
        or input_value.dtype not in (torch.float16, torch.bfloat16)
        or input_value.layout is not torch.strided
        or not input_value.is_contiguous()
        or any(
            weight.ndim != 2
            or weight.dtype is not torch.int8
            or weight.device != input_value.device
            or weight.layout is not torch.strided
            or not weight.is_contiguous()
            for weight in concrete_weights
        )
        or any(
            scale.dtype is not torch.float32
            or scale.device != input_value.device
            or scale.layout is not torch.strided
            or not scale.is_contiguous()
            for scale in concrete_scales
        )
        or _backend.select_linear_backend(input_value) is None
    ):
        return False

    source_weights = concrete_weights[:-1]
    source_scales = concrete_scales[:-1]
    down_weight = concrete_weights[-1]
    down_scale = concrete_scales[-1]
    input_features = source_weights[0].shape[1]
    intermediate_features = source_weights[0].shape[0]
    output_features = down_weight.shape[0]
    if (
        not _dimension_matches(input_value.shape[-1], input_features)
        or any(
            any(
                not _dimension_matches(left, right)
                for left, right in zip(source_weights[0].shape, source.shape, strict=True)
            )
            for source in source_weights[1:]
        )
        or not _dimension_matches(down_weight.shape[1], intermediate_features)
        or any(scale.shape != (intermediate_features, 1) for scale in source_scales)
        or down_scale.shape != (output_features, 1)
        or output_value.dtype is not input_value.dtype
        or output_value.device != input_value.device
        or output_value.ndim != input_value.ndim
        or any(
            not _dimension_matches(output_dimension, input_dimension)
            for output_dimension, input_dimension in zip(
                output_value.shape[:-1],
                input_value.shape[:-1],
                strict=True,
            )
        )
        or not _dimension_matches(output_value.shape[-1], output_features)
    ):
        return False
    group_sizes = tuple(match.kwargs[f"{prefix}_group_size"] for prefix in projection_prefixes)
    if any(
        isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1
        for group_size in group_sizes
    ):
        return False
    source_group_sizes = group_sizes[:-1]
    down_group_size = group_sizes[-1]
    if (
        any(group_size != source_group_sizes[0] for group_size in source_group_sizes[1:])
        or (isinstance(input_features, int) and input_features % source_group_sizes[0])
        or (isinstance(intermediate_features, int) and intermediate_features % down_group_size)
    ):
        return False
    return all(
        _valid_bias(
            match,
            f"{prefix}_bias",
            features=features,
            input_value=input_value,
        )
        for prefix, features in (
            *((prefix, intermediate_features) for prefix in source_prefixes),
            ("down", output_features),
        )
    )


__all__ = [
    "argument_metadata",
    "semantic_linear_pattern",
    "source_files",
    "valid_semantic_ffn",
]
