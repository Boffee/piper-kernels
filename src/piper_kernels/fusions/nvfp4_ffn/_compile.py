"""Format-level compiler matching shared by bounded NVFP4 FFNs."""

from __future__ import annotations

import torch
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, Match

from piper_kernels.linear import _bias
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx
from piper_kernels.linear.nvfp4 import _storage as nvfp4_storage
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation


def source_files() -> tuple[str, ...]:
    """Return source files whose changes invalidate format-level matching."""
    return tuple(
        file_name
        for file_name in (
            __file__,
            _bias.__file__,
            preparation_sharing.__file__,
            nvfp4_compile_fx.__file__,
            nvfp4_storage.__file__,
            nvfp4_validation.__file__,
        )
        if file_name is not None
    )


def semantic_linear_pattern(
    input_pattern: object,
    prefix: str,
    users: int | None,
    *,
    target: torch._ops.OpOverload,
    with_group_size: bool,
    with_high_first: bool,
) -> CallFunction:
    """Match one semantic NVFP4 projection with named operands."""
    arguments = (
        input_pattern,
        KeywordArg(f"{prefix}_weight_qdata"),
        KeywordArg(f"{prefix}_weight_scale"),
        KeywordArg(f"{prefix}_weight_per_tensor_scale"),
        KeywordArg(f"{prefix}_activation_per_tensor_scale"),
        KeywordArg(f"{prefix}_bias"),
        KeywordArg(f"{prefix}_dynamic_activation_scale"),
        *((KeywordArg(f"{prefix}_group_size"),) if with_group_size else ()),
        *((KeywordArg(f"{prefix}_high_first"),) if with_high_first else ()),
    )
    if users is None:
        return CallFunction(target, *arguments)
    return CallFunction(target, *arguments, _users=users)


def projection_call_matches(node: torch.fx.Node, match: Match, prefix: str) -> bool:
    """Identify a captured projection by all operands, not potentially shared weights."""
    if node.kwargs or not isinstance(node.target, torch._ops.OpOverload):
        return False
    # Plain NVFP4 callers need not have registered the optional ConvRot operator.
    convrot = node.target._schema.name == "piper_kernels::convrot_nvfp4_linear"
    if convrot != (f"{prefix}_group_size" in match.kwargs):
        return False
    names = (
        "weight_qdata",
        "weight_scale",
        "weight_per_tensor_scale",
        "activation_per_tensor_scale",
        "bias",
        "dynamic_activation_scale",
        *(("group_size",) if convrot else ()),
        "high_first",
    )
    expected = tuple(
        match.kwargs.get(f"{prefix}_{name}", False)
        if name == "high_first"
        else match.kwargs[f"{prefix}_{name}"]
        for name in names
    )
    arguments = node.args[1:]
    if len(arguments) == len(expected) - 1:
        arguments = (*arguments, False)
    return arguments == expected


def matched_projections(
    match: Match,
    prefixes: tuple[str, ...],
) -> tuple[nvfp4_compile_fx.SemanticLinearNodes, ...] | None:
    """Parse a topology's named standard NVFP4 projection calls."""
    calls = [
        node
        for node in match.nodes
        if node.op == "call_function"
        and node.target == torch.ops.piper_kernels.nvfp4_linear.default
    ]
    if len(calls) != len(prefixes):
        return None
    parsed = []
    for prefix in prefixes:
        call = next(
            (node for node in calls if projection_call_matches(node, match, prefix)),
            None,
        )
        if call is None:
            return None
        projection = nvfp4_compile_fx.SemanticLinearNodes.from_call(call)
        if projection is None:
            return None
        parsed.append(projection)
    return tuple(parsed)


def _dimension_matches(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


def valid_semantic_ffn(
    match: Match,
    sources: tuple[nvfp4_compile_fx.SemanticLinearNodes, ...],
    down: nvfp4_compile_fx.SemanticLinearNodes,
) -> bool:
    """Validate metadata shared by one- and two-source semantic NVFP4 FFNs."""
    if len(sources) not in (1, 2):
        return False
    validated_sources = tuple(
        nvfp4_compile_fx.validated_semantic_linear(
            source,
            f"NVFP4 FFN compiler source projection {index}",
        )
        for index, source in enumerate(sources)
    )
    validated_down = nvfp4_compile_fx.validated_semantic_linear(
        down,
        "NVFP4 FFN compiler down projection",
    )
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    if (
        validated_down is None
        or output_value is None
        or any(validated is None for validated in validated_sources)
    ):
        return False
    source_values = tuple(validated for validated in validated_sources if validated is not None)
    input_value, source_shape = source_values[0]
    down_input, down_shape = validated_down
    return bool(
        all(source.input is sources[0].input for source in sources[1:])
        and input_value.dtype in (torch.float16, torch.bfloat16)
        and input_value.layout is torch.strided
        and input_value.is_contiguous()
        and all(value.dtype is input_value.dtype for value, _shape in source_values[1:])
        and down_input.dtype is input_value.dtype
        and all(
            _dimension_matches(source_shape.rows, shape.rows) for _value, shape in source_values
        )
        and _dimension_matches(source_shape.rows, down_shape.rows)
        and all(
            _dimension_matches(source_shape.input_features, shape.input_features)
            for _value, shape in source_values
        )
        and all(
            _dimension_matches(source_shape.output_features, shape.output_features)
            for _value, shape in source_values
        )
        and _dimension_matches(down_shape.input_features, source_shape.output_features)
        and all(source.high_first == sources[0].high_first for source in sources[1:])
        and output_value.dtype is input_value.dtype
        and output_value.device == input_value.device
        and output_value.ndim == input_value.ndim
        and all(
            _dimension_matches(output_dimension, input_dimension)
            for output_dimension, input_dimension in zip(
                output_value.shape[:-1],
                input_value.shape[:-1],
                strict=True,
            )
        )
        and _dimension_matches(output_value.shape[-1], down_shape.output_features)
    )


__all__ = [
    "matched_projections",
    "projection_call_matches",
    "semantic_linear_pattern",
    "source_files",
    "valid_semantic_ffn",
]
