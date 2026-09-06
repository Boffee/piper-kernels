"""Compiler folding for a bounded-workspace ConvRot INT8 SwiGLU FFN."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from piper_kernels.fusions.swiglu_ffn import _compile as swiglu_ffn_compile
from piper_kernels.fusions.swiglu_ffn import _pattern as swiglu_ffn_pattern
from piper_kernels.fusions.swiglu_ffn import triton as swiglu_ffn_triton
from piper_kernels.linear import _bias
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _backend
from piper_kernels.linear.convrot.int8 import _compile as convrot_int8_compile

from . import triton as ffn_backend

_COMPILE_PASS_VERSION = "convrot-int8-swiglu-ffn-compile-v7"


def _semantic_linear_pattern(
    input_pattern: object,
    prefix: str,
    users: int | None,
) -> CallFunction:
    arguments = (
        input_pattern,
        KeywordArg(f"{prefix}_weight_qdata"),
        KeywordArg(f"{prefix}_weight_scale"),
        KeywordArg(f"{prefix}_bias"),
        KeywordArg(f"{prefix}_group_size"),
    )
    if users is None:
        return CallFunction(torch.ops.piper_kernels.convrot_int8_linear.default, *arguments)
    return CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        *arguments,
        _users=users,
    )


def _metadata(match: Match, name: str) -> torch.Tensor | None:
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
    value = _metadata(match, name)
    return bool(
        value is not None
        and value.ndim == 1
        and _dimension_matches(value.shape[0], features)
        and _bias.is_supported_dtype(value.dtype)
        and value.device == input_value.device
        and value.layout is torch.strided
        and value.is_contiguous()
    )


def _valid_semantic_ffn(  # noqa: PLR0911
    match: Match,
    *,
    promote_gate: bool | None,
) -> bool:
    input_value = _metadata(match, "ffn_input")
    gate_weight = _metadata(match, "gate_weight_qdata")
    gate_scale = _metadata(match, "gate_weight_scale")
    value_weight = _metadata(match, "value_weight_qdata")
    value_scale = _metadata(match, "value_weight_scale")
    down_weight = _metadata(match, "down_weight_qdata")
    down_scale = _metadata(match, "down_weight_scale")
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    if any(
        value is None
        for value in (
            input_value,
            gate_weight,
            gate_scale,
            value_weight,
            value_scale,
            down_weight,
            down_scale,
            output_value,
        )
    ):
        return False
    assert input_value is not None
    assert gate_weight is not None
    assert gate_scale is not None
    assert value_weight is not None
    assert value_scale is not None
    assert down_weight is not None
    assert down_scale is not None
    assert output_value is not None
    weights = gate_weight, value_weight, down_weight
    scales = gate_scale, value_scale, down_scale
    if (
        input_value.ndim == 0
        or input_value.dtype is not torch.bfloat16
        or input_value.layout is not torch.strided
        or not input_value.is_contiguous()
        or any(
            weight.ndim != 2
            or weight.dtype is not torch.int8
            or weight.device != input_value.device
            or weight.layout is not torch.strided
            or not weight.is_contiguous()
            for weight in weights
        )
        or any(
            scale.dtype is not torch.float32
            or scale.device != input_value.device
            or scale.layout is not torch.strided
            or not scale.is_contiguous()
            for scale in scales
        )
        or _backend.select_linear_backend(input_value) is None
    ):
        return False

    input_features = gate_weight.shape[1]
    intermediate_features = gate_weight.shape[0]
    output_features = down_weight.shape[0]
    if (
        not _dimension_matches(input_value.shape[-1], input_features)
        or any(
            not _dimension_matches(left, right)
            for left, right in zip(gate_weight.shape, value_weight.shape, strict=True)
        )
        or not _dimension_matches(down_weight.shape[1], intermediate_features)
        or gate_scale.shape != (intermediate_features, 1)
        or value_scale.shape != (intermediate_features, 1)
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
    if promote_gate is True and match.kwargs["logical_dtype"] is not input_value.dtype:
        return False
    group_sizes = tuple(
        match.kwargs[f"{prefix}_group_size"] for prefix in ("gate", "value", "down")
    )
    if any(
        isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1
        for group_size in group_sizes
    ):
        return False
    gate_group_size, value_group_size, down_group_size = group_sizes
    if (
        gate_group_size != value_group_size
        or (isinstance(input_features, int) and input_features % gate_group_size)
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
            ("gate", intermediate_features),
            ("value", intermediate_features),
            ("down", output_features),
        )
    )


def _replace_semantic_ffn(  # noqa: PLR0913, PLR0917
    match: Match,
    ffn_input: torch.fx.Node,
    gate_weight_qdata: torch.fx.Node,
    gate_weight_scale: torch.fx.Node,
    gate_bias: torch.fx.Node | None,
    gate_group_size: int,
    value_weight_qdata: torch.fx.Node,
    value_weight_scale: torch.fx.Node,
    value_bias: torch.fx.Node | None,
    value_group_size: int,
    down_weight_qdata: torch.fx.Node,
    down_weight_scale: torch.fx.Node,
    down_bias: torch.fx.Node | None,
    down_group_size: int,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    with graph.inserting_before(original):
        replacement = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default,
            args=(
                ffn_input,
                gate_weight_qdata,
                gate_weight_scale,
                gate_bias,
                gate_group_size,
                value_weight_qdata,
                value_weight_scale,
                value_bias,
                value_group_size,
                down_weight_qdata,
                down_weight_scale,
                down_bias,
                down_group_size,
                ffn_backend._DEFAULT_CHUNK_ROWS,
            ),
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


def _replace_semantic_ffn_gated_updates(  # noqa: PLR0913, PLR0917
    match: Match,
    ffn_input: torch.fx.Node,
    gate_weight_qdata: torch.fx.Node,
    gate_weight_scale: torch.fx.Node,
    gate_bias: torch.fx.Node | None,
    gate_group_size: int,
    value_weight_qdata: torch.fx.Node,
    value_weight_scale: torch.fx.Node,
    value_bias: torch.fx.Node | None,
    value_group_size: int,
    down_weight_qdata: torch.fx.Node,
    down_weight_scale: torch.fx.Node,
    down_bias: torch.fx.Node | None,
    down_group_size: int,
    base: torch.fx.Node,
    reusable_update: torch.fx.Node,
    update_gate: torch.fx.Node,
    gate_indices: torch.fx.Node,
    ffn_gate: torch.fx.Node,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    python_indexing = swiglu_ffn_compile.uses_python_indexing(match)
    with graph.inserting_before(original):
        mutation = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_swiglu_ffn_gated_updates_.default,
            args=(
                ffn_input,
                gate_weight_qdata,
                gate_weight_scale,
                gate_bias,
                gate_group_size,
                value_weight_qdata,
                value_weight_scale,
                value_bias,
                value_group_size,
                down_weight_qdata,
                down_weight_scale,
                down_bias,
                down_group_size,
                base,
                reusable_update,
                update_gate,
                ffn_gate,
                gate_indices,
                python_indexing,
                ffn_backend._DEFAULT_CHUNK_ROWS,
            ),
        )
    mutation.meta["val"] = None
    original.replace_all_uses_with(reusable_update)
    match.erase_nodes()


_gated_updates_patterns = PatternMatcherPass("convrot_int8_swiglu_ffn_gated_updates")
_patterns = PatternMatcherPass("convrot_int8_swiglu_ffn")
for _promote_gate in (None, False, True):
    for _reverse_multiply in (False, True):
        _semantic_ffn_pattern = swiglu_ffn_pattern.semantic_ffn_pattern(
            _semantic_linear_pattern,
            _semantic_linear_pattern,
            _semantic_linear_pattern,
            promote_gate=_promote_gate,
            reverse_multiply=_reverse_multiply,
        )
        for _use_aten_index in (False, True):
            register_graph_pattern(
                swiglu_ffn_pattern.gated_updates_pattern(
                    _semantic_ffn_pattern,
                    use_aten_index=_use_aten_index,
                ),
                extra_check=lambda match, promote_gate=_promote_gate: (
                    swiglu_ffn_compile.valid_gated_updates(
                        match,
                        partial(_valid_semantic_ffn, promote_gate=promote_gate),
                    )
                ),
                pass_dict=_gated_updates_patterns,  # pyright: ignore[reportArgumentType]
            )(_replace_semantic_ffn_gated_updates)
        register_graph_pattern(
            _semantic_ffn_pattern,
            extra_check=lambda match, promote_gate=_promote_gate: _valid_semantic_ffn(
                match,
                promote_gate=promote_gate,
            ),
            pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
        )(_replace_semantic_ffn)


def _fold_chunked_ffn(graph: torch.fx.Graph) -> bool:
    changes = _gated_updates_patterns.apply(graph)
    changes += _patterns.apply(graph)
    changed = changes > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold semantic gate/value FFNs before ordinary linear normalization."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            _fold_chunked_ffn(graph)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            tuple(
                file_name
                for file_name in (
                    __file__,
                    _bias.__file__,
                    ffn_backend.__file__,
                    swiglu_ffn_compile.__file__,
                    swiglu_ffn_triton.__file__,
                    swiglu_ffn_pattern.__file__,
                )
                if file_name is not None
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def convrot_int8_swiglu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install semantic FFN folding before ordinary ConvRot INT8 folding."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (compile_pass, convrot_int8_compile.compile_pass),
    )


__all__ = ["convrot_int8_swiglu_ffn_compile_options"]
