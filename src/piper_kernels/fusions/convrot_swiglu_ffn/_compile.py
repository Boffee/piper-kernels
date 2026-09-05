"""Compiler folding for a bounded-workspace ConvRot SwiGLU FFN."""

from __future__ import annotations

import operator
from collections.abc import Mapping

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

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.swiglu_ffn import _compile as swiglu_ffn_compile
from piper_kernels.fusions.swiglu_ffn import _pattern as swiglu_ffn_pattern
from piper_kernels.fusions.swiglu_ffn import triton as swiglu_ffn_triton
from piper_kernels.linear import _bias
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _compile as convrot_compile

from . import triton as ffn_backend

_COMPILE_PASS_VERSION = "convrot-swiglu-ffn-compile-v5"


def _normalized_ffn_pattern(*, explicit_up_activation: bool) -> CallFunction:
    """Match the stable ConvRot graph produced by input-activation folding."""
    up_arguments: tuple[object, ...] = (
        KeywordArg("ffn_input"),
        KeywordArg("up_weight_qdata"),
        KeywordArg("up_weight_scale"),
        KeywordArg("up_bias"),
        KeywordArg("up_group_size"),
    )
    if explicit_up_activation:
        up_arguments = (*up_arguments, None)
    packed = CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        *up_arguments,
        _users=1,
    )
    prepared = CallFunction(
        torch.ops.piper_kernels.convrot_int8_prepare_input.default,
        packed,
        KeywordArg("down_group_size"),
        "swiglu",
        _users=2,
    )
    prepared_qdata = CallFunction(operator.getitem, prepared, 0, _users=1)
    prepared_scale = CallFunction(operator.getitem, prepared, 1, _users=1)
    return CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear_prepared.default,
        prepared_qdata,
        prepared_scale,
        KeywordArg("down_weight_qdata"),
        KeywordArg("down_weight_scale"),
        KeywordArg("down_bias"),
        KeywordArg("logical_dtype"),
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


def _valid_normalized_ffn(match: Match) -> bool:
    input_value = _metadata(match, "ffn_input")
    up_weight = _metadata(match, "up_weight_qdata")
    up_scale = _metadata(match, "up_weight_scale")
    down_weight = _metadata(match, "down_weight_qdata")
    down_scale = _metadata(match, "down_weight_scale")
    output_value = preparation_sharing.tensor_metadata(match.output_node())
    if any(
        value is None
        for value in (
            input_value,
            up_weight,
            up_scale,
            down_weight,
            down_scale,
            output_value,
        )
    ):
        return False
    assert input_value is not None
    assert up_weight is not None
    assert up_scale is not None
    assert down_weight is not None
    assert down_scale is not None
    assert output_value is not None

    if (
        input_value.ndim == 0
        or input_value.dtype is not torch.bfloat16
        or input_value.device.type != "cuda"
        or input_value.layout is not torch.strided
        or not input_value.is_contiguous()
        or up_weight.ndim != 2
        or down_weight.ndim != 2
        or up_weight.dtype is not torch.int8
        or down_weight.dtype is not torch.int8
        or up_scale.dtype is not torch.float32
        or down_scale.dtype is not torch.float32
        or up_weight.device != input_value.device
        or up_scale.device != input_value.device
        or down_weight.device != input_value.device
        or down_scale.device != input_value.device
        or any(
            value.layout is not torch.strided or not value.is_contiguous()
            for value in (up_weight, up_scale, down_weight, down_scale)
        )
    ):
        return False
    target = AcceleratorTarget.from_device(input_value.device)
    if not target.cuda_capability_at_least(7, 5):
        return False

    input_features = up_weight.shape[1]
    intermediate_features = down_weight.shape[1]
    output_features = down_weight.shape[0]
    if (
        not _dimension_matches(input_value.shape[-1], input_features)
        or not _dimension_matches(up_weight.shape[0], 2 * intermediate_features)
        or up_scale.shape != (up_weight.shape[0], 1)
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
        or match.kwargs["logical_dtype"] is not input_value.dtype
    ):
        return False

    up_group_size = match.kwargs["up_group_size"]
    down_group_size = match.kwargs["down_group_size"]
    if (
        isinstance(up_group_size, bool)
        or not isinstance(up_group_size, int)
        or isinstance(down_group_size, bool)
        or not isinstance(down_group_size, int)
        or up_group_size < 1
        or down_group_size < 1
        or (isinstance(input_features, int) and input_features % up_group_size)
        or (isinstance(intermediate_features, int) and intermediate_features % down_group_size)
    ):
        return False
    return _valid_bias(
        match,
        "up_bias",
        features=up_weight.shape[0],
        input_value=input_value,
    ) and _valid_bias(
        match,
        "down_bias",
        features=output_features,
        input_value=input_value,
    )


def _valid_gated_updates(match: Match) -> bool:
    return swiglu_ffn_compile.valid_gated_updates(match, _valid_normalized_ffn)


def _replace_normalized_ffn(
    match: Match,
    ffn_input: torch.fx.Node,
    up_weight_qdata: torch.fx.Node,
    up_weight_scale: torch.fx.Node,
    up_bias: torch.fx.Node | None,
    up_group_size: int,
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
            torch.ops.piper_kernels.convrot_swiglu_ffn.default,
            args=(
                ffn_input,
                up_weight_qdata,
                up_weight_scale,
                up_bias,
                up_group_size,
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


def _replace_normalized_ffn_gated_updates(  # noqa: PLR0913, PLR0917
    match: Match,
    ffn_input: torch.fx.Node,
    up_weight_qdata: torch.fx.Node,
    up_weight_scale: torch.fx.Node,
    up_bias: torch.fx.Node | None,
    up_group_size: int,
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
            torch.ops.piper_kernels.convrot_swiglu_ffn_gated_updates_.default,
            args=(
                ffn_input,
                up_weight_qdata,
                up_weight_scale,
                up_bias,
                up_group_size,
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


_gated_updates_patterns = PatternMatcherPass("convrot_swiglu_ffn_gated_updates")
_patterns = PatternMatcherPass("convrot_swiglu_ffn")
for _explicit_up_activation in (False, True):
    for _use_aten_index in (False, True):
        register_graph_pattern(
            swiglu_ffn_pattern.gated_updates_pattern(
                _normalized_ffn_pattern(explicit_up_activation=_explicit_up_activation),
                use_aten_index=_use_aten_index,
            ),
            extra_check=_valid_gated_updates,
            pass_dict=_gated_updates_patterns,  # pyright: ignore[reportArgumentType]
        )(_replace_normalized_ffn_gated_updates)
    register_graph_pattern(
        _normalized_ffn_pattern(explicit_up_activation=_explicit_up_activation),
        extra_check=_valid_normalized_ffn,
        pass_dict=_patterns,  # pyright: ignore[reportArgumentType]
    )(_replace_normalized_ffn)


def _fold_chunked_ffn(graph: torch.fx.Graph) -> bool:
    changes = _gated_updates_patterns.apply(graph)
    changes += _patterns.apply(graph)
    changed = changes > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold normalized ConvRot SwiGLU FFNs after ordinary ConvRot rewriting."""

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


def convrot_swiglu_ffn_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install chunked FFN folding immediately after ordinary ConvRot folding."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (convrot_compile.compile_pass, compile_pass),
    )


__all__ = ["convrot_swiglu_ffn_compile_options"]
