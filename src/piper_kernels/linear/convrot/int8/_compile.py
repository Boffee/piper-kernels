"""ConvRot INT8 inference graph optimizations for Inductor."""

from __future__ import annotations

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

from piper_kernels.linear import _input_activation_compile as input_activation_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot import triton as convrot_triton

from . import _compile_fx

_COMPILE_PASS_VERSION = "convrot-compile-v6"
type _PreparedInputNodes = _compile_fx.PreparedInputNodes


def _emit_linear_prepared(
    graph: torch.fx.Graph,
    prepared: _PreparedInputNodes,
    weight_qdata: torch.fx.Node,
    weight_scale: torch.fx.Node,
    bias: torch.fx.Node | None,
) -> torch.fx.Node:
    input_qdata, input_scale, logical_dtype = prepared
    return graph.call_function(
        torch.ops.piper_kernels.convrot_int8_linear_prepared.default,
        args=(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            logical_dtype,
        ),
    )


class _PreparationRule:
    """Describe how semantic ConvRot linears share prepared inputs."""

    linear_target = torch.ops.piper_kernels.convrot_int8_linear.default

    def match_key(
        self,
        node: torch.fx.Node,
    ) -> preparation_sharing.PreparationMatchKey | None:
        if node.kwargs or len(node.args) not in (5, 6):
            return None
        arguments = (*node.args, None) if len(node.args) == 5 else node.args
        input_node, weight_qdata, _weight_scale, _bias, group_size, activation_fn = arguments
        if (
            not isinstance(input_node, torch.fx.Node)
            or not isinstance(weight_qdata, torch.fx.Node)
            or not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or activation_fn is not None
        ):
            return None

        input_value = preparation_sharing.tensor_metadata(input_node)
        weight_qdata_value = preparation_sharing.tensor_metadata(weight_qdata)
        if (
            input_value is None
            or input_value.ndim == 0
            or weight_qdata_value is None
            or weight_qdata_value.ndim != 2
        ):
            return None

        # Exact FX identity prevents sharing across functionalized mutations.
        key = (
            input_node,
            group_size,
            preparation_sharing.dimension_key(weight_qdata_value.shape[1]),
            input_value.dtype,
        )
        return key, key

    def prepare(
        self,
        graph: torch.fx.Graph,
        first: torch.fx.Node,
    ) -> _PreparedInputNodes:
        arguments = (*first.args, None) if len(first.args) == 5 else first.args
        input_node, _weight_qdata, _weight_scale, _bias, group_size, _activation_fn = arguments
        assert isinstance(input_node, torch.fx.Node)
        assert isinstance(group_size, int)
        assert not isinstance(group_size, bool)
        input_value = preparation_sharing.tensor_metadata(input_node)
        assert input_value is not None
        return _compile_fx.emit_prepared_input(
            graph,
            input_node,
            group_size,
            None,
            tuple(input_value.shape),
        )

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: _PreparedInputNodes,
    ) -> torch.fx.Node:
        arguments = (*node.args, None) if len(node.args) == 5 else node.args
        _input, weight_qdata, weight_scale, bias, _group_size, _activation_fn = arguments
        assert isinstance(weight_qdata, torch.fx.Node)
        assert isinstance(weight_scale, torch.fx.Node)
        assert bias is None or isinstance(bias, torch.fx.Node)
        return _emit_linear_prepared(
            graph,
            prepared,
            weight_qdata,
            weight_scale,
            bias,
        )


_PREPARATION_RULES = (_PreparationRule(),)


_gelu_tanh_patterns = PatternMatcherPass("convrot_gelu_tanh_inputs")


def _linear_pattern(input_pattern: CallFunction) -> CallFunction:
    return CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        input_pattern,
        KeywordArg("weight_qdata"),
        KeywordArg("weight_scale"),
        KeywordArg("bias"),
        KeywordArg("group_size"),
    )


def _valid_gelu_tanh(match: Match, *, promote_input: bool) -> bool:
    weight_qdata = match.kwargs["weight_qdata"]
    if not isinstance(weight_qdata, torch.fx.Node):
        return False
    weight_value = preparation_sharing.tensor_metadata(weight_qdata)
    if weight_value is None or weight_value.ndim != 2:
        return False
    return input_activation_compile.valid_gelu_tanh(
        match,
        promote_input=promote_input,
        input_features=weight_value.shape[1],
    )


def _replace_gelu_tanh(
    match: Match,
    input: torch.fx.Node,  # noqa: A002 - pattern keyword
    weight_qdata: torch.fx.Node,
    weight_scale: torch.fx.Node,
    bias: torch.fx.Node | None,
    group_size: int,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    input_value = preparation_sharing.tensor_metadata(input)
    assert input_value is not None
    with graph.inserting_before(original):
        prepared = _compile_fx.emit_prepared_input(
            graph,
            input,
            group_size,
            "gelu_tanh",
            tuple(input_value.shape),
        )
        replacement = _emit_linear_prepared(
            graph,
            prepared,
            weight_qdata,
            weight_scale,
            bias,
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


for _promote_input in (False, True):
    register_graph_pattern(
        input_activation_compile.gelu_tanh_pattern(
            _linear_pattern,
            promote_input=_promote_input,
        ),
        extra_check=lambda match, promote_input=_promote_input: _valid_gelu_tanh(
            match,
            promote_input=promote_input,
        ),
        pass_dict=_gelu_tanh_patterns,  # pyright: ignore[reportArgumentType]
    )(_replace_gelu_tanh)


def _fold_gelu_tanh_inputs(graph: torch.fx.Graph) -> bool:
    changed = _gelu_tanh_patterns.apply(graph) > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Fold GELU-tanh before sharing ordinary ConvRot input preparation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        _fold_gelu_tanh_inputs(graph)
        preparation_sharing.share_preparation(graph, _PREPARATION_RULES)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            (
                __file__,
                input_activation_compile.__file__,
                preparation_sharing.__file__,
                convrot_triton.__file__,
                _compile_fx.__file__,
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def convrot_int8_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return Inductor options for ConvRot inference graph optimizations.

    Existing post-grad pre-passes run first and are preserved. Reapplying this
    helper is idempotent, and the input mapping and any contained list remain
    unchanged. Pass these options to ``torch.compile`` without also supplying
    its mutually exclusive ``mode`` argument.
    """
    return preparation_sharing.add_post_grad_pass(options, compile_pass)


__all__ = ["convrot_int8_compile_options"]
