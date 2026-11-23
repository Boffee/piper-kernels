"""ConvRot INT8 inference graph optimizations for Inductor."""

from __future__ import annotations

import torch

from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _input_activation_compile as input_activation_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _compile_fx, _distributed_compile

type _PreparedInputNodes = _compile_fx.PreparedInputNodes
_LINEAR_TARGET = torch.ops.piper_kernels.convrot_int8_linear.default


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


def _bind_unactivated_linear(
    node: torch.fx.Node,
) -> (
    tuple[
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node | None,
        int,
        torch.fx.Node | None,
    ]
    | None
):
    bound = linear_fx.bind_call(node, _LINEAR_TARGET)
    if bound is None:
        return None
    input_node = bound["input"]
    weight_qdata = bound["weight_qdata"]
    weight_scale = bound["weight_scale"]
    bias = bound["bias"]
    group_size = bound["group_size"]
    activation_fn = bound["activation_fn"]
    input_scale = bound["input_scale"]
    if (
        not isinstance(input_node, torch.fx.Node)
        or not isinstance(weight_qdata, torch.fx.Node)
        or not isinstance(weight_scale, torch.fx.Node)
        or (bias is not None and not isinstance(bias, torch.fx.Node))
        or not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or activation_fn is not None
        or (input_scale is not None and not isinstance(input_scale, torch.fx.Node))
    ):
        return None
    return input_node, weight_qdata, weight_scale, bias, group_size, input_scale


class _PreparationRule:
    """Describe how semantic ConvRot linears share prepared inputs."""

    linear_target = _LINEAR_TARGET

    def match_key(
        self,
        node: torch.fx.Node,
    ) -> preparation_sharing.PreparationMatchKey | None:
        parsed = _bind_unactivated_linear(node)
        if parsed is None:
            return None
        input_node, weight_qdata, _weight_scale, _bias, group_size, input_scale = parsed

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
            input_scale,
        )
        return key, key

    def prepare(
        self,
        graph: torch.fx.Graph,
        first: torch.fx.Node,
    ) -> _PreparedInputNodes:
        parsed = _bind_unactivated_linear(first)
        assert parsed is not None
        input_node, _weight_qdata, _weight_scale, _bias, group_size, input_scale = parsed
        input_value = preparation_sharing.tensor_metadata(input_node)
        assert input_value is not None
        return _compile_fx.emit_prepared_input(
            graph,
            input_node,
            group_size,
            None,
            tuple(input_value.shape),
            input_scale,
        )

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: _PreparedInputNodes,
    ) -> torch.fx.Node:
        parsed = _bind_unactivated_linear(node)
        assert parsed is not None
        _input, weight_qdata, weight_scale, bias, _group_size, _input_scale = parsed
        return _emit_linear_prepared(
            graph,
            prepared,
            weight_qdata,
            weight_scale,
            bias,
        )


_PREPARATION_RULES = (_PreparationRule(),)


def _fold_gelu_tanh_inputs(graph: torch.fx.Graph) -> bool:
    """Fold exclusive GELU-tanh decompositions into prepared ConvRot INT8 inputs."""
    changed = False
    for node in list(graph.nodes):
        parsed = _bind_unactivated_linear(node)
        if parsed is None:
            continue
        activated, weight_qdata, weight_scale, bias, group_size, input_scale = parsed
        gelu = input_activation_compile.match_gelu_tanh_input(activated)
        if gelu is None:
            continue
        source, logical_dtype = gelu
        weight_value = preparation_sharing.tensor_metadata(weight_qdata)
        if weight_value is None or weight_value.ndim != 2:
            continue
        if not input_activation_compile.valid_gelu_tanh_input(
            source,
            logical_dtype,
            input_features=weight_value.shape[1],
        ):
            continue
        source_value = preparation_sharing.tensor_metadata(source)
        assert source_value is not None
        with graph.inserting_before(node):
            prepared = _compile_fx.emit_prepared_input(
                graph,
                source,
                group_size,
                "gelu_tanh",
                tuple(source_value.shape),
                input_scale,
            )
            replacement = _emit_linear_prepared(
                graph,
                prepared,
                weight_qdata,
                weight_scale,
                bias,
            )
        replacement.meta = node.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


def fold_linear(graph: torch.fx.Graph) -> bool:
    """Fold GELU-tanh inputs, share INT8 preparation, and hoist quantized all-gathers."""
    changed = _fold_gelu_tanh_inputs(graph)
    changed = preparation_sharing.share_preparation(graph, _PREPARATION_RULES) or changed
    changed = _distributed_compile.fold_quantized_all_gather(graph) or changed
    return changed


__all__ = ["fold_linear"]
