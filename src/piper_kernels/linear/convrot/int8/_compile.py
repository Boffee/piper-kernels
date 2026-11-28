"""ConvRot INT8 inference graph optimizations for Inductor."""

from __future__ import annotations

import torch

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


class _PreparationRule:
    """Describe how semantic ConvRot linears share prepared inputs."""

    linear_target = _LINEAR_TARGET

    def match_key(
        self,
        node: torch.fx.Node,
    ) -> preparation_sharing.PreparationMatchKey | None:
        bound = _compile_fx.BoundLinearNodes.from_call(node)
        if bound is None:
            return None

        input_value = preparation_sharing.tensor_metadata(bound.input)
        weight_qdata_value = preparation_sharing.tensor_metadata(bound.weight_qdata)
        if (
            input_value is None
            or input_value.ndim == 0
            or weight_qdata_value is None
            or weight_qdata_value.ndim != 2
        ):
            return None

        # Exact FX identity prevents sharing across functionalized mutations.
        key = (
            bound.input,
            bound.group_size,
            preparation_sharing.dimension_key(weight_qdata_value.shape[1]),
            input_value.dtype,
            bound.input_scale,
        )
        return key, key

    def prepare(
        self,
        graph: torch.fx.Graph,
        first: torch.fx.Node,
    ) -> _PreparedInputNodes:
        bound = _compile_fx.BoundLinearNodes.from_call(first)
        assert bound is not None
        input_value = preparation_sharing.tensor_metadata(bound.input)
        assert input_value is not None
        return _compile_fx.emit_prepared_input(
            graph,
            bound.input,
            bound.group_size,
            None,
            tuple(input_value.shape),
            bound.input_scale,
        )

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: _PreparedInputNodes,
    ) -> torch.fx.Node:
        bound = _compile_fx.BoundLinearNodes.from_call(node)
        assert bound is not None
        return _emit_linear_prepared(
            graph,
            prepared,
            bound.weight_qdata,
            bound.weight_scale,
            bound.bias,
        )


_PREPARATION_RULES = (_PreparationRule(),)


def _fold_gelu_tanh_inputs(graph: torch.fx.Graph) -> bool:
    """Fold exclusive GELU-tanh decompositions into prepared ConvRot INT8 inputs."""
    changed = False
    for node in list(graph.nodes):
        bound = _compile_fx.BoundLinearNodes.from_call(node)
        if bound is None:
            continue
        gelu = input_activation_compile.match_gelu_tanh_input(bound.input)
        if gelu is None:
            continue
        source, logical_dtype = gelu
        weight_value = preparation_sharing.tensor_metadata(bound.weight_qdata)
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
                bound.group_size,
                "gelu_tanh",
                tuple(source_value.shape),
                bound.input_scale,
            )
            replacement = _emit_linear_prepared(
                graph,
                prepared,
                bound.weight_qdata,
                bound.weight_scale,
                bound.bias,
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
