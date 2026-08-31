"""NVFP4 inference graph optimizations for Inductor."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)

from piper_kernels.linear import _preparation_sharing as preparation_sharing

from . import _compile_fx, _ops

_COMPILE_PASS_VERSION = "nvfp4-compile-v1"
type _PreparedInputNodes = _compile_fx.PreparedInputNodes


def _arguments(node: torch.fx.Node) -> tuple[object, ...] | None:
    if node.kwargs or len(node.args) != 7:
        return None
    return node.args


class _PreparationRule:
    """Describe how compatible semantic NVFP4 linears share prepared inputs."""

    linear_target = torch.ops.piper_kernels.nvfp4_linear.default

    def match_key(self, node: torch.fx.Node) -> Hashable | None:
        arguments = _arguments(node)
        if arguments is None:
            return None
        (
            input_node,
            weight_qdata,
            weight_scale,
            weight_per_tensor_scale,
            activation_per_tensor_scale,
            bias,
            dynamic_activation_scale,
        ) = arguments
        if (
            not isinstance(input_node, torch.fx.Node)
            or not isinstance(weight_qdata, torch.fx.Node)
            or not isinstance(weight_scale, torch.fx.Node)
            or (
                weight_per_tensor_scale is not None
                and not isinstance(weight_per_tensor_scale, torch.fx.Node)
            )
            or (
                activation_per_tensor_scale is not None
                and not isinstance(activation_per_tensor_scale, torch.fx.Node)
            )
            or (bias is not None and not isinstance(bias, torch.fx.Node))
            or not isinstance(dynamic_activation_scale, bool)
        ):
            return None
        input_value = preparation_sharing.tensor_metadata(input_node)
        weight_value = preparation_sharing.tensor_metadata(weight_qdata)
        weight_scale_value = preparation_sharing.tensor_metadata(weight_scale)
        if (
            input_value is None
            or input_value.ndim == 0
            or weight_value is None
            or weight_value.ndim != 2
            or weight_value.dtype is not torch.uint8
            or not isinstance(input_value.shape[-1], int)
            or input_value.shape[-1] % 16 != 0
            or preparation_sharing.dimension_key(input_value.shape[-1])
            != preparation_sharing.dimension_key(2 * weight_value.shape[1])
            or weight_scale_value is None
            or weight_scale_value.dtype is not torch.float8_e4m3fn
        ):
            return None
        # Exact FX identity prevents sharing across functionalized mutations and
        # weights calibrated with different static activation scales.
        return (
            input_node,
            activation_per_tensor_scale,
            dynamic_activation_scale,
            preparation_sharing.dimension_key(weight_value.shape[1]),
            input_value.dtype,
        )

    def prepare(
        self,
        graph: torch.fx.Graph,
        first: torch.fx.Node,
    ) -> _PreparedInputNodes:
        arguments = _arguments(first)
        assert arguments is not None
        input_node = arguments[0]
        activation_per_tensor_scale = arguments[4]
        dynamic_activation_scale = arguments[6]
        assert isinstance(input_node, torch.fx.Node)
        assert activation_per_tensor_scale is None or isinstance(
            activation_per_tensor_scale, torch.fx.Node
        )
        assert isinstance(dynamic_activation_scale, bool)
        return _compile_fx.emit_prepared_input(
            graph,
            input_node,
            activation_per_tensor_scale,
            dynamic_activation_scale,
        )

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: _PreparedInputNodes,
    ) -> torch.fx.Node:
        arguments = _arguments(node)
        assert arguments is not None
        input_qdata, input_scale, input_per_tensor_scale, leading_shape = prepared
        weight_qdata = arguments[1]
        weight_scale = arguments[2]
        weight_per_tensor_scale = arguments[3]
        bias = arguments[5]
        assert isinstance(weight_qdata, torch.fx.Node)
        assert isinstance(weight_scale, torch.fx.Node)
        input_node = arguments[0]
        assert isinstance(input_node, torch.fx.Node)
        assert weight_per_tensor_scale is None or isinstance(weight_per_tensor_scale, torch.fx.Node)
        assert bias is None or isinstance(bias, torch.fx.Node)
        input_value = preparation_sharing.tensor_metadata(input_node)
        weight_value = preparation_sharing.tensor_metadata(weight_qdata)
        assert input_value is not None
        assert weight_value is not None
        projected = graph.call_function(
            torch.ops.piper_kernels.nvfp4_linear_prepared.default,
            args=(
                input_qdata,
                input_scale,
                input_per_tensor_scale,
                weight_qdata,
                weight_scale,
                weight_per_tensor_scale,
                bias,
                input_value.dtype,
            ),
        )
        projected.meta["val"] = input_value.new_empty(
            (input_qdata.meta["val"].shape[0], weight_value.shape[0])
        )
        if len(leading_shape) == 1:
            return projected
        return graph.call_function(
            torch.ops.aten.reshape.default,
            args=(projected, (*leading_shape, weight_value.shape[0])),
        )


_PREPARATION_RULES = (_PreparationRule(),)


class _CompilePass(CustomInferenceAwareGraphPass):
    """Share compatible NVFP4 activation preparation in inference graphs."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            preparation_sharing.share_preparation(graph, _PREPARATION_RULES)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            (__file__, preparation_sharing.__file__, _compile_fx.__file__, _ops.__file__),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def nvfp4_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return Inductor options that share compatible NVFP4 preparation."""
    return preparation_sharing.add_post_grad_pass(options, compile_pass)


__all__ = ["nvfp4_compile_options"]
