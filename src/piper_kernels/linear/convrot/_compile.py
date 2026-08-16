"""ConvRot rule for automatic preparation sharing under Inductor."""

from __future__ import annotations

import operator
from collections.abc import Hashable, Mapping

import torch

from piper_kernels.linear._preparation_sharing import (
    PreparationSharingPass,
    add_post_grad_pass,
    dimension_key,
    tensor_metadata,
)

_PASS_VERSION = "convrot-shared-input-preparation-v4"

type _PreparedInputNodes = tuple[torch.fx.Node, torch.fx.Node, torch.dtype]


class _ConvRotPreparationRule:
    """Describe how semantic ConvRot linears share prepared inputs."""

    linear_target = torch.ops.piper_kernels.convrot_int8_linear.default

    def match_key(self, node: torch.fx.Node) -> Hashable | None:
        if node.kwargs or len(node.args) != 5:
            return None
        input_node, weight_qdata, _weight_scale, _bias, group_size = node.args
        if (
            not isinstance(input_node, torch.fx.Node)
            or not isinstance(weight_qdata, torch.fx.Node)
            or not isinstance(group_size, int)
            or isinstance(group_size, bool)
        ):
            return None

        input_value = tensor_metadata(input_node)
        weight_qdata_value = tensor_metadata(weight_qdata)
        if (
            input_value is None
            or input_value.ndim == 0
            or weight_qdata_value is None
            or weight_qdata_value.ndim != 2
        ):
            return None

        # Exact FX identity prevents sharing across functionalized mutations.
        return (
            input_node,
            group_size,
            dimension_key(weight_qdata_value.shape[1]),
            input_value.dtype,
        )

    def prepare(
        self,
        graph: torch.fx.Graph,
        first: torch.fx.Node,
    ) -> _PreparedInputNodes:
        input_node, _weight_qdata, _weight_scale, _bias, group_size = first.args
        assert isinstance(input_node, torch.fx.Node)
        input_value = tensor_metadata(input_node)
        assert input_value is not None

        input_qdata_value = input_value.new_empty(input_value.shape, dtype=torch.int8)
        input_scale_value = input_value.new_empty(
            input_value.shape[:-1],
            dtype=torch.float32,
        )
        prepared = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_prepare_input.default,
            args=(input_node, group_size),
        )
        prepared.meta["val"] = (input_qdata_value, input_scale_value)
        input_qdata = graph.call_function(operator.getitem, args=(prepared, 0))
        input_qdata.meta["val"] = input_qdata_value
        input_scale = graph.call_function(operator.getitem, args=(prepared, 1))
        input_scale.meta["val"] = input_scale_value
        return input_qdata, input_scale, input_value.dtype

    def replace(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        prepared: _PreparedInputNodes,
    ) -> torch.fx.Node:
        _input, weight_qdata, weight_scale, bias, group_size = node.args
        input_qdata, input_scale, logical_dtype = prepared
        return graph.call_function(
            torch.ops.piper_kernels.convrot_int8_linear_prepared.default,
            args=(
                input_qdata,
                input_scale,
                weight_qdata,
                weight_scale,
                bias,
                group_size,
                logical_dtype,
            ),
        )


share_preparation_pass = PreparationSharingPass(
    (_ConvRotPreparationRule(),),
    version=_PASS_VERSION,
    source_files=(__file__,),
)


def convrot_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return Inductor options with automatic ConvRot preparation reuse.

    Existing post-grad pre-passes run first and are preserved. Reapplying this
    helper is idempotent, and the input mapping and any contained list remain
    unchanged. Pass these options to ``torch.compile`` without also supplying
    its mutually exclusive ``mode`` argument.
    """
    return add_post_grad_pass(options, share_preparation_pass)


__all__ = ["convrot_compile_options"]
