"""Inductor graph integration for the MiniMax-H3 video VAE specialization."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot.int8 import _compile as convrot_int8_compile
from piper_kernels.linear.convrot.int8 import _compile_fx as convrot_int8_compile_fx
from piper_kernels.linear.convrot.int8._nvidia import triton as convrot_int8_backend

from . import _ops

_COMPILE_PASS_VERSION = "minimax-h3-vae-convrot-int8-compile-v3"

type _LinearShape = tuple[int, int, int]
type _MatmulSchedule = tuple[int, int, int, int, int]

# Rows, output features, input features. These schedules apply only to the fixed
# 256x256 H3 VAE tile on exact SM120. The first shape covers the attention
# projection linears; this pass never replaces the attention operator itself.
_SM120_MATMUL_SCHEDULES: dict[_LinearShape, _MatmulSchedule] = {
    (1_797, 2_048, 2_048): (128, 128, 128, 4, 2),
    (1_797, 16_384, 2_048): (128, 128, 64, 8, 3),
    (1_797, 2_048, 8_192): (128, 128, 128, 4, 2),
    (3_594, 2_048, 2_048): (128, 128, 64, 8, 3),
    (3_594, 16_384, 2_048): (128, 128, 64, 8, 3),
    (3_594, 2_048, 8_192): (128, 128, 64, 8, 3),
    (7_188, 2_048, 2_048): (128, 128, 128, 4, 2),
    (7_188, 16_384, 2_048): (128, 128, 64, 8, 3),
    (7_188, 2_048, 8_192): (128, 128, 128, 4, 2),
}

_LINEAR_TARGET = torch.ops.piper_kernels.convrot_int8_linear.default
_PREPARED_TARGET = torch.ops.piper_kernels.convrot_int8_linear_prepared.default
_SPECIALIZED_PREPARED_TARGET = (
    torch.ops.piper_kernels.minimax_h3_vae_convrot_int8_linear_prepared.default
)


def _static_linear_shape(
    node: torch.fx.Node,
    *,
    weight_index: int,
) -> tuple[_LinearShape, torch.Tensor, torch.Tensor] | None:
    if node.kwargs or len(node.args) <= weight_index:
        return None
    input_node = node.args[0]
    weight_node = node.args[weight_index]
    if not isinstance(input_node, torch.fx.Node) or not isinstance(weight_node, torch.fx.Node):
        return None
    input_value = preparation_sharing.tensor_metadata(input_node)
    weight_value = preparation_sharing.tensor_metadata(weight_node)
    if (
        input_value is None
        or input_value.ndim == 0
        or weight_value is None
        or weight_value.ndim != 2
        or any(type(dimension) is not int for dimension in input_value.shape[:-1])
        or any(type(dimension) is not int for dimension in weight_value.shape)
    ):
        return None
    rows = math.prod(input_value.shape[:-1])
    out_features, in_features = weight_value.shape
    assert isinstance(rows, int)
    assert isinstance(out_features, int)
    assert isinstance(in_features, int)
    return (rows, out_features, in_features), input_value, weight_value


def _schedule_for(
    shape: _LinearShape,
    *,
    target: AcceleratorTarget,
) -> _MatmulSchedule | None:
    if not target.is_cuda_capability(12, 0):
        return None
    return _SM120_MATMUL_SCHEDULES.get(shape)


def _specialize_linears(
    graph: torch.fx.Graph,
    *,
    target: AcceleratorTarget | None = None,
) -> bool:
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target == _LINEAR_TARGET:
            weight_index = 1
        elif node.target == _PREPARED_TARGET:
            weight_index = 2
        else:
            continue
        shape_and_device = _static_linear_shape(node, weight_index=weight_index)
        if shape_and_device is None:
            continue
        shape, input_value, weight_value = shape_and_device
        schedule = _schedule_for(
            shape,
            target=(
                target if target is not None else AcceleratorTarget.from_device(weight_value.device)
            ),
        )
        if schedule is None:
            continue
        with graph.inserting_before(node):
            if node.target == _LINEAR_TARGET:
                arguments = (*node.args, None) if len(node.args) == 5 else node.args
                input_node, weight_qdata, weight_scale, bias, group_size, activation_fn = arguments
                assert isinstance(input_node, torch.fx.Node)
                assert isinstance(weight_qdata, torch.fx.Node)
                assert isinstance(weight_scale, torch.fx.Node)
                assert bias is None or isinstance(bias, torch.fx.Node)
                assert isinstance(group_size, int)
                assert activation_fn is None or isinstance(activation_fn, str)
                prepared = convrot_int8_compile_fx.emit_prepared_input(
                    graph,
                    input_node,
                    group_size,
                    activation_fn,
                    (*input_value.shape[:-1], weight_value.shape[1]),
                )
                input_qdata, input_scale, logical_dtype = prepared
                replacement_args = (
                    input_qdata,
                    input_scale,
                    weight_qdata,
                    weight_scale,
                    bias,
                    logical_dtype,
                    list(schedule),
                )
            else:
                replacement_args = (*node.args, list(schedule))
            replacement = graph.call_function(
                _SPECIALIZED_PREPARED_TARGET,
                args=replacement_args,
            )
        replacement.meta = node.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
    return changed


class _CompilePass(CustomInferenceAwareGraphPass):
    """Apply measured schedules without changing the H3 attention implementation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            _specialize_linears(graph)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            (
                __file__,
                _ops.__file__,
                convrot_int8_backend.__file__,
                convrot_int8_compile_fx.__file__,
            ),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def minimax_h3_vae_convrot_int8_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install ordinary ConvRot rewriting followed by the H3 VAE schedules."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (convrot_int8_compile.compile_pass, compile_pass),
    )


__all__ = ["minimax_h3_vae_convrot_int8_compile_options"]
