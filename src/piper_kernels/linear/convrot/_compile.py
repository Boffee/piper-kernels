"""ConvRot inference graph optimizations for Inductor."""

from __future__ import annotations

import operator
from collections.abc import Hashable, Mapping

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

from piper_kernels.linear import _preparation_sharing as preparation_sharing

_COMPILE_PASS_VERSION = "convrot-compile-v2"

type _PreparedInputNodes = tuple[torch.fx.Node, torch.fx.Node, torch.dtype]


def _emit_prepared_input(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    group_size: int,
    activation_fn: str | None,
    prepared_shape: tuple[int | torch.SymInt, ...],
) -> _PreparedInputNodes:
    input_value = preparation_sharing.tensor_metadata(input_node)
    assert input_value is not None
    input_qdata_value = input_value.new_empty(prepared_shape, dtype=torch.int8)
    input_scale_value = input_value.new_empty(prepared_shape[:-1], dtype=torch.float32)
    prepared = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_prepare_input.default,
        args=(input_node, group_size, activation_fn),
    )
    prepared.meta["val"] = (input_qdata_value, input_scale_value)
    input_qdata = graph.call_function(operator.getitem, args=(prepared, 0))
    input_qdata.meta["val"] = input_qdata_value
    input_scale = graph.call_function(operator.getitem, args=(prepared, 1))
    input_scale.meta["val"] = input_scale_value
    return input_qdata, input_scale, input_value.dtype


def _emit_linear_prepared(
    graph: torch.fx.Graph,
    prepared: _PreparedInputNodes,
    weight_qdata: torch.fx.Node,
    weight_scale: torch.fx.Node,
    bias: torch.fx.Node | None,
    group_size: int,
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
            group_size,
            logical_dtype,
        ),
    )


class _ConvRotPreparationRule:
    """Describe how semantic ConvRot linears share prepared inputs."""

    linear_target = torch.ops.piper_kernels.convrot_int8_linear.default

    def match_key(self, node: torch.fx.Node) -> Hashable | None:
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
        return (
            input_node,
            group_size,
            preparation_sharing.dimension_key(weight_qdata_value.shape[1]),
            input_value.dtype,
        )

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
        return _emit_prepared_input(
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
        _input, weight_qdata, weight_scale, bias, group_size, _activation_fn = arguments
        assert isinstance(weight_qdata, torch.fx.Node)
        assert isinstance(weight_scale, torch.fx.Node)
        assert bias is None or isinstance(bias, torch.fx.Node)
        assert isinstance(group_size, int)
        assert not isinstance(group_size, bool)
        return _emit_linear_prepared(
            graph,
            prepared,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
        )


_PREPARATION_RULES = (_ConvRotPreparationRule(),)


_packed_swiglu_patterns = PatternMatcherPass("convrot_packed_swiglu")


def _packed_swiglu_pattern(
    *,
    promote_gate: bool | None,
    reverse_multiply: bool,
) -> CallFunction:
    """Build one exclusive packed-SwiGLU pattern in Inductor's normalized IR."""
    split = CallFunction(
        torch.ops.aten.split.Tensor,
        KeywordArg("packed"),
        KeywordArg("split_size"),
        -1,
        _users=2,
    )
    up = CallFunction(operator.getitem, split, 0, _users=1)
    gate_users = 2 if promote_gate is False else 1
    gate = CallFunction(operator.getitem, split, 1, _users=gate_users)
    if promote_gate is None:
        silu = CallFunction(torch.ops.aten.silu.default, gate, _users=1)
    else:
        gate_value = (
            CallFunction(
                torch.ops.prims.convert_element_type.default,
                gate,
                torch.float32,
                _users=2,
            )
            if promote_gate
            else gate
        )
        silu = CallFunction(
            torch.ops.aten.div.Tensor,
            gate_value,
            CallFunction(
                torch.ops.aten.add.Tensor,
                CallFunction(
                    torch.ops.aten.exp.default,
                    CallFunction(torch.ops.aten.neg.default, gate_value, _users=1),
                    _users=1,
                ),
                1,
                _users=1,
            ),
            _users=1,
        )
        if promote_gate:
            silu = CallFunction(
                torch.ops.prims.convert_element_type.default,
                silu,
                KeywordArg("logical_dtype"),
                _users=1,
            )
    multiply_args = (silu, up) if reverse_multiply else (up, silu)
    return CallFunction(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        CallFunction(torch.ops.aten.mul.Tensor, *multiply_args, _users=1),
        KeywordArg("weight_qdata"),
        KeywordArg("weight_scale"),
        KeywordArg("bias"),
        KeywordArg("group_size"),
    )


def _valid_packed_swiglu(match: Match, *, promote_gate: bool | None) -> bool:
    packed = match.kwargs["packed"]
    weight_qdata = match.kwargs["weight_qdata"]
    split_size = match.kwargs["split_size"]
    if not isinstance(packed, torch.fx.Node) or not isinstance(weight_qdata, torch.fx.Node):
        return False
    packed_value = preparation_sharing.tensor_metadata(packed)
    weight_value = preparation_sharing.tensor_metadata(weight_qdata)
    if (
        packed_value is None
        or packed_value.ndim == 0
        or weight_value is None
        or weight_value.ndim != 2
        or isinstance(split_size, bool)
        or not isinstance(split_size, (int, torch.SymInt))
    ):
        return False
    in_features = weight_value.shape[1]
    dimensions_match = preparation_sharing.dimension_key(
        split_size
    ) == preparation_sharing.dimension_key(in_features) and preparation_sharing.dimension_key(
        packed_value.shape[-1]
    ) == preparation_sharing.dimension_key(2 * in_features)
    if promote_gate is True:
        return dimensions_match and match.kwargs["logical_dtype"] is packed_value.dtype
    if promote_gate is False:
        return dimensions_match and packed_value.dtype is torch.float32
    return dimensions_match


def _replace_packed_swiglu(
    match: Match,
    packed: torch.fx.Node,
    weight_qdata: torch.fx.Node,
    weight_scale: torch.fx.Node,
    bias: torch.fx.Node | None,
    group_size: int,
    **_unused: object,
) -> None:
    original = match.output_node()
    graph = match.graph
    packed_value = preparation_sharing.tensor_metadata(packed)
    assert packed_value is not None
    input_shape = (*packed_value.shape[:-1], packed_value.shape[-1] // 2)
    with graph.inserting_before(original):
        prepared = _emit_prepared_input(
            graph,
            packed,
            group_size,
            "swiglu",
            input_shape,
        )
        replacement = _emit_linear_prepared(
            graph,
            prepared,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
        )
    replacement.meta = original.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    original.replace_all_uses_with(replacement)
    match.erase_nodes()


# SiLU may remain direct, lower without casts for FP32, or lower through FP32
# for FP16/BF16. Multiplication is commutative, so accept either operand order.
for _promote_gate in (None, False, True):
    for _reverse_multiply in (False, True):
        register_graph_pattern(
            _packed_swiglu_pattern(
                promote_gate=_promote_gate,
                reverse_multiply=_reverse_multiply,
            ),
            extra_check=lambda match, promote_gate=_promote_gate: _valid_packed_swiglu(
                match,
                promote_gate=promote_gate,
            ),
            # PyTorch's concrete pass has a parameter-name-only protocol mismatch.
            pass_dict=_packed_swiglu_patterns,  # pyright: ignore[reportArgumentType]
        )(_replace_packed_swiglu)


def _fold_packed_swiglu(graph: torch.fx.Graph) -> bool:
    changed = _packed_swiglu_patterns.apply(graph) > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


class _ConvRotCompilePass(CustomInferenceAwareGraphPass):
    """Apply ConvRot graph folds before sharing ordinary input preparation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        _fold_packed_swiglu(graph)
        preparation_sharing.share_preparation(graph, _PREPARATION_RULES)

    def uuid(self) -> bytes:
        return get_hash_for_files(
            (__file__, preparation_sharing.__file__),
            extra=_COMPILE_PASS_VERSION,
        )


convrot_compile_pass = _ConvRotCompilePass()


def convrot_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return Inductor options for packed SwiGLU folding and preparation reuse.

    Existing post-grad pre-passes run first and are preserved. Reapplying this
    helper is idempotent, and the input mapping and any contained list remain
    unchanged. Pass these options to ``torch.compile`` without also supplying
    its mutually exclusive ``mode`` argument.
    """
    return preparation_sharing.add_post_grad_pass(options, convrot_compile_pass)


__all__ = ["convrot_compile_options"]
